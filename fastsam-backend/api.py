"""FastAPI transport for the segmentation pipeline."""

import base64
import io
import time

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sam_runtime import *
from segmentation_policy import *
from mask_ops import *
from candidate_selection import *
from matte_ops import *
from segmentation_core import *
from sam_diagnostics import sam_diagnostic_request, sam_diagnostic_event

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
    return {"ok": True}

@app.post("/segment")
async def segment(request: Request):
    try:
        data = await request.json()
        request_id = str(
            data.get("taskId") or data.get("requestId") or
            f"segment-{int(time.time() * 1000)}"
        )
        requested_engine = normalize_requested_engine(data.get("engine"))
        image_b64 = data.get("image")
        bboxes_norm = data.get("bboxes", [])
        layer_ids = data.get("layerIds", [])
        layer_metas = data.get("layers", [])
        context_layers = data.get("contextLayers", layer_metas)
        quality_profile = normalize_sam_quality_profile(data.get("qualityProfile"))
        print(
            f"SAM request start id={request_id} profile={quality_profile} "
            f"engine={requested_engine} requestedLayers={len(layer_ids)} "
            f"contextLayers={len(context_layers) if isinstance(context_layers, list) else 0}"
        )

        if not image_b64:
            return JSONResponse(status_code=400, content={"error": "No image provided"})

        img = base64_to_cv2(image_b64)
        print(f"SAM request image id={request_id} sourceSize={img.shape[1]}x{img.shape[0]}")
        h, w = img.shape[:2]

        # Determine if we should use bounding box prompts
        if bboxes_norm and len(bboxes_norm) > 0:
            print(f"Using {len(bboxes_norm)} bounding box prompts with engine={requested_engine}")
            # Convert 0-1000 normalized bboxes back to pixel coordinates [x1, y1, x2, y2]
            pixel_bboxes = []
            original_target_bboxes = []
            sam_prompt_bboxes = []
            for bbox in bboxes_norm:
                ymin_n, xmin_n, ymax_n, xmax_n = bbox
                y1 = int((ymin_n / 1000.0) * h)
                x1 = int((xmin_n / 1000.0) * w)
                y2 = int((ymax_n / 1000.0) * h)
                x2 = int((xmax_n / 1000.0) * w)
                output_bbox = [
                    clamp(x1, 0, w - 1),
                    clamp(y1, 0, h - 1),
                    clamp(x2, 1, w),
                    clamp(y2, 1, h)
                ]
                original_target_bboxes.append(output_bbox)
                layer_meta = layer_metas[len(pixel_bboxes)] if (
                    isinstance(layer_metas, list) and len(pixel_bboxes) < len(layer_metas)
                ) else {}
                # Compile explicit adjacent-instance evidence once per layer.
                # The guard is used by candidate arbitration and probe review;
                # it does not by itself enable ownership point prompts.
                spatial_guard = build_spatial_ownership_guard(
                    layer_meta or {},
                    context_layers if isinstance(context_layers, list) else [],
                    output_bbox,
                    w,
                    h
                )
                if spatial_guard.get("enabled"):
                    layer_meta["_spatialOwnershipGuard"] = spatial_guard
                else:
                    layer_meta.pop("_spatialOwnershipGuard", None)
                resolved_layer_strategy = get_layer_strategy(layer_meta or {})
                layer_policy = resolve_mask_policy(layer_meta or {}, quality_profile)
                prompt_bbox = expand_bbox(*output_bbox, w, h)
                pixel_bboxes.append(output_bbox)
                # Prompt context and output geometry are separate. Output
                # expands only when the recovery audit establishes evidence.
                sam_prompt_bboxes.append(prompt_bbox)

            print(
                f"Using expanded SAM prompts by {int(BBOX_EXPAND_RATIO * 100)}% "
                "with semantic-bbox output clipping unless recovery verifies an extension"
            )
            if requested_engine == "sam":
                def sam_mask_provider(target_bbox, layer_meta, index):
                    layer_policy = resolve_mask_policy(layer_meta or {}, quality_profile)
                    resolved_layer_strategy = get_layer_strategy(layer_meta or {})
                    strategy_type = layer_policy["selectionType"]
                    prompt_bbox = sam_prompt_bboxes[index]
                    layer_name = layer_meta.get("name") or layer_ids[index] or index
                    completion_layer = is_completion_segmentation_layer(layer_meta)
                    completion_observation = (
                        decode_completion_observation_mask(layer_meta.get("completionObservationMask"), w, h)
                        if completion_layer else None
                    )
                    completion_occlusion_mask = (
                        decode_completion_occlusion_mask(layer_meta.get("completionOcclusionMask"), w, h)
                        if completion_layer else None
                    )
                    completion_flat_hard_edge = bool(
                        completion_layer and
                        layer_policy["profile"] == "flat_design" and
                        "hard_edge" in layer_policy["features"] and
                        not layer_policy["softEdge"]
                    )
                    spatial_canonical_completion = bool(
                        completion_layer and layer_policy["spatialCanonicalCompletion"]
                    )
                    # The post-processing stage may issue one local recovery
                    # query. Start every SAM pass with B; the normal quality
                    # gates below decide whether this request needs L. The
                    # accepted model is recorded so refinement and matte
                    # processing follow the selected candidate.
                    initial_sam_variant = forced_sam_model_variant("b")
                    layer_meta["_samModelVariant"] = initial_sam_variant
                    print(
                        f"SAM layer policy id={request_id} layer={layer_name} "
                        f"profile={layer_policy['profile']} phase={layer_policy['phase']} "
                        f"selector={layer_policy['selector']} matte={layer_policy['matteType']} "
                        f"spatialCanonical={layer_policy['spatialCanonicalCompletion']} "
                        f"boundaryRecovery={bool(layer_policy.get('boundaryRecovery', {}).get('enabled'))} "
                        f"boundaryMatte={bool(layer_policy.get('boundaryMatte', {}).get('enabled'))} "
                        f"spatialOwnership={bool((layer_policy.get('spatialOwnership') or {}).get('enabled'))} "
                        f"instanceUnion={bool(layer_policy.get('completionOccluder') and strategy_type == 'furniture')} "
                        f"features={','.join(layer_policy['features'])} "
                        f"imgsz={layer_policy['samImgSize']} "
                        f"forcedModel={initial_sam_variant.upper() if sam_l_forced() else 'off'}"
                    )
                    if (
                        layer_policy["selector"] == "compound_food" and
                        not layer_policy["completionOccluder"] and
                        not (layer_policy.get("compoundComponentDiscovery") or {}).get("enabled")
                    ):
                        print(
                            f"SAM prompts for {layer_meta.get('name') or layer_ids[index] or index}: "
                            f"bbox-only strategy=food_product"
                        )
                        results = run_sam_bbox_inference(
                            img,
                            prompt_bbox,
                            multimask_output=True,
                            imgsz=1024,
                            model_variant=initial_sam_variant
                        )
                        return normalize_result_masks(results, w, h)

                    soft_edge_prompts = None
                    semantic_ownership_prompts = None
                    if strategy_type == "soft_edge":
                        soft_edge_prompts = build_soft_edge_prompt_inputs(img, target_bbox)
                        print(
                            f"SAM prompts for {layer_meta.get('name') or layer_ids[index] or index}: "
                            f"+{len(soft_edge_prompts['positive'])} -{len(soft_edge_prompts['negative'])} "
                            "strategy=soft_edge_color_guided"
                        )
                    else:
                        ownership_config = layer_policy.get("semanticOwnershipPrompt") or {}
                        if ownership_config.get("enabled"):
                            semantic_ownership_prompts = build_sam_prompt_inputs(
                                layer_meta,
                                context_layers,
                                target_bbox,
                                w,
                                h,
                                include_boundary_negatives=bool(
                                    ownership_config.get("includeBoundaryNegatives")
                                ),
                                prompt_bbox=prompt_bbox
                            )
                            print(
                                f"SAM prompts for {layer_name}: "
                                f"+{len((semantic_ownership_prompts.get('points') or [[]])[0])} "
                                f"ownershipNegatives="
                                f"{sum(1 for label in ((semantic_ownership_prompts.get('labels') or [[]])[0]) if label == 0)} "
                                f"contextExcludes={semantic_ownership_prompts.get('ownershipExcludeCount', 0)} "
                                f"intent={semantic_ownership_prompts.get('segmentationIntentAudit', {}).get('reason')} "
                                f"intentComponents={semantic_ownership_prompts.get('segmentationIntentAudit', {}).get('componentCount', 0)} "
                                f"spatialOwnership={bool((layer_policy.get('spatialOwnership') or {}).get('enabled'))} "
                                "strategy=semantic_ownership"
                            )
                        else:
                            ownership_state = layer_policy.get("spatialOwnership") or {}
                            ownership_suffix = (
                                " ownership=single_component_bbox_baseline"
                                if ownership_state.get("enabled") else ""
                            )
                            print(
                                f"SAM prompts for {layer_name}: "
                                f"bbox-only strategy={strategy_type}{ownership_suffix}"
                            )
                    prompt_inputs = semantic_ownership_prompts or soft_edge_prompts
                    results = run_sam_bbox_inference(
                        img,
                        prompt_bbox,
                        multimask_output=True,
                        imgsz=(
                            layer_policy["samImgSize"]
                        ),
                        points=prompt_inputs["points"] if prompt_inputs else None,
                        labels=prompt_inputs["labels"] if prompt_inputs else None,
                        model_variant=initial_sam_variant
                    )
                    # Preserve the established raster mode for tables/chairs;
                    # profile classification must not silently change their
                    # mask edge behavior.
                    use_subpixel_masks = (
                        layer_policy["softEdge"] or
                        layer_policy["baseStrategyType"] in HARD_EDGE_STRATEGIES
                    )
                    candidate_masks = normalize_result_masks(
                        results,
                        w,
                        h,
                        interpolation=cv2.INTER_LINEAR if use_subpixel_masks else cv2.INTER_NEAREST,
                        debug_label=(
                            f"{layer_meta.get('name') or layer_ids[index] or index} "
                            f"strategy={strategy_type}"
                        )
                    )
                    # The normalized masks are CPU numpy arrays. Drop the
                    # Ultralytics result before a possible B -> L switch so
                    # its GPU tensors do not keep B's inference memory alive.
                    del results
                    sam_diagnostic_event(
                        "b_candidates", masks=candidate_masks,
                        targetBbox=target_bbox, promptBbox=prompt_bbox,
                        policy=layer_policy,
                        points=prompt_inputs["points"] if prompt_inputs else None,
                        labels=prompt_inputs["labels"] if prompt_inputs else None
                    )
                    if soft_edge_prompts:
                        candidate_masks = filter_soft_edge_masks_by_points(candidate_masks, soft_edge_prompts)
                    positive_probe_enabled = bool(
                        SAM_POSITIVE_PROBE_ENABLED and
                        (layer_policy.get("positiveProbe") or {}).get("enabled")
                    )
                    if positive_probe_enabled:
                        candidate_masks, positive_probe_audit = run_positive_probe(
                            img,
                            candidate_masks,
                            target_bbox,
                            prompt_bbox,
                            layer_name,
                            strategy_type=strategy_type,
                            layer_meta=layer_meta,
                            context_layers=context_layers,
                            quality_profile=quality_profile
                        )
                        layer_meta["_positiveProbeAudit"] = positive_probe_audit
                        sam_diagnostic_event("probe_decision", audit=positive_probe_audit)
                        if positive_probe_audit.get("status") == "accepted":
                            layer_meta["_samSelectionRoute"] = "b_positive_probe"
                            if not (layer_policy.get("spatialOwnership") or {}).get("strictOutput"):
                                layer_meta["_samEffectiveBbox"] = positive_probe_audit.get(
                                    "effectiveBbox"
                                )
                        print(
                            f"SAM positive probe for {layer_name}: "
                            f"status={positive_probe_audit.get('status')} "
                            f"reason={positive_probe_audit.get('reason')} "
                            f"points={positive_probe_audit.get('positivePoints', 0)} "
                            f"fillGain={positive_probe_audit.get('fillGain', 0)} "
                            f"addedPixels={positive_probe_audit.get('addedPixels', 0)} "
                            f"preserve={positive_probe_audit.get('preserve')} "
                            f"preserveCore={positive_probe_audit.get('preserveCore')} "
                            f"imageReview={positive_probe_audit.get('imageReview')} "
                            f"connectedGrowth={positive_probe_audit.get('connectedGrowth')} "
                            f"growth={positive_probe_audit.get('growth')} "
                            f"growthGain={positive_probe_audit.get('growthGain')} "
                            f"outsideAddedRatio={positive_probe_audit.get('outsideAddedRatio')} "
                            f"contextConflictRatio={positive_probe_audit.get('contextConflictRatio')} "
                            f"effectiveBbox={positive_probe_audit.get('effectiveBbox')} "
                            f"rejections={positive_probe_audit.get('rejectionCounts', {})}"
                        )
                    compound_discovery_config = layer_policy.get("compoundComponentDiscovery") or {}
                    if compound_discovery_config.get("enabled"):
                        candidate_masks, compound_discovery_audit = run_compound_component_discovery(
                            img,
                            candidate_masks,
                            target_bbox,
                            layer_name,
                            layer_meta=layer_meta,
                            context_layers=context_layers,
                            quality_profile=quality_profile,
                            policy=layer_policy
                        )
                        layer_meta["_compoundComponentDiscovery"] = compound_discovery_audit
                        sam_diagnostic_event(
                            "compound_discovery_decision",
                            masks=candidate_masks,
                            audit=compound_discovery_audit,
                            targetBbox=target_bbox
                        )
                        print(
                            f"SAM compound component discovery for {layer_name}: "
                            f"status={compound_discovery_audit.get('status')} "
                            f"reason={compound_discovery_audit.get('reason')} "
                            f"selected={compound_discovery_audit.get('selectedIndex')} "
                            f"verifiedAddedPixels={compound_discovery_audit.get('verifiedAddedPixels')} "
                            f"baselinePixels={compound_discovery_audit.get('baselinePixels')} "
                            f"candidateCount={compound_discovery_audit.get('candidateCount')} "
                            f"promptBbox={compound_discovery_audit.get('promptBbox')}"
                        )
                    if layer_policy["allowLocalUpscale"]:
                        local_upscale, local_reason = should_run_local_upscale(
                            candidate_masks,
                            target_bbox,
                            strategy_type
                        )
                        if local_upscale:
                            print(
                                f"SAM local-upscale route for {layer_name}: "
                                f"model=B reason={local_reason}"
                            )
                            local_masks = run_upscaled_hard_edge_bbox_inference(
                                img,
                                prompt_bbox,
                                target_bbox,
                                w,
                                h,
                                layer_name,
                                model_variant=initial_sam_variant,
                                imgsz=LOCAL_UPSCALE_SAM_IMGSZ
                            )
                            if local_masks is not None and len(local_masks) > 0:
                                # normalize_result_masks returns an ndarray,
                                # while candidate augmentation is list-based.
                                candidate_masks = [
                                    np.asarray(mask, dtype=np.float32)
                                    for mask in candidate_masks
                                ]
                                before_count = len(candidate_masks)
                                append_unique_masks(
                                    candidate_masks,
                                    local_masks,
                                    min_pixels=64,
                                    dedupe_iou=0.96
                                )
                                print(
                                    f"SAM local-upscale candidates for {layer_name}: "
                                    f"before={before_count} after={len(candidate_masks)}"
                                )
                    boundary_config = layer_policy.get("boundaryRecovery") or {}
                    if boundary_config.get("enabled"):
                        boundary_evidence = boundary_recovery_evidence(
                            candidate_masks,
                            target_bbox,
                            strategy_type=strategy_type,
                            layer_meta=layer_meta,
                            context_layers=context_layers,
                            policy=layer_policy
                        )
                        print(
                            f"SAM boundary recovery evidence for {layer_name}: "
                            f"needed={bool(boundary_evidence.get('needed'))} "
                            f"boundaryTruncated={boundary_evidence.get('boundary_truncated')} "
                            f"candidateDisagreement={boundary_evidence.get('candidate_disagreement')} "
                            f"contextConflict={boundary_evidence.get('context_conflict')} "
                            f"preserveCore={boundary_evidence.get('preserve_core')} "
                            f"edgeType={boundary_evidence.get('edge_type')} "
                            f"reason={boundary_evidence.get('reason')}"
                        )
                        if boundary_evidence.get("needed"):
                            try:
                                recovered_mask, recovery_audit = recover_boundary_with_sam_l(
                                    img,
                                    candidate_masks,
                                    target_bbox,
                                    prompt_bbox,
                                    layer_name,
                                    strategy_type=strategy_type,
                                    layer_meta=layer_meta,
                                    context_layers=context_layers,
                                    policy=layer_policy
                                )
                                print(
                                    f"SAM boundary recovery result for {layer_name}: "
                                    f"{recovery_audit}"
                                )
                                layer_meta["_boundaryRecovery"] = {
                                    **boundary_evidence,
                                    **(recovery_audit or {})
                                }
                                if recovered_mask is not None and np.any(recovered_mask > 0.5):
                                    candidate_masks = np.stack([recovered_mask], axis=0)
                                    layer_meta["_samModelVariant"] = "l"
                                    layer_meta["_samSelectionRoute"] = "boundary_recovery_l"
                                    layer_meta["_samEffectiveBbox"] = recovery_audit.get("effectiveBbox")
                                    return candidate_masks
                                release_sam_model("l", reason="boundary_recovery_rejected")
                            except Exception as error:
                                print(
                                    f"SAM boundary recovery failed for {layer_name}: {error}"
                                )
                                release_sam_model("l", reason="boundary_recovery_failed")
                            # This capability has already performed its dedicated L
                            # review. Preserve the established B result rather
                            # than running a second, generic L arbitration pass.
                            layer_meta["_samSelectionRoute"] = "boundary_recovery_b_fallback"
                            return candidate_masks
                    if layer_policy["allowModelEscalation"] and not sam_l_forced():
                        arbitration_bbox, uses_effective_bbox = resolve_verified_effective_bbox(
                            target_bbox,
                            layer_meta.get("_samEffectiveBbox"),
                            w,
                            h
                        )
                        if (layer_policy.get("spatialOwnership") or {}).get("strictOutput"):
                            arbitration_bbox, uses_effective_bbox = target_bbox, False
                        if uses_effective_bbox:
                            print(
                                f"SAM B/L effective bbox for {layer_name}: "
                                f"original={target_bbox} effective={arbitration_bbox}"
                            )
                        escalate, escalation_reason = should_escalate_sam_to_l(
                            candidate_masks,
                            arbitration_bbox,
                            strategy_type,
                            quality_profile=quality_profile,
                            completion_observation=completion_observation,
                            policy=layer_policy
                        )
                        layer_label = layer_meta.get("name") or layer_ids[index] or index
                        if escalate:
                            print(
                                f"SAM route for {layer_label}: model=B -> L "
                                f"reason={escalation_reason}"
                            )
                            try:
                                l_points = (
                                    semantic_ownership_prompts["points"]
                                    if semantic_ownership_prompts else None
                                )
                                l_labels = (
                                    semantic_ownership_prompts["labels"]
                                    if semantic_ownership_prompts else None
                                )
                                l_results, l_imgsz = run_sam_l_with_retry(
                                    img,
                                    prompt_bbox,
                                    strategy_type,
                                    layer_label,
                                    policy=layer_policy,
                                    points=l_points,
                                    labels=l_labels
                                )
                                l_masks = normalize_result_masks(
                                    l_results,
                                    w,
                                    h,
                                    interpolation=cv2.INTER_LINEAR if use_subpixel_masks else cv2.INTER_NEAREST,
                                    debug_label=(
                                        f"{layer_label} strategy={strategy_type} "
                                        f"model=L imgsz={l_imgsz}"
                                    )
                                )
                                del l_results
                                local_l_upscale, local_l_reason = should_run_l_local_upscale(
                                    candidate_masks,
                                    arbitration_bbox,
                                    strategy_type
                                )
                                if local_l_upscale:
                                    print(
                                        f"SAM local-upscale route for {layer_label}: "
                                        f"model=L reason={local_l_reason}"
                                    )
                                    local_l_masks = run_upscaled_hard_edge_bbox_inference(
                                        img,
                                        prompt_bbox,
                                        arbitration_bbox,
                                        w,
                                        h,
                                        layer_label,
                                        model_variant="l",
                                        imgsz=LOCAL_UPSCALE_SAM_IMGSZ
                                    )
                                    if local_l_masks is not None and len(local_l_masks) > 0:
                                        l_masks = [
                                            np.asarray(mask, dtype=np.float32)
                                            for mask in l_masks
                                        ]
                                        before_count = len(l_masks)
                                        append_unique_masks(
                                            l_masks,
                                            local_l_masks,
                                            min_pixels=64,
                                            dedupe_iou=0.96
                                        )
                                        print(
                                            f"SAM local-upscale L candidates for {layer_label}: "
                                            f"before={before_count} after={len(l_masks)}"
                                        )
                                b_candidate_masks_for_arbitration = candidate_masks
                                candidate_masks, arbitration = arbitrate_sam_b_l_masks(
                                    candidate_masks,
                                    l_masks,
                                    arbitration_bbox,
                                    strategy_type=strategy_type,
                                    b_failure_reason=escalation_reason,
                                    completion_recovery=(
                                        completion_layer and not spatial_canonical_completion
                                    ),
                                    completion_observation=completion_observation,
                                    completion_occlusion_mask=completion_occlusion_mask,
                                    completion_base_strategy_type=layer_policy["baseStrategyType"],
                                    completion_spatial=layer_policy["spatial"],
                                    completion_flat_hard_edge=completion_flat_hard_edge,
                                    # A completion occluder may represent one
                                    # semantic layer containing several visible
                                    # instances. Preserve a safe SAM-L union;
                                    # do not require the semantic model to have
                                    # emitted child layers for this ownership
                                    # mask.
                                    composite_instance_union=bool(
                                        layer_policy.get("completionOccluder") and
                                        strategy_type == "furniture"
                                    )
                                )
                                if layer_policy["completionOccluder"]:
                                    l_selected = (
                                        arbitration.startswith("l_") or
                                        arbitration.startswith("hybrid_added=")
                                    )
                                    layer_meta["_completionOccluderAudit"] = {
                                        "status": "verified" if l_selected else "unverified",
                                        "selectedModel": "l" if l_selected else "b",
                                        "reason": arbitration
                                    }
                                if arbitration == "completion_observation_fallback":
                                    layer_meta["_completionObservationFallback"] = True
                                elif arbitration.startswith("completion_full_scene_sam="):
                                    layer_meta["_completionFullSceneMask"] = True
                                print(
                                    f"SAM arbitration for {layer_label}: "
                                    f"{arbitration} candidates={len(candidate_masks)}"
                                )
                                if arbitration.startswith("l_"):
                                    if strategy_type == "furniture" and len(candidate_masks) > 0:
                                        peer_merge_count = 0
                                        for arbitration_token in arbitration.split():
                                            if arbitration_token.startswith("peers="):
                                                try:
                                                    peer_merge_count = int(arbitration_token.split("=", 1)[1])
                                                except (TypeError, ValueError):
                                                    peer_merge_count = 0
                                                break
                                        if (
                                            (
                                                peer_merge_count == 0 or
                                                layer_policy.get("completionOccluder")
                                            ) and
                                            b_candidate_masks_for_arbitration is not None and
                                            len(b_candidate_masks_for_arbitration) > 0
                                        ):
                                            cross_mask, cross_changed, cross_debug = refine_furniture_mask_with_cross_model_evidence(
                                                img,
                                                b_candidate_masks_for_arbitration,
                                                candidate_masks[0],
                                                target_bbox,
                                                layer_label,
                                                allow_disconnected_components=bool(
                                                    layer_policy.get("completionOccluder")
                                                )
                                            )
                                            print(
                                                f"Furniture cross-model refine for {layer_label}: "
                                                f"status={cross_debug.get('status')} "
                                                f"components={cross_debug.get('components', 0)} "
                                                f"pixels={cross_debug.get('pixels', 0)} "
                                                f"bEvidence={cross_debug.get('bEvidencePixels', 0)} "
                                                f"disagreement={cross_debug.get('disagreementPixels', 0)} "
                                                f"points={cross_debug.get('positivePoints', 0)}/"
                                                f"{cross_debug.get('negativePoints', 0)} "
                                                f"disconnectedAllowed={cross_debug.get('disconnectedComponentsAllowed', False)} "
                                                f"accepted={bool(cross_changed)}"
                                            )
                                            if cross_changed and np.any(cross_mask > 0.5):
                                                candidate_masks = np.stack([cross_mask], axis=0)
                                    layer_meta["_samModelVariant"] = "l"
                                elif arbitration == "b_kept_l_rejected":
                                    # The selected silhouette is still B-led
                                    # even when L supplied an accepted bounded
                                    # edge measurement. Keep the B marker so a
                                    # category-specific later post-process is
                                    # not accidentally enabled by this generic
                                    # boundary capability.
                                    layer_meta["_samModelVariant"] = "b"
                                    release_sam_model("l", reason="b_arbitration_kept")
                                else:
                                    layer_meta["_samModelVariant"] = "l"
                            except Exception as error:
                                print(f"SAM-L escalation failed for {layer_label}: {error}")
                                # An OOM can leave the L predictor and its
                                # allocator reservation alive. Release it so
                                # the request can still finish with B and the
                                # next invocation starts from a clean state.
                                release_sam_model("l", reason="escalation_failed")
                                # A completion occluder is later used as the
                                # edit mask for GPT inpainting.  Falling back
                                # to an unverified B mask after a programming
                                # or inference error propagates a bad mask into
                                # the next stage and makes the eventual 422
                                # misleading.  Only CUDA OOM is recoverable;
                                # surface every other failure immediately.
                                if layer_policy["completionOccluder"] and not is_cuda_oom(error):
                                    raise
                                layer_meta["_samModelVariant"] = "b"
                        else:
                            if layer_policy["completionOccluder"]:
                                layer_meta["_completionOccluderAudit"] = {
                                    "status": "unverified",
                                    "selectedModel": "b",
                                    "reason": escalation_reason
                                }
                            print(
                                f"SAM route for {layer_label}: model=B "
                                f"reason={escalation_reason}"
                            )
                    return candidate_masks

                shared_embedding_request = (
                    SAM_SHARED_EMBEDDING_ENABLED and len(pixel_bboxes) > 1
                )
                positive_probe_request = bool(
                    SAM_POSITIVE_PROBE_ENABLED and
                    any(
                        isinstance(meta, dict) and
                        (resolve_mask_policy(meta, quality_profile).get("positiveProbe") or {}).get("enabled")
                        for meta in layer_metas
                    ) if isinstance(layer_metas, list) else False
                )
                def run_prompted_sam_cutouts():
                    if shared_embedding_request or positive_probe_request:
                        # Keep the shared predictor features alive for this
                        # request. The layer prompts, multimask output and all
                        # candidate/post-processing decisions remain independent.
                        with sam_runtime_lock, sam_embedding_cache():
                            return process_prompted_cutouts(
                                img,
                                pixel_bboxes,
                                layer_ids,
                                layer_metas,
                                context_layers,
                                sam_mask_provider,
                                "sam_bbox_prompt",
                                refine_masks=True,
                                original_target_bboxes=original_target_bboxes,
                                quality_profile=quality_profile
                            )
                    return process_prompted_cutouts(
                        img,
                        pixel_bboxes,
                        layer_ids,
                        layer_metas,
                        context_layers,
                        sam_mask_provider,
                        "sam_bbox_prompt",
                        refine_masks=True,
                        original_target_bboxes=original_target_bboxes,
                        quality_profile=quality_profile
                    )

                with sam_diagnostic_request(data, img, request_id), sam_request_metrics() as sam_metrics:
                    cutouts = run_prompted_sam_cutouts()
                    print(
                        f"SAM request metrics id={request_id}: "
                        f"{sam_metrics.summary()}"
                    )
                completion_request = any(
                    is_completion_segmentation_layer(meta)
                    for meta in layer_metas
                ) if isinstance(layer_metas, list) else False
                if completion_request and len(layer_ids) == 1 and not cutouts:
                    # A successful HTTP response with no cutout is not a
                    # successful completion. Surface this as an error so the
                    # client records the actual segmentation failure.
                    return JSONResponse(
                        status_code=422,
                        content={
                            "success": False,
                            "error": "补全模式未通过候选筛选，未返回 cutout",
                            "layerId": layer_ids[0] if layer_ids else None,
                            "qualityProfile": quality_profile
                        }
                    )
                return JSONResponse(content={"success": True, "engine": "sam", "cutouts": cutouts})

            # Default FastSAM path: global candidate masks + existing merge logic.
            fastsam = get_fastsam_model()
            results = fastsam(img, retina_masks=True, imgsz=1024, conf=0.25, iou=0.9)
            masks = normalize_result_masks(results, w, h)

            def fastsam_mask_provider(target_bbox, layer_meta, index):
                return masks

            cutouts = process_prompted_cutouts(
                img,
                pixel_bboxes,
                layer_ids,
                layer_metas,
                context_layers,
                fastsam_mask_provider,
                "fastsam_multi_mask",
                refine_masks=False,
                original_target_bboxes=original_target_bboxes,
                quality_profile=quality_profile
            )
            return JSONResponse(content={"success": True, "engine": "fastsam", "cutouts": cutouts})
        else:
            # Fallback to everything=True segmentation if no bboxes provided
            if requested_engine == "sam":
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "高精 SAM 当前仅支持带 bbox prompt 的分割请求"}
                )

            print("No bounding boxes provided, fallback to segment everything")
            fastsam = get_fastsam_model()
            results = fastsam(img, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

            layers = []
            if len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                boxes = results[0].boxes.data.cpu().numpy()

                for i, (mask, box) in enumerate(zip(masks, boxes)):
                    if mask.shape != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    layer_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    layer_img[:, :, 3] = mask * 255

                    x1, y1, x2, y2 = map(int, box[:4])
                    cropped_img = layer_img[y1:y2, x1:x2]

                    if cropped_img.size == 0:
                        continue

                    layer_b64 = cv2_to_base64(cropped_img)
                    norm_ymin = int((y1 / h) * 1000)
                    norm_xmin = int((x1 / w) * 1000)
                    norm_ymax = int((y2 / h) * 1000)
                    norm_xmax = int((x2 / w) * 1000)

                    layers.append({
                        "id": f"fastsam-layer-{i}-{int(time.time())}",
                        "name": f"FastSAM 层 {i+1}",
                        "layerType": "OBJECT",
                        "bbox": [norm_ymin, norm_xmin, norm_ymax, norm_xmax],
                        "image": layer_b64,
                        "assetStatus": "idle"
                    })

            return JSONResponse(content={"success": True, "engine": "fastsam", "layers": layers})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
