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
            legacy_food_output_count = 0
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
                prompt_bbox = expand_bbox(*output_bbox, w, h)
                original_target_bboxes.append(output_bbox)
                layer_meta = layer_metas[len(pixel_bboxes)] if (
                    isinstance(layer_metas, list) and len(pixel_bboxes) < len(layer_metas)
                ) else {}
                resolved_layer_strategy = get_layer_strategy(layer_meta or {})
                layer_policy = resolve_mask_policy(layer_meta or {}, quality_profile)
                is_food_product = layer_policy["selector"] == "compound_food"
                # Restore the previous cloud-parity food path: food products
                # use the expanded bbox for candidate/output parity.
                pixel_bboxes.append(prompt_bbox if is_food_product else output_bbox)
                if is_food_product:
                    legacy_food_output_count += 1
                # The expanded prompt gives SAM enough context to recover the
                # complete hard object. The output remains constrained by the
                # original bbox, so this must not be confused with output
                # expansion.
                sam_prompt_bboxes.append(prompt_bbox)

            print(
                f"Using expanded SAM prompts by {int(BBOX_EXPAND_RATIO * 100)}% "
                "with strict original-bbox output clipping"
            )
            if legacy_food_output_count:
                print(
                    f"Food product legacy bbox parity enabled for {legacy_food_output_count} layer(s): "
                    "expanded prompt bbox is also used for output"
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
                    # query. Record which model owns the accepted mask so that
                    # recovery does not accidentally reload the other variant.
                    # The completion marker identifies the second SAM pass
                    # unambiguously. Do not allow a caller-provided model hint
                    # to downgrade this pass back to B.
                    force_completion_sam_l = completion_layer
                    layer_meta["_samModelVariant"] = "l" if force_completion_sam_l else "b"
                    print(
                        f"SAM layer policy id={request_id} layer={layer_name} "
                        f"profile={layer_policy['profile']} phase={layer_policy['phase']} "
                        f"selector={layer_policy['selector']} matte={layer_policy['matteType']} "
                        f"spatialCanonical={layer_policy['spatialCanonicalCompletion']} "
                        f"instanceUnion={bool(layer_policy.get('completionOccluder') and strategy_type == 'furniture')} "
                        f"features={','.join(layer_policy['features'])} "
                        f"imgsz={layer_policy['samImgSize']}"
                    )
                    if force_completion_sam_l:
                        # This is exclusively the post-inpaint, second-SAM
                        # path. It must not start with B or let B/L arbitration
                        # retain a fragmented B mask. Initial extraction,
                        # including all category-specific routes, is unchanged.
                        print(
                            f"SAM route for {layer_name}: model=L forced "
                            "reason=completion_forced_l"
                        )
                        l_results, l_imgsz = run_sam_l_with_retry(
                            img,
                            prompt_bbox,
                            strategy_type,
                            layer_name,
                            policy=layer_policy
                        )
                        candidate_masks = normalize_result_masks(
                            l_results,
                            w,
                            h,
                            interpolation=(
                                cv2.INTER_LINEAR
                                if (layer_policy["softEdge"] or layer_policy["baseStrategyType"] in HARD_EDGE_STRATEGIES)
                                else cv2.INTER_NEAREST
                            ),
                            debug_label=(
                                f"{layer_name} strategy={strategy_type} "
                                f"model=L forced imgsz={l_imgsz}"
                            )
                        )
                        del l_results
                        if strategy_type == "soft_edge":
                            forced_soft_edge_prompts = build_soft_edge_prompt_inputs(img, target_bbox)
                            candidate_masks = filter_soft_edge_masks_by_points(
                                candidate_masks,
                                forced_soft_edge_prompts
                            )
                        layer_meta["_samSelectionRoute"] = "completion_forced_l"
                        return candidate_masks
                    if (
                        layer_policy["selector"] == "compound_food" and
                        not layer_policy["completionOccluder"]
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
                            model_variant="b"
                        )
                        return normalize_result_masks(results, w, h)

                    soft_edge_prompts = None
                    if strategy_type == "soft_edge":
                        soft_edge_prompts = build_soft_edge_prompt_inputs(img, target_bbox)
                        print(
                            f"SAM prompts for {layer_meta.get('name') or layer_ids[index] or index}: "
                            f"+{len(soft_edge_prompts['positive'])} -{len(soft_edge_prompts['negative'])} "
                            "strategy=soft_edge_color_guided"
                        )
                    else:
                        print(
                            f"SAM prompts for {layer_name}: "
                            f"bbox-only strategy={strategy_type}"
                        )
                    results = run_sam_bbox_inference(
                        img,
                        prompt_bbox,
                        multimask_output=True,
                        imgsz=(
                            layer_policy["samImgSize"]
                        ),
                        points=soft_edge_prompts["points"] if soft_edge_prompts else None,
                        labels=soft_edge_prompts["labels"] if soft_edge_prompts else None,
                        model_variant="b"
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
                    if soft_edge_prompts:
                        candidate_masks = filter_soft_edge_masks_by_points(candidate_masks, soft_edge_prompts)
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
                                model_variant="b",
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
                    if layer_policy["allowModelEscalation"]:
                        escalate, escalation_reason = should_escalate_sam_to_l(
                            candidate_masks,
                            target_bbox,
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
                                l_results, l_imgsz = run_sam_l_with_retry(
                                    img,
                                    prompt_bbox,
                                    strategy_type,
                                    layer_label,
                                    policy=layer_policy
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
                                    target_bbox,
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
                                        target_bbox,
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
                                    target_bbox,
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

                cutouts = process_prompted_cutouts(
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
