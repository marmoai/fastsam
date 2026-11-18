"""V5 transport: one evidence pipeline behind the existing /segment contract.

The client keeps sending the same payload it sends to ``api.py`` (0-1000
normalized bboxes, layer metadata, context layers). What changes is what
happens to it:

- the semantic hint box is prompt *context*, never an output clip;
- SAM/FastSAM masks are proposals only;
- every pixel decision (background cleanup, subject completion, band alpha)
  is measured from image colour and gradient evidence in
  ``segmentation_core_v5``, with capabilities resolved by
  ``segmentation_policy_v5``.

No object category chooses a pipeline here, so wording changes cannot switch
post-processing branches.
"""

import base64
import json
import os
import time
from contextvars import ContextVar

import cv2
import numpy as np
from ownership_v5 import build_ownership_evidence, review_ownership_candidates, neighbour_consensus, _interior_points
from matting_v5 import polarity_prompt
from hq_candidate_review_v5 import review_hq_candidates
from candidate_pool_v5 import choose_candidate_pool, enabled as _v5_enabled
from query_candidates_v5 import validated_candidates
from local_completeness_v5 import augment_ownership
from independent_neighbour_v5 import verify_neighbours
from joint_ownership_v5 import reconcile_layers
from component_recovery_v5 import (component_proposal, review_component_candidate, merge_component_alpha,
                                   local_component_inputs, merge_verified_completion, local_fusion_proposals,
                                   review_local_component)
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sam_diagnostics import (sam_diagnostic_event, sam_diagnostic_request,
                             sam_diagnostic_output, sam_diagnostic_layer)
from sam_runtime_v5 import (
    forced_sam_model_variant,
    get_fastsam_model,
    run_sam_bbox_inference,
    sam_embedding_cache,
    sam_request_metrics,
    sam_runtime_lock,
)
import segmentation_core_v5 as _core_v5
from segmentation_core_v5 import (
    CORE_VERSION_V5,
    build_cutout_entry_v5,
    process_layer_v5,
    select_candidate_v5,
)

# Partial historical snapshots predate this optional prompt experiment.
# Do not transplant newer segmentation logic into a restored core.
propose_background_negative_points_v5 = getattr(
    _core_v5, 'propose_background_negative_points_v5', None)
from segmentation_policy_v5 import (
    POLICY_VERSION_V5,
    expand_hint_bbox_v5,
    resolve_policy_v5,
)

API_VERSION_V5 = "v5-hql-evidence-transport-5-validated-query-boundary"

# Retrying a failed layer with the larger SAM model is evidence-neutral: it
# only offers more proposals to the same selector.
_ESCALATE_ENABLED = str(
    os.environ.get("SAMV5_ESCALATE", "1")
).strip().lower() not in {"0", "false", "no", "off"}
_SAM_IMGSZ = int(os.environ.get("SAMV5_IMGSZ", "1024"))
# A/B switch only. Default remains the existing three-mask path. Setting this
# to 1 uses the dedicated single-mask head only for the ownership retry.
_SAM_SINGLE_MASK_AB = str(os.environ.get("SAMV5_SINGLE_MASK_AB", "0")).lower() in {"1", "true", "yes", "on"}
_SAM_OUTPUT_EVIDENCE = str(os.environ.get('SAMV5_OUTPUT_EVIDENCE', '1')).lower() not in {'0','false','no','off'}
# D is opt-in: decoder logits are used only as a dense feedback prompt.
_SAM_MASK_FEEDBACK_AB = str(os.environ.get('SAMV5_MASK_FEEDBACK_AB', '0')).lower() in {'1','true','yes','on'}
# E preserves the established one-shot local component fallback by default;
# the switch makes its contribution measurable without changing the contract.
_SAM_LOCAL_CROP_AB = str(os.environ.get('SAMV5_LOCAL_CROP_AB', '1')).lower() in {'1','true','yes','on'}
_SAM_BACKGROUND_NEGATIVE_AB = (
    callable(propose_background_negative_points_v5) and
    str(os.environ.get('SAMV5_BACKGROUND_NEGATIVE_AB', '0')).lower() in {'1','true','yes','on'})
if not callable(propose_background_negative_points_v5):
    print(f'SAMV5 compatibility core={CORE_VERSION_V5}: '
          'background-negative retry unavailable; skipped for historical core', flush=True)
_sam_output_stats = ContextVar("sam_v5_output_stats", default=None)
_sam_feedback_logits = ContextVar("sam_v5_feedback_logits", default=None)
_single_head_allowed = ContextVar('sam_v5_single_head_allowed', default=False)
_hq_branch_evidence = ContextVar('v5_hq_branch_evidence', default=None)

app = FastAPI(title="MarmoAI segmentation (V5 evidence)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _log(message):
    print(f"SAMV5 transport {message}", flush=True)


def _decode_image(encoded):
    """Decode a data URL or bare base64 PNG/JPEG into a BGR image."""
    if not encoded:
        return None
    payload = encoded.split(",", 1)[1] if "," in encoded else encoded
    payload = payload.replace(" ", "").replace("\n", "").replace("\r", "")
    missing = len(payload) % 4
    if missing:
        payload += "=" * (4 - missing)
    try:
        raw = base64.b64decode(payload)
    except Exception:
        return None
    buffer = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _normalized_engine(value):
    text = str(value or "").strip().lower()
    if text in {"fastsam", "fast_sam", "yolo", "yolo_fastsam"}:
        return "fastsam"
    return "sam"


def _normalized_profile(value):
    text = str(value or "").strip().lower()
    return text or "publish"


def _pixel_hint(bbox_norm, width, height):
    """0-1000 [ymin, xmin, ymax, xmax] -> pixel [x1, y1, x2, y2]."""
    ymin, xmin, ymax, xmax = [float(v) for v in bbox_norm]
    x1 = int(round(xmin / 1000.0 * width))
    y1 = int(round(ymin / 1000.0 * height))
    x2 = int(round(xmax / 1000.0 * width))
    y2 = int(round(ymax / 1000.0 * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def _result_masks(results, width, height):
    """Pull full-image masks out of an Ultralytics result, without rescaling
    decisions into the candidate space: masks are resampled only to image size.
    """
    if results is None or len(results) == 0:
        return []
    entry = results[0]
    masks = getattr(entry, "masks", None)
    if masks is None or masks.data is None:
        return []
    scores = None
    boxes = getattr(entry, "boxes", None)
    if boxes is not None and getattr(boxes, "conf", None) is not None:
        try:
            scores = boxes.conf.detach().cpu().numpy().astype(float).tolist()
        except Exception:
            scores = None
    raw = masks.data.cpu().numpy()
    out = []
    for mask in raw:
        array = np.asarray(mask, dtype=np.float32)
        if array.shape != (height, width):
            array = cv2.resize(array, (width, height), interpolation=cv2.INTER_NEAREST)
        out.append(array)
    _sam_output_stats.set({
        "maskCount": len(out),
        "modelScores": scores or [],
    })
    return out


def _sam_candidates(image, prompt_bbox, model_variant, points=None, labels=None, masks=None):
    _sam_output_stats.set(None)
    _sam_feedback_logits.set(None)
    # B isolates the dedicated head for already guided prompts. Box-only
    # discovery stays identical to A, avoiding a change to two variables.
    single = _SAM_SINGLE_MASK_AB and points is not None and _single_head_allowed.get()
    # HQ branch validation is required even when diagnostic dumping is off.
    evidence = {}
    results = run_sam_bbox_inference(
        image,
        prompt_bbox,
        multimask_output=not single,
        imgsz=_SAM_IMGSZ,
        model_variant=model_variant,
        points=points,
        labels=labels,
        masks=masks,
        output_evidence=evidence,
    )
    height, width = image.shape[:2]
    masks = _result_masks(results, width, height)
    stats = _sam_output_stats.get() or {"maskCount":len(masks), "modelScores":[]}
    stats['outputMode'] = 'hq_single' if single else 'hq_multimask'
    _hq_branch_evidence.set(evidence)
    if evidence and 'binaryParity' in evidence:
        stats.update(binaryParity=evidence['binaryParity'],
                     headIndices=evidence['retainedHeadIndices'].tolist(),
                     thresholdStability=evidence['thresholdStability'].tolist())
    if (evidence and evidence.get('binaryParity') is True
            and evidence.get('fullResolutionLogits') is not None
            and np.asarray(evidence['fullResolutionLogits']).shape == (len(masks), height, width)
            and np.isfinite(evidence['fullResolutionLogits']).all()):
        # Use source-aligned logits; the predictor performs its own prompt
        # resize. Keeping the low-resolution tensor here would bypass the
        # predictor's geometry transform for SAM-L/SAM2 variants.
        _sam_feedback_logits.set(evidence['fullResolutionLogits'])
    elif evidence and evidence.get('captureError'):
        stats['captureError'] = evidence['captureError']
    if evidence and evidence.get('fullResolutionLogits') is not None:
        stats['feedbackEligible'] = _sam_feedback_logits.get() is not None
    if evidence and _SAM_OUTPUT_EVIDENCE:
        sam_diagnostic_output(evidence)
    _sam_output_stats.set(stats)
    del results
    return masks


def _validated_query(image, box, variant, points, labels, validator, purpose):
    defaults = _sam_candidates(image, box, variant, points=points, labels=labels)
    if os.environ.get('SAMV5_GUIDED_CANDIDATES', '1').strip().lower() in {'0','false','off'}:
        return defaults
    masks, review = validated_candidates(defaults, _hq_branch_evidence.get(),
                                         image.shape, points, labels, validator)
    # Return one accepted native proposal. Multiple heads from this query must
    # not manufacture a two-query consensus in downstream pixel voting.
    if len(masks) > 1:
        selected, _ = select_candidate_v5(image, masks, box, {'verbose':False}, image.shape)
        masks = [selected] if selected is not None else []
    review['purpose'] = purpose
    sam_diagnostic_event('hq_guided_candidate_review', masks=masks, review=review,
                         points=points, labels=labels)
    if review['status'] != 'kept_default':
        _log(f"guided_candidate purpose={purpose} status={review['status']} candidates={len(masks)}")
        _sam_feedback_logits.set(None)
    stats = dict(_sam_output_stats.get() or {})
    stats.update(guidedCandidateReview=review, maskCount=len(masks))
    _sam_output_stats.set(stats)
    return masks


def _ownership_candidates(image, prompt_bbox, variant, evidence):
    # B2: keep the original multimask candidates and add the dedicated head as
    # one more proposal. Existing ownership/completeness checks decide whether
    # it is useful; the single head never replaces the original set.
    token = _single_head_allowed.set(False)
    try:
        multimask = _validated_query(image, prompt_bbox, variant,
            evidence['points'], evidence['labels'],
            lambda mask: bool(review_ownership_candidates([mask], evidence)[0]), 'target_ownership')
    finally:
        _single_head_allowed.reset(token)
    if not _SAM_SINGLE_MASK_AB:
        return multimask
    token = _single_head_allowed.set(True)
    try:
        single = _sam_candidates(image, prompt_bbox, variant,
                                 points=evidence['points'], labels=evidence['labels'])
    finally:
        _single_head_allowed.reset(token)
    combined = list(multimask)
    for candidate in single:
        array = np.asarray(candidate) > .5
        if not any(np.array_equal(array, np.asarray(existing) > .5) for existing in combined):
            combined.append(candidate)
    return combined


def _background_negative_retry(image, prompt_bbox, variant, candidates,
                               hint, policy, layer_id):
    """Ask the same SAM model to reject only evidence-backed backdrop rims.

    This is deliberately a candidate proposal pass.  It never edits pixels
    directly and keeps the original proposal when the retry loses its deep
    foreground support or fails to honour the negative anchors.
    """
    if not _SAM_BACKGROUND_NEGATIVE_AB or not candidates:
        return candidates
    reference, ranking = select_candidate_v5(
        image, [np.asarray(mask) > .5 for mask in candidates], hint,
        dict(policy, verbose=False), image.shape)
    if reference is None:
        return candidates
    negatives = propose_background_negative_points_v5(image, reference, max_points=4)
    if not negatives:
        _log(f"background_negative layer={layer_id} status=skipped reason=no_connected_evidence")
        return candidates
    distance = cv2.distanceTransform(reference.astype(np.uint8), cv2.DIST_L2, 5)
    core = reference & (distance >= max(3.0, min(image.shape[:2]) * .006))
    if int(core.sum()) < 32:
        return candidates
    ys, xs = np.where(core)
    order = np.argsort(distance[ys, xs])[::-1]
    positives = []
    for pos in order:
        point = [int(xs[pos]), int(ys[pos])]
        if all((point[0]-p[0])**2 + (point[1]-p[1])**2 >= 16**2 for p in positives):
            positives.append(point)
        if len(positives) >= 4:
            break
    if not positives:
        return candidates
    try:
        guided = _sam_candidates(
            image, prompt_bbox, variant,
            points=[positives + negatives],
            labels=[[1] * len(positives) + [0] * len(negatives)])
    except Exception as error:
        _log(f"background_negative layer={layer_id} status=error reason={error}")
        return candidates
    valid = []
    base_area = max(1, int(reference.sum()))
    core_area = max(1, int(core.sum()))
    for proposal in guided:
        trial = np.asarray(proposal) > .5
        retained = int((trial & core).sum()) / core_area
        area_ratio = int(trial.sum()) / base_area
        neg_hit = sum(bool(trial[y, x]) for x, y in negatives)
        if (retained >= .94 and .35 <= area_ratio <= 1.08 and
                neg_hit == 0 and all(trial[y, x] for x, y in positives)):
            valid.append(proposal)
    if valid:
        # Keep all safe alternatives for the normal image-evidence selector.
        candidates = list(candidates) + valid
    sam_diagnostic_event('background_negative_candidates', masks=guided,
                         layerId=layer_id,
                         sourceIndex=ranking[0]['index'] if ranking else None,
                         points=[positives + negatives],
                         labels=[[1] * len(positives) + [0] * len(negatives)],
                         negativePoints=negatives,
                         accepted=bool(valid),
                         validCount=len(valid))
    _log(f"background_negative layer={layer_id} status={'applied' if valid else 'skipped'} "
         f"negativePoints={len(negatives)} accepted={len(valid)}")
    return candidates


def _client_quality(result, hint, layer_meta):
    """Expose the V5 evidence report in the shape the client already gates on.

    Preserve the core's accept/hold decision, including reasons. A diagnostic
    cutout is not automatically a usable runtime layer.
    """
    quality = dict(result.get("quality") or {})
    mask_pixels = int(result.get("quality", {}).get("maskPixels") or 0)
    hint_area = max(1.0, float((hint[2] - hint[0]) * (hint[3] - hint[1])))
    quality.update({
        "apiVersion": API_VERSION_V5,
        "policyVersion": POLICY_VERSION_V5,
        "coreVersion": CORE_VERSION_V5,
        "strategy": "image_evidence_v5",
        "strategyProfile": "image_evidence_v5",
        "qualityProfile": _normalized_profile(
            quality.get("qualityProfile") or
            (layer_meta.get("_qualityProfile") if isinstance(layer_meta, dict) else "")
        ),
        "targetFillRatio": round(mask_pixels / hint_area, 4),
        "maskAudit": {
            "pixels": mask_pixels,
            "components": _mask_component_count(result.get("mask")),
        },
    })
    if quality.get("runtimeAction") not in {"accept", "hold"}:
        quality.update(status="low_quality", runtimeAction="hold",
                       shouldGenerateRuntimeLayer=False, needsHigherPrecision=True,
                       issues=[*(quality.get("issues") or []), "missing_v5_validation"])
    row = quality.get("candidate")
    if isinstance(row, dict):
        quality["selectedIndexes"] = [row.get("index")]
        quality["debugCandidates"] = [dict(row, selected=True)]
    return quality


def _mask_component_count(mask):
    if mask is None:
        return None
    labels_count, _ = cv2.connectedComponents(mask.astype(np.uint8))
    return int(max(0, labels_count - 1))


def _process_one(image, hint, candidates, layer_meta, context_layers, policy):
    result = process_layer_v5(
        image,
        hint,
        candidates,
        layer_meta=layer_meta,
        context_layers=context_layers,
        policy=policy,
    )
    if result is None:
        return None
    result["quality"] = _client_quality(result, hint, layer_meta)
    return result


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "apiVersion": API_VERSION_V5,
        "policyVersion": POLICY_VERSION_V5,
        "coreVersion": CORE_VERSION_V5,
    }


@app.get("/v5/info")
async def v5_info():
    """Describe the contract so a request can be debugged without reading code."""
    return {
        "apiVersion": API_VERSION_V5,
        "policyVersion": POLICY_VERSION_V5,
        "coreVersion": CORE_VERSION_V5,
        "samOutputEvidence": _SAM_OUTPUT_EVIDENCE,
        "serviceGeneration": 5,
        "defaultPort": 8005,
        "modelVariant": forced_sam_model_variant(),
        "hqModelPathEnv": "SAM_HQ_L_MODEL_PATH",
        "hqFusionReview": "legacy cancellation recovery plus source-labelled original/expanded fusion and conditional HQ-only pool",
        "samSingleMaskAB": _SAM_SINGLE_MASK_AB,
        "samMaskFeedbackAB": _SAM_MASK_FEEDBACK_AB,
        "samLocalCropAB": _SAM_LOCAL_CROP_AB,
        "samBackgroundNegativeAB": _SAM_BACKGROUND_NEGATIVE_AB,
        "componentMerge": "verified local additions; preserve existing alpha and mask",
        "boundaryAudit": "per-stage geometry, alpha and exported RGB; source-image pixel distances",
        "hqBoundaryGuard": os.environ.get('SAMV5_HQ_BOUNDARY_GUARD', '1').lower() not in {'0','false','off'},
        "hqBoundaryGuardScope": "shallow image-aligned cleanup cuts only; independent neighbour and background evidence wins",
        "hqOnlySelection": _v5_enabled('SAMV5_HQ_ONLY_CANDIDATES'),
        "candidatePool": _v5_enabled('SAMV5_CANDIDATE_POOL'),
        "dualBoxCandidates": _v5_enabled('SAMV5_DUAL_BOX_CANDIDATES'),
        "alphaBoundaryConstraint": _v5_enabled('SAMV5_ALPHA_BOUNDARY_GUARD'),
        "alphaBoundaryScope": "local color mixture evidence on original opaque hard boundaries; no fixed alpha floor",
        "singleMaskScope": "ownership retry only; box discovery and neighbour consensus unchanged",
        "contract": {
            "bboxes": "0-1000 normalized [ymin, xmin, ymax, xmax]",
            "bboxRole": "hint geometry; accepted subject alpha may extend past it; SAMV5_EXPORT_BBOX=hint opts into clipping",
            "routing": "single evidence pipeline; no category routing",
        },
    }


@app.post("/segment")
async def segment(request: Request):
    try:
        data = await request.json()
        request_id = str(
            data.get("taskId") or data.get("requestId") or
            f"segment-v5-{int(time.time() * 1000)}"
        )
        engine = _normalized_engine(data.get("engine"))
        quality_profile = _normalized_profile(data.get("qualityProfile"))

        image = _decode_image(data.get("image"))
        if image is None:
            return JSONResponse(status_code=400, content={"error": "No image provided"})
        height, width = image.shape[:2]

        bboxes_norm = data.get("bboxes") or []
        layer_ids = data.get("layerIds") or []
        layer_metas = data.get("layers") or []
        context_layers = data.get("contextLayers", layer_metas)
        if not isinstance(context_layers, list):
            context_layers = []
        _log(
            f"request id={request_id} profile={quality_profile} engine={engine} "
            f"layers={len(layer_ids)} image={width}x{height}"
        )

        if not bboxes_norm:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "V5 只支持带 bbox hint 的分割请求",
                },
            )

        hints = [_pixel_hint(bbox, width, height) for bbox in bboxes_norm]

        def meta_for(index):
            if isinstance(layer_metas, list) and index < len(layer_metas):
                return layer_metas[index] or {}
            return {}

        def policy_for(index):
            meta = dict(meta_for(index))
            if context_layers:
                meta["_contextLayers"] = context_layers
            return resolve_policy_v5(meta, quality_profile)

        def prompt_for(index):
            policy = policy_for(index)
            return [
                int(round(v)) for v in expand_hint_bbox_v5(
                    hints[index], width, height, policy.get("hintExpandRatio", 0.18)
                )
            ]

        cutouts = []
        with sam_diagnostic_request(data, image, request_id):
            if engine == "fastsam":
                cutouts = _run_fastsam(image, hints, layer_ids, meta_for, policy_for)
                return JSONResponse(content={
                    "success": True, "engine": "fastsam", "cutouts": cutouts,
                    "apiVersion": API_VERSION_V5,
                })

            with sam_runtime_lock, sam_embedding_cache(), sam_request_metrics() as metrics:
                cutouts = _run_sam(
                    image, hints, layer_ids, layer_metas, context_layers,
                    meta_for, policy_for, prompt_for, request_id,
                )
            _log(f"metrics id={request_id}: {metrics.summary()}")

        if len(layer_ids) == 1 and not cutouts:
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": "V5 证据流水线未通过候选筛选，未返回 cutout",
                    "layerId": layer_ids[0] if layer_ids else None,
                    "qualityProfile": quality_profile,
                },
            )
        return JSONResponse(content={
            "success": True, "engine": "sam", "cutouts": cutouts,
            "apiVersion": API_VERSION_V5,
        })
    except Exception as error:  # pragma: no cover - transport guard
        print(f"SAMV5 transport failed: {error}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(error)})


def _run_sam(image, hints, layer_ids, layer_metas, context_layers,
             meta_for, policy_for, prompt_for, request_id):
    cutouts = []
    pending = []
    for index, layer_id in enumerate(layer_ids):
        sam_diagnostic_layer(index, layer_id)
        layer_meta = dict(meta_for(index))
        policy = policy_for(index)
        prompt_bbox = prompt_for(index)
        hint = hints[index]
        variant = forced_sam_model_variant(str(layer_meta.get("samModelVariant") or "l"))

        candidates = _sam_candidates(image, prompt_bbox, variant)
        # Keep the native default for geometry accounting, never as a forced
        # output. Retry policies inherit this immutable reference.
        if candidates:
            policy = dict(policy, _hqDefaultBoundaryReference=np.asarray(candidates[0]) > .5)
        sam_diagnostic_event('hq_default_fusion', masks=candidates, layerId=layer_id)
        branch_evidence = _hq_branch_evidence.get()
        if branch_evidence and 'allFullResolutionLogits' in branch_evidence:
            full = np.asarray(branch_evidence['allFullResolutionLogits'])
            if full.shape == (9, *image.shape[:2]) and 'maskThreshold' in branch_evidence:
                sam_diagnostic_event('hq_internal_diagnostic_candidates',
                    masks=[full[i] > branch_evidence['maskThreshold'] for i in (4, 6, 7, 8)],
                    sources=['hq_only', 'fused1', 'fused2', 'fused3'],
                    candidateSelectionChanged=False, layerId=layer_id)
        candidates, hq_review = review_hq_candidates(
            image, hint, candidates, _hq_branch_evidence.get(), policy)
        tight_evidence = None
        prompt_review = dict(status='skipped',reason='disabled_or_same_box')
        if (_v5_enabled('SAMV5_CANDIDATE_POOL') and _v5_enabled('SAMV5_DUAL_BOX_CANDIDATES')
                and list(hint)!=list(prompt_bbox)):
            saved_stats,saved_logits=_sam_output_stats.get(),_sam_feedback_logits.get()
            try:
                _sam_candidates(image,hint,variant)
                tight_evidence=_hq_branch_evidence.get()
                prompt_review=dict(status='decoded',bbox=list(hint))
            except Exception as error:
                prompt_review=dict(status='unresolved',reason=str(error))
            finally:
                _sam_output_stats.set(saved_stats);_sam_feedback_logits.set(saved_logits)
        def verify_pool_islands(reference):
            saved = (_sam_output_stats.get(), _sam_feedback_logits.get(),
                     _hq_branch_evidence.get())
            try:
                cut, review = verify_neighbours(
                    image, hint, reference, policy,
                    lambda box, points, labels: _sam_candidates(
                        image, box, variant, points=points, labels=labels),
                    decode_validated=lambda box,p,l,v,why:_validated_query(image,box,variant,p,l,v,why))
                sam_diagnostic_event('candidate_island_ownership', masks=[cut],
                                     layerId=layer_id, review=review)
                return cut, review
            finally:
                _sam_output_stats.set(saved[0]); _sam_feedback_logits.set(saved[1])
                _hq_branch_evidence.set(saved[2])
        candidates,pool_review=choose_candidate_pool(
            image,hint,candidates,branch_evidence,tight_evidence,policy,
            verify_uncertain=verify_pool_islands)
        pool_review['originalPrompt']=prompt_review
        sam_diagnostic_event('hq_candidate_pool',masks=candidates,layerId=layer_id,review=pool_review)
        _log(f"candidate_pool layer={layer_id} status={pool_review['status']} source={pool_review['selectedSource']}")
        _hq_branch_evidence.set(None)
        if pool_review['status']=='accepted':
            _sam_output_stats.set(dict(maskCount=len(candidates),modelScores=[],
                outputMode='hq_verified_prompt_branch_pool',source=pool_review['selectedSource'],
                scoreMeaning='fixed subject/part/ownership retention and image improvement; not predicted IoU'))
            _sam_feedback_logits.set(None)
        elif hq_review['status'] in {'accepted', 'proposed'}:
            # Default IoU/logits belong to the discarded head. Keep provenance
            # explicit; do not feed stale logits into a later feedback retry.
            _sam_output_stats.set(dict(maskCount=len(candidates), modelScores=[],
                outputMode='hq_verified_alternative', headIndices=[hq_review['selectedHead']],
                scoreMeaning='fusion validated by image/ownership evidence, not predicted IoU'))
            _sam_feedback_logits.set(None)
        sam_diagnostic_event('hq_candidate_review', masks=candidates,
                             layerId=layer_id, hqReview=hq_review)
        _log(f"hq_candidate_review layer={layer_id} status={hq_review['status']} "
             f"reason={hq_review['reason']} head={hq_review.get('selectedHead', hq_review.get('defaultHead'))}")
        if _SAM_MASK_FEEDBACK_AB:
            feedback_logits = _sam_feedback_logits.get()
            feedback = []
            if feedback_logits is not None:
                logits = np.asarray(feedback_logits, dtype=np.float32)
                if logits.ndim == 4:
                    logits = logits.reshape(-1, *logits.shape[-2:])
                if logits.ndim == 3:
                    for logit in logits[:3]:
                        try:
                            feedback.extend(_sam_candidates(
                                image, prompt_bbox, variant, masks=[logit]))
                        except Exception as error:
                            # Some installed SAM2 checkpoints expose a dense
                            # prompt shape incompatible with the Ultralytics
                            # predictor wrapper. Fall back to sparse points
                            # sampled from the same logits; this keeps D an
                            # evidence feedback pass without changing alpha.
                            x1,y1,x2,y2 = [int(v) for v in prompt_bbox]
                            crop = np.asarray(logit)[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                            if crop.size:
                                py,px = np.unravel_index(int(np.argmax(crop)), crop.shape)
                                ny,nx = np.unravel_index(int(np.argmin(crop)), crop.shape)
                                pts = [[x1+int(px), y1+int(py)], [x1+int(nx), y1+int(ny)]]
                                try:
                                    feedback.extend(_sam_candidates(
                                        image, prompt_bbox, variant,
                                        points=[pts], labels=[[1, 0]]))
                                    _log(f"mask_feedback layer={layer_id} mode=sparse_fallback")
                                except Exception as point_error:
                                    _log(f"mask_feedback layer={layer_id} status=error reason={point_error}")
                            else:
                                _log(f"mask_feedback layer={layer_id} status=error reason={error}")
            existing = [np.asarray(m) > .5 for m in candidates]
            for candidate in feedback:
                binary = np.asarray(candidate) > .5
                if not any(np.array_equal(binary, prior) for prior in existing):
                    candidates.append(candidate)
                    existing.append(binary)
            sam_diagnostic_event('mask_feedback_candidates', masks=feedback,
                                 layerId=layer_id,
                                 status='applied' if feedback else 'skipped',
                                 sourceHeads=int(len(feedback_logits)) if feedback_logits is not None else 0)
            _log(f"mask_feedback layer={layer_id} status={'applied' if feedback else 'skipped'} candidates={len(feedback)}")
        polarity = polarity_prompt(image, hint, candidates)
        if polarity is not None:
            guided = _sam_candidates(image, prompt_bbox, variant,
                                     points=polarity['points'], labels=polarity['labels'])
            valid = [m for m in guided
                     if all(m[y,x]>.5 for x,y in polarity['positive'])
                     and not any(m[y,x]>.5 for x,y in polarity['negative'])
                     and ((np.asarray(m)>.5)&polarity['reference']).sum()
                         /max(1,polarity['reference'].sum())>=.80
                     and (np.asarray(m)>.5).sum()<=polarity['reference'].sum()*1.45]
            if valid:
                candidates = valid
            sam_diagnostic_event('polarity_review', masks=guided, layerId=layer_id,
                                 points=polarity['points'], labels=polarity['labels'],
                                 status='accepted' if valid else 'unresolved')
            _log(f"polarity layer={layer_id} status={'accepted' if valid else 'unresolved'} model={variant}")
        # Candidate-level background rejection.  It runs after the existing
        # polarity pass so both prompts remain independently observable.
        candidates = _background_negative_retry(
            image, prompt_bbox, variant, candidates, hint, policy, layer_id)


        sam_diagnostic_event(
            "candidates",
            masks=candidates,
            hintBbox=hint,
            promptBbox=prompt_bbox,
            policyVersion=POLICY_VERSION_V5,
            modelVariant=variant,
            layerId=layer_id,
            outputMode=(_sam_output_stats.get() or {}).get('outputMode','unknown'),
            outputStats=_sam_output_stats.get() or {},
        )
        # Keep the baseline evidence fixed while asking the SAME model to
        # resolve instance ownership. Shared request embedding avoids encoding
        # the image again; only its prompt decoder runs a second time.
        reference, original_ranking = select_candidate_v5(
            image, [np.asarray(mask) > .5 for mask in candidates], hint,
            dict(policy, verbose=False), image.shape) if candidates else (None, [])
        sam_diagnostic_event('candidate_selection', layerId=layer_id,
                             candidateRanking=original_ranking,
                             outputMode=(_sam_output_stats.get() or {}).get('outputMode','unknown'),
                             outputStats=_sam_output_stats.get() or {})
        _log(f"candidate_selection layer={layer_id} model={variant} "
             f"top={original_ranking[0] if original_ranking else None}")
        evidence = build_ownership_evidence(hint, candidates, policy, image.shape,
                                            reference=reference)
        if evidence is None and reference is not None:
            neighbour, independent_review = verify_neighbours(
                image, hint, reference, policy,
                lambda box, points, labels: _sam_candidates(
                    image,box,variant,points=points,labels=labels),
                decode_validated=lambda box,p,l,v,why:_validated_query(image,box,variant,p,l,v,why))
            if independent_review:
                policy = dict(policy,_ownershipNeighbourMask=neighbour,
                              _independentNeighbourReview=independent_review)
                sam_diagnostic_event('independent_neighbour',masks=[neighbour],
                                     layerId=layer_id,review=independent_review)
                _log(f"independent_neighbour layer={layer_id} removedPixels={int(neighbour.sum())} review={independent_review}")
        _, baseline_review = review_ownership_candidates(
            [np.asarray(mask) > .5 for mask in candidates], evidence)
        if evidence is not None:
            policy = dict(policy, _ownershipEvidence=evidence,
                          _ownershipBaselineMasks=[np.asarray(mask) > .5 for mask in candidates])
            best_index = original_ranking[0]['index'] if original_ranking else None
            disputed = (best_index is not None and
                        not baseline_review['candidates'][best_index]['accepted'])
            if disputed:
                guided = _ownership_candidates(image, prompt_bbox, variant, evidence)
                valid, review = review_ownership_candidates(
                    [np.asarray(mask) > .5 for mask in guided], evidence)
                # Preserve the established retry as fallback. Spatial prompts
                # are a separate bounded attempt, never a new reason to turn a
                # previously accepted whole subject into hold.
                spatial = augment_ownership(evidence, reference)
                if valid and spatial is not evidence:
                    retained, spatial_review = review_ownership_candidates(valid, spatial)
                    if not retained:
                        local_guided = _ownership_candidates(image, prompt_bbox, variant, spatial)
                        retained, spatial_review = review_ownership_candidates(local_guided, spatial)
                        sam_diagnostic_event('local_integrity_verification', masks=local_guided,
                            layerId=layer_id, points=spatial['points'], labels=spatial['labels'],
                            localIntegrityReview=spatial_review)
                        _log(f"local_integrity layer={layer_id} status={'accepted' if retained else 'kept_baseline'}")
                    if retained:
                        valid = retained
                    review['localIntegrity'] = spatial_review
                review['baseline'] = baseline_review
                review['outputMode'] = (_sam_output_stats.get() or {}).get('outputMode','unknown')
                review['modelVariant'] = variant
                review['promptPasses'] = 2
                # A failed ownership retry is reported as a hold by the core;
                # do not silently fall back to an ambiguous merged proposal.
                policy['_ownershipRetryUnresolved'] = not bool(valid)
                policy['_ownershipReview'] = review
                sam_diagnostic_event('ownership_candidates', masks=guided,
                                     layerId=layer_id, ownershipReview=review,
                                     points=evidence['points'], labels=evidence['labels'])
                if valid:
                    candidates = valid
                    # Ask for the competing instance too: an occluded leg can
                    # be disconnected from its visible upper structure.
                    groups = evidence.get('neighbourGroups') or [
                        {'region':evidence['excluded'],'negative':evidence['negative']}]
                    neighbour = np.zeros(image.shape[:2], bool)
                    group_reviews = []
                    for group_index, group in enumerate(groups):
                        if not group['negative']:
                            group_reviews.append({'index':group_index,'status':'unseeded'})
                            continue
                        group_evidence = dict(evidence,negative=group['negative'],excluded=group['region'])
                        x, y, w, h = cv2.boundingRect(group['region'].astype(np.uint8))
                        neighbour_points = [group['negative'] + evidence['positive']]
                        neighbour_labels = [[1]*len(group['negative']) + [0]*len(evidence['positive'])]
                        competing = _validated_query(
                            image, [x,y,x+w,y+h], variant, neighbour_points, neighbour_labels,
                            lambda m: True, 'ownership_competing_instance')
                        # A multihead query is still one query. Confirmation
                        # needs a second prompt, with fewer neighbour seeds.
                        confirmation = _validated_query(
                            image, [x,y,x+w,y+h], variant,
                            [group['negative'][:1] + evidence['positive']],
                            [[1] + [0]*len(evidence['positive'])],
                            lambda m: all(m[py,px] for px,py in group['negative']),
                            'ownership_competing_confirmation') if competing and len(group['negative'])>1 else []
                        confirmed = neighbour_consensus(competing + confirmation, group_evidence)
                        neighbour |= confirmed
                        group_reviews.append({'index':group_index,'status':'confirmed' if confirmed.any() else 'unresolved',
                                              'pixels':int(confirmed.sum())})
                        sam_diagnostic_event('ownership_neighbour', masks=competing,
                                             layerId=layer_id, ownershipReview=review,
                                             groupIndex=group_index,points=[group['negative']+evidence['positive']],
                                             labels=[[1]*len(group['negative'])+[0]*len(evidence['positive'])])
                    policy['_ownershipNeighbourMask'] = neighbour
                    review['neighbourGroups'] = group_reviews
                    review['promptPasses'] = (2 + sum(bool(g['negative']) for g in groups)
                                              + sum(len(g['negative'])>1 for g in groups))
                    review['neighbourConsensusPixels'] = int(neighbour.sum())
                _log(f"ownership layer={layer_id} status={review['status']} "
                     f"model={variant} head={review['outputMode']} positives={len(evidence['positive'])} "
                     f"negatives={len(evidence['negative'])}")
            else:
                policy['_ownershipReview'] = baseline_review
        result = _process_one(image, hint, candidates, layer_meta, context_layers, policy)

        if result is None and _ESCALATE_ENABLED and variant not in {"l", "hq-l"}:
            _log(f"escalate layer={layer_id} reason=no_evidence_result")
            candidates = _sam_candidates(image, prompt_bbox, "l")
            sam_diagnostic_event(
                "candidates_escalated",
                masks=candidates,
                hintBbox=hint,
                promptBbox=prompt_bbox,
                policyVersion=POLICY_VERSION_V5,
                modelVariant="l",
                layerId=layer_id,
            )
            result = _process_one(image, hint, candidates, layer_meta, context_layers, policy)

        if result is None:
            _log(f"layer_failed layer={layer_id} hint={hint}")
            continue

        completion = result.get('completionEvidence')
        if completion is not None and result['quality'].get('runtimeAction') == 'accept':
            # Image evidence proposes a missing region; SAM must independently
            # include its interior while preserving the existing whole target.
            positive = (evidence['positive'] if evidence is not None else
                        _interior_points(result['mask'], 4))
            negative = evidence['negative'] if evidence is not None else []
            points = positive + completion['points'] + negative
            labels = [1]*(len(positive)+len(completion['points'])) + [0]*len(negative)
            def completion_valid(m):
                old=result['mask']
                return ((m & old).sum()/max(1,old.sum())>=.90
                        and m.sum()<=old.sum()*1.45
                        and (evidence is None or bool(review_ownership_candidates([m],evidence)[0])))
            verified = _validated_query(image, prompt_bbox, variant,
                [points], [labels], completion_valid, 'completion')
            accepted = []
            old_mask = result['mask']
            for proposal in verified:
                proposal = np.asarray(proposal) > .5
                retains = int((proposal & old_mask).sum()) / max(1, int(old_mask.sum()))
                if (retains >= .90 and proposal.sum() <= old_mask.sum()*1.45 and
                        all(proposal[y, x] for x, y in positive + completion['points']) and
                        not any(proposal[y, x] for x, y in negative)):
                    accepted.append(proposal)
            if evidence is not None:
                accepted, _ = review_ownership_candidates(accepted, evidence)
            completion_policy = dict(policy, _ownershipEvidence=evidence,
                                     _ownershipEvidenceFrozen=True)
            replacement = (_process_one(image, hint, accepted, layer_meta, context_layers, completion_policy)
                           if accepted else None)
            adopted = bool(replacement is not None and
                           replacement['quality'].get('runtimeAction') == 'accept' and
                           all(replacement['alpha'][y, x] >= 128 for x, y in completion['points']))
            if adopted:
                replacement = merge_verified_completion(result,replacement,completion)
                valid_final, _ = review_ownership_candidates([replacement['alpha'] >= 128],evidence)
                adopted = bool(valid_final and all(replacement['alpha'][y,x] >= 128
                                                   for x,y in completion['points']))
            review = {'status': 'accepted' if adopted else 'unresolved',
                      'points': completion['points'], 'proposedPixels': int(completion['mask'].sum()),
                      'retentionFloor': .90, 'modelVariant': variant}
            if adopted:
                review['addedPixels'] = int((replacement['mask'] & ~old_mask).sum())
                result = replacement
            else:
                # Image-only proposals can be backdrop or shadow. Failed
                # verification is uncertainty, not proof of subject loss; the
                # original whole-target ownership/quality gates still apply.
                result['quality']['warnings'] = [*(result['quality'].get('warnings') or []),
                                                'completion_verification_unresolved']
            result['quality']['completionReview'] = review
            sam_diagnostic_event('completion_verification', masks=verified,
                                 layerId=layer_id, completionReview=review)
            _log(f"completion_verification layer={layer_id} status={review['status']} "
                 f"addedPixels={review.get('addedPixels', 0)}")

        if (variant in {'l', 'hq-l'} and result['quality'].get('runtimeAction') == 'accept'
                and result.get('policy',{}).get('material') == 'opaque'):
            proposal = component_proposal(image, result['mask'], policy)
            if proposal is not None:
                old_mask = result['mask']
                positive = evidence['positive'] if evidence is not None else _interior_points(old_mask,4)
                negative = evidence['negative'] if evidence is not None else []
                points = positive + proposal['points'] + negative
                labels = [1]*(len(positive)+len(proposal['points']))+[0]*len(negative)
                verified = _validated_query(image,prompt_bbox,variant,[points],[labels],
                    lambda m:review_component_candidate(m,old_mask,proposal,positive,negative,evidence),
                    'component_recovery')
                candidate_reviews=[]
                valid=[]
                for m in verified:
                    check={'source':'global_prompt'}
                    if review_component_candidate(m,old_mask,proposal,positive,negative,evidence,audit=check):
                        valid.append(np.asarray(m)>.5)
                    candidate_reviews.append(check)
                source = 'global_prompt'
                sam_diagnostic_event('component_verification',masks=verified,layerId=layer_id,
                                     points=[points],labels=[labels],status='accepted' if valid else 'unresolved')
                if not valid and _SAM_LOCAL_CROP_AB:
                    local = local_component_inputs(image,old_mask,proposal,evidence)
                    local_masks = _sam_candidates(local['image'],local['bbox'],variant,
                                                  points=local['points'],labels=local['labels'])
                    local_evidence=_hq_branch_evidence.get()
                    x,y,X,Y=local['crop']
                    local_proposals=local_fusion_proposals(local_masks,local_evidence)
                    local_support_masks=[m for m,_ in local_proposals]
                    for m,head_source in local_proposals:
                        full=np.zeros_like(old_mask);full[y:Y,x:X]=np.asarray(m)>.5
                        candidate=old_mask|full
                        # A local SAM part may omit the body because the body
                        # is outside its prompt. Existing pixels are kept, but
                        # the requested new anchors must be in the local mask.
                        check={'source':'local_crop','head':head_source}
                        if not all(full[py,px] for px,py in proposal['points']):
                            check.update(accepted=False,reason='local_positive_anchor_missing')
                        elif any(full[py,px] for px,py in local['negative']):
                            check.update(accepted=False,reason='local_background_anchor_included')
                        elif review_local_component(candidate,old_mask,proposal,
                                                    positive,negative,evidence,
                                                    local_support_masks,(x,y,X,Y),audit=check):
                            valid.append(candidate)
                        candidate_reviews.append(check)
                        if valid:
                            break  # Keep a valid default; alternatives are failure fallback only.
                    source='local_crop'
                    sam_diagnostic_event('component_local_verification',masks=local_masks,
                                         layerId=layer_id,crop=local['crop'],promptBbox=local['bbox'],
                                         points=local['points'],labels=local['labels'],
                                         candidateReviews=candidate_reviews,
                                         status='accepted' if valid else 'unresolved')
                replacement=None
                if valid:
                    # Choose the smallest verified extension, not the broadest
                    # proposal that could absorb an adjacent surface.
                    candidate=min(valid,key=lambda m:int((m&~old_mask).sum()))
                    confirmed=candidate & proposal['region']
                    recovery_policy=dict(policy,_componentVerifiedSupport=confirmed,
                                         _ownershipEvidence=evidence,_ownershipEvidenceFrozen=True)
                    replacement=_process_one(image,hint,[candidate],layer_meta,context_layers,recovery_policy)
                adopted=bool(replacement is not None
                             and replacement['quality'].get('runtimeAction')=='accept'
                             and all(replacement['alpha'][py,px]>=128 for px,py in proposal['points'])
                             and (replacement['mask']&old_mask).sum()/max(1,old_mask.sum())>=.97)
                if adopted:
                    replacement=merge_component_alpha(result,replacement,proposal)
                    # Validate the actual merged alpha, not the discarded
                    # full-frame reprocessing result.
                    adopted=bool(all(replacement['alpha'][py,px]>=128
                                     for px,py in proposal['points'])
                                 and (replacement['mask'] & ~old_mask).any()
                                 and (review_component_candidate(
                                     replacement['mask'],old_mask,proposal,
                                     positive,negative,evidence)
                                      or (source=='local_crop' and review_local_component(
                                          replacement['mask'],old_mask,proposal,positive,negative,evidence,
                                          local_support_masks,(x,y,X,Y)))))
                review={'status':'accepted' if adopted else 'unresolved','source':source,
                        'points':proposal['points'],'proposedPixels':int(proposal['mask'].sum()),
                        'modelVariant':variant,'retentionFloor':.97,'candidateReviews':candidate_reviews}
                if adopted:
                    review['addedPixels']=int((replacement['mask']&~old_mask).sum())
                    result=replacement
                else:
                    result['quality']['warnings']=[*(result['quality'].get('warnings') or []),
                                                    'component_completeness_unverified']
                    result['quality']['needsComponentReview']=True
                result['quality']['componentRecovery']=review
                result['quality']['validation']=dict(result['quality'].get('validation') or {},
                    componentCompleteness='verified_proposal' if adopted else 'unverified')
                _log(f"component_recovery layer={layer_id} status={review['status']} "
                     f"source={source} addedPixels={review.get('addedPixels',0)} checks={candidate_reviews}")
        if hq_review['status'] == 'proposed':
            hq_review['finalOwnershipStatus'] = (policy.get('_ownershipReview') or {}).get('status')
            hq_review['finalRuntimeAction'] = result['quality'].get('runtimeAction')
        result['quality']['hqCandidateReview'] = hq_review
        result['quality']['candidatePoolReview'] = pool_review
        pending.append({'id':layer_id,'meta':layer_meta,'hint':hint,'policy':policy,
                        'result':result,'reference':reference,'evidence':evidence})
    reconcile_layers(pending)
    for export_index, record in enumerate(pending):
        layer_id, result, hint = record['id'], record['result'], record['hint']
        sam_diagnostic_layer(export_index, layer_id)
        result['quality'] = _client_quality(result,hint,record['meta'])
        if result['quality'].get('jointOwnership'):
            review=result['quality']['jointOwnership']
            _log(f"joint_ownership layer={layer_id} status={review['status']} "
                 f"removedPixels={review.get('removedPixels',0)}")
        audit = result.get('_boundaryAudit')
        if audit is not None:
            audit.record('transport_final_after_recovery_and_joint_ownership', result['alpha'] >= 128,
                         alpha=result['alpha'])
            result['quality']['boundaryAudit'] = audit.summary()
        entry = build_cutout_entry_v5(image, result, layer_id, extract_engine="sam_bbox_prompt_v5")
        if entry is None:
            continue
        entry["quality"]["hintBbox"] = hint
        sam_diagnostic_event("alpha", alpha=result["alpha"], bbox=entry["bbox"], layerId=layer_id)
        cutouts.append(entry)
        _log(
            f"layer_done layer={layer_id} bbox={result['bbox']} "
            f"alphaPixels={result['quality'].get('alphaPixels')} "
            f"edge={result['policy'].get('edge')} material={result['policy'].get('material')}"
        )
    return cutouts


def _run_fastsam(image, hints, layer_ids, meta_for, policy_for):
    fastsam = get_fastsam_model()
    results = fastsam(image, retina_masks=True, imgsz=_SAM_IMGSZ, conf=0.25, iou=0.9)
    height, width = image.shape[:2]
    candidates = _result_masks(results, width, height)
    del results

    cutouts = []
    for index, layer_id in enumerate(layer_ids):
        layer_meta = dict(meta_for(index))
        result = _process_one(
            image, hints[index], candidates, layer_meta, [], policy_for(index)
        )
        if result is None:
            continue
        entry = build_cutout_entry_v5(image, result, layer_id, extract_engine="fastsam_multi_mask_v5")
        if entry is None:
            continue
        cutouts.append(entry)
    return cutouts
