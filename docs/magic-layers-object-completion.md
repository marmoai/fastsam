# Magic Layers Object Completion Contract

## Scope

This document defines the automatic post-extraction completion pipeline for verified occluded hard objects. It runs as part of Magic Layers and changes only the eligible extracted child asset, never the base scene or unrelated layers.

## Two Asset Representations

Every approved candidate can own two independent representations:

- `observedCutout`: the pixels actually visible in the original image. It remains immutable provenance for the original extraction.
- `canonicalAsset`: the completed transparent object generated for a verified occlusion. On success it replaces the corresponding exploded child in place and becomes the publishable layer asset.

The completed child keeps its original z-index and transform. Therefore a table, person, or other foreground layer remains in front of a completed sofa or chair; completion fills the hidden form without changing the scene's occlusion order.

## Eligibility And Validation

`layerGraph-v2` first applies a bbox pre-screen. It excludes backgrounds, text, panels, logos, vector graphics, groups, and soft-edge subjects. Both the target and the occluder must be credible hard entities.

After extraction, alpha masks are sampled locally in the target bbox. A candidate becomes automatic only when the masks show actual occluder coverage adjacent to the target silhouette and the estimated coverage is between 8% and 35%.

Small, large, unavailable-mask, and ambiguous cases remain `manual_review`. Rejected candidates never get a completion asset contract.

## Stored Contract

Approved candidates are written to `item.semanticViews.completionAssets`:

```js
{
  id: 'completion_asset_<layerId>',
  targetLayerId,
  taskId,
  taskSignature,
  status: 'ready_for_completion',
  observedCutout: { cutoutUrl, maskUrl, bbox, immutable: true },
  canonicalAsset: { status: 'not_requested', cutoutUrl: null, maskUrl: null },
  composition: {
    activeRepresentation: 'observed_cutout',
    preserveOriginalOcclusion: true,
    originalZIndex,
    occluderLayerIds,
    placementPolicy: 'replace_split_child_preserve_z_index'
  },
  generation: { state: 'not_started', attempts: [], lastError: null }
}
```

`taskSignature` combines the target, occluders, and missing-region bbox. A canonical result is retained only while that relation remains unchanged. A changed relation invalidates the prior contract instead of placing stale pixels into the scene.

## Automatic Execution Lifecycle

1. `ready_for_completion`: candidate is verified and has a frozen observed cutout.
2. `completing` -> `generating` -> `matting` -> `sam_review` when needed: automatic generation pipeline states.
3. `canonical_ready`: the canonical transparent asset has replaced the matching exploded child at the preflight crop footprint, while retaining its original z-index, rotation, and scene parent.
4. Failed or ambiguous output becomes internal `manual_review`; the old observed child remains visible and no incomplete result is published.

## Preflight Plan

`completion-preflight-v1` is now prepared for every verified automatic candidate. It records only source references, normalized crop geometry, prompt context, matte strategy, and quality thresholds. It never stores a generated image or calls a model.

The crop expands the target bbox by 20 percent on each axis, with a minimum normalized size of 180 by 180. It preserves the crop aspect ratio and requests a 1024px preferred long edge rather than forcing a square output. The transient crop bitmap is materialized only when a future executor is ready to make the model request.

The output gate rejects an opaque scene-like result, an empty subject, or an asset without a transparent border. Ambiguous results are sent to manual review; they do not replace the visible scene layer.

## Automatic Execution

After all layer masks are extracted, Magic Layers validates the candidate using the target and occluder alpha masks. The same post-extraction coordinator also runs after a user extracts one layer from the layer list. It waits until both the target and every required foreground occluder have extracted masks, then calls `executeObjectCompletion(item, assetId)` only for verified candidates. This lets individually extracted layers become eligible later when their occluder is extracted, without requiring a new full Magic Layers run. The executor materializes an expanded local crop, calls the currently selected image-edit model, runs local solid-background Matte, and invokes SAM only when the Matte result fails the quality gate.

Only an accepted result is uploaded and written to `canonicalAsset`. `layer-manager` then replaces the matching exploded child rather than creating a new Workbench item. It preserves the existing child z-index and transform, updates the active layer version, and keeps the defective observed cutout only as non-publishable provenance. There is no layer-list button or user confirmation step.
