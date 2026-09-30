#!/usr/bin/env python3
"""Offline image-only acceptance audit for local SAM contour proposals."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sam_label_evaluation import LABELS, boundary, event_mask, find_diagnostics, load_truth, metrics, rectangle
from sam_variable_test import CASES, DIAGNOSTICS, ROOT
from mask_ops import cleanup_mask
from matte_ops import generate_alpha_matte


def candidate_consensus(directory: Path, event: dict, shape: tuple[int, int]) -> np.ndarray:
    masks = []
    for entry in event.get("images") or []:
        candidate = directory / Path(entry["file"]).with_suffix(".npy")
        if candidate.exists():
            masks.append(np.load(candidate) > 0.5)
    if not masks:
        return np.zeros(shape, dtype=np.uint8)
    return np.sum(np.stack(masks), axis=0, dtype=np.uint8)


def image_evidence(image: np.ndarray, base: np.ndarray, proposal: np.ndarray, votes: np.ndarray) -> dict:
    added = proposal & ~base
    removed = base & ~proposal
    changed = added | removed
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    local_gradient = cv2.dilate(gradient, np.ones((3, 3), dtype=np.uint8))
    change_neighborhood = cv2.dilate(changed.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) > 0
    old_boundary = boundary(base) & change_neighborhood
    new_boundary = boundary(proposal) & change_neighborhood
    old_strength = float(np.median(local_gradient[old_boundary])) if old_boundary.any() else 0.0
    new_strength = float(np.median(local_gradient[new_boundary])) if new_boundary.any() else 0.0
    stable_core = cv2.erode(base.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=2) > 0
    base_boundary_band = cv2.dilate(boundary(base).astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) > 0
    return {
        "added": int(added.sum()),
        "removed": int(removed.sum()),
        "stableCoreRemoved": int((removed & stable_core).sum()),
        "editsOutside2pxBoundaryBand": int((changed & ~base_boundary_band).sum()),
        "oldEdgeStrength": round(old_strength, 2),
        "newEdgeStrength": round(new_strength, 2),
        "newToOldEdgeStrength": round(new_strength / max(1.0, old_strength), 3),
        "addedSupportedBy2BCandidates": round(float(np.mean(votes[added] >= 2)), 3) if added.any() else None,
        "removedSupportedBy2BCandidates": round(float(np.mean(votes[removed] >= 2)), 3) if removed.any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    test_report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    directories = find_diagnostics(test_report)
    output = run / f"refine-gate-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output.mkdir()
    report = {}
    for case_name, case in test_report.items():
        baseline = case["runs"]["baseline"]
        directory = directories[baseline["requestHash"]]
        events = json.loads((directory / "summary.json").read_text(encoding="utf-8"))["events"]
        with Image.open(DIAGNOSTICS / CASES[case_name] / "source.png") as source:
            image = np.asarray(source.convert("RGB"))
            size = source.size
        for layer_id in baseline["layers"]:
            truth, _, _ = load_truth(LABELS / f"{layer_id}.png", size)
            stages = {
                event["stage"]: event
                for event in events if (event.get("layer") or {}).get("id") == layer_id
            }
            if not {"selected_mask", "postprocess_mask", "b_candidates"} <= stages.keys():
                raise ValueError(f"Missing stage for {case_name}/{layer_id}")
            target = rectangle(size, stages["selected_mask"]["targetBbox"])
            prior_event = stages.get("interior_hole_decision", stages["selected_mask"])
            prior = event_mask(directory, prior_event) & target
            proposal = event_mask(directory, stages["postprocess_mask"]) & target
            target_bbox = stages["selected_mask"]["targetBbox"]
            votes = candidate_consensus(directory, stages["b_candidates"], prior.shape)
            evidence = image_evidence(image, prior, proposal, votes)
            blur_edge_ratios = {
                str(sigma): image_evidence(
                    cv2.GaussianBlur(image, (0, 0), sigmaX=sigma), prior, proposal, votes
                )["newToOldEdgeStrength"]
                for sigma in (0.6, 1.2)
            }
            policy = stages["b_candidates"].get("policy") or {}
            strategy = policy.get("selectionType")
            local_refine_eligible = bool(policy.get("allowLocalRefine"))
            replayed_cleanup = cleanup_mask(prior.astype(np.float32), target_bbox) > 0.5
            replayed_cleanup &= target
            cleanup_matches_post = bool(np.array_equal(replayed_cleanup, proposal))
            # This audit isolates the cleanup step only where its output exactly
            # reproduces the saved postprocess mask; other paths remain untouched.
            eligible = strategy == "hard_product" and cleanup_matches_post and not local_refine_eligible
            area = int(prior.sum())
            core_limit = max(8, round(area * 0.0005))
            core_safe = evidence["stableCoreRemoved"] <= core_limit
            edge_better = evidence["newToOldEdgeStrength"] >= 1.0
            # These are predeclared, conservative offline gates; none reads the label.
            core_gate_accept = not eligible or core_safe
            core_edge_gate_accept = not eligible or (core_safe and edge_better)
            image_gate_mask = proposal if core_edge_gate_accept else prior
            variants = {
                "prior": metrics(prior, truth),
                "currentPostprocess": metrics(proposal, truth),
                "corePreservationGate": metrics(proposal if core_gate_accept else prior, truth),
                "coreAndEdgeGate": metrics(image_gate_mask, truth),
            }
            alpha_repeats = {}
            replayed_post_alpha = None
            if eligible:
                source_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                for name, mask in (("priorAlphaCurrentCode", prior), ("postAlphaCurrentCode", proposal), ("imageGateAlphaCurrentCode", image_gate_mask)):
                    repeated = []
                    for _ in range(3):
                        alpha = generate_alpha_matte(
                            source_bgr, mask.astype(np.float32), target_bbox,
                            strategy_type=policy.get("matteType")
                        )
                        alpha_binary = (alpha >= 128) & target
                        repeated.append(metrics(alpha_binary, truth))
                        if name == "postAlphaCurrentCode" and replayed_post_alpha is None:
                            replayed_post_alpha = alpha_binary
                    variants[name] = repeated[0]
                    alpha_repeats[name] = [
                        {"iou": item["iou"], "boundaryF1_2px": item["boundaryF1_2px"]}
                        for item in repeated
                    ]
            report[f"{case_name}/{layer_id}"] = {
                "diagnostics": directory.name,
                "strategy": strategy,
                "localRefineEligible": local_refine_eligible,
                "cleanupReplayMatchesPostprocess": cleanup_matches_post,
                "cleanupReplayDifferentPixels": int((replayed_cleanup ^ proposal).sum()),
                "cleanupGateEligible": eligible,
                "coreRemovalLimit": core_limit,
                "evidence": evidence,
                "blurredSourceEdgeRatios": blur_edge_ratios,
                "variants": variants,
                "alphaRepeatability": alpha_repeats,
                "decisions": {
                    "corePreservationGate": "accept" if core_gate_accept else "reject",
                    "coreAndEdgeGate": "accept" if core_edge_gate_accept else "reject",
                },
            }
            if eligible and "final_alpha" in stages:
                final_saved = event_mask(directory, stages["final_alpha"]) & target
                report[f"{case_name}/{layer_id}"]["postAlphaReplayDifferentPixels"] = int(
                    (final_saved ^ replayed_post_alpha).sum()
                )
            print(f"EVALUATED {case_name}/{layer_id}", flush=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
