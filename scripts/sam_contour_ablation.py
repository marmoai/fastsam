#!/usr/bin/env python3
"""Offline contour ablations against saved SAM masks and manual labels."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from sam_label_evaluation import LABELS, boundary, event_mask, find_diagnostics, load_truth, metrics, rectangle
from sam_variable_test import CASES, DIAGNOSTICS, ROOT
from matte_ops import build_hard_edge_alpha


RUN = ROOT / "fastsam-backend" / "runs" / "variable-tests" / "20260924-093700"


def edge_signal(image: np.ndarray, truth: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    distance = distance_transform_edt(~boundary(truth))
    on_edge = gradient[distance <= 1]
    near_edge = gradient[(distance >= 3) & (distance <= 7)]
    return {
        "medianGradientOnLabelEdge": round(float(np.median(on_edge)), 2),
        "medianGradientNearby": round(float(np.median(near_edge)), 2),
        "edgeToNearbyRatio": round(float(np.median(on_edge) / max(1, np.median(near_edge))), 3),
        "fractionLabelEdgeWithGradientBelow20": round(float(np.mean(on_edge < 20)), 3),
    }


def boundary_band_merge(base: np.ndarray, proposed: np.ndarray, radius: int) -> np.ndarray:
    edge_distance = distance_transform_edt(~boundary(base))
    editable = edge_distance <= radius
    return (base & ~editable) | (proposed & editable)


def main() -> None:
    run_report = json.loads((RUN / "report.json").read_text(encoding="utf-8"))
    diagnostic_dirs = find_diagnostics(run_report)
    output = RUN / f"contour-ablation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output.mkdir()
    report = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for case_name, case in run_report.items():
        run = case["runs"]["baseline"]
        directory = diagnostic_dirs[run["requestHash"]]
        events = json.loads((directory / "summary.json").read_text(encoding="utf-8"))["events"]
        with Image.open(DIAGNOSTICS / CASES[case_name] / "source.png") as source:
            image = np.asarray(source.convert("RGB"))
            size = source.size
        for layer_id in run["layers"]:
            truth, _, _ = load_truth(LABELS / f"{layer_id}.png", size)
            stages = {
                event["stage"]: event
                for event in events if (event.get("layer") or {}).get("id") == layer_id
            }
            target = rectangle(size, stages["selected_mask"]["targetBbox"])
            selected = event_mask(directory, stages["selected_mask"]) & target
            post = event_mask(directory, stages["postprocess_mask"]) & target
            final = event_mask(directory, stages["final_alpha"]) & target
            hole_event = stages.get("interior_hole_decision")
            hole = (event_mask(directory, hole_event) & target) if hole_event else None
            close = (cv2.morphologyEx(post.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0) & target
            median = (cv2.medianBlur(post.astype(np.uint8), 3) > 0) & target
            row = {
                "sourceSize": size,
                "edgeSignal": edge_signal(image, truth),
                "variants": {
                    "selected": metrics(selected, truth),
                    "postprocess": metrics(post, truth),
                    "finalAlpha": metrics(final, truth),
                    "close3": metrics(close, truth),
                    "median3": metrics(median, truth),
                },
            }
            if hole is not None:
                row["variants"]["afterHoleRecovery"] = metrics(hole, truth)
                if case_name == "food":
                    bbox = stages["selected_mask"]["targetBbox"]
                    for name, source_mask in (("afterHoleAlphaCurrentCode", hole), ("postprocessAlphaCurrentCode", post)):
                        alpha = build_hard_edge_alpha(source_mask.astype(np.float32), bbox)
                        alpha = (alpha >= 128) & source_mask & target
                        row["variants"][name] = metrics(alpha, truth)
                for radius in (2, 4, 6):
                    merged = boundary_band_merge(hole, post, radius) & target
                    row["variants"][f"postOnlyNearPriorEdge{radius}px"] = metrics(merged, truth)
                row["holeToPostprocess"] = {
                    "removedPixels": int((hole & ~post).sum()),
                    "addedPixels": int((post & ~hole).sum()),
                    "holeAddedPixelsRetained": int(((hole & ~selected) & post).sum()),
                    "holeAddedPixels": int((hole & ~selected).sum()),
                }
            report[f"{case_name}/{layer_id}"] = row
            print(f"EVALUATED {case_name}/{layer_id}", flush=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
