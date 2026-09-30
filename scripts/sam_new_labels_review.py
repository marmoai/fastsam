#!/usr/bin/env python3
"""Render source-sized extraction errors for the new-label A/B report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sam_label_evaluation import LABELS, load_truth, metrics, output_alpha_from_cutout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    review = run / "visual-review"
    review.mkdir(exist_ok=True)
    rows = {}
    for case_name, case in report["cases"].items():
        with Image.open(LABELS / case["source"]) as source:
            rgb = np.asarray(source.convert("RGB"))
            size = source.size
        truth, label_gray, label_info = load_truth(LABELS / case["label"], size)
        predictions = {}
        for variant in ("A_cleanup_on", "B_skip_open"):
            entry = case["variants"][variant][0]
            path = run / variant / f"{case_name}-1.png"
            prediction = output_alpha_from_cutout(path, entry["bbox"], size)
            predictions[variant] = prediction
            overlay = rgb.copy()
            overlay[prediction & ~truth] = (230, 50, 45)
            overlay[~prediction & truth] = (45, 115, 230)
            Image.fromarray(overlay).save(review / f"{case_name}-{variant}-errors.png")
        a = predictions["A_cleanup_on"]
        b = predictions["B_skip_open"]
        difference = rgb.copy()
        difference[a & ~b] = (45, 115, 230)
        difference[b & ~a] = (230, 50, 45)
        Image.fromarray(difference).save(review / f"{case_name}-ab-difference.png")
        false_positive = a & ~truth
        count, _, stats, _ = cv2.connectedComponentsWithStats(false_positive.astype(np.uint8), connectivity=8)
        components = []
        for index in range(1, count):
            x, y, width, height, pixels = [int(v) for v in stats[index]]
            components.append({"pixels": pixels, "bbox": [x, y, x + width, y + height]})
        rows[case_name] = {
            "label": label_info,
            "abAddedPixels": int((b & ~a).sum()),
            "abRemovedPixels": int((a & ~b).sum()),
            "labelThresholdSensitivity": {
                str(threshold): {
                    variant: {
                        "iou": metrics(prediction, label_gray >= threshold)["iou"],
                        "boundaryF1_2px": metrics(prediction, label_gray >= threshold)["boundaryF1_2px"],
                    }
                    for variant, prediction in predictions.items()
                }
                for threshold in (64, 128, 192)
            },
            "largestBaselineFalsePositiveComponents": sorted(components, key=lambda item: item["pixels"], reverse=True)[:5],
        }
    (review / "report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(review)


if __name__ == "__main__":
    main()
