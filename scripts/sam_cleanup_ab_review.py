#!/usr/bin/env python3
"""Summarize final-cutout A/B changes, including lost true subject regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sam_label_evaluation import LABELS, load_truth, output_alpha_from_cutout
from sam_variable_test import CASES, DIAGNOSTICS


def components(mask: np.ndarray) -> list[dict]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    result = []
    for index in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[index]]
        result.append({"pixels": area, "bbox": [x, y, x + width, y + height]})
    return sorted(result, key=lambda item: item["pixels"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", default="A_cleanup_on")
    parser.add_argument("--candidate", default="B_cleanup_off")
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    output = run / f"visual-review-{args.candidate}"
    output.mkdir(exist_ok=True)
    summary = {}
    for case_name, case in report["cases"].items():
        source_path = DIAGNOSTICS / CASES[case_name] / "source.png"
        with Image.open(source_path) as source:
            rgb = np.asarray(source.convert("RGB"))
            size = source.size
        for layer_id in case["variants"][args.baseline][0]["layers"]:
            truth, _, _ = load_truth(LABELS / f"{layer_id}.png", size)
            rows = []
            count = min(len(case["variants"][args.baseline]), len(case["variants"][args.candidate]))
            for index in range(count):
                a_entry = case["variants"][args.baseline][index]["layers"][layer_id]
                b_entry = case["variants"][args.candidate][index]["layers"][layer_id]
                a = output_alpha_from_cutout(
                    run / args.baseline / f"{case_name}-{layer_id}-{index + 1}.png",
                    a_entry["bbox"], size,
                )
                b = output_alpha_from_cutout(
                    run / args.candidate / f"{case_name}-{layer_id}-{index + 1}.png",
                    b_entry["bbox"], size,
                )
                lost_true = a & ~b & truth
                recovered_true = ~a & b & truth
                removed_false = a & ~b & ~truth
                added_false = ~a & b & ~truth
                rows.append({
                    "repeat": index + 1,
                    "lostTrue": int(lost_true.sum()),
                    "recoveredTrue": int(recovered_true.sum()),
                    "removedFalse": int(removed_false.sum()),
                    "addedFalse": int(added_false.sum()),
                    "largestLostTrueComponents": components(lost_true)[:5],
                })
                if index == 0:
                    overlay = rgb.copy()
                    overlay[lost_true] = (230, 45, 50)
                    overlay[recovered_true] = (50, 190, 85)
                    overlay[removed_false] = (60, 115, 235)
                    overlay[added_false] = (240, 160, 40)
                    Image.fromarray(overlay).save(output / f"{layer_id}-changes.png")
            summary[layer_id] = rows
    (output / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
