#!/usr/bin/env python3
"""Compare hard-edge alpha rasterizations using saved SAM masks and labels."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sam_label_evaluation import LABELS, event_mask, find_diagnostics, load_truth, metrics
from sam_variable_test import CASES, DIAGNOSTICS, ROOT

sys.path.insert(0, str(ROOT / "fastsam-backend"))
from matte_ops import build_hard_edge_alpha


RUN = ROOT / "fastsam-backend" / "runs" / "variable-tests" / "20260924-093700"
PRIOR = ROOT / "fastsam-backend" / "runs" / "variable-tests" / "20260924-002527"


def centered_contour_alpha(mask: np.ndarray, box: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = np.asarray(mask[y1:y2, x1:x2] > 0.5, dtype=np.uint8)
    scale = 8
    hi = np.zeros((crop.shape[0] * scale, crop.shape[1] * scale), dtype=np.uint8)
    contours, hierarchy = cv2.findContours(crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is not None:
        offset = scale // 2
        for index, contour in enumerate(contours):
            scaled = contour.astype(np.int32) * scale + offset
            cv2.drawContours(hi, [scaled], 0, 0 if hierarchy[0][index][3] >= 0 else 255, thickness=cv2.FILLED)
    alpha = np.zeros(mask.shape, dtype=np.uint8)
    alpha[y1:y2, x1:x2] = cv2.resize(hi, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)
    return np.where(mask > 0.5, alpha, 0).astype(np.uint8)


def main() -> None:
    output = RUN / f"alpha-ab-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output.mkdir()
    result = {}
    for run_root, include in ((RUN, "baseline"), (PRIOR, "publish")):
        report = json.loads((run_root / "report.json").read_text(encoding="utf-8"))
        indexed = find_diagnostics(report)
        for case_name, case in report.items():
            if include not in case["runs"]:
                continue
            if include == "publish" and case_name != "armchair":
                continue
            run = case["runs"][include]
            directory = indexed[run["requestHash"]]
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            with Image.open(DIAGNOSTICS / CASES[case_name] / "source.png") as source:
                size = source.size
            for layer_id in run["layers"]:
                truth, _, _ = load_truth(LABELS / f"{layer_id}.png", size)
                events = [event for event in summary["events"] if event.get("layer", {}).get("id") == layer_id]
                by_stage = {event["stage"]: event for event in events}
                mask_stage = by_stage.get("boundary_matte") or by_stage.get("postprocess_mask")
                mask = event_mask(directory, mask_stage)
                final = event_mask(directory, by_stage["final_alpha"])
                box = by_stage["final_alpha"]["targetBbox"]
                direct = np.where(mask, build_hard_edge_alpha(mask.astype(np.float32), box), 0).astype(np.uint8)
                centered = centered_contour_alpha(mask, box)
                conservative = np.where(mask, np.maximum(direct, 128), 0).astype(np.uint8)
                row = {"sourceDiagnostics": directory.name, "maskStage": mask_stage["stage"], "maskPixels": int(mask.sum())}
                for name, alpha in (("production", final.astype(np.uint8) * 255), ("recreated", direct), ("centered", centered), ("preserveCore", conservative)):
                    binary = alpha >= 128
                    row[name] = {
                        **metrics(binary, truth),
                        "acceptedMaskPixelsLost": int((mask & ~binary).sum()),
                        "pixelsOutsideAcceptedMask": int((binary & ~mask).sum()),
                    }
                    if name in {"centered", "preserveCore"}:
                        Image.fromarray(alpha).save(output / f"{layer_id}-{include}-{name}.png")
                row["recreatedMatchesProduction"] = bool(np.array_equal(direct >= 128, final))
                result[f"{case_name}/{layer_id}/{include}"] = row
                print(f"EVALUATED {case_name}/{layer_id}/{include}", flush=True)
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
