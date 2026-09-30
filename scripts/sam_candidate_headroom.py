#!/usr/bin/env python3
"""Locate first-pass mask errors across saved SAM candidate and alpha stages."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from sam_label_evaluation import LABELS, event_mask, find_diagnostics, load_truth, metrics, rectangle
from sam_variable_test import CASES, DIAGNOSTICS, ROOT


RUN = ROOT / "fastsam-backend" / "runs" / "variable-tests" / "20260924-093700"
STAGES = ("b_candidates", "probe_candidates", "provider_candidates")


def candidate_masks(directory: Path, event: dict) -> list[np.ndarray]:
    stem = event.get("_fileStem")
    result = []
    for index, _ in enumerate(event.get("images") or []):
        path = directory / f"{stem}-{index}.npy"
        if path.exists():
            result.append(np.load(path) > 0.5)
    return result


def main() -> None:
    run_report = json.loads((RUN / "report.json").read_text(encoding="utf-8"))
    indexed = find_diagnostics(run_report)
    output = RUN / f"candidate-headroom-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output.mkdir()
    report = {}
    for case_name, case in run_report.items():
        run = case["runs"]["baseline"]
        directory = indexed[run["requestHash"]]
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        with Image.open(DIAGNOSTICS / CASES[case_name] / "source.png") as image:
            size = image.size
        for layer_id in run["layers"]:
            truth, _, _ = load_truth(LABELS / f"{layer_id}.png", size)
            stages = {}
            layer_events = []
            for index, event in enumerate(summary["events"]):
                if event.get("layer", {}).get("id") != layer_id:
                    continue
                event = {**event, "_fileStem": f"{index:03d}"}
                layer_events.append(event)
                stages[event["stage"]] = event
            target = stages["selected_mask"]["targetBbox"]
            target_mask = rectangle(size, target)
            selected_indexes = (stages["selected_mask"].get("quality") or {}).get("selectedIndexes") or []
            row = {
                "diagnostics": directory.name,
                "targetBbox": target,
                "groundTruthOutsideTarget": int((truth & ~target_mask).sum()),
                "selectedIndexes": selected_indexes,
                "stages": {},
            }
            for stage_name in STAGES:
                event = stages.get(stage_name)
                if not event:
                    continue
                masks = candidate_masks(directory, event)
                if not masks:
                    continue
                candidates = []
                union = np.zeros_like(truth)
                for index, raw in enumerate(masks):
                    clipped = raw & target_mask
                    union |= clipped
                    candidates.append({"index": index, **metrics(clipped, truth)})
                row["stages"][stage_name] = {
                    "candidates": candidates,
                    "bestIoUIndexUsingGroundTruth": max(candidates, key=lambda item: item["iou"])["index"],
                    "bestIoUUsingGroundTruth": max(item["iou"] for item in candidates),
                    "unionCoverage": metrics(union, truth),
                }
            for stage_name in ("selected_mask", "interior_hole_decision", "ownership_boundary_repair", "postprocess_mask", "boundary_matte", "final_alpha"):
                event = stages.get(stage_name)
                if event:
                    mask = event_mask(directory, event)
                    if mask is not None:
                        row["stages"][stage_name] = metrics(mask & target_mask, truth)
            report[f"{case_name}/{layer_id}"] = row
            print(f"EVALUATED {case_name}/{layer_id}", flush=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
