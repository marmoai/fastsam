#!/usr/bin/env python3
"""Evaluate saved first-pass SAM diagnostics against source-sized label masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt

from sam_variable_test import CASES, DIAGNOSTICS, ROOT, digest

sys.path.insert(0, str(ROOT / "fastsam-backend"))
from boundary_matte import recover_safe_outer_boundary


LABELS = ROOT / "sam-test-labels"
STAGES = {"selected_mask", "interior_hole_decision", "ownership_boundary_repair", "postprocess_mask", "boundary_matte", "final_alpha"}


def boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)


def metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    pred_edge = boundary(predicted)
    truth_edge = boundary(truth)
    if pred_edge.any() and truth_edge.any():
        pred_near = distance_transform_edt(~truth_edge)[pred_edge] <= 2
        truth_near = distance_transform_edt(~pred_edge)[truth_edge] <= 2
        edge_precision = float(pred_near.mean())
        edge_recall = float(truth_near.mean())
        edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if edge_precision + edge_recall else 0.0
    else:
        edge_precision = edge_recall = edge_f1 = 0.0
    return {
        "groundTruthPixels": int(truth.sum()),
        "predictedPixels": int(predicted.sum()),
        "truePositive": tp,
        "falsePositive": fp,
        "falseNegative": fn,
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "iou": round(tp / (tp + fp + fn), 5) if tp + fp + fn else 1.0,
        "boundaryF1_2px": round(edge_f1, 5),
        "boundaryPrecision_2px": round(edge_precision, 5),
        "boundaryRecall_2px": round(edge_recall, 5),
    }


def load_truth(path: Path, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict]:
    with Image.open(path) as image:
        if image.size != size:
            raise ValueError(f"Label size {image.size} differs from source {size}: {path}")
        gray = np.asarray(image.convert("RGB").convert("L"))
        white = gray >= 128
        uncertain = (gray > 16) & (gray < 239)
        if not white.any() or white.all():
            raise ValueError(f"Label is empty or covers entire source: {path}")
        return white, gray, {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "whitePixels": int(white.sum()),
            "intermediateGrayPixels": int(uncertain.sum()),
            "threshold": 128,
        }


def event_mask(directory: Path, event: dict) -> np.ndarray | None:
    files = event.get("images") or []
    if not files:
        return None
    file = files[0].get("file")
    if not file:
        return None
    path = directory / file
    if path.suffix == ".png" and not path.name.endswith("-alpha.png"):
        path = path.with_suffix(".npy")
    if path.suffix == ".npy":
        return np.asarray(np.load(path) > 0.5, dtype=bool)
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) >= 128


def find_diagnostics(test_report: dict) -> dict[str, Path]:
    needed = {
        run["requestHash"]: (case_name, variant)
        for case_name, case in test_report.items()
        for variant, run in case["runs"].items()
    }
    found: dict[str, Path] = {}
    for summary_path in DIAGNOSTICS.glob("sam-*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        request_id = str(summary.get("requestId", ""))
        if not request_id.startswith("variable-test-"):
            continue
        request_path = summary_path.parent / "request.json"
        if not request_path.exists():
            continue
        recorded = json.loads(request_path.read_text(encoding="utf-8"))
        request_hash = digest(recorded)
        if request_hash in needed:
            found[request_hash] = summary_path.parent
            continue
        # The API can inject private ownership evidence into the in-memory
        # request before the diagnostic snapshot is written.
        public_recorded = {
            **recorded,
            "layers": [
                {key: value for key, value in layer.items() if not key.startswith("_")}
                for layer in recorded.get("layers") or []
            ],
        }
        for expected_hash, (case_name, variant) in needed.items():
            if expected_hash in found or not request_id.startswith(f"variable-test-{case_name}-{variant}-"):
                continue
            source_request = json.loads(
                (DIAGNOSTICS / CASES[case_name] / "request.json").read_text(encoding="utf-8")
            )
            if variant == "no_context":
                source_request["contextLayers"] = []
            elif variant == "publish":
                source_request["qualityProfile"] = "publish"
            source_request["taskId"] = request_id
            source_request["layers"] = [
                {key: value for key, value in layer.items() if not key.startswith("_")}
                for layer in source_request.get("layers") or []
            ]
            if digest(public_recorded) == digest(source_request):
                found[expected_hash] = summary_path.parent
    missing = needed.keys() - found.keys()
    if missing:
        raise FileNotFoundError(f"Missing saved diagnostics for {[(needed[key]) for key in missing]}")
    return found


def rectangle(size: tuple[int, int], box: list[int], pad: int = 0) -> np.ndarray:
    width, height = size
    x1, y1, x2, y2 = [int(v) for v in box]
    result = np.zeros((height, width), dtype=bool)
    result[max(0, y1 - pad):min(height, y2 + pad), max(0, x1 - pad):min(width, x2 + pad)] = True
    return result


def error_image(predicted: np.ndarray, truth: np.ndarray, path: Path) -> None:
    rgb = np.full((*truth.shape, 3), 245, dtype=np.uint8)
    rgb[predicted & truth] = (65, 135, 95)
    rgb[predicted & ~truth] = (226, 58, 55)
    rgb[~predicted & truth] = (36, 112, 224)
    Image.fromarray(rgb).save(path)


def output_alpha_from_cutout(cutout_path: Path, normalized_box: list[int], size: tuple[int, int]) -> np.ndarray:
    with Image.open(cutout_path) as image:
        cutout = image.convert("RGBA").getchannel("A")
    height, width = size[1], size[0]
    ymin, xmin, ymax, xmax = [int(value) for value in normalized_box]

    def recover_start(normalized_start: int, normalized_end: int, dimension: int, extent: int) -> int:
        matches = [
            start for start in range(dimension - extent + 1)
            if int(start / dimension * 1000) == normalized_start
            and int((start + extent) / dimension * 1000) == normalized_end
        ]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely recover cutout placement: {normalized_start}, {normalized_end}, {dimension}, {extent}")
        return matches[0]

    x1 = recover_start(xmin, xmax, width, cutout.width)
    y1 = recover_start(ymin, ymax, height, cutout.height)
    alpha = Image.new("L", size)
    alpha.paste(cutout, (x1, y1))
    return np.asarray(alpha) >= 128


def evaluate_cutouts(test_run: Path, test_report: dict) -> None:
    output = test_run / f"label-cutout-evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output.mkdir()
    result = {"sourceTestRun": str(test_run), "threshold": 128, "boundaryTolerancePixels": 2, "cases": {}}
    for case_name, case in test_report.items():
        source = DIAGNOSTICS / CASES[case_name] / "source.png"
        with Image.open(source) as image:
            size = image.size
        case_result = result["cases"].setdefault(case_name, {})
        for variant, run in case["runs"].items():
            for layer_id, entry in run["layers"].items():
                truth, _, label_info = load_truth(LABELS / f"{layer_id}.png", size)
                cutout = test_run / case_name / variant / f"{layer_id}.png"
                final = output_alpha_from_cutout(cutout, entry["bbox"], size)
                layer = case_result.setdefault(layer_id, {"label": label_info, "variants": {}})
                layer["variants"][variant] = {"final": metrics(final, truth), "outputBbox": entry["bbox"]}
                error_image(final, truth, output / f"{layer_id}-{variant}-errors.png")
        print(f"EVALUATED {case_name}", flush=True)
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-run", type=Path, default=ROOT / "fastsam-backend" / "runs" / "variable-tests" / "20260924-002527")
    parser.add_argument("--cutouts-only", action="store_true", help="Evaluate actual response PNGs without diagnostic snapshots")
    args = parser.parse_args()
    test_report = json.loads((args.test_run / "report.json").read_text(encoding="utf-8"))
    if args.cutouts_only:
        evaluate_cutouts(args.test_run, test_report)
        return
    directories = find_diagnostics(test_report)
    output = args.test_run / f"label-evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output.mkdir()
    result = {"sourceTestRun": str(args.test_run), "threshold": 128, "boundaryTolerancePixels": 2, "cases": {}}
    for case_name, case in test_report.items():
        source = DIAGNOSTICS / CASES[case_name] / "source.png"
        original_request = json.loads((source.parent / "request.json").read_text(encoding="utf-8"))
        with Image.open(source) as image:
            size = image.size
        case_result = result["cases"].setdefault(case_name, {})
        for variant, run in case["runs"].items():
            directory = directories[run["requestHash"]]
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            layers = case_result.setdefault("layers", {})
            for layer_id in run["layers"]:
                truth, label_gray, truth_info = load_truth(LABELS / f"{layer_id}.png", size)
                layer = layers.setdefault(layer_id, {"label": truth_info, "variants": {}})
                events = [
                    event for event in summary["events"]
                    if event.get("layer", {}).get("id") == layer_id and event.get("stage") in STAGES
                ]
                by_stage = {event["stage"]: event for event in events}
                final = event_mask(directory, by_stage["final_alpha"])
                if final is None or final.shape != truth.shape:
                    raise ValueError(f"Invalid final alpha for {case_name}/{variant}/{layer_id}")
                evaluated = {
                    "diagnostics": directory.name,
                    "final": metrics(final, truth),
                    "labelThresholdSensitivity": {},
                }
                for threshold in (64, 192):
                    alternate = metrics(final, label_gray >= threshold)
                    evaluated["labelThresholdSensitivity"][str(threshold)] = {
                        "iou": alternate["iou"],
                        "boundaryF1_2px": alternate["boundaryF1_2px"],
                    }
                layer["variants"][variant] = evaluated
                error_image(final, truth, output / f"{layer_id}-{variant}-errors.png")

                if variant != "baseline":
                    continue
                selected_event = by_stage.get("selected_mask")
                if selected_event:
                    selected = event_mask(directory, selected_event)
                    box = selected_event["targetBbox"]
                    bbox_test = {"targetBboxPixels": box, "groundTruthOutsideTarget": int((truth & ~rectangle(size, box)).sum())}
                    layer_index = original_request["layerIds"].index(layer_id)
                    ymin, xmin, ymax, xmax = original_request["bboxes"][layer_index]
                    semantic_box = [
                        int(xmin * size[0] / 1000), int(ymin * size[1] / 1000),
                        int(xmax * size[0] / 1000), int(ymax * size[1] / 1000),
                    ]
                    bbox_test["semanticBboxPixels"] = semantic_box
                    bbox_test["groundTruthOutsideSemantic"] = int((truth & ~rectangle(size, semantic_box)).sum())
                    for pad in (0, 4, 8, 16):
                        candidate = selected & rectangle(size, box, pad)
                        bbox_test[f"selectedMaskWithinBoxPlus{pad}px"] = metrics(candidate, truth)
                    evaluated["bboxOffline"] = bbox_test
                repair_event = by_stage.get("ownership_boundary_repair") or by_stage.get("boundary_matte")
                if repair_event:
                    repair_index = events.index(repair_event)
                    before_event = next((event for event in reversed(events[:repair_index]) if event_mask(directory, event) is not None), None)
                    after = event_mask(directory, repair_event)
                    before = event_mask(directory, before_event) if before_event else None
                    if before is not None and after is not None:
                        box = repair_event.get("targetBbox")
                        if box:
                            before &= rectangle(size, box)
                            after &= rectangle(size, box)
                        evaluated["boundaryOffline"] = {
                            "beforeStage": before_event["stage"],
                            "afterStage": repair_event["stage"],
                            "audit": repair_event.get("audit"),
                            "before": metrics(before, truth),
                            "after": metrics(after, truth),
                            "addedVsPreviousStage": int((after & ~before).sum()),
                            "removedVsPreviousStage": int((before & ~after).sum()),
                        }
                if repair_event and repair_event.get("stage") == "ownership_boundary_repair" and selected_event:
                    policy_event = next((event for event in summary["events"] if event.get("layer", {}).get("id") == layer_id and event.get("stage") == "b_candidates"), None)
                    config = ((policy_event or {}).get("policy") or {}).get("ownershipBoundaryRepair") or {}
                    base = selected & rectangle(size, box)
                    expected_pixels = (repair_event.get("audit") or {}).get("basePixels")
                    production = event_mask(directory, repair_event)
                    input_matches = (
                        int(base.sum()) == expected_pixels if expected_pixels is not None
                        else production is not None and np.array_equal(base, production)
                    )
                    if config.get("enabled") and input_matches:
                        blocked = np.zeros_like(base)
                        for context in repair_event.get("contextBboxes") or []:
                            blocked |= rectangle(size, context["bbox"])
                        image_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
                        sweep = {}
                        for agreement in (float(config["minAgreement"]), 0.975, 0.97):
                            variant_config = {**config, "minAgreement": agreement}
                            repaired, audit = recover_safe_outer_boundary(
                                image_bgr, base.astype(np.float32), box,
                                blocked_additions_mask=blocked, config=variant_config,
                            )
                            repaired = np.asarray(repaired > 0.5)
                            sweep[str(agreement)] = {
                                "audit": audit,
                                "metrics": metrics(repaired, truth),
                                "matchesProductionStage": bool(np.array_equal(repaired, production)) if agreement == float(config["minAgreement"]) else None,
                            }
                        evaluated["ownershipBoundaryAgreementSweep"] = sweep
                    else:
                        evaluated["ownershipBoundaryAgreementSweep"] = {
                            "skipped": "saved_selected_mask_does_not_match_repair_input",
                            "selectedPixelsInsideBox": int(base.sum()),
                            "repairInputPixels": expected_pixels,
                        }
        print(f"EVALUATED {case_name}", flush=True)
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
