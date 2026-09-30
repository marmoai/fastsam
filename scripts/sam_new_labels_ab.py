#!/usr/bin/env python3
"""Replay new source images against isolated cleanup A/B backends."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from sam_ab_regression import BACKEND, PYTHON, model_environment, wait_for_health
from sam_cleanup_ab import verify_port_free
from sam_label_evaluation import LABELS, load_truth, metrics, output_alpha_from_cutout
from sam_variable_test import digest, selected_candidate, send


# Pixel bboxes were chosen from the source images, not from the label masks.
# Transport coordinates are [ymin, xmin, ymax, xmax] on a 0..1000 scale.
CASES = {
    "chair": {
        "source": "chairs.jpg",
        "label": "chair.png",
        "sourceBbox": [400, 345, 650, 683],
        "id": "chair",
        "name": "left blue dining chair",
        "semanticType": "furniture_chair",
        "designRole": "scene_object",
        "extractionProfile": "multi_part_hard_object",
        "includedComponents": [
            {"name": "chair back and seat", "sourceBbox": [400, 345, 642, 510]},
            {"name": "chair legs", "sourceBbox": [410, 490, 650, 683]},
        ],
    },
    "bud": {
        "source": "buds.png",
        "label": "bud.png",
        "sourceBbox": [90, 270, 335, 590],
        "id": "bud",
        "name": "blue wireless earbuds with charging case",
        "semanticType": "product_electronics",
        "designRole": "product_image",
        "extractionProfile": "multi_part_hard_product",
        "includedComponents": [
            {"name": "left earbud", "sourceBbox": [105, 350, 193, 475]},
            {"name": "right earbud", "sourceBbox": [205, 275, 300, 400]},
            {"name": "charging case", "sourceBbox": [95, 395, 332, 585]},
        ],
    },
}
VARIANTS = {"A_cleanup_on": "", "B_skip_open": "open"}


def normalized_bbox(box: list[int], size: tuple[int, int]) -> list[int]:
    x1, y1, x2, y2 = box
    width, height = size
    return [
        int(y1 * 1000 / height), int(x1 * 1000 / width),
        int(y2 * 1000 / height), int(x2 * 1000 / width),
    ]


def payload_for_case(case_name: str, case: dict) -> tuple[dict, tuple[int, int]]:
    source_path = LABELS / case["source"]
    with Image.open(source_path) as image:
        size = image.size
    raw = source_path.read_bytes()
    mime = "image/jpeg" if source_path.suffix.lower() == ".jpg" else "image/png"
    bbox = normalized_bbox(case["sourceBbox"], size)
    layer = {
        "id": case["id"],
        "name": case["name"],
        "semanticType": case["semanticType"],
        "designRole": case["designRole"],
        "renderMode": "raster_cutout",
        "extractionProfile": case["extractionProfile"],
        "category": "object",
        "runtimeType": "semantic_object",
        "compositeRole": "atomic_object",
        "childLayerIds": [],
        "completionSegmentation": False,
        "bbox": bbox,
        "segmentationIntent": {
            "targetKind": "atomic_object",
            "bboxRole": "tight_subject",
            "includedComponents": [
                {
                    "name": component["name"],
                    "bbox": normalized_bbox(component["sourceBbox"], size),
                    "reason": "visible part of requested object",
                }
                for component in case["includedComponents"]
            ],
            "excludedAdjacentObjects": [],
            "confidence": 0.95,
        },
    }
    return {
        "engine": "sam",
        "qualityProfile": "completion",
        "image": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}",
        "bboxes": [bbox],
        "layerIds": [case["id"]],
        "layers": [layer],
        "contextLayers": [],
    }, size


def run_variant(name: str, skip_step: str, port: int, output: Path, report: dict) -> None:
    verify_port_free(port)
    directory = output / name
    directory.mkdir()
    env = model_environment()
    env["SAM_AB_SKIP_HARD_PRODUCT_CLEANUP"] = "0"
    env["SAM_AB_HARD_PRODUCT_SKIP_STEP"] = skip_step
    env["SAM_DEBUG_DUMP"] = "0"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with (directory / "backend.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(PYTHON), "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=BACKEND, env=env, stdout=log, stderr=subprocess.STDOUT,
            text=True, creationflags=creationflags,
        )
        try:
            wait_for_health(port, process, timeout=90)
            for case_name, case in CASES.items():
                payload, size = payload_for_case(case_name, case)
                truth, _, label_info = load_truth(LABELS / case["label"], size)
                case_report = report["cases"].setdefault(case_name, {
                    "source": case["source"],
                    "label": case["label"],
                    "sourceBbox": case["sourceBbox"],
                    "sourceSha256": hashlib.sha256((LABELS / case["source"]).read_bytes()).hexdigest(),
                    "labelSha256": label_info["sha256"],
                    "semanticHash": digest({key: payload.get(key) for key in ("bboxes", "layerIds", "layers", "contextLayers")}),
                    "variants": {},
                })
                runs = case_report["variants"].setdefault(name, [])
                for repeat in range(1, 4):
                    payload["taskId"] = f"new-labels-{name}-{case_name}-{repeat}-{int(time.time())}"
                    print(f"START {name} {case_name} repeat={repeat}", flush=True)
                    started = time.monotonic()
                    response = send(f"http://127.0.0.1:{port}", payload)
                    entries = response.get("cutouts") or []
                    if not response.get("success") or len(entries) != 1:
                        raise RuntimeError(f"Invalid response for {name}/{case_name}/{repeat}")
                    entry = entries[0]
                    if (entry.get("layerId") or entry.get("id")) != case["id"]:
                        raise RuntimeError(f"Unexpected layer for {name}/{case_name}/{repeat}")
                    path = directory / f"{case_name}-{repeat}.png"
                    path.write_bytes(base64.b64decode(entry["image"].split(",", 1)[-1]))
                    predicted = output_alpha_from_cutout(path, entry["bbox"], size)
                    quality = entry.get("quality") or {}
                    run = {
                        "repeat": repeat,
                        "durationSeconds": round(time.monotonic() - started, 2),
                        "metrics": metrics(predicted, truth),
                        "pngSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bbox": entry["bbox"],
                        "strategy": quality.get("strategy"),
                        "selectedCandidate": selected_candidate(quality),
                        "selectedIndexes": quality.get("selectedIndexes"),
                    }
                    runs.append(run)
                    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"END {name} {case_name} repeat={repeat} {run['durationSeconds']}s", flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-base", type=int, default=18010)
    args = parser.parse_args()
    output = BACKEND / "runs" / "new-labels-ab" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    report = {"variants": VARIANTS, "repeats": 3, "cases": {}}
    print(f"OUTPUT {output}", flush=True)
    for index, (name, skip_step) in enumerate(VARIANTS.items()):
        run_variant(name, skip_step, args.port_base + index, output, report)
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
