#!/usr/bin/env python3
"""Replay first-pass requests with hard-product local refinement on/off."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sam_ab_regression import BACKEND, PYTHON, model_environment, wait_for_health
from sam_cleanup_ab import verify_port_free
from sam_label_evaluation import LABELS, load_truth, metrics
from sam_variable_test import DIAGNOSTICS, cutout_on_source, digest, selected_candidate, send


CASES = {
    "charging_case": ("sam-qyue6tyb", None),
    "monstera": ("sam-x__ss6jc", None),
    "person": ("sam-mwi0vt_m", "person_screaming_woman.png"),
}
VARIANTS = {"A_current": "0", "B_no_local_refine": "1"}


def run_variant(name: str, flag: str, port: int, output: Path, report: dict) -> None:
    verify_port_free(port)
    variant_dir = output / name
    variant_dir.mkdir()
    env = model_environment()
    env["SAM_AB_SKIP_HARD_PRODUCT_LOCAL_REFINE"] = flag
    env["SAM_DEBUG_DUMP"] = "0"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with (variant_dir / "backend.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(PYTHON), "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=BACKEND, env=env, stdout=log, stderr=subprocess.STDOUT,
            text=True, creationflags=creationflags,
        )
        try:
            wait_for_health(port, process, timeout=90)
            for case_name, (diagnostic_name, label_name) in CASES.items():
                case_dir = DIAGNOSTICS / diagnostic_name
                payload = json.loads((case_dir / "request.json").read_text(encoding="utf-8"))
                if len(payload.get("layerIds") or []) != 1 or any(
                    layer.get("completionSegmentation") for layer in payload.get("layers") or []
                ):
                    raise ValueError(f"Not a single-layer first-pass case: {case_name}")
                payload["taskId"] = f"local-refine-ab-{name}-{case_name}-{int(time.time())}"
                with Image.open(case_dir / "source.png") as source:
                    size = source.size
                case_report = report["cases"].setdefault(case_name, {
                    "diagnostics": diagnostic_name,
                    "sourceSize": size,
                    "label": label_name,
                    "semanticHash": digest({key: payload.get(key) for key in (
                        "bboxes", "layerIds", "layers", "contextLayers"
                    )}),
                    "variants": {},
                })
                print(f"START {name} {case_name}", flush=True)
                start = time.monotonic()
                response = send(f"http://127.0.0.1:{port}", payload)
                entries = response.get("cutouts") or []
                if not response.get("success") or len(entries) != 1:
                    raise RuntimeError(f"Unexpected response for {name}/{case_name}: {len(entries)} cutouts")
                entry = entries[0]
                layer_id = entry.get("layerId") or entry.get("id")
                if layer_id != payload["layerIds"][0]:
                    raise RuntimeError(f"Wrong layer: {layer_id}")
                path = variant_dir / f"{case_name}.png"
                mask, _, preview = cutout_on_source(entry, size, path)
                preview.save(variant_dir / f"{case_name}-preview.png")
                quality = entry.get("quality") or {}
                result = {
                    "durationSeconds": round(time.monotonic() - start, 2),
                    "pngSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bbox": entry.get("bbox"),
                    "alphaPixels128": int(mask.sum()),
                    "selectedCandidate": selected_candidate(quality),
                    "selectedIndexes": quality.get("selectedIndexes"),
                    "strategy": quality.get("strategy"),
                }
                if label_name:
                    truth, _, _ = load_truth(LABELS / label_name, size)
                    result["labelMetrics"] = metrics(mask, truth)
                case_report["variants"][name] = result
                (output / "report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"END {name} {case_name} {result['durationSeconds']}s", flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def compare(output: Path, report: dict) -> None:
    for case_name, (diagnostic_name, _) in CASES.items():
        size = tuple(report["cases"][case_name]["sourceSize"])
        masks = {}
        previews = {}
        for name in VARIANTS:
            entry = report["cases"][case_name]["variants"][name]
            cutout = output / name / f"{case_name}.png"
            # cutout_on_source uses the response's placement contract.
            cutout_entry = {
                "bbox": entry["bbox"],
                "image": base64.b64encode(cutout.read_bytes()).decode("ascii"),
            }
            masks[name], _, _ = cutout_on_source(cutout_entry, size, cutout)
            previews[name] = Image.open(output / name / f"{case_name}-preview.png").convert("RGB")
        a, b = (masks[name] for name in VARIANTS)
        added = int((b & ~a).sum())
        lost = int((a & ~b).sum())
        union = int((a | b).sum())
        report["cases"][case_name]["comparison"] = {
            "addedInB": added, "lostInB": lost,
            "A_B_IoU": round(int((a & b).sum()) / union, 5) if union else 1.0,
        }
        with Image.open(DIAGNOSTICS / diagnostic_name / "source.png") as source:
            source = source.convert("RGB")
        montage = Image.new("RGB", (size[0] * 3, size[1] + 32), "white")
        draw = ImageDraw.Draw(montage)
        for index, (title, image) in enumerate([
            ("source", source), ("A current", previews["A_current"]),
            ("B no local refine", previews["B_no_local_refine"]),
        ]):
            montage.paste(image, (index * size[0], 32))
            draw.text((index * size[0] + 8, 10), title, fill="black")
        montage.save(output / f"{case_name}-comparison.png")
        diff = np.zeros((*a.shape, 3), dtype=np.uint8)
        diff[a & b] = (175, 175, 175)
        diff[b & ~a] = (60, 160, 80)
        diff[a & ~b] = (220, 60, 60)
        Image.fromarray(diff).save(output / f"{case_name}-alpha-diff.png")
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-base", type=int, default=18120)
    args = parser.parse_args()
    output = BACKEND / "runs" / "local-refine-ab" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    report = {"variants": VARIANTS, "cases": {}}
    print(f"OUTPUT {output}", flush=True)
    for index, (name, flag) in enumerate(VARIANTS.items()):
        run_variant(name, flag, args.port_base + index, output, report)
    compare(output, report)
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
