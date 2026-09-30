#!/usr/bin/env python3
"""Run three end-to-end cleanup A/B passes on isolated SAM backends."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from sam_ab_regression import BACKEND, PYTHON, model_environment, wait_for_health
from sam_label_evaluation import LABELS, load_truth, metrics, output_alpha_from_cutout
from sam_variable_test import CASES, DIAGNOSTICS, digest, selected_candidate, send


VARIANTS = {"A_cleanup_on": "0", "B_cleanup_off": "1"}


def verify_port_free(port: int) -> None:
    try:
        connection = socket.create_connection(("127.0.0.1", port), timeout=1)
    except OSError:
        return
    connection.close()
    raise RuntimeError(f"Port {port} already has a backend; refusing to use it")


def run_variant(
    name: str, switch: str, port: int, output: Path, report: dict,
    *, skip_step: str | None = None, case_repeats: dict[str, int] | None = None,
) -> None:
    verify_port_free(port)
    variant_dir = output / name
    variant_dir.mkdir()
    env = model_environment()
    env["SAM_AB_SKIP_HARD_PRODUCT_CLEANUP"] = switch
    env["SAM_AB_HARD_PRODUCT_SKIP_STEP"] = skip_step or ""
    env["SAM_DEBUG_DUMP"] = "0"
    startup = [str(PYTHON), "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", str(port)]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with (variant_dir / "backend.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            startup, cwd=BACKEND, env=env, stdout=log, stderr=subprocess.STDOUT,
            text=True, creationflags=creationflags,
        )
        try:
            wait_for_health(port, process, timeout=90)
            for case_name, diagnostic_name in CASES.items():
                source_dir = DIAGNOSTICS / diagnostic_name
                source = source_dir / "source.png"
                payload_template = json.loads((source_dir / "request.json").read_text(encoding="utf-8"))
                with Image.open(source) as image:
                    size = image.size
                case_report = report["cases"].setdefault(case_name, {
                    "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "semanticHash": digest({
                        key: payload_template.get(key)
                        for key in ("bboxes", "layerIds", "layers", "contextLayers")
                    }),
                    "variants": {},
                })
                runs = case_report["variants"].setdefault(name, [])
                repeat_count = (case_repeats or {}).get(case_name, 3)
                for repeat in range(1, repeat_count + 1):
                    payload = dict(payload_template)
                    payload["taskId"] = f"cleanup-ab-{name}-{case_name}-{repeat}-{int(time.time())}"
                    print(f"START {name} {case_name} repeat={repeat}", flush=True)
                    started = time.monotonic()
                    response = send(f"http://127.0.0.1:{port}", payload)
                    if not response.get("success"):
                        raise RuntimeError(f"Backend failure for {name}/{case_name}/{repeat}")
                    entries = {
                        entry.get("layerId") or entry.get("id"): entry
                        for entry in response.get("cutouts") or []
                    }
                    expected = set(payload.get("layerIds") or [])
                    if expected != set(entries):
                        raise RuntimeError(f"Cutout mismatch {name}/{case_name}/{repeat}: {expected ^ set(entries)}")
                    run = {"repeat": repeat, "durationSeconds": round(time.monotonic() - started, 2), "layers": {}}
                    for layer_id, entry in entries.items():
                        path = variant_dir / f"{case_name}-{layer_id}-{repeat}.png"
                        encoded = entry.get("image") or ""
                        if not encoded:
                            raise RuntimeError(f"Missing image for {layer_id}")
                        path.write_bytes(base64.b64decode(encoded.split(",", 1)[-1]))
                        truth, _, label_info = load_truth(LABELS / f"{layer_id}.png", size)
                        predicted = output_alpha_from_cutout(path, entry["bbox"], size)
                        quality = entry.get("quality") or {}
                        run["layers"][layer_id] = {
                            "metrics": metrics(predicted, truth),
                            "labelSha256": label_info["sha256"],
                            "pngSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "bbox": entry["bbox"],
                            "selectedCandidate": selected_candidate(quality),
                            "selectedIndexes": quality.get("selectedIndexes"),
                            "strategy": quality.get("strategy"),
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
    parser.add_argument("--port-base", type=int, default=18003)
    args = parser.parse_args()
    output = BACKEND / "runs" / "cleanup-ab" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    report = {"variants": VARIANTS, "repeats": 3, "cases": {}}
    print(f"OUTPUT {output}", flush=True)
    for index, (name, switch) in enumerate(VARIANTS.items()):
        run_variant(name, switch, args.port_base + index, output, report)
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
