#!/usr/bin/env python3
"""Replay saved SAM diagnostics against isolated backend snapshots.

This tool is deliberately process-isolated: each snapshot is copied to a
temporary backend directory, started on its own port, and receives the exact
saved HTTP request.  It never switches or edits the active workspace.

Examples:
  python scripts/sam_ab_regression.py discover
  python scripts/sam_ab_regression.py run \
    --case fastsam-backend/runs/diagnostics/sam-yq9mvd5h \
    --case fastsam-backend/runs/diagnostics/sam-pdf712nv
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "fastsam-backend"
DIAGNOSTICS = BACKEND / "runs" / "diagnostics"
PYTHON = BACKEND / ".venv" / "Scripts" / "python.exe"
SNAPSHOTS = {
    "flat_legacy": ROOT / "backups" / "semantic-sam-before-20260919-115916",
    "spatial_optimized": ROOT / "backups" / "workspace-before-restore-20260921-20260921-094046",
}
IGNORED_BACKEND_NAMES = {".venv", "runs", "logs", "__pycache__", "Ultralytics"}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def image_hash(case_dir: Path) -> str:
    return hashlib.sha256((case_dir / "source.png").read_bytes()).hexdigest()


def request_metadata(request: dict[str, Any]) -> dict[str, Any]:
    layer = (request.get("layers") or [{}])[0]
    intent = layer.get("segmentationIntent") or {}
    return {
        "layerId": layer.get("id"),
        "layerName": layer.get("name"),
        "bbox": layer.get("bbox"),
        "contextCount": len(request.get("contextLayers") or []),
        "hasSegmentationIntent": bool(intent),
        "intentKind": intent.get("targetKind"),
        "intentComponents": len(intent.get("includedComponents") or []),
        "intentExclusions": len(intent.get("excludedAdjacentObjects") or []),
        "parentLayerId": layer.get("parentLayerId"),
        "compositeRole": layer.get("compositeRole"),
    }


def semantic_signature(request: dict[str, Any]) -> str:
    """Hash only semantic transport fields, never the embedded raster."""
    payload = {
        "bboxes": request.get("bboxes"),
        "layerIds": request.get("layerIds"),
        "layers": request.get("layers"),
        "contextLayers": request.get("contextLayers"),
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def discover_cases(diagnostics_root: Path = DIAGNOSTICS) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request_path in sorted(diagnostics_root.glob("*/request.json")):
        case_dir = request_path.parent
        source = case_dir / "source.png"
        if not source.exists():
            continue
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        groups[image_hash(case_dir)].append({
            "case": str(case_dir.relative_to(ROOT)),
            "sourceHash": image_hash(case_dir),
            "semanticSignature": semantic_signature(request),
            **request_metadata(request),
        })
    return [
        {"sourceHash": key, "cases": value}
        for key, value in groups.items()
        if len({item["semanticSignature"] for item in value}) > 1
    ]


def copy_backend_snapshot(name: str, destination: Path) -> Path:
    snapshot = SNAPSHOTS[name]
    if not snapshot.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot}")
    backend_copy = destination / name / "fastsam-backend"
    shutil.copytree(
        BACKEND,
        backend_copy,
        ignore=shutil.ignore_patterns(*IGNORED_BACKEND_NAMES),
    )
    snapshot_backend = snapshot / "fastsam-backend"
    for source in snapshot_backend.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(snapshot_backend)
        target = backend_copy / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return backend_copy


def model_environment() -> dict[str, str]:
    env = os.environ.copy()
    defaults = {
        "FASTSAM_MODEL_PATH": r"E:\fastsam-temp\fastsam-main\fastsam-backend\FastSAM-x.pt",
        "SAM_B_MODEL_PATH": r"E:\fastsam-temp\fastsam-main\fastsam-backend\sam_b.pt",
        "SAM_L_MODEL_PATH": r"E:\fastsam-temp\fastsam-main\fastsam-backend\sam_l.pt",
        "SAM_SHARED_EMBEDDING": "1",
        "SAM_DEBUG_DUMP": "1",
        "SAM_FORCE_MODEL_VARIANT": "off",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
    return env


def wait_for_health(port: int, process: subprocess.Popen[str], timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/healthz"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Backend exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.35)
    raise TimeoutError(f"Backend did not become healthy on port {port}")


def post_json(port: int, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/segment",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def decode_cutout(entry: dict[str, Any], destination: Path) -> Image.Image | None:
    encoded = str(entry.get("image") or "")
    if not encoded:
        return None
    encoded = encoded.split(",", 1)[-1]
    destination.write_bytes(base64.b64decode(encoded))
    with Image.open(destination) as image:
        return image.convert("RGBA")


def response_metrics(response: dict[str, Any]) -> dict[str, Any]:
    cutout = (response.get("cutouts") or [{}])[0]
    quality = cutout.get("quality") or {}
    selected = next(
        (row for row in quality.get("debugCandidates") or [] if row.get("selected")),
        {},
    )
    return {
        "success": bool(response.get("success")),
        "cutoutCount": len(response.get("cutouts") or []),
        "outputBbox": cutout.get("bbox"),
        "outputSize": [cutout.get("width"), cutout.get("height")],
        "strategy": quality.get("strategy"),
        "selectedFill": selected.get("fill"),
        "selectedScore": selected.get("score"),
        "selectedReason": selected.get("rejectReason"),
        "maskPixels": (quality.get("finalMaskAudit") or {}).get("pixels"),
        "timingMs": quality.get("layerTimingMs"),
    }


def composite_to_source(source: Path, cutout: dict[str, Any], cutout_image: Image.Image) -> Image.Image:
    with Image.open(source) as original:
        canvas = Image.new("RGBA", original.size, (245, 245, 245, 255))
        y1, x1, y2, x2 = [int(value) for value in cutout["bbox"]]
        left = round(x1 / 1000 * original.width)
        top = round(y1 / 1000 * original.height)
        width = max(1, round((x2 - x1) / 1000 * original.width))
        height = max(1, round((y2 - y1) / 1000 * original.height))
        layer = cutout_image.resize((width, height))
        canvas.alpha_composite(layer, (left, top))
        return canvas


def write_montage(source: Path, images: list[tuple[str, Image.Image]], destination: Path) -> None:
    with Image.open(source) as original:
        base = original.convert("RGBA")
    tiles = [("source", base), *images]
    width = max(tile.width for _, tile in tiles)
    height = max(tile.height for _, tile in tiles)
    header = 42
    montage = Image.new("RGBA", (width * len(tiles), height + header), "white")
    draw = ImageDraw.Draw(montage)
    for index, (label, tile) in enumerate(tiles):
        left = index * width
        montage.alpha_composite(tile, (left, header))
        draw.text((left + 8, 12), label, fill="black")
    montage.convert("RGB").save(destination)


def replay_case(case_dir: Path, output_root: Path, port_base: int) -> dict[str, Any]:
    request = json.loads((case_dir / "request.json").read_text(encoding="utf-8"))
    case_name = case_dir.name
    case_output = output_root / case_name
    case_output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "case": str(case_dir.relative_to(ROOT)),
        "sourceHash": image_hash(case_dir),
        "semantic": request_metadata(request),
        "semanticSignature": semantic_signature(request),
        "runs": {},
    }
    montage_images: list[tuple[str, Image.Image]] = []

    for offset, snapshot_name in enumerate(SNAPSHOTS):
        run_output = case_output / snapshot_name
        run_output.mkdir(parents=True, exist_ok=True)
        workspace = copy_backend_snapshot(
            snapshot_name, output_root / "workspaces" / case_name
        )
        port = port_base + offset
        env = model_environment()
        env["SAM_DIAGNOSTICS_DIR"] = str(run_output / "diagnostics")
        payload = dict(request)
        payload["taskId"] = f"ab-{snapshot_name}-{case_name}"
        log_path = run_output / "backend.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    str(PYTHON), "-m", "uvicorn", "api:app",
                    "--host", "127.0.0.1", "--port", str(port),
                ], cwd=workspace,
                env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
            )
            try:
                wait_for_health(port, process)
                response = post_json(port, payload)
            finally:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        (run_output / "response.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        metrics = response_metrics(response)
        report["runs"][snapshot_name] = metrics
        cutouts = response.get("cutouts") or []
        if cutouts:
            image = decode_cutout(cutouts[0], run_output / "cutout.png")
            if image is not None:
                montage_images.append((snapshot_name, composite_to_source(case_dir / "source.png", cutouts[0], image)))

    if montage_images:
        write_montage(case_dir / "source.png", montage_images, case_output / "comparison.png")
    (case_output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("discover", help="list same-source diagnostics with distinct semantic requests")
    run = subparsers.add_parser("run", help="replay saved request(s) against both backend snapshots")
    run.add_argument("--case", action="append", required=True, help="diagnostic directory, relative to repository root")
    run.add_argument("--port-base", type=int, default=18100)
    args = parser.parse_args()

    if args.command == "discover":
        print(json.dumps(discover_cases(), ensure_ascii=False, indent=2))
        return 0

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = BACKEND / "runs" / "ab-regression" / timestamp
    output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, raw_case in enumerate(args.case):
        case_dir = (ROOT / raw_case).resolve()
        if not (case_dir / "request.json").exists() or not (case_dir / "source.png").exists():
            raise FileNotFoundError(f"Not a diagnostic case directory: {case_dir}")
        reports.append(replay_case(case_dir, output_root, args.port_base + index * 10))
    (output_root / "summary.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
