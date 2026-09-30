#!/usr/bin/env python3
"""Run controlled first-pass SAM replays against an already-running backend."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "fastsam-backend" / "runs" / "diagnostics"
CASES = {
    "person": "sam-mwi0vt_m",
    "armchair": "sam-e622i7ao",
    "tiered_table": "sam-7ysu6tv_",
    "food": "sam-pnlq7wu0",
}
VARIANTS = ("baseline", "no_context", "publish")


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def send(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + "/segment",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def cutout_on_source(entry: dict, source_size: tuple[int, int], output: Path):
    encoded = entry.get("image") or ""
    if not encoded:
        raise ValueError("Cutout has no image")
    raw = base64.b64decode(encoded.split(",", 1)[-1])
    output.write_bytes(raw)
    image = Image.open(io.BytesIO(raw)).convert("RGBA")
    y1, x1, y2, x2 = [float(v) for v in entry["bbox"]]
    left = round(x1 * source_size[0] / 1000)
    top = round(y1 * source_size[1] / 1000)
    width = max(1, round((x2 - x1) * source_size[0] / 1000))
    height = max(1, round((y2 - y1) * source_size[1] / 1000))
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    alpha = Image.new("L", source_size)
    alpha.paste(image.getchannel("A"), (left, top))
    preview = Image.new("RGBA", source_size, "white")
    preview.alpha_composite(image, (left, top))
    return np.asarray(alpha) >= 128, np.asarray(alpha), preview.convert("RGB")


def mask_metrics(mask: np.ndarray, alpha: np.ndarray) -> dict:
    edge = mask.copy()
    edge[1:, :] &= mask[:-1, :]
    edge[:-1, :] &= mask[1:, :]
    edge[:, 1:] &= mask[:, :-1]
    edge[:, :-1] &= mask[:, 1:]
    return {
        "maskPixels": int(mask.sum()),
        "edgePixels4Neighbor": int((mask & ~edge).sum()),
        "partialAlphaPixels": int(((alpha > 0) & (alpha < 255)).sum()),
    }


def selected_candidate(quality: dict) -> dict:
    candidates = quality.get("debugCandidates") or []
    selected = next((item for item in candidates if item.get("selected")), None)
    if selected is None:
        return {}
    return {
        "index": selected.get("index"),
        "score": selected.get("score"),
        "fill": selected.get("fill"),
        "reason": selected.get("rejectReason"),
    }


def write_montage(source: Path, previews: dict[str, Image.Image], output: Path) -> None:
    original = Image.open(source).convert("RGB")
    tiles = [("source", original), *[(name, previews[name]) for name in VARIANTS if name in previews]]
    header = 34
    montage = Image.new("RGB", (original.width * len(tiles), original.height + header), "white")
    draw = ImageDraw.Draw(montage)
    for position, (label, tile) in enumerate(tiles):
        x = position * original.width
        montage.paste(tile, (x, header))
        draw.text((x + 10, 10), label, fill="black")
    montage.save(output)


def run_case(name: str, case_dir: Path, output_dir: Path, url: str, report: dict, variants: list[str]) -> None:
    request = json.loads((case_dir / "request.json").read_text(encoding="utf-8"))
    if any(layer.get("completionSegmentation") for layer in request.get("layers") or []):
        raise ValueError(f"{name} is a completion request, not first-pass extraction")
    source = case_dir / "source.png"
    source_size = Image.open(source).size
    case_output = output_dir / name
    case_output.mkdir(parents=True, exist_ok=True)
    report[name] = {
        "diagnosticCase": case_dir.name,
        "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sourceSize": source_size,
        "layerIds": request.get("layerIds"),
        "semanticHash": digest({key: request.get(key) for key in ("bboxes", "layerIds", "layers", "contextLayers")}),
        "runs": {},
    }
    masks: dict[str, dict[str, np.ndarray]] = {}
    previews: dict[str, dict[str, Image.Image]] = {}
    for variant in variants:
        payload = dict(request)
        if variant == "no_context":
            payload["contextLayers"] = []
        elif variant == "publish":
            payload["qualityProfile"] = "publish"
        payload["taskId"] = f"variable-test-{name}-{variant}-{int(time.time())}"
        run_dir = case_output / variant
        run_dir.mkdir(exist_ok=True)
        run_report = {
            "requestHash": digest(payload),
            "qualityProfile": payload.get("qualityProfile"),
            "contextCount": len(payload.get("contextLayers") or []),
            "layers": {},
        }
        report[name]["runs"][variant] = run_report
        print(f"START {name}/{variant}", flush=True)
        start = time.monotonic()
        try:
            response = send(url, payload)
            run_report["durationSeconds"] = round(time.monotonic() - start, 2)
            run_report["success"] = bool(response.get("success"))
            for entry in response.get("cutouts") or []:
                layer_id = entry.get("layerId") or entry.get("id")
                if not layer_id:
                    raise ValueError("Cutout missing layerId")
                mask, alpha, preview = cutout_on_source(entry, source_size, run_dir / f"{layer_id}.png")
                quality = entry.get("quality") or {}
                run_report["layers"][layer_id] = {
                    "bbox": entry.get("bbox"),
                    "cutoutSize": [entry.get("width"), entry.get("height")],
                    "selectedCandidate": selected_candidate(quality),
                    "selectedIndexes": quality.get("selectedIndexes"),
                    "finalMaskAudit": quality.get("finalMaskAudit"),
                    "strategy": quality.get("strategy"),
                    **mask_metrics(mask, alpha),
                }
                masks.setdefault(layer_id, {})[variant] = mask
                previews.setdefault(layer_id, {})[variant] = preview
            expected = set(request.get("layerIds") or [])
            run_report["missingLayers"] = sorted(expected - set(run_report["layers"]))
            if not response.get("success") or run_report["missingLayers"]:
                run_report["error"] = "Backend returned failure or missing cutouts"
        except Exception as exc:
            run_report["durationSeconds"] = round(time.monotonic() - start, 2)
            run_report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"END {name}/{variant}: {run_report['durationSeconds']}s {run_report.get('error', 'ok')}", flush=True)
        (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for layer_id, variants in masks.items():
        baseline = variants.get("baseline")
        if baseline is not None:
            for variant, mask in variants.items():
                if variant == "baseline":
                    continue
                intersection = int((baseline & mask).sum())
                union = int((baseline | mask).sum())
                report[name]["runs"][variant]["layers"][layer_id]["vsBaseline"] = {
                    "iou": round(intersection / union, 5) if union else 1,
                    "addedPixels": int((mask & ~baseline).sum()),
                    "lostPixels": int((baseline & ~mask).sum()),
                }
        write_montage(source, previews[layer_id], case_output / f"{layer_id}-comparison.png")
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--case", action="append", choices=CASES, help="Run only selected cases")
    parser.add_argument("--variant", action="append", choices=VARIANTS, help="Run only selected request variants")
    args = parser.parse_args()
    with urllib.request.urlopen(args.url.rstrip("/") + "/healthz", timeout=5) as health:
        if health.status != 200:
            raise RuntimeError("Backend is not healthy")
    output = ROOT / "fastsam-backend" / "runs" / "variable-tests" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    print(f"OUTPUT {output}", flush=True)
    report = {}
    for name in args.case or CASES:
        run_case(name, DIAGNOSTICS / CASES[name], output, args.url, report, args.variant or list(VARIANTS))


if __name__ == "__main__":
    main()
