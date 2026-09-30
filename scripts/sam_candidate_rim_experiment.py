#!/usr/bin/env python3
"""Inspect conservative candidate-consensus rim recovery on saved SAM masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from sam_variable_test import DIAGNOSTICS


def analyze(directory: Path, radius: int, neighbor_box: tuple[int, int, int, int] | None, mode: str):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    events = summary["events"]
    selected = next(event for event in events if event["stage"] == "selected_mask")
    provider = next(event for event in events if event["stage"] == "provider_candidates")
    selected_index = selected["quality"]["selectedIndexes"][0]
    base = np.load(directory / Path(selected["images"][0]["file"]).with_suffix(".npy")) > .5
    peers = [
        np.load(directory / Path(entry["file"]).with_suffix(".npy")) > .5
        for index, entry in enumerate(provider["images"])
        if index != selected_index
    ]
    if len(peers) < 2:
        raise ValueError("Requires two peer proposals")
    consensus = peers[0] & peers[1]
    target = np.zeros_like(base)
    x1, y1, x2, y2 = selected["targetBbox"]
    target[y1:y2, x1:x2] = True
    blocked = np.zeros_like(base)
    if neighbor_box:
        x1, y1, x2, y2 = neighbor_box
        blocked[y1:y2, x1:x2] = True
    outer_rim = cv2.distanceTransform(consensus.astype(np.uint8), cv2.DIST_L2, 5) <= radius
    nearby = cv2.dilate(base.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))) > 0
    added = consensus & ~base & target & ~blocked & nearby
    if mode == "rim":
        added &= outer_rim
    result = base | added
    output = directory / f"candidate-{mode}-{radius}.png"
    with Image.open(directory / "source.png") as source:
        source = source.convert("RGB")
        panel = Image.new("RGB", (source.width * 3, source.height + 28), "white")
        draw = ImageDraw.Draw(panel)
        for index, (title, mask) in enumerate((("source", None), ("selected", base), ("rim proposal", result))):
            tile = source if mask is None else Image.composite(source, Image.new("RGB", source.size, "white"), Image.fromarray((mask * 255).astype(np.uint8)))
            panel.paste(tile, (index * source.width, 28))
            draw.text((index * source.width + 6, 7), title, fill="black")
        panel.save(output)
    print(json.dumps({
        "case": directory.name, "radius": radius, "mode": mode, "selectedIndex": selected_index,
        "basePixels": int(base.sum()), "peerConsensusPixels": int(consensus.sum()),
        "addedPixels": int(added.sum()), "blockedPeerPixels": int(((consensus & ~base) & blocked).sum()),
        "preview": str(output),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic")
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--mode", choices=("rim", "full"), default="rim")
    parser.add_argument("--neighbor-box", type=int, nargs=4)
    args = parser.parse_args()
    analyze(DIAGNOSTICS / args.diagnostic, args.radius, tuple(args.neighbor_box) if args.neighbor_box else None, args.mode)
