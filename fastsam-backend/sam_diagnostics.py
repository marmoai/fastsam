"""Opt-in request snapshots; never run models or modify inference arrays."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np


_trace = ContextVar("sam_diagnostic_trace", default=None)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported diagnostic type: {type(value).__name__}")


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8"
    )


@contextmanager
def sam_diagnostic_request(payload, image, request_id):
    state = None
    if os.environ.get("SAM_DEBUG_DUMP", "0").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            root = Path(__file__).resolve().parent / "runs" / "diagnostics"
            root.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="sam-", dir=str(root)))
            # Capture before the provider annotates the request layer metadata.
            _write_json(directory / "request.json", payload)
            encoded, png = cv2.imencode(".png", image)
            if not encoded:
                raise OSError("Could not encode source image")
            png.tofile(str(directory / "source.png"))
            state = {"directory": directory, "requestId": request_id, "layer": None, "events": []}
            print(f"SAM diagnostics directory={directory}")
        except Exception as error:
            print(f"SAM diagnostics unavailable: {error}")
    token = _trace.set(state)
    try:
        yield
    finally:
        if state is not None:
            try:
                _write_json(state["directory"] / "summary.json", {
                    "requestId": request_id, "events": state["events"]
                })
            except Exception as error:
                print(f"SAM diagnostics summary failed: {error}")
        _trace.reset(token)


def sam_diagnostic_layer(index, layer_id):
    state = _trace.get()
    if state is not None:
        state["layer"] = {"index": index, "id": layer_id}


def sam_diagnostic_event(stage, masks=None, alpha=None, **metadata):
    state = _trace.get()
    if state is None:
        return
    try:
        prefix = f"{len(state['events']):03d}"
        event = {"stage": stage, "layer": state["layer"], **metadata, "images": []}
        arrays = enumerate(masks) if masks is not None else []
        if alpha is not None:
            arrays = [("alpha", alpha)]
        for index, array in arrays:
            values = np.asarray(array)
            raster = (
                np.clip(values, 0, 255).astype(np.uint8)
                if alpha is not None else (values > 0.5).astype(np.uint8) * 255
            )
            filename = f"{prefix}-{index}.png"
            encoded, png = cv2.imencode(".png", raster)
            if not encoded:
                raise OSError("Could not encode diagnostic mask")
            png.tofile(str(state["directory"] / filename))
            # Preserve subpixel candidates too, so later comparisons need no
            # second encoder run. Only the current array is written at a time.
            if alpha is None:
                np.save(state["directory"] / f"{prefix}-{index}.npy", values, allow_pickle=False)
            ys, xs = np.nonzero(raster)
            event["images"].append({
                "file": filename,
                "sha256": hashlib.sha256(raster.tobytes()).hexdigest(),
                "pixels": int(xs.size),
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if xs.size else None
            })
        # Quality dictionaries acquire post-processing fields later. Freeze
        # this stage so summary.json describes the state when it was captured.
        event = json.loads(json.dumps(event, default=_json_default))
        _write_json(state["directory"] / f"{prefix}.json", event)
        state["events"].append(event)
    except Exception as error:
        # Diagnostic disk failures must not alter the segmentation response.
        print(f"SAM diagnostics event failed stage={stage}: {error}")
        _trace.set(None)
