"""Serve the V5 evidence pipeline for A/B comparison with the existing API.

Usage:
    .venv/Scripts/python.exe main_v5.py

Listens on port 8005 by default so it can run beside the original service on
8001. The frontend URL for A/B testing is a client setting and is not changed
by this file.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HOST = os.environ.get("SAMV5_HOST", "0.0.0.0")
PORT = int(os.environ.get("SAMV5_PORT", "8005"))


if __name__ == "__main__":
    import uvicorn

    from api_v5 import API_VERSION_V5, app
    from segmentation_core_v5 import CORE_VERSION_V5
    from segmentation_policy_v5 import POLICY_VERSION_V5
    from sam_runtime_v5 import forced_sam_model_variant, HQ_L_MODEL_PATH

    print("=" * 64)
    print("MarmoAI segmentation V5 (image evidence pipeline)")
    print(f"  api    : {API_VERSION_V5}")
    print(f"  policy : {POLICY_VERSION_V5}")
    print(f"  core   : {CORE_VERSION_V5}")
    print(f"  model  : {forced_sam_model_variant()} (official vit_l)")
    print(f"  weight : {HQ_L_MODEL_PATH}")
    print(f"  listen : {HOST}:{PORT}")
    print("=" * 64)

    uvicorn.run(app, host=HOST, port=PORT, log_level="info", access_log=True)
