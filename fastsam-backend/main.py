import os

# Configure PyTorch before Ultralytics imports it. This reduces allocator
# fragmentation during the mutually-exclusive SAM-B/SAM-L model switch.
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from api import app

from diagnostics import run_deterministic_self_tests

if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv or os.getenv("SAM_SELF_TEST") == "1":
        run_deterministic_self_tests()
    else:
        import uvicorn
        print("Starting FastSAM Backend on http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
