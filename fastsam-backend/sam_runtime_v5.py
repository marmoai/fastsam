"""Private official HQ-SAM runtime for V5; other services are never patched."""
import os
import sys
import threading
import time
import urllib.request
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MODEL_CACHE_DIR = Path(os.environ.get('MODEL_CACHE_DIR', '/tmp/models')).expanduser()
HQ_L_MODEL_PATH = os.environ.get('SAM_HQ_L_MODEL_PATH', str(MODEL_CACHE_DIR / 'sam_hq_vit_l.pth'))
HQ_L_MODEL_URL = os.environ.get('SAM_HQ_L_MODEL_URL', 'https://www.marmoai.cn/models/fastsam/sam_hq_vit_l.pth')
sam_runtime_lock = threading.RLock()
_cache = ContextVar('v5_embeddings', default=None)
_metrics = ContextVar('v5_metrics', default=None)
_predictor = None


def forced_sam_model_variant(model_variant='hq-l'):
    # V5 has one supported architecture; stale V2/V3 environment must not
    # silently load a different model under HQ branch validation.
    return 'hq-l'


class Metrics:
    def __init__(self):
        self.start = time.perf_counter()
        self.encoder_count = self.decoder_count = 0
        self.encoder_ms = self.decoder_ms = 0.

    def summary(self):
        return dict(totalMs=round((time.perf_counter()-self.start)*1000, 1),
                    imageEncoderCount=self.encoder_count, imageEncoderMs=round(self.encoder_ms, 1),
                    promptDecoderCount=self.decoder_count, promptDecoderMs=round(self.decoder_ms, 1))


@contextmanager
def sam_request_metrics():
    metrics = Metrics(); token = _metrics.set(metrics)
    try:
        yield metrics
    finally:
        _metrics.reset(token)


@contextmanager
def sam_embedding_cache():
    cache = {}; token = _cache.set(cache)
    try:
        yield cache
    finally:
        cache.clear()
        if _predictor is not None:
            _predictor.reset_image()
        _cache.reset(token)


def get_predictor():
    global _predictor
    if _predictor is None:
        path = Path(HQ_L_MODEL_PATH).expanduser().resolve()
        if not path.is_file():
            if not HQ_L_MODEL_URL:
                raise FileNotFoundError(f'HQ-L checkpoint missing: {path}; set SAM_HQ_L_MODEL_URL or SAM_HQ_L_MODEL_PATH')
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f'{path.name}.{uuid.uuid4().hex}.part')
            try:
                print('SAMV5 downloading HQ-L checkpoint from configured OSS model URL', flush=True)
                request = urllib.request.Request(HQ_L_MODEL_URL, headers={'User-Agent': 'MarmoAI-SAMV5/1.0'})
                with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as output:
                    if response.status != 200:
                        raise RuntimeError(f'HQ-L checkpoint download returned HTTP {response.status}')
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if partial.stat().st_size < 100 * 1024 * 1024:
                    raise RuntimeError('HQ-L checkpoint download is unexpectedly small; verify the OSS object URL and permissions')
                os.replace(partial, path)
                print(f'SAMV5 cached HQ-L checkpoint at {path} ({path.stat().st_size / (1024 ** 2):.1f} MiB)', flush=True)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
        vendor = ROOT / 'vendor_hq_sam'
        # Load the vendored official package under a V5-specific package name,
        # avoiding collision with any separately installed segment_anything.
        import importlib.util
        name = '_marmo_hq_sam_v5'
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, vendor/'segment_anything/__init__.py',
                        submodule_search_locations=[str(vendor/'segment_anything')])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module; spec.loader.exec_module(module)
        package = sys.modules[name]
        torch.set_num_threads(max(1, int(os.environ.get('SAMV5_CPU_THREADS', '8'))))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'SAMV5 loading official HQ-SAM vit_l checkpoint={path} device={device}', flush=True)
        model = package.sam_model_registry['vit_l'](checkpoint=str(path)).to(device).eval()
        _predictor = package.SamPredictor(model)
    return _predictor


def _prepare(predictor, image):
    cache = _cache.get()
    key = (id(image), image.shape, str(image.dtype))
    state = cache.get(key) if cache is not None else None
    if state is not None and state[0] is image:
        for name, value in state[1].items():
            setattr(predictor, name, value)
        return
    start = time.perf_counter()
    predictor.set_image(image, image_format='BGR')
    metrics = _metrics.get()
    if metrics is not None:
        metrics.encoder_count += 1; metrics.encoder_ms += (time.perf_counter()-start)*1000
    if cache is not None:
        cache[key] = (image, {k:getattr(predictor, k) for k in
                      ['features', 'interm_features', 'input_size', 'original_size', 'is_image_set']})


def run_sam_bbox_inference(img, target_bbox, multimask_output=True, imgsz=1024,
                           points=None, labels=None, masks=None, model_variant='hq-l',
                           output_evidence=None, **kwargs):
    with sam_runtime_lock, torch.inference_mode():
        predictor = get_predictor(); _prepare(predictor, img)
        coords = None if points is None else np.asarray(points, np.float32).reshape(-1, 2)
        tags = None if labels is None else np.asarray(labels, np.int32).reshape(-1)
        if coords is not None and (tags is None or len(tags) != len(coords)):
            raise ValueError('Point labels must match point coordinates')
        dense = None
        if masks is not None:
            dense = np.asarray(masks, np.float32).reshape(-1, *np.asarray(masks).shape[-2:])
            if dense.shape != (1, 256, 256):
                raise ValueError('HQ-SAM requires native 1x256x256 mask logits')
        decoder = predictor.model.mask_decoder
        captured = {}
        original = decoder.predict_masks
        def capture(*args, **kw):
            raw, scores = original(*args, **kw)
            captured.update(raw=raw.detach(), scores=scores.detach())
            return raw, scores
        decoder.predict_masks = capture
        start = time.perf_counter()
        try:
            official, scores, low = predictor.predict(box=np.asarray(target_bbox, np.float32),
                point_coords=coords, point_labels=tags, mask_input=dense,
                multimask_output=multimask_output, hq_token_only=False, return_logits=True)
        finally:
            decoder.predict_masks = original
        metrics = _metrics.get()
        if metrics is not None:
            metrics.decoder_count += 1; metrics.decoder_ms += (time.perf_counter()-start)*1000
        raw = captured['raw']; ious = captured['scores'][0].float().cpu().numpy()
        head = 1+int(np.argmax(ious[1:4])) if multimask_output else 0
        # Preserve exact official addition, unweighted; score belongs to the
        # SAM head and must never be presented as a fused-mask quality score.
        all_low = torch.cat([raw[:, :4], raw[:, 4:5], raw[:, :4]+raw[:, 4:5]], dim=1)
        full = predictor.model.postprocess_masks(all_low, predictor.input_size, predictor.original_size)[0].float().cpu().numpy()
        threshold = float(predictor.model.mask_threshold)
        parity = bool(np.array_equal(full[5+head] > threshold, official[0] > threshold))
        if not parity:
            raise RuntimeError('HQ fusion reconstruction differs from official predictor')
        if output_evidence is not None:
            output_evidence.update(rawLowResolutionLogits=raw.float().cpu().numpy(),
                allFullResolutionLogits=full, predictedIoU=ious,
                lowResolutionLogits=low, fullResolutionLogits=official,
                retainedHeadIndices=np.asarray([head]), thresholdStability=np.asarray([]),
                binaryParity=parity, maskThreshold=threshold, defaultHead=head,
                branchNames=['sam0','sam1','sam2','sam3','hq','fused0','fused1','fused2','fused3'],
                scoreMeaning='base SAM predicted IoU; not rescored fusion quality',
                mode='hq_multimask' if multimask_output else 'hq_single',
                geometry=dict(sourceShape=list(img.shape[:2]), actualEncoderSize=1024,
                    resize='official SAM longest-side resize, padding removal and bilinear postprocess',
                    bbox=list(target_bbox), points=points, labels=labels, modelVariant='hq-l'))
        binary = torch.from_numpy(official > threshold)
        return [SimpleNamespace(masks=SimpleNamespace(data=binary),
                  boxes=SimpleNamespace(conf=torch.from_numpy(np.asarray(scores, np.float32))))]


def get_fastsam_model():
    raise ValueError('V5 supports HQ-L box/point segmentation only; use V2 for FastSAM')
