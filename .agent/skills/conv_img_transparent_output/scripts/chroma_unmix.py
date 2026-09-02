"""Neural chroma unmixing: C = a*F + (1-a)*B with known backdrop B."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

MODEL_DIR = Path(__file__).resolve().parent / "models"
_FALLBACK_ONNX = (
    Path(__file__).resolve().parent.parent.parent
    / "summer-to-winter-sprites"
    / "scripts"
    / "models"
    / "chroma_unmix.onnx"
)
ONNX_PATH = MODEL_DIR / "chroma_unmix.onnx"
if not ONNX_PATH.is_file() and _FALLBACK_ONNX.is_file():
    ONNX_PATH = _FALLBACK_ONNX
SIZE = 256

PRESETS = {
    "magenta": np.array([1.0, 0.0, 1.0], dtype=np.float32),
    "green": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "black": np.array([0.0, 0.0, 0.0], dtype=np.float32),
}


def _pad_multiple(h: int, w: int, m: int = 16) -> tuple[int, int]:
    return ((h + m - 1) // m) * m, ((w + m - 1) // m) * m


def build_input(rgb: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """N,6,H,W float32 in [0,1]. rgb is HxWx3 0-255."""
    img = rgb.astype(np.float32) / 255.0
    h, w, _ = img.shape
    ph, pw = _pad_multiple(h, w)
    canvas = np.zeros((ph, pw, 6), dtype=np.float32)
    canvas[:h, :w, :3] = img
    canvas[:, :, 3:] = bg_rgb.reshape(1, 1, 3)
    canvas[h:, :, :3] = bg_rgb.reshape(1, 1, 3)
    canvas[:, w:, :3] = bg_rgb.reshape(1, 1, 3)
    x = np.transpose(canvas, (2, 0, 1))[None]
    return x, (h, w)


def infer_unmix(rgb_u8: np.ndarray, bg_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (fg_rgb_0_255, alpha_0_1) or None if ONNX missing/fails."""
    if not ONNX_PATH.is_file():
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    bg = PRESETS.get(bg_name, PRESETS["magenta"])
    h0, w0 = rgb_u8.shape[:2]
    scale = SIZE / float(max(h0, w0))
    nw, nh = max(16, int(w0 * scale)), max(16, int(h0 * scale))
    small = np.array(Image.fromarray(rgb_u8).resize((nw, nh), Image.Resampling.BILINEAR))
    x, (h, w) = build_input(small, bg)
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    y = sess.run(None, {name: x})[0]
    fg = np.clip(y[0, :3, :h, :w].transpose(1, 2, 0), 0.0, 1.0)
    alpha = np.clip(y[0, 3, :h, :w], 0.0, 1.0)
    fg_img = Image.fromarray((fg * 255.0).astype(np.uint8), "RGB").resize((w0, h0), Image.Resampling.BILINEAR)
    a_img = Image.fromarray((alpha * 255.0).astype(np.uint8), "L").resize((w0, h0), Image.Resampling.BILINEAR)
    return np.array(fg_img).astype(np.float32), np.array(a_img).astype(np.float32) / 255.0
