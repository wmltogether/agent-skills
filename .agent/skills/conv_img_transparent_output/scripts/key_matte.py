"""
Soft matte for Image-gen sprites on magenta (#FF00FF) or green (#00FF00).

Pipeline:
  default rembg
  --method unmix only when requested (uses frozen ONNX, no per-job training)
  fallback pymatting then chroma
Never use a hard RGB-max threshold on black.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chroma_unmix import infer_unmix

PRESETS = {
    "magenta": np.array([255.0, 0.0, 255.0], dtype=np.float32),
    "green": np.array([0.0, 255.0, 0.0], dtype=np.float32),
    "black": np.array([0.0, 0.0, 0.0], dtype=np.float32),
}


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def flood_background(is_bg: np.ndarray) -> np.ndarray:
    h, w = is_bg.shape
    visited = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def seed(y: int, x: int) -> None:
        if is_bg[y, x] and not visited[y, x]:
            visited[y, x] = True
            q.append((y, x))

    for x in range(w):
        seed(0, x)
        seed(h - 1, x)
    for y in range(h):
        seed(y, 0)
        seed(y, w - 1)
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and is_bg[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))
    return visited


def drop_small_islands(alpha_u8: np.ndarray, min_keep: int) -> np.ndarray:
    h, w = alpha_u8.shape
    opaque = alpha_u8 > 8
    labels = np.zeros((h, w), dtype=np.int32)
    lab = 0
    sizes: dict[int, int] = {}
    for y in range(h):
        for x in range(w):
            if opaque[y, x] and labels[y, x] == 0:
                lab += 1
                q: deque[tuple[int, int]] = deque([(y, x)])
                labels[y, x] = lab
                n = 0
                while q:
                    cy, cx = q.popleft()
                    n += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and opaque[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = lab
                            q.append((ny, nx))
                sizes[lab] = n
    if not sizes:
        return alpha_u8
    keep = {i for i, s in sizes.items() if s >= min_keep} or {max(sizes, key=sizes.get)}
    out = alpha_u8.copy()
    out[(labels > 0) & ~np.isin(labels, list(keep))] = 0
    return out


def detect_bg(rgb: np.ndarray) -> str:
    samples = np.stack(
        [
            rgb[0, 0],
            rgb[0, -1],
            rgb[-1, 0],
            rgb[-1, -1],
            rgb[0, rgb.shape[1] // 2],
            rgb[rgb.shape[0] // 2, 0],
        ]
    ).astype(np.float32)
    mean = samples.mean(axis=0)
    return min(PRESETS, key=lambda k: float(np.linalg.norm(mean - PRESETS[k])))


def chroma_spill_score(rgb: np.ndarray, bg_name: str) -> np.ndarray:
    """Higher = more like the backdrop (should become transparent)."""
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    if bg_name == "green":
        return np.clip(g - np.maximum(r, b), 0.0, 255.0)
    if bg_name == "magenta":
        return np.clip(np.minimum(r, b) - g, 0.0, 255.0)
    return 255.0 - np.maximum(np.maximum(r, g), b)


def chroma_alpha(rgb: np.ndarray, bg_name: str, inner: float, outer: float) -> np.ndarray:
    spill = chroma_spill_score(rgb, bg_name)
    bg = PRESETS[bg_name]
    dist = np.linalg.norm(rgb - bg.reshape(1, 1, 3), axis=2)
    visited = flood_background(dist <= max(outer, 40.0) if bg_name == "black" else (spill >= inner))
    t = 1.0 - (spill - inner) / max(1.0, outer - inner)
    soft = smoothstep(t)
    return np.where(visited, 0.0, np.clip(soft, 0.0, 1.0))


def suppress_chroma_alpha(rgb: np.ndarray, alpha: np.ndarray, bg_name: str, hard: float, fade: float) -> np.ndarray:
    """Zero alpha on ALL chroma pixels, including enclosed holes flood-fill never reaches."""
    spill = chroma_spill_score(rgb, bg_name)
    kill = smoothstep((spill - hard) / max(1.0, fade - hard))
    return np.clip(alpha * (1.0 - kill), 0.0, 1.0)


def channel_despill(rgb: np.ndarray, bg_name: str) -> np.ndarray:
    """Pull chroma toward the opposite channel. No /alpha (that oversaturates)."""
    out = rgb.copy()
    r, g, b = out[:, :, 0], out[:, :, 1], out[:, :, 2]
    if bg_name == "magenta":
        excess = np.clip(np.minimum(r, b) - g, 0.0, 255.0)
        out[:, :, 0] = np.clip(r - excess, 0.0, 255.0)
        out[:, :, 2] = np.clip(b - excess, 0.0, 255.0)
    elif bg_name == "green":
        excess = np.clip(g - np.maximum(r, b), 0.0, 255.0)
        out[:, :, 1] = np.clip(g - excess, 0.0, 255.0)
    return out


def cap_saturation(rgb: np.ndarray, alpha: np.ndarray, amount: float) -> np.ndarray:
    """amount 1 = unchanged, <1 mutes oversaturated gens."""
    if amount >= 0.999:
        return rgb
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    from PIL import ImageEnhance
    muted = np.array(ImageEnhance.Color(img).enhance(amount), dtype=np.float32)
    return np.where(alpha[:, :, None] > 0, muted, 0.0)


def alpha_rembg(rgb_u8: np.ndarray) -> np.ndarray | None:
    try:
        from rembg import new_session, remove
    except ImportError:
        return None
    img = Image.fromarray(rgb_u8, "RGB")
    session = None
    for model in ("birefnet-general", "u2net"):
        try:
            session = new_session(model)
            break
        except Exception:
            continue
    if session is None:
        return None
    cut = remove(img, session=session)
    if not isinstance(cut, Image.Image):
        cut = Image.fromarray(np.array(cut))
    return np.array(cut.convert("RGBA"))[:, :, 3].astype(np.float32) / 255.0


def _resize_rgb(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side:
        return rgb, (w, h)
    scale = max_side / float(max(h, w))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").resize((nw, nh), Image.Resampling.BILINEAR)
    return np.array(img).astype(np.float32), (w, h)


def alpha_pymatting(rgb: np.ndarray, bg_name: str, inner: float, outer: float) -> np.ndarray | None:
    try:
        from pymatting import estimate_alpha_cf
    except ImportError:
        return None
    small, orig_wh = _resize_rgb(rgb, 768)
    spill = chroma_spill_score(small, bg_name)
    dist = np.linalg.norm(small - PRESETS[bg_name].reshape(1, 1, 3), axis=2)
    is_bg = (spill >= outer) if bg_name != "black" else (dist <= outer)
    visited = flood_background(is_bg)
    trimap = np.full(spill.shape, 0.5, dtype=np.float64)
    trimap[visited] = 0.0
    fg = (~visited) & (spill <= inner if bg_name != "black" else dist >= outer + 20)
    er = fg.copy()
    er[1:] &= fg[:-1]
    er[:-1] &= fg[1:]
    er[:, 1:] &= fg[:, :-1]
    er[:, :-1] &= fg[:, 1:]
    trimap[er] = 1.0
    rgb01 = (small / 255.0).astype(np.float64)
    alpha_s = np.clip(estimate_alpha_cf(rgb01, trimap), 0.0, 1.0)
    a_img = Image.fromarray((alpha_s * 255.0).astype(np.uint8), "L").resize(orig_wh, Image.Resampling.BILINEAR)
    return np.array(a_img).astype(np.float32) / 255.0


def finish(
    rgb: np.ndarray,
    alpha: np.ndarray,
    bg_name: str,
    blur: float,
    min_keep: int,
    sat: float,
    hard: float,
    fade: float,
    fg_pred: np.ndarray | None = None,
) -> np.ndarray:
    alpha = suppress_chroma_alpha(rgb, alpha, bg_name, hard, fade)
    fg = fg_pred if fg_pred is not None else channel_despill(rgb, bg_name)
    if fg_pred is not None:
        fg = channel_despill(fg, bg_name)
    fg = cap_saturation(fg, alpha, sat)
    rgba = np.dstack([fg, np.clip(alpha * 255.0, 0.0, 255.0)])
    img = Image.fromarray(np.round(rgba).astype(np.uint8), "RGBA")
    if blur > 0:
        img.putalpha(img.split()[3].filter(ImageFilter.GaussianBlur(radius=blur)))
    arr = np.array(img)
    arr[:, :, 3] = drop_small_islands(arr[:, :, 3], min_keep)
    arr[arr[:, :, 3] == 0, 0:3] = 0
    return arr


def key_image(
    rgb_u8: np.ndarray,
    bg_name: str,
    method: str,
    inner: float,
    outer: float,
    min_keep: int,
    blur: float,
    sat: float,
    hard: float,
    fade: float,
) -> tuple[np.ndarray, str]:
    rgb = rgb_u8.astype(np.float32)
    used = method
    alpha = None
    fg_pred = None
    chroma_a = chroma_alpha(rgb, bg_name, inner, outer)
    if method == "unmix":
        un = infer_unmix(rgb_u8, bg_name)
        if un is not None:
            fg_pred, alpha = un
            used = "unmix"
    if alpha is None and method in ("auto", "rembg"):
        alpha = alpha_rembg(rgb_u8)
        used = "rembg" if alpha is not None else used
    if alpha is None and method in ("auto", "pymatting"):
        alpha = alpha_pymatting(rgb, bg_name, inner, outer)
        used = "pymatting" if alpha is not None else used
    if alpha is None:
        alpha = chroma_a
        used = "chroma"
    else:
        alpha = np.minimum(alpha, chroma_a)
        used = f"{used}+chroma"
    return finish(rgb, alpha, bg_name, blur, min_keep, sat, hard, fade, fg_pred), used


def process_file(src: Path, dest: Path, args: argparse.Namespace) -> None:
    rgb = np.array(Image.open(src).convert("RGB"))
    bg = args.bg if args.bg != "auto" else detect_bg(rgb)
    out, used = key_image(
        rgb, bg, args.method, args.inner, args.outer, args.min_keep, args.blur, args.sat, args.hard, args.fade
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, "RGBA").save(dest, "PNG", optimize=True)
    opaque = int((out[:, :, 3] > 0).sum())
    print(f"OK {dest.name} bg={bg} method={used} {out.shape[1]}x{out.shape[0]} opaque={opaque}")


def main() -> None:
    p = argparse.ArgumentParser(description="Soft-key magenta/green AI sprites to RGBA")
    p.add_argument("src")
    p.add_argument("dest")
    p.add_argument("--bg", choices=["auto", "magenta", "green", "black"], default="auto")
    p.add_argument("--method", choices=["auto", "unmix", "rembg", "pymatting", "chroma"], default="rembg")
    p.add_argument("--inner", type=float, default=18.0)
    p.add_argument("--outer", type=float, default=80.0)
    p.add_argument("--min-keep", type=int, default=1500)
    p.add_argument("--blur", type=float, default=0.45)
    p.add_argument("--sat", type=float, default=1.0, help="HSV saturation scale; 1 keeps gen colors, <1 mutes")
    p.add_argument("--hard", type=float, default=28.0, help="spill above this starts killing alpha (enclosed holes)")
    p.add_argument("--fade", type=float, default=72.0, help="spill at/above this is fully transparent")
    args = p.parse_args()
    src, dest = Path(args.src), Path(args.dest)
    if src.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.png")):
            process_file(f, dest / f.name, args)
    else:
        process_file(src, dest, args)


if __name__ == "__main__":
    main()
