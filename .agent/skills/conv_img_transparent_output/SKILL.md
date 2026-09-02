---
name: conv_img_transparent_output
description: >-
  Keys Image-gen (or any) sprites to a true RGBA transparent PNG. Generate on
  solid #FF00FF magenta or #00FF00 green, never black/white/checkerboard, then
  run scripts/key_matte.py (default rembg). Use when the user wants transparent
  background, alpha PNG, knockout, rembg, chroma key, or after image generation
  the last deliverable must have a real transparent bottom.
---

# Transparent PNG after image-gen

Do not invent a knockout script. Do not flood-fill black. Do not leave a checkerboard baked into RGB.

## Generate

Backdrop is **solid even chroma**, full frame:

- Prefer **magenta `#FF00FF`**
- If the subject is pink/purple/magenta, use **green `#00FF00`**
- *Never* black, white, or a semi-transparent grid

Prompt must include: `solid even #FF00FF magenta backdrop, no checkerboard, no black background` (or `#00FF00` when green).

Image-gen cannot emit true alpha. The file on disk is RGB on chroma until keyed.

## Key

Default is **rembg**. Do not retrain when keying.

```bash
pip install -r .cursor/skills/conv_img_transparent_output/scripts/requirements.txt
pip install "rembg[cpu]"

python .cursor/skills/conv_img_transparent_output/scripts/key_matte.py "<generated.png>" "<out.png>"
```

`--method unmix` uses `scripts/models/chroma_unmix.onnx` if present (inference only). `--method auto` is rembg → pymatting → chroma. Never auto-picks unmix.

Defaults: `--bg auto --method rembg --inner 18 --outer 80 --blur 0.45 --sat 1.0`

Always: min with chroma so enclosed magenta/green holes go to α=0; channel despill; never `RGB/α`.

Pass `--sat 0.72` only when you also want muted colors (e.g. winter sprites).

## Verify

Corners `(0,0,0,0)`; alpha extrema `(0,255)`; fringe not almost all 255. Magenta/green leftover → `--bg magenta|green` and re-key.
