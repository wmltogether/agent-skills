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

Backdrop is **solid even chroma**, full frame. It is a keying plate, not a color grade.

- Prefer **magenta `#FF00FF`**
- If the subject is pink/purple/magenta, use **green `#00FF00`**
- *Never* black, white, or a semi-transparent grid

Prompt must include: `solid even #FF00FF magenta backdrop, no checkerboard, no black background` (or `#00FF00` when green).

When style/reference images exist: copy their **line weight and hard shadow shapes**. Match their **hue and saturation**; do not ask for punchier, neon, or “more vivid” color. Cel-shading means flat fields and hard shade, not higher chroma.

Image-gen cannot emit true alpha. The file on disk is RGB on chroma until keyed. If the keyed result looks oversaturated vs refs, **re-generate** — do not raise `--sat` and do not unpremultiply (`RGB/α`).

## Key

Default is **rembg**. Do not retrain when keying.

```bash
pip install -r .agent/skills/conv_img_transparent_output/scripts/requirements.txt
pip install "rembg[cpu]"

python .agent/skills/conv_img_transparent_output/scripts/key_matte.py "<generated.png>" "<out.png>"
```

`--method unmix` uses `scripts/models/chroma_unmix.onnx` if present (inference only). `--method auto` is rembg → pymatting → chroma. Never auto-picks unmix.

Defaults: `--bg auto --method rembg --inner 18 --outer 80 --blur 0.45 --sat 1.0`

`--sat 1.0` keeps generated colors (icons and general work). **Never `--sat` > 1.**

Always: min with chroma so enclosed magenta/green holes go to α=0; **fringe-only** channel despill (interior RGB unchanged); never `RGB/α` (that oversaturates edges).

Magenta surround makes the plate look duller; after keying, the same RGB reads stronger. That is display contrast, not a cue to boost saturation.

## Verify

Corners `(0,0,0,0)`; alpha extrema `(0,255)`; fringe not almost all 255. Magenta/green leftover → `--bg magenta|green` and re-key. Interior color should match the chroma plate, not a graded version.
