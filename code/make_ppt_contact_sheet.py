# -*- coding: utf-8 -*-
"""Create a contact sheet from exported PPT slide PNGs."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
RENDER_DIR = ROOT / "ppt_render"

slides = [Image.open(path).convert("RGB") for path in sorted(RENDER_DIR.glob("slide_*.png"))]
thumb_w, thumb_h = 400, 225
canvas = Image.new("RGB", (thumb_w * 4, thumb_h * 4), (247, 241, 229))
draw = ImageDraw.Draw(canvas)

for idx, slide in enumerate(slides):
    thumb = ImageOps.contain(slide, (thumb_w - 12, thumb_h - 22))
    x = (idx % 4) * thumb_w + 6
    y = (idx // 4) * thumb_h + 16
    canvas.paste(thumb, (x, y))
    draw.text((x, y - 14), f"{idx + 1:02d}", fill=(63, 58, 54))

canvas.save(RENDER_DIR / "contact_sheet.png")
