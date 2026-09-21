"""Turns the chosen generated icon (source-midnight-violet-cyan.png: generate-image skill, ERNIE-Image-Turbo,
seed 8507, 1024x1024 on a white background) into the app icon files.

The tile sits at (172, 174)-(851, 847) with a corner radius of about 150 px; everything outside that rounded
square (white background, drop shadow) becomes transparent.

Usage: uv run --project PoC/tools python installer/icon/prepare_icon.py
Writes rvc-app.png (1024 px, transparent corners) and rvc-app.ico (16-256 px) next to this script.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
TILE = (172, 174, 852, 848)  # left, top, right, bottom (exclusive)
RADIUS = 150
SS = 4
ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]


def main():
    src = Image.open(HERE / "source-midnight-violet-cyan.png").convert("RGBA")
    tile = src.crop(TILE)
    w, h = tile.size

    # anti-aliased rounded-square mask drawn at 4x
    mask = Image.new("L", (w * SS, h * SS), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w * SS - 1, h * SS - 1], radius=RADIUS * SS, fill=255)
    mask = mask.resize((w, h), Image.LANCZOS)
    tile.putalpha(mask)

    # square canvas with a small transparent margin, as Windows icons usually have
    side = 1024
    inner = round(side * 0.94)
    icon = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    icon.alpha_composite(tile.resize((inner, inner), Image.LANCZOS), ((side - inner) // 2, (side - inner) // 2))
    icon.save(HERE / "rvc-app.png")

    frames = []
    for n in ICO_SIZES:
        f = icon.resize((n, n), Image.LANCZOS)
        if n <= 48:
            f = f.filter(ImageFilter.UnsharpMask(radius=0.6, percent=60, threshold=1))
        frames.append(f)
    frames[-1].save(HERE / "rvc-app.ico", format="ICO", sizes=[(n, n) for n in ICO_SIZES], append_images=frames[:-1])

    # preview on dark and light grounds at 256 / 64 / 32 / 24 / 16
    shown = [256, 64, 32, 24, 16]
    width = sum(shown) + 24 * (len(shown) + 1)
    sheet = Image.new("RGBA", (width, 608), (0, 0, 0, 0))
    for row, ground in enumerate([(32, 33, 38, 255), (240, 241, 238, 255)]):
        band = Image.new("RGBA", (width, 304), ground)
        x = 24
        for n, f in zip(shown, [frames[ICO_SIZES.index(n)] for n in shown]):
            band.alpha_composite(f, (x, 24 + 256 - n))
            x += n + 24
        sheet.alpha_composite(band, (0, row * 304))
    sheet.save(HERE / "preview.png")
    print("wrote", HERE)


if __name__ == "__main__":
    main()
