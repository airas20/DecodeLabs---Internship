

import random
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"

LINES = [
    "DECODELABS INDUSTRIAL SUPPLY",
    "INVOICE #0042",
    "DATE: 2026-03-14",
    "",
    "ITEM              QTY   TOTAL",
    "SERVER RACK UNIT   1   $499.00",
    "CABLE BUNDLE       4    $60.00",
    "COOLING FAN        2    $35.00",
    "",
    "SUBTOTAL:              $594.00",
    "TAX:                    $47.52",
    "TOTAL:                 $641.52",
]


def build_clean_image():
    w, h = 900, 700
    img = Image.new("L", (w, h), color=250)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 26)

    y = 40
    for line in LINES:
        draw.text((60, y), line, fill=10, font=font)
        y += 45

    return img


def add_noise(img, amount=18):
    px = img.load()
    w, h = img.size
    for x in range(w):
        for y in range(h):
            if random.random() < 0.15:
                jitter = random.randint(-amount, amount)
                val = px[x, y] + jitter
                px[x, y] = max(0, min(255, val))
    return img


def main():
    img = build_clean_image()
    img = add_noise(img)
    # tilt it a couple degrees so the deskew step has something to fix
    img = img.rotate(-3.5, expand=True, fillcolor=250)
    out_path = "input_samples/sample_invoice.png"
    img.save(out_path)
    print(f"wrote {out_path}  ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
