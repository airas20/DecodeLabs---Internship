#!/usr/bin/env python3
"""
Project 4 - Image/Text Recognition (Basic)
Pipeline: grayscale -> blur -> adaptive threshold -> deskew -> tesseract OCR
Only detections with confidence >= 80% get kept, per the project's accuracy
requirement. Anything below that is dropped rather than shown to the user.

Usage:
    python3 ocr_pipeline.py <path_to_image> [--psm 6] [--min-conf 80]
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pytesseract

MIN_CONFIDENCE_DEFAULT = 80


def to_grayscale(img):
    # collapses the 3-channel RGB matrix down to a single intensity channel
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray):
    # gaussian blur smooths out sensor noise / jpeg artifacts before we
    # threshold, otherwise the binary step amplifies every speck of noise
    return cv2.GaussianBlur(gray, (5, 5), 0)


def binarize(gray):
    # otsu picks the cutoff automatically instead of us hardcoding a value
    thresh_val, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary, thresh_val


def deskew(binary_img, original_gray):
    """
    Finds the dominant text angle from the binary mask and rotates the
    original grayscale image to straighten it out. Tesseract's accuracy
    drops off fast once text tilts more than a couple degrees.
    """
    coords = np.column_stack(np.where(binary_img > 0))
    if len(coords) < 20:
        # not enough foreground pixels to trust an angle estimate
        return original_gray, 0.0

    angle = cv2.minAreaRect(coords)[-1]

    # minAreaRect angle convention is a little annoying, normalize it
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = original_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        original_gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def run_ocr(img, psm=6, min_conf=MIN_CONFIDENCE_DEFAULT):
    """
    Runs tesseract with image_to_data so we get per-word confidence scores,
    not just a blob of text. Words under min_conf get dropped entirely.
    """
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(
        img, config=config, output_type=pytesseract.Output.DICT
    )

    kept = []
    dropped = 0

    n = len(data["text"])
    for i in range(n):
        word = data["text"][i].strip()
        conf = float(data["conf"][i])

        if not word:
            continue

        if conf < 0:
            # tesseract uses -1 for non-text regions, skip those quietly
            continue

        if conf >= min_conf:
            kept.append({
                "text": word,
                "conf": conf,
                "x": data["left"][i],
                "y": data["top"][i],
                "w": data["width"][i],
                "h": data["height"][i],
            })
        else:
            dropped += 1

    return kept, dropped


def annotate(img_bgr, words):
    out = img_bgr.copy()
    for w in words:
        x, y, bw, bh = w["x"], w["y"], w["w"], w["h"]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
        label = f'{w["text"]} ({w["conf"]:.0f}%)'
        cv2.putText(
            out, label, (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Basic OCR recognition pipeline")
    parser.add_argument("image", help="path to input image")
    parser.add_argument("--psm", type=int, default=6,
                         help="tesseract page segmentation mode (default 6, single block)")
    parser.add_argument("--min-conf", type=float, default=MIN_CONFIDENCE_DEFAULT,
                         help="minimum confidence to keep a detection (default 80)")
    parser.add_argument("--out", default="output/annotated.png",
                         help="where to save the annotated result")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"can't find input image: {args.image}")
        sys.exit(1)

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print("opencv couldn't load that image, is it a valid file?")
        sys.exit(1)

    print(f"loaded {args.image}  {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    gray = to_grayscale(img_bgr)
    blurred = denoise(gray)
    binary, thresh_val = binarize(blurred)
    print(f"otsu threshold landed at {thresh_val:.1f}")

    straightened, angle = deskew(binary, gray)
    print(f"deskew rotated by {angle:.2f} degrees")

    words, dropped_count = run_ocr(straightened, psm=args.psm, min_conf=args.min_conf)

    print()
    print(f"--- recognized text (confidence >= {args.min_conf:.0f}%) ---")
    if not words:
        print("(nothing cleared the confidence bar)")
    else:
        line = " ".join(w["text"] for w in words)
        print(line)
        avg_conf = sum(w["conf"] for w in words) / len(words)
        print()
        print(f"words kept: {len(words)}   words dropped (low confidence): {dropped_count}")
        print(f"average confidence of kept words: {avg_conf:.1f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    straightened_bgr = cv2.cvtColor(straightened, cv2.COLOR_GRAY2BGR)
    annotated = annotate(straightened_bgr, words)
    cv2.imwrite(args.out, annotated)
    print(f"\nannotated image saved to {args.out}")


if __name__ == "__main__":
    main()
