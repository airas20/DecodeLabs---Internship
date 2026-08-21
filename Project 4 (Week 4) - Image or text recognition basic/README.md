# Project 4 - Image/Text Recognition (Basic)

DecodeLabs AI Engineer track, optional mastery phase. Goal was to wire up a
pre-trained recognition library and prove it can read raw visual input,
not train anything from scratch.

Went with **Path 1 (OCR)** using pytesseract, since it doesn't require
downloading external model weights and tesseract's already a solid,
well-tested engine for this kind of task.

## What it does

`ocr_pipeline.py` takes an image, runs it through a preprocessing chain,
then extracts text with per-word confidence scores:

1. **Grayscale** - drops the 3-channel RGB down to one intensity channel
2. **Gaussian blur** - knocks out sensor noise before thresholding
3. **Adaptive threshold (Otsu)** - forces a clean black/white split
4. **Deskew** - finds the tilt angle from the binary mask and rotates it flat
5. **Tesseract OCR** - reads the straightened image, word by word

Any word tesseract isn't at least 80% confident about gets dropped instead
of shown to the user (that threshold is configurable via `--min-conf` but
80 is the project's minimum standard).

The result is printed to the console and saved as an annotated image with
green bounding boxes + confidence labels next to each kept word.

## Files

- `ocr_pipeline.py` - the actual deliverable
- `make_sample.py` - generates a synthetic noisy/tilted "invoice" image
  for testing, since I didn't have a real scanned doc handy
- `input_samples/sample_invoice.png` - output of the above
- `output/annotated.png` - result of running the pipeline on the sample

## Setup

```
pip install pytesseract opencv-python numpy pillow
```

Also need the tesseract binary itself installed on the system
(`apt install tesseract-ocr` on Debian/Ubuntu).

## Usage

```
python3 ocr_pipeline.py input_samples/sample_invoice.png
```

Options:

```
--psm N          tesseract page segmentation mode (default 6 - single
                  uniform block, works well for invoices/documents)
--min-conf N      confidence cutoff, 0-100 (default 80)
--out PATH        where to write the annotated image
```

Example with a sparser layout:

```
python3 ocr_pipeline.py some_photo.jpg --psm 11 --min-conf 75
```

## Sample run

```
$ python3 ocr_pipeline.py input_samples/sample_invoice.png
loaded input_samples/sample_invoice.png  942x754
otsu threshold landed at 186.0
deskew rotated by 3.53 degrees

--- recognized text (confidence >= 80%) ---
DECODELABS INDUSTRIAL SUPPLY INVOICE #0042 DATE: 2026-03-14 ITEM QTY TOTAL
SERVER RACK UNIT CABLE BUNDLE 4 $60.00 COOLING FAN 2 $35.00 SUBTOTAL:
$594.00 TAX: $47.52 TOTAL: $641.52

words kept: 27   words dropped (low confidence): 2
average confidence of kept words: 94.6%

annotated image saved to output/annotated.png
```

## Notes

- The two dropped words in the sample run are small dollar figures that
  got a bit distorted by the synthetic noise + rotate - a decent real
  example of the confidence gate actually doing its job instead of just
  being a formality.
- Deskew only handles rotation, not perspective warp - fine for phone
  photos/scans taken roughly straight-on, not for skewed camera angles.
