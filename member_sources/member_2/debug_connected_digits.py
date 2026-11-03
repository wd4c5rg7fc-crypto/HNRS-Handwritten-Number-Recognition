from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "member_sources" / "member_2"))

from Segmentation import find_split_position

IMAGE_PATH = ROOT / "test_images" / "gui_segments" / "component_0.png"
OUTPUT_DIR = ROOT / "test_results" / "connected_digits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

_, binary = cv2.threshold(
    image,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

height, width = binary.shape

split_x = find_split_position(binary)

print("\nCONNECTED DIGIT SPLITTING TEST")
print("==============================")
print(f"Image dimensions: {width} x {height}")
print(f"Suggested split position: {split_x}")

if split_x is None:
    print("No suitable split found.")
else:
    left = binary[:, :split_x]
    right = binary[:, split_x:]

    cv2.imwrite(str(OUTPUT_DIR / "left.png"), left)
    cv2.imwrite(str(OUTPUT_DIR / "right.png"), right)

    preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    cv2.line(
        preview,
        (split_x, 0),
        (split_x, height - 1),
        (0, 0, 255),
        2
    )

    cv2.imwrite(str(OUTPUT_DIR / "split_preview.png"), preview)

    print(f"Left part saved: {OUTPUT_DIR / 'left.png'}")
    print(f"Right part saved: {OUTPUT_DIR / 'right.png'}")
    print(f"Preview saved: {OUTPUT_DIR / 'split_preview.png'}")