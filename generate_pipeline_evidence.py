"""
HNRS - Preprocessing and Segmentation Evidence Generator

Purpose:
Generate visual evidence for the HNRS final project report.

Outputs:
E10_original.png
E11_preprocessed.png
E12_v74_binary.png
E13_segmentation_boxes.png
E14_pipeline_comparison.png
Extracted_Characters/

This script does not modify the HNRS application.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

from pathlib import Path

from hnrs.preprocessing import (
    to_grayscale,
    remove_noise,
    apply_threshold
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

IMAGE_PATH = (
    BASE
    / "member_sources"
    / "member_2"
    / "testimages"
    / "testcase1.jpeg"
)

OUTPUT = (
    BASE
    / "HNRS_Final_Evidence"
    / "Preprocessing_Segmentation"
)

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD MEMBER 2 SEGMENTATION MODULE
# ============================================================

segmentation_path = (
    BASE
    / "member_sources"
    / "member_2"
    / "Segmentation.py"
)

if not segmentation_path.exists():
    raise FileNotFoundError(
        f"Segmentation module not found: {segmentation_path}"
    )

spec = importlib.util.spec_from_file_location(
    "member2_segmentation_evidence",
    segmentation_path
)

segmentation_module = importlib.util.module_from_spec(spec)

spec.loader.exec_module(segmentation_module)

ensure_binary = segmentation_module.ensure_binary
segment_digits = segmentation_module.segment_digits

print("Member 2 segmentation module loaded successfully.")


# ============================================================
# 3. LOAD ORIGINAL HANDWRITTEN IMAGE
# ============================================================

print("\nLoading handwritten test image...")

original = cv2.imread(str(IMAGE_PATH))

if original is None:
    raise FileNotFoundError(
        f"Could not load test image: {IMAGE_PATH}"
    )

cv2.imwrite(
    str(OUTPUT / "E10_original.png"),
    original
)

print("Original image saved.")


# ============================================================
# 4. PREPROCESSING
# ============================================================

print("\nGenerating preprocessing evidence...")

# Convert original image to grayscale
gray = to_grayscale(original)

# Apply Gaussian noise reduction
blurred = remove_noise(
    gray,
    method="gaussian"
)

# Apply Otsu thresholding
thresholded = apply_threshold(
    blurred,
    method="otsu"
)

cv2.imwrite(
    str(OUTPUT / "E11_preprocessed.png"),
    thresholded
)

print("Preprocessed image saved.")


# ============================================================
# 5. BINARY CONVERSION
# ============================================================

print("\nGenerating binary image...")

binary = ensure_binary(original)

# Convert binary image to uint8 for saving
binary_display = np.asarray(binary)

if binary_display.dtype != np.uint8:

    if binary_display.max() <= 1.0:
        binary_display = (
            binary_display * 255
        ).astype(np.uint8)

    else:
        binary_display = np.clip(
            binary_display,
            0,
            255
        ).astype(np.uint8)

cv2.imwrite(
    str(OUTPUT / "E12_v74_binary.png"),
    binary_display
)

print("Binary image saved.")


# ============================================================
# 6. CHARACTER SEGMENTATION
# ============================================================

print("\nRunning character segmentation...")

characters, boxes = segment_digits(
    original,
    method="improved",
    group_division=True
)

print(f"Detected characters: {len(characters)}")
print(f"Detected bounding boxes: {len(boxes)}")


# ============================================================
# 7. DRAW SEGMENTATION BOUNDING BOXES
# ============================================================

boxed = original.copy()

for index, box in enumerate(boxes):

    x, y, w, h = map(int, box[:4])

    # Draw bounding box
    cv2.rectangle(
        boxed,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Add region number
    cv2.putText(
        boxed,
        str(index + 1),
        (x, max(15, y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )

cv2.imwrite(
    str(OUTPUT / "E13_segmentation_boxes.png"),
    boxed
)

print("Segmentation bounding boxes saved.")


# ============================================================
# 8. SAVE INDIVIDUAL EXTRACTED CHARACTERS
# ============================================================

print("\nSaving extracted characters...")

CHAR_OUTPUT = OUTPUT / "Extracted_Characters"

CHAR_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

for index, character in enumerate(characters):

    character = np.asarray(character)

    if character.dtype != np.uint8:

        if character.max() <= 1.0:

            character = (
                character * 255
            ).astype(np.uint8)

        else:

            character = np.clip(
                character,
                0,
                255
            ).astype(np.uint8)

    filename = (
        CHAR_OUTPUT
        / f"character_{index + 1:02d}.png"
    )

    cv2.imwrite(
        str(filename),
        character
    )

print(f"Saved {len(characters)} extracted characters.")


# ============================================================
# 9. GENERATE PIPELINE COMPARISON FIGURE
# ============================================================

print("\nGenerating pipeline comparison figure...")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)

# Original image
axes[0, 0].imshow(
    cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )
)

axes[0, 0].set_title(
    "1. Original Handwritten Image"
)

# Grayscale image
axes[0, 1].imshow(
    gray,
    cmap="gray"
)

axes[0, 1].set_title(
    "2. Grayscale Conversion"
)

# Binary image
axes[1, 0].imshow(
    binary_display,
    cmap="gray"
)

axes[1, 0].set_title(
    "3. Binary Conversion"
)

# Segmented image
axes[1, 1].imshow(
    cv2.cvtColor(
        boxed,
        cv2.COLOR_BGR2RGB
    )
)

axes[1, 1].set_title(
    f"4. Character Segmentation ({len(boxes)} regions)"
)

# Remove axis labels
for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()

comparison_path = (
    OUTPUT
    / "E14_pipeline_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Pipeline comparison figure saved.")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("HNRS EVIDENCE GENERATION COMPLETED")
print("=" * 60)

print(f"\nTest image: {IMAGE_PATH.name}")

print(f"Characters detected: {len(characters)}")

print(f"Bounding boxes detected: {len(boxes)}")

print(f"\nEvidence saved to:\n{OUTPUT}")

print("\nGenerated evidence:")

for file in sorted(OUTPUT.glob("*.png")):
    print(f"  {file.name}")

print("\nDone!")