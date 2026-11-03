"""
HNRS Batch Segmentation Evidence Generator

Processes 24 handwritten test images using Member 2's
improved segmentation algorithm.

Generates:
1. Segmentation screenshots for each test case
2. CSV containing detected character counts
3. Contact sheet showing all segmentation results

Does not modify the HNRS application.
"""

import cv2
import csv
import math
import importlib.util
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

INPUT_DIR = (
    BASE / "member_sources" / "member_2" / "testimages"
)

OUTPUT_DIR = (
    BASE / "HNRS_Final_Evidence" / "Batch_Segmentation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD SEGMENTATION MODULE
# ============================================================

module_path = (
    BASE / "member_sources" / "member_2" / "Segmentation.py"
)

spec = importlib.util.spec_from_file_location(
    "member2_segmentation_batch",
    module_path
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

segment_digits = module.segment_digits

print("Segmentation module loaded successfully.")


# ============================================================
# 3. PROCESS TEST IMAGES
# ============================================================

results = []
figures = []

image_paths = sorted(
    INPUT_DIR.glob("testcase*.jpeg"),
    key=lambda p: int(p.stem.replace("testcase", ""))
)

for image_path in image_paths:

    test_id = image_path.stem

    print(f"Processing {test_id}...")

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"WARNING: Could not read {image_path.name}")

        results.append({
            "Test_ID": test_id,
            "Image": image_path.name,
            "Detected_Regions": "",
            "Expected_Regions": "",
            "Result": "IMAGE_ERROR"
        })
        continue

    try:
        characters, boxes = segment_digits(
            image,
            method="improved",
            group_division=True
        )

        boxed = image.copy()

        for index, box in enumerate(boxes):

            x, y, w, h = map(int, box[:4])

            cv2.rectangle(
                boxed,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                boxed,
                str(index + 1),
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        output_path = OUTPUT_DIR / f"{test_id}_segmented.png"

        cv2.imwrite(str(output_path), boxed)

        results.append({
            "Test_ID": test_id,
            "Image": image_path.name,
            "Detected_Regions": len(boxes),
            "Expected_Regions": 6 if test_id == "testcase1" else "",
            "Result": "PASS_VISUAL" if test_id == "testcase1" else "NOT_REVIEWED"
        })

        figures.append((test_id, boxed, len(boxes)))

        print(f"  Detected {len(boxes)} regions")

    except Exception as error:

        print(f"  ERROR: {error}")

        results.append({
            "Test_ID": test_id,
            "Image": image_path.name,
            "Detected_Regions": "",
            "Expected_Regions": "",
            "Result": "SEGMENTATION_ERROR"
        })


# ============================================================
# 4. SAVE CSV RESULTS
# ============================================================

csv_path = OUTPUT_DIR / "segmentation_results.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Test_ID",
            "Image",
            "Detected_Regions",
            "Expected_Regions",
            "Result"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print(f"\nCSV saved: {csv_path}")


# ============================================================
# 5. GENERATE CONTACT SHEET
# ============================================================

if figures:

    columns = 3
    rows = math.ceil(len(figures) / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, rows * 3.5),
        squeeze=False
    )

    for ax in axes.flat:
        ax.axis("off")

    for index, (test_id, image, count) in enumerate(figures):

        ax = axes.flat[index]

        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        ax.set_title(
            f"{test_id}: {count} regions",
            fontsize=11
        )

        ax.axis("off")

    plt.tight_layout()

    contact_sheet = OUTPUT_DIR / "all_24_segmentation_results.png"

    plt.savefig(
        contact_sheet,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Contact sheet saved: {contact_sheet}")


# ============================================================
# 6. SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("HNRS BATCH SEGMENTATION EVIDENCE")
print("=" * 55)

print(f"Images found: {len(image_paths)}")
print(f"Images processed successfully: {len(figures)}")
print(f"Results saved to: {OUTPUT_DIR}")

print("\nIMPORTANT:")
print("Only testcase1 has been visually verified.")
print("Review the remaining segmentation images before")
print("assigning PASS or FAIL results.")

print("\nDone!")