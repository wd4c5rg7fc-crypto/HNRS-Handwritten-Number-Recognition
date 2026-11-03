import cv2
import numpy as np
from pathlib import Path


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

RESULTS_DIR = (
    PROJECT_DIR
    / "member_sources"
    / "member_2"
    / "results"
)

# Test both division and multiplication.
TEST_CASES = {
    "testcase7": "24/6",
    "testcase11": "6*4"
}


# ==================================================
# ANALYSE CHARACTER SHAPE
# ==================================================

def analyse_character(image_path):
    """
    Analyse one segmented character.

    Measurements:
        - Aspect ratio
        - Horizontal strength
        - Vertical strength
        - Corner density
        - Diagonal strengths
        - Overall density
        - Connected component count
    """

    # Step 1: Load the character image.
    gray = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        print(f"ERROR: Cannot load {image_path.name}")
        return

    # Step 2: Convert to binary.
    # White foreground on black background.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 3: Find character boundaries.
    points = cv2.findNonZero(binary)

    if points is None:
        print(f"{image_path.stem}: Empty character")
        return

    x, y, w, h = cv2.boundingRect(points)

    if w == 0 or h == 0:
        print(f"{image_path.stem}: Invalid dimensions")
        return

    # Step 4: Crop the character.
    crop = binary[y:y + h, x:x + w]

    # Step 5: Calculate aspect ratio.
    aspect_ratio = w / h

    # ==================================================
    # CONNECTED COMPONENT ANALYSIS
    # ==================================================

    # Count separate white components in the image.
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            crop,
            connectivity=8
        )
    )

    # Subtract the background component.
    component_count = num_labels - 1

    # Get component sizes.
    component_areas = []

    for index in range(1, num_labels):
        area = int(
            stats[index, cv2.CC_STAT_AREA]
        )

        component_areas.append(area)

    # ==================================================
    # SHAPE ANALYSIS
    # ==================================================

    # Resize to 40x40 for consistent measurements.
    resized = cv2.resize(
        crop,
        (40, 40),
        interpolation=cv2.INTER_NEAREST
    )

    foreground = resized > 0

    # Horizontal stroke strength.
    horizontal = np.mean(
        foreground[17:23, :]
    )

    # Vertical stroke strength.
    vertical = np.mean(
        foreground[:, 17:23]
    )

    # Corner density.
    corners = [
        foreground[:10, :10],
        foreground[:10, -10:],
        foreground[-10:, :10],
        foreground[-10:, -10:]
    ]

    corner_density = np.mean([
        np.mean(corner)
        for corner in corners
    ])

    # Diagonal strengths.
    diagonal_1 = np.mean(
        np.diag(foreground)
    )

    diagonal_2 = np.mean(
        np.diag(np.fliplr(foreground))
    )

    # Overall foreground density.
    overall_density = np.mean(foreground)

    # ==================================================
    # DISPLAY RESULTS
    # ==================================================

    print(f"\n{image_path.stem}")

    print(f"  Width:                {w}")
    print(f"  Height:               {h}")
    print(f"  Aspect ratio:         {aspect_ratio:.2f}")

    print(f"  Horizontal strength:  {horizontal:.2f}")
    print(f"  Vertical strength:    {vertical:.2f}")

    print(f"  Corner density:       {corner_density:.2f}")

    print(f"  Diagonal 1:           {diagonal_1:.2f}")
    print(f"  Diagonal 2:           {diagonal_2:.2f}")

    print(f"  Overall density:      {overall_density:.2f}")

    print(f"  Connected components: {component_count}")
    print(f"  Component areas:      {component_areas}")


# ==================================================
# ANALYSE A TESTCASE
# ==================================================

def analyse_testcase(testcase, expression):
    """
    Analyse all segmented characters for a testcase.
    """

    folder = (
        RESULTS_DIR
        / testcase
        / "connected_components"
    )

    print("\n========================================")
    print(f"TESTCASE: {testcase}")
    print(f"EXPECTED EXPRESSION: {expression}")
    print("========================================")

    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        return

    # Find region images only.
    image_files = sorted([
        path
        for path in folder.iterdir()
        if path.stem.startswith("region_")
        and path.suffix.lower() in {
            ".png", ".jpg", ".jpeg"
        }
    ])

    if not image_files:
        print("No segmented character images found.")
        return

    # Analyse each region.
    for image_path in image_files:
        analyse_character(image_path)


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():

    print("\n========================================")
    print("HNRS OPERATOR SHAPE ANALYSIS")
    print("========================================")

    for testcase, expression in TEST_CASES.items():

        analyse_testcase(
            testcase,
            expression
        )

    print("\n========================================")
    print("ANALYSIS COMPLETED")
    print("========================================")


if __name__ == "__main__":
    main()