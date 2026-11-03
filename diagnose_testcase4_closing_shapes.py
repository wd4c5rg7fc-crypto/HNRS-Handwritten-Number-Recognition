import cv2
import numpy as np
from pathlib import Path

files = {
    "Original": Path("test_images/gui_segments/component_4.png"),
    "Closing 3x3": Path("test_results/testcase4_closing/closed_3.png"),
    "Closing 5x5": Path("test_results/testcase4_closing/closed_5.png"),
    "Closing 7x7": Path("test_results/testcase4_closing/closed_7.png"),
}

print("TESTCASE4 MULTIPLICATION SHAPE COMPARISON\n")

for label, path in files.items():
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if gray is None:
        print(f"{label}: file not found")
        continue

    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    points = cv2.findNonZero(binary)

    if points is None:
        continue

    x, y, w, h = cv2.boundingRect(points)
    crop = binary[y:y+h, x:x+w]

    resized = cv2.resize(
        crop, (40, 40),
        interpolation=cv2.INTER_NEAREST
    )

    foreground = resized > 0

    horizontal = np.mean(foreground[17:23, :])
    vertical = np.mean(foreground[:, 17:23])

    diagonal1 = np.mean(np.diag(foreground))
    diagonal2 = np.mean(
        np.diag(np.fliplr(foreground))
    )

    count, _, _, _ = cv2.connectedComponentsWithStats(
        crop, connectivity=8
    )

    print(f"{label}:")
    print(f"  Components: {count - 1}")
    print(f"  Aspect ratio: {w/h:.3f}")
    print(f"  Horizontal: {horizontal:.3f}")
    print(f"  Vertical: {vertical:.3f}")
    print(f"  Diagonal 1: {diagonal1:.3f}")
    print(f"  Diagonal 2: {diagonal2:.3f}")
    print()