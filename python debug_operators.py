import cv2
import numpy as np
from pathlib import Path

FOLDER = Path(
    "member_sources/member_2/results/testcase1/connected_components"
)

for path in sorted(FOLDER.glob("region_*")):

    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        continue

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if gray is None:
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
        crop,
        (40, 40),
        interpolation=cv2.INTER_NEAREST
    )

    foreground = resized > 0

    horizontal = np.mean(foreground[17:23, :])
    vertical = np.mean(foreground[:, 17:23])

    corners = [
        foreground[:10, :10],
        foreground[:10, -10:],
        foreground[-10:, :10],
        foreground[-10:, -10:]
    ]

    corner_density = np.mean([
        np.mean(corner) for corner in corners
    ])

    print(
        f"{path.stem}: "
        f"aspect={w/h:.2f}, "
        f"horizontal={horizontal:.2f}, "
        f"vertical={vertical:.2f}, "
        f"corners={corner_density:.2f}"
    )