import cv2
from pathlib import Path

source = Path("test_images/gui_segments/component_4.png")
image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(source)

_, binary = cv2.threshold(
    image, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

output = Path("test_results/testcase4_closing")
output.mkdir(parents=True, exist_ok=True)

for size in [3, 5, 7]:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size)
    )

    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    count, _, _, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    filename = output / f"closed_{size}.png"
    cv2.imwrite(str(filename), closed)

    print(
        f"Kernel {size}x{size}: "
        f"{count - 1} components"
    )

print(f"Images saved to: {output.resolve()}")