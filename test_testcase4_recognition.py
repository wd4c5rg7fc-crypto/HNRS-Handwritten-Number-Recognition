import cv2
from pathlib import Path
from operator_recognition import recognise_operator

folder = Path("test_results/testcase4_closing")

for name in ["closed_3.png", "closed_5.png", "closed_7.png"]:
    path = folder / name
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"{name}: image not found")
        continue

    prediction = recognise_operator(image)

    print(f"{name}: operator={prediction!r}")