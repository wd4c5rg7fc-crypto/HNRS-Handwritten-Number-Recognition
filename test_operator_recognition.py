import cv2
from pathlib import Path
from operator_recognition import recognise_operator

# Folder containing teammate's segmented characters
FOLDER = Path(
    "member_sources/member_2/results/testcase1/connected_components"
)

# Expected characters
EXPECTED = {
    "region_01": None,  # Digit 1
    "region_02": None,  # Digit 2
    "region_03": "-",   # Minus
    "region_04": None,  # Digit 5
    "region_05": "+",   # Plus
    "region_06": None,  # Digit 8
}

print("\n=== OPERATOR RECOGNITION TEST ===\n")

correct = 0

for name, expected in EXPECTED.items():

    # Find image regardless of extension
    matches = [
        p for p in FOLDER.glob(f"{name}.*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]

    if not matches:
        print(f"{name}: IMAGE NOT FOUND")
        continue

    image = cv2.imread(str(matches[0]), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"{name}: COULD NOT LOAD IMAGE")
        continue

    prediction = recognise_operator(image)

    passed = prediction == expected

    if passed:
        correct += 1

    print(
        f"{name} | "
        f"Expected: {str(expected):5} | "
        f"Predicted: {str(prediction):5} | "
        f"{'PASS' if passed else 'FAIL'}"
    )

print(f"\nCorrect: {correct}/{len(EXPECTED)}")