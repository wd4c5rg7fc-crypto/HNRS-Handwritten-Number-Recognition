import cv2
from pathlib import Path
import compare_four_plus_v54 as v54
from HNRS_app_v74_smart import App

images = v54.base.IMAGES

print("V7.4 SINGLE-DIGIT CANDIDATE CHECK")
print("-" * 65)

candidate_count = 0

for number in range(1, 24):
    path = next(
        (p for p in images.glob(f"testcase{number}.*")
         if p.suffix.lower() in (".jpg", ".jpeg", ".png")),
        None
    )

    if path is None:
        print(f"testcase{number}: IMAGE MISSING")
        continue

    image = cv2.imread(str(path))

    if image is None:
        print(f"testcase{number}: CANNOT READ IMAGE")
        continue

    candidate = App._single_digit_candidate(None, image)
    detected = candidate is not None

    if detected:
        candidate_count += 1

    print(
        f"testcase{number:02d}: "
        f"{'SINGLE-DIGIT CANDIDATE' if detected else 'HYBRID PIPELINE CANDIDATE'}"
    )

print("-" * 65)
print(f"Single-digit candidates: {candidate_count}/23")
print(f"Other images: {23 - candidate_count}/23")
