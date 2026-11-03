"""Read-only V4.5 vs revised V4.6 test for a handwritten '+(' pair.

Place alongside compare_testcase13_v46.py and its dependencies.
Run: python compare_testcase13_v46_revised.py
Does not modify the GUI, trained models, or recognizer source files.
"""
from pathlib import Path
import numpy as np
import compare_testcase13_v46 as experiment


def open_paren_geometry_revised(char):
    """Test a LEFT-bulging '('; earlier rule assumed a rightward bulge.

    This is a hypothesis, not a guarantee: validate all labelled images.
    The original experiment already requires an adjacent CNN '4','6' pair
    and a geometrically cross-like '+' before this rule can take effect.
    """
    mask = experiment.mask_of(char)
    if mask is None:
        return False, 'empty'
    h, w = mask.shape
    if h < 45 or w < 20 or not 0.35 <= w / h <= 1.05:
        return False, 'aspect'

    centers = []
    for lo, hi in ((0.10, 0.25), (0.42, 0.58), (0.75, 0.90)):
        band = mask[int(h * lo):max(int(h * hi), int(h * lo) + 1)]
        _, xs = np.nonzero(band)
        if len(xs) < 10:
            return False, 'sparse-band'
        centers.append(float(np.median(xs)) / w)

    top, middle, bottom = centers
    bend = middle - (top + bottom) / 2
    # The uploaded testcase13 '(' has centers [0.39, 0.08, 0.13]
    # and bend -0.18. Its ink bulges LEFT at the middle.
    accepted = (
        bend <= -0.12
        and top - middle >= 0.18
        and abs(top - bottom) <= 0.38
    )
    return accepted, (f'left-bend={bend:.2f} '
                      f'centers={[round(x, 2) for x in centers]}')


def main():
    # Replace only the in-memory geometry function used by the existing
    # read-only comparison pipeline. No imported file is changed on disk.
    experiment.open_paren_geometry = open_paren_geometry_revised
    experiment.OUTPUT = (Path(__file__).resolve().parent / 'test_results'
                         / 'testcase13_v46_revised_comparison.csv')
    experiment.main()


if __name__ == '__main__':
    main()
