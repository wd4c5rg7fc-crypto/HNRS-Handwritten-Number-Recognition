
import cv2
import numpy as np
import pytest

from Segmentation import (
    ensure_binary,
    segment_digits,
    segment_contours,
    segment_connected_components,
    segment_vertical_projection,
    resize_digit,
    normalize_image,
    group_division_symbols
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_test_image(count=4, spacing=20):
    """
    Create an artificial image containing separated
    rectangular characters.
    """
    width = 20 + count * (15 + spacing)

    image = np.zeros((70, width), dtype=np.uint8)

    for i in range(count):
        x = 10 + i * (15 + spacing)

        cv2.rectangle(
            image,
            (x, 15),
            (x + 10, 50),
            255,
            -1
        )

    return image


def make_division_image():
    """
    Create an artificial division symbol:
         dot
         bar
         dot
    """
    image = np.zeros((100, 120), dtype=np.uint8)

    cv2.circle(image, (60, 30), 3, 255, -1)
    cv2.line(image, (45, 50), (75, 50), 255, 3)
    cv2.circle(image, (60, 70), 3, 255, -1)

    return image


# ============================================================
# TEST 1: BASIC SEGMENTATION METHODS
# ============================================================

@pytest.mark.parametrize("count", [1, 2, 4, 6])
@pytest.mark.parametrize(
    "method",
    [
        "contours",
        "connected_components",
        "vertical_projection"
    ]
)
def test_basic_segmentation(count, method):

    image = make_test_image(count)

    digits, boxes = segment_digits(
        image,
        method=method,
        group_division=False
    )

    assert len(digits) == count
    assert len(boxes) == count

    # Check correct left-to-right ordering.
    assert boxes == sorted(
        boxes,
        key=lambda b: (b[0], b[1])
    )


# ============================================================
# TEST 2: DIFFERENT DIGIT SPACING
# ============================================================

@pytest.mark.parametrize("spacing", [5, 10, 20, 35])
def test_different_spacing(spacing):

    image = make_test_image(
        count=4,
        spacing=spacing
    )

    digits, boxes = segment_digits(
        image,
        method="contours",
        group_division=False
    )

    assert len(digits) == 4
    assert len(boxes) == 4


# ============================================================
# TEST 3: SINGLE DIGIT
# ============================================================

def test_single_digit():

    image = make_test_image(count=1)

    digits, boxes = segment_digits(
        image,
        method="contours"
    )

    assert len(digits) == 1
    assert len(boxes) == 1


# ============================================================
# TEST 4: EMPTY IMAGE
# ============================================================

@pytest.mark.parametrize(
    "method",
    [
        "contours",
        "connected_components",
        "vertical_projection",
        "hybrid",
        "improved"
    ]
)
def test_empty_image(method):

    image = np.zeros((100, 200), dtype=np.uint8)

    digits, boxes = segment_digits(
        image,
        method=method
    )

    assert digits == []
    assert boxes == []


# ============================================================
# TEST 5: SMALL NOISE
# ============================================================

def test_noise_component():

    image = make_test_image(count=2)

    # Add one isolated noise pixel.
    image[2, 2] = 255

    digits, boxes = segment_digits(
        image,
        method="contours",
        group_division=False
    )

    assert len(digits) == 2
    assert len(boxes) == 2


# ============================================================
# TEST 6: HYBRID METHOD WITH SEPARATED DIGITS
# ============================================================

def test_hybrid_separated_digits():

    image = make_test_image(count=4)

    digits, boxes = segment_digits(
        image,
        method="hybrid",
        group_division=False
    )

    assert len(digits) == 4
    assert len(boxes) == 4


# ============================================================
# TEST 7: TOUCHING DIGITS
# ============================================================

def test_touching_digits_baseline():

    image = np.zeros((70, 130), dtype=np.uint8)

    cv2.rectangle(
        image, (30, 15), (50, 50), 255, -1
    )

    cv2.rectangle(
        image, (50, 15), (75, 50), 255, -1
    )

    digits, boxes = segment_digits(
        image,
        method="contours",
        group_division=False
    )

    # Basic contours see the connected shape as one.
    assert len(digits) == 1


def test_hybrid_touching_digits():

    image = np.zeros((70, 150), dtype=np.uint8)

    # Separated reference character.
    cv2.rectangle(
        image, (5, 15), (20, 50), 255, -1
    )

    # Two connected regions with a narrow bridge.
    cv2.rectangle(
        image, (40, 15), (55, 50), 255, -1
    )

    cv2.rectangle(
        image, (75, 15), (90, 50), 255, -1
    )

    cv2.line(
        image, (55, 32), (75, 32), 255, 1
    )

    digits, boxes = segment_digits(
        image,
        method="hybrid",
        reference_width=16,
        group_division=False
    )

    # One reference digit and two separated regions.
    assert len(digits) == 3
    assert len(boxes) == 3


# ============================================================
# TEST 8: DIVISION SYMBOL GROUPING
# ============================================================

def test_division_symbol_grouping():

    image = make_division_image()

    digits, boxes = segment_digits(
        image,
        method="connected_components",
        group_division=True
    )

    # Three disconnected components must form one symbol.
    assert len(digits) == 1
    assert len(boxes) == 1

    resized = resize_digit(digits[0])

    assert resized.shape == (28, 28)
    assert np.count_nonzero(resized) > 0


# ============================================================
# TEST 9: DIVISION WITHOUT GROUPING
# ============================================================

def test_division_without_grouping():

    image = make_division_image()

    digits, boxes = segment_digits(
        image,
        method="connected_components",
        group_division=False
    )

    # Upper dot, bar and lower dot remain separate.
    assert len(digits) == 3
    assert len(boxes) == 3


# ============================================================
# TEST 10: MINUS SIGN MUST REMAIN SEPARATE
# ============================================================

def test_minus_sign():

    image = np.zeros((100, 150), dtype=np.uint8)

    cv2.line(
        image,
        (30, 50),
        (65, 50),
        255,
        3
    )

    digits, boxes = segment_digits(
        image,
        method="connected_components",
        group_division=True
    )

    assert len(digits) == 1
    assert len(boxes) == 1


# ============================================================
# TEST 11: DIVISION INSIDE AN EXPRESSION
# ============================================================

def test_division_inside_expression():

    image = np.zeros((100, 260), dtype=np.uint8)

    # First digit.
    cv2.rectangle(
        image, (15, 30), (35, 70), 255, -1
    )

    # Second digit.
    cv2.rectangle(
        image, (50, 30), (70, 70), 255, -1
    )

    # Division symbol.
    cv2.circle(
        image, (115, 30), 3, 255, -1
    )

    cv2.line(
        image, (100, 50), (130, 50), 255, 3
    )

    cv2.circle(
        image, (115, 70), 3, 255, -1
    )

    # Final digit.
    cv2.rectangle(
        image, (165, 30), (185, 70), 255, -1
    )

    digits, boxes = segment_digits(
        image,
        method="connected_components",
        group_division=True
    )

    # Expected: digit, digit, division, digit.
    assert len(digits) == 4
    assert len(boxes) == 4

    assert boxes == sorted(
        boxes,
        key=lambda b: (b[0], b[1])
    )

    for digit in digits:
        resized = resize_digit(digit)
        assert resized.shape == (28, 28)


# ============================================================
# TEST 12: DIRECT DIVISION GROUPING
# ============================================================

def test_direct_division_grouping():

    image = make_division_image()

    original_boxes = segment_connected_components(image)

    assert len(original_boxes) == 3

    grouped_boxes = group_division_symbols(
        image,
        original_boxes
    )

    assert len(grouped_boxes) == 1


# ============================================================
# TEST 13: 28x28 RESIZING
# ============================================================

def test_resize_digit():

    image = make_test_image(count=1)

    digits, boxes = segment_digits(
        image,
        method="contours"
    )

    resized = resize_digit(digits[0])

    assert resized.shape == (28, 28)
    assert resized.dtype == np.uint8
    assert np.count_nonzero(resized) > 0


# ============================================================
# TEST 14: RESIZE EMPTY IMAGE
# ============================================================

def test_resize_empty_image():

    image = np.zeros((40, 40), dtype=np.uint8)

    resized = resize_digit(image)

    assert resized.shape == (28, 28)
    assert np.count_nonzero(resized) == 0


# ============================================================
# TEST 15: NORMALISATION
# ============================================================

def test_normalization():

    image = np.array(
        [[0, 128, 255]],
        dtype=np.uint8
    )

    normalized = normalize_image(image)

    assert normalized.shape == image.shape
    assert normalized.dtype == np.float32

    assert normalized.min() >= 0
    assert normalized.max() <= 1

    assert normalized[0, 0] == 0.0
    assert normalized[0, 2] == 1.0


# ============================================================
# TEST 16: COLOUR IMAGE INPUT
# ============================================================

def test_colour_image():

    image = make_test_image(count=2)

    colour_image = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    digits, boxes = segment_digits(
        colour_image,
        method="contours"
    )

    assert len(digits) == 2
    assert len(boxes) == 2


# ============================================================
# TEST 17: INVALID METHOD
# ============================================================

def test_invalid_method():

    image = make_test_image(count=2)

    with pytest.raises(ValueError):
        segment_digits(
            image,
            method="invalid"
        )


# ============================================================
# TEST 18: INVALID IMAGE INPUT
# ============================================================

def test_none_image():

    with pytest.raises(ValueError):
        ensure_binary(None)


def test_zero_size_image():

    image = np.zeros((0, 0), dtype=np.uint8)

    with pytest.raises(ValueError):
        ensure_binary(image)


# ============================================================
# TEST 19: BOUNDING BOX ORDER
# ============================================================

def test_bounding_box_order():

    image = make_test_image(
        count=6,
        spacing=15
    )

    digits, boxes = segment_digits(
        image,
        method="connected_components"
    )

    x_coordinates = [box[0] for box in boxes]

    assert x_coordinates == sorted(x_coordinates)




@pytest.mark.parametrize("count", [1, 2, 4, 6])
def test_all_digits_resize(count):

    image = make_test_image(count=count)

    digits, boxes = segment_digits(
        image,
        method="contours"
    )

    assert len(digits) == count

    for digit in digits:
        resized = resize_digit(digit)

        assert resized.shape == (28, 28)
        assert resized.dtype == np.uint8
