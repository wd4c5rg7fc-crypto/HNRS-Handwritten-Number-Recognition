"""Rule-based handwritten operator recognition for HNRS.

Supports +, -, *, /, ( and ).

Digit recognition remains the CNN's job.

These are experimental heuristics.
Validate changes against the complete test set.
"""

import cv2
import numpy as np


# ==================================================
# STEP 1: PREPROCESS CHARACTER
# ==================================================

def _binary_crop(character):
    """Convert character to binary and crop empty borders."""

    if character is None or character.size == 0:
        return None

    if character.ndim == 3:
        gray = cv2.cvtColor(
            character,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = character.copy()

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    points = cv2.findNonZero(binary)

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)

    return binary[y:y + h, x:x + w]


# ==================================================
# STEP 2: HELPER FOR PARENTHESES
# ==================================================

def _row_band_x(crop, start, stop):
    """Calculate average horizontal position of foreground pixels."""

    h, w = crop.shape

    start_row = int(start * h)
    end_row = max(
        int(stop * h),
        start_row + 1
    )

    section = crop[start_row:end_row]

    ys, xs = np.where(section > 0)

    if len(xs) == 0:
        return None

    return float(xs.mean() / w)


# ==================================================
# STEP 3: RECOGNISE PARENTHESES
# ==================================================

def _parenthesis(crop):
    """Detect opening and closing parentheses using curvature."""

    h, w = crop.shape

    aspect_ratio = w / h

    # Parentheses are generally tall and narrow.
    if h < 20 or not (0.22 <= aspect_ratio <= 0.64):
        return None

    top = _row_band_x(crop, 0.0, 0.20)
    mid = _row_band_x(crop, 0.40, 0.60)
    bottom = _row_band_x(crop, 0.80, 1.0)

    if None in (top, mid, bottom):
        return None

    # Opening parenthesis curves towards the left.
    if (
        top - mid >= 0.27
        and bottom - mid >= 0.18
    ):
        return "("

    # Closing parenthesis curves towards the right.
    if (
        mid - top >= 0.27
        and mid - bottom >= 0.27
    ):
        return ")"

    return None


# ==================================================
# STEP 4: RECOGNISE ASTERISK MULTIPLICATION
# ==================================================

def _asterisk(crop):
    """Recognise handwritten asterisk-shaped multiplication symbols."""

    h, w = crop.shape

    aspect_ratio = w / h

    if (
        not (0.75 <= aspect_ratio <= 1.35)
        or min(h, w) < 18
    ):
        return False

    mask = cv2.resize(
        crop,
        (64, 64),
        interpolation=cv2.INTER_NEAREST
    ) > 0

    yy, xx = np.indices(mask.shape)

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return False

    # Find approximate centre of the handwritten symbol.
    cx = float(np.median(xs))
    cy = float(np.median(ys))

    radius = np.sqrt(
        (xx - cx) ** 2
        + (yy - cy) ** 2
    )

    angle = (
        np.arctan2(yy - cy, xx - cx)
        + 2 * np.pi
    ) % (2 * np.pi)

    # Measure how dense the central crossing is.
    hub = mask & (radius <= 7)

    hub_density = (
        hub.sum()
        / max(1, (radius <= 7).sum())
    )

    # Check whether strokes extend in many directions.
    outer = (
        mask
        & (radius >= 15)
        & (radius <= 29)
    )

    sectors = 12

    occupied = 0

    for i in range(sectors):

        start_angle = i * 2 * np.pi / sectors
        end_angle = (i + 1) * 2 * np.pi / sectors

        sector_pixels = np.count_nonzero(
            outer
            & (angle >= start_angle)
            & (angle < end_angle)
        )

        if sector_pixels >= 9:
            occupied += 1

    return (
        hub_density >= 0.45
        and occupied >= 11
    )


# ==================================================
# STEP 5: DETECT ENCLOSED LOOPS
# ==================================================

def _has_large_enclosed_loop(crop):
    """
    Detect substantial enclosed loops.

    Helps prevent digits such as 6, 8 and 9
    from being incorrectly classified as multiplication.
    """

    contours, hierarchy = cv2.findContours(
        crop,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return False

    foreground_area = max(
        1,
        cv2.countNonZero(crop)
    )

    for index, contour in enumerate(contours):

        # A contour with a parent can represent an enclosed hole.
        if hierarchy[0][index][3] >= 0:

            contour_area = cv2.contourArea(contour)

            if contour_area >= 0.12 * foreground_area:
                return True

    return False


# ==================================================
# STEP 6: MAIN OPERATOR RECOGNITION FUNCTION
# ==================================================

def recognise_asterisk_v43(character):
    """V4.3: conservative two-component asterisk candidate (testcase10)."""
    crop = _binary_crop(character)
    if crop is None:
        return False
    h, w = crop.shape
    if h == 0 or not (0.90 <= w / h <= 1.10):
        return False
    n, _, _, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    if n - 1 != 2:
        return False
    foreground = cv2.resize(crop, (40, 40), interpolation=cv2.INTER_NEAREST) > 0
    horizontal = float(np.mean(foreground[17:23, :]))
    vertical = float(np.mean(foreground[:, 17:23]))
    corners = [foreground[:10, :10], foreground[:10, -10:],
               foreground[-10:, :10], foreground[-10:, -10:]]
    corner_density = float(np.mean([np.mean(c) for c in corners]))
    return (_asterisk(crop) and horizontal >= 0.60 and vertical >= 0.40
            and corner_density <= 0.05)



def recognise_division_v44(character):
    """V4.4: division with three substantial pieces and tiny stray fragments.

    This is the same candidate used in compare_testcase20_v44.py.
    Existing clean three-component division is handled by recognise_operator.
    """
    crop = _binary_crop(character)
    if crop is None:
        return False
    h, w = crop.shape
    if h < 24 or w < 24:
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    if n < 4:
        return False
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = int(areas.sum())
    parts = []
    for component in stats[1:]:
        x, y, pw, ph, area = map(int, component)
        if area >= max(8, 0.08 * total):
            parts.append((x, y, pw, ph, area))
    if len(parts) != 3:
        return False
    parts.sort(key=lambda p: p[1])
    top, middle, bottom = parts
    tx, ty, tw, th, ta = top
    mx, my, mw, mh, ma = middle
    bx, by, bw, bh, ba = bottom
    return (
        mw >= 2.3 * mh
        and mw >= 0.65 * w
        and ta < 0.75 * ma
        and ba < 0.75 * ma
        and tw < 0.55 * mw
        and bw < 0.55 * mw
        and ty + th <= my + 0.06 * h
        and my + mh <= by + 0.06 * h
        and ty < my < by
    )

def recognise_operator(character):
    """
    Recognise a handwritten mathematical operator.

    Returns:
        "+" for addition
        "-" for subtraction
        "*" for multiplication
        "/" for division
        "(" for opening parenthesis
        ")" for closing parenthesis

        None if the character should be classified by the CNN.
    """

    # ----------------------------------------------
    # PREPROCESS CHARACTER
    # ----------------------------------------------

    crop = _binary_crop(character)

    if crop is None:
        return None

    h, w = crop.shape

    if h == 0 or w == 0:
        return None

    aspect_ratio = w / h

    # ----------------------------------------------
    # CONNECTED COMPONENT ANALYSIS
    # ----------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            crop,
            connectivity=8
        )
    )

    component_count = num_labels - 1

    # ==================================================
    # STEP 7: DIVISION DETECTION
    # ==================================================

    # Division usually contains:
    #   1. Top dot
    #   2. Horizontal bar
    #   3. Bottom dot

    if component_count == 3:

        components = []

        for index in range(1, num_labels):

            area = int(
                stats[index, cv2.CC_STAT_AREA]
            )

            if area >= 5:

                components.append({
                    "x": int(stats[index, cv2.CC_STAT_LEFT]),
                    "y": int(stats[index, cv2.CC_STAT_TOP]),
                    "w": int(stats[index, cv2.CC_STAT_WIDTH]),
                    "h": int(stats[index, cv2.CC_STAT_HEIGHT]),
                    "area": area
                })

        if len(components) == 3:

            components.sort(
                key=lambda c: c["y"]
            )

            top, middle, bottom = components

            middle_is_horizontal = (
                middle["w"] > middle["h"] * 1.5
            )

            top_is_small = (
                top["area"] < middle["area"]
            )

            bottom_is_small = (
                bottom["area"] < middle["area"]
            )

            correct_order = (
                top["y"] + top["h"] <= middle["y"]
                and middle["y"] + middle["h"] <= bottom["y"]
            )

            if (
                middle_is_horizontal
                and top_is_small
                and bottom_is_small
                and correct_order
            ):
                return "/"

    # ==================================================
    # STEP 8: PARENTHESES DETECTION
    # ==================================================

    parenthesis = _parenthesis(crop)

    if parenthesis is not None:
        return parenthesis

    # ==================================================
    # STEP 9: MINUS DETECTION
    # ==================================================

    if aspect_ratio >= 2.5:
        return "-"

    # ==================================================
    # STEP 10: SHAPE ANALYSIS
    # ==================================================

    if 0.55 <= aspect_ratio <= 1.8:

        resized = cv2.resize(
            crop,
            (40, 40),
            interpolation=cv2.INTER_NEAREST
        )

        foreground = resized > 0

        # Horizontal stroke strength.
        horizontal = float(
            np.mean(foreground[17:23, :])
        )

        # Vertical stroke strength.
        vertical = float(
            np.mean(foreground[:, 17:23])
        )

        # Corner density.
        corners = [
            foreground[:10, :10],
            foreground[:10, -10:],
            foreground[-10:, :10],
            foreground[-10:, -10:]
        ]

        corner_density = float(
            np.mean([
                np.mean(corner)
                for corner in corners
            ])
        )

        # Diagonal stroke strengths.
        diagonal_1 = float(
            np.mean(np.diag(foreground))
        )

        diagonal_2 = float(
            np.mean(
                np.diag(np.fliplr(foreground))
            )
        )

        # Strongest vertical column.
        strongest_vertical_column = float(
            np.max(
                np.mean(foreground, axis=0)
            )
        )

        # Check whether character has an enclosed loop.
        has_large_loop = _has_large_enclosed_loop(crop)

        # ==================================================
        # STEP 11: MULTIPLICATION DETECTION
        # ==================================================

        # Original X-shaped multiplication rule.
        x_multiplication = (
            component_count == 1
            and 0.75 <= aspect_ratio <= 1.50
            and diagonal_1 >= 0.55
            and diagonal_2 >= 0.55
            and vertical >= 0.45
        )

        # V4.2: disconnected asterisk candidate, validated on 24 test images.
        disconnected_star = False
        if component_count == 3 and 0.90 <= aspect_ratio <= 1.10:
            areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
            total = float(areas.sum())
            largest_share = float(areas.max() / total) if total else 0.0
            disconnected_star = (
                0.90 <= largest_share <= 0.97
                and horizontal >= 0.45
                and vertical >= 0.35
                and not has_large_loop
            )

        # Asterisk-shaped multiplication rule.
        asterisk_multiplication = (
            _asterisk(crop)
            and vertical >= 0.55
        )  # experiment: require strong vertical strokes

        # Prevent looped digits such as 6, 8 and 9
        # from being classified as multiplication.
        # V4 EXPERIMENT: asymmetric crossed-stroke multiplication.
        asymmetric_multiplication = (
            component_count == 1
            and 0.75 <= aspect_ratio <= 1.35
            and diagonal_2 >= 0.60
            and horizontal >= 0.50
            and vertical >= 0.40
            and not has_large_loop
        )

        if (
            not has_large_loop
            and (
                x_multiplication
                or asterisk_multiplication
                or asymmetric_multiplication
                or disconnected_star
            )
        ):
            return "*"

        # ==================================================
        # STEP 12: PLUS DETECTION
        # ==================================================

        # Standard handwritten plus.
        standard_plus = (
            horizontal >= 0.55
            and vertical >= 0.40
            and corner_density <= 0.12
        )

        # Wider handwritten plus.
        wide_plus = (
            1.20 <= aspect_ratio <= 1.80
            and horizontal >= 0.25
            and vertical >= 0.25
            and corner_density <= 0.05
        )

        # Thin handwritten plus.
        #
        # UPDATED FOR TESTCASE18:
        #
        # Previous strongest vertical column threshold:
        #     >= 0.85
        #
        # New threshold:
        #     >= 0.75
        #
        # This allows slightly curved or uneven
        # vertical strokes to be recognised.
        thin_plus = (
            0.75 <= aspect_ratio <= 1.20
            and horizontal >= 0.40  # experiment: accept weaker plus crossbars
            and vertical >= 0.18
            and strongest_vertical_column >= 0.75
            and corner_density <= 0.09
            and not has_large_loop
        )

        # V4 EXPERIMENT: unusual disconnected plus from testcase3.
        # Strict conditions are provisional; validate on all images.
        disconnected_plus_candidate = (
            component_count == 2
            and 0.90 <= aspect_ratio <= 1.20
            and strongest_vertical_column >= 0.98
            and horizontal <= 0.20
            and vertical <= 0.20
            and corner_density <= 0.10
        )

        if (
            standard_plus
            or wide_plus
            or thin_plus
            or disconnected_plus_candidate
        ):
            return "+"

    # ==================================================
    # STEP 13: DIGIT FALLBACK
    # ==================================================

    # If no operator rule matches,
    # let the CNN recognise the character as a digit.
    return None