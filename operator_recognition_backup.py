import cv2
import numpy as np


def recognise_operator(character):
    """
    Recognise handwritten mathematical operators.

    Supported operators:
        +  Addition
        -  Subtraction
        *  Multiplication
        /  Division

    Returns:
        Operator symbol if recognised.
        None if the character should be classified by the digit CNN.
    """

    # ==================================================
    # STEP 1: CONVERT TO GRAYSCALE
    # ==================================================

    if len(character.shape) == 3:
        gray = cv2.cvtColor(
            character,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = character.copy()

    # ==================================================
    # STEP 2: CONVERT TO BINARY
    # ==================================================

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ==================================================
    # STEP 3: FIND CHARACTER BOUNDARIES
    # ==================================================

    points = cv2.findNonZero(binary)

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)

    if w == 0 or h == 0:
        return None

    crop = binary[y:y + h, x:x + w]

    aspect_ratio = w / h

    # ==================================================
    # STEP 4: CONNECTED COMPONENT ANALYSIS
    # ==================================================

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            crop,
            connectivity=8
        )
    )

    component_count = num_labels - 1

    # ==================================================
    # STEP 5: DIVISION DETECTION
    # ==================================================

    # A handwritten division symbol usually contains:
    #   1. Top dot
    #   2. Horizontal line
    #   3. Bottom dot

    if component_count == 3:

        # Ignore extremely small components.
        components = []

        for index in range(1, num_labels):

            area = stats[index, cv2.CC_STAT_AREA]

            if area >= 5:
                components.append({
                    "x": stats[index, cv2.CC_STAT_LEFT],
                    "y": stats[index, cv2.CC_STAT_TOP],
                    "w": stats[index, cv2.CC_STAT_WIDTH],
                    "h": stats[index, cv2.CC_STAT_HEIGHT],
                    "area": area
                })

        if len(components) == 3:

            # Sort components from top to bottom.
            components.sort(
                key=lambda component: component["y"]
            )

            top = components[0]
            middle = components[1]
            bottom = components[2]

            # The middle component should be a
            # horizontal line.
            middle_is_horizontal = (
                middle["w"] > middle["h"] * 1.5
            )

            # Top and bottom should be smaller than
            # the horizontal line.
            top_is_small = (
                top["area"] < middle["area"]
            )

            bottom_is_small = (
                bottom["area"] < middle["area"]
            )

            # Ensure components appear vertically
            # in the correct order.
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
    # STEP 6: MINUS DETECTION
    # ==================================================

    if aspect_ratio >= 2.5:
        return "-"

    # ==================================================
    # STEP 7: SHAPE ANALYSIS
    # ==================================================

    if 0.55 <= aspect_ratio <= 1.8:

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

        # ==================================================
        # STEP 8: MULTIPLICATION DETECTION
        # ==================================================

        multiplication = (
            component_count == 1
            and 0.75 <= aspect_ratio <= 1.50
            and diagonal_1 >= 0.55
            and diagonal_2 >= 0.55
            and vertical >= 0.45
        )

        if multiplication:
            return "*"

        # ==================================================
        # STEP 9: PLUS DETECTION
        # ==================================================

        standard_plus = (
            horizontal >= 0.55
            and vertical >= 0.40
            and corner_density <= 0.12
        )

        wide_plus = (
            1.20 <= aspect_ratio <= 1.80
            and horizontal >= 0.25
            and vertical >= 0.25
            and corner_density <= 0.05
        )

        if standard_plus or wide_plus:
            return "+"

    # ==================================================
    # STEP 10: NOT AN OPERATOR
    # ==================================================

    return None