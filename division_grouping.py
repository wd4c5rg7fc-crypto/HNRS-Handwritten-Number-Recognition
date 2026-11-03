import cv2
import numpy as np


def merge_division_parts(characters, boxes):
    """
    Merge a division symbol split into two regions.

    Expected pattern:
        Region A: upper dot + horizontal line
        Region B: small lower dot

    Bounding boxes must be (x, y, width, height).

    Returns:
        updated_characters, updated_boxes
    """

    if len(characters) != len(boxes):
        raise ValueError(
            "Character count and bounding-box count must match."
        )

    merged_characters = []
    merged_boxes = []

    i = 0

    while i < len(characters):

        # Check whether another region follows.
        if i + 1 < len(characters):

            x1, y1, w1, h1 = map(int, boxes[i])
            x2, y2, w2, h2 = map(int, boxes[i + 1])

            # Horizontal centres.
            center1 = x1 + w1 / 2
            center2 = x2 + w2 / 2

            # Check whether the second region is a
            # small dot below the first region.
            small_dot = (
                w2 <= w1 * 0.35
                and h2 <= h1 * 0.35
            )

            horizontally_aligned = (
                abs(center1 - center2) <= w1 * 0.25
            )

            below_first = (
                y2 >= y1 + h1
            )

            vertical_gap = (
                y2 - (y1 + h1)
            )

            reasonably_close = (
                vertical_gap <= h1 * 1.0
            )

            # Require the first region to contain
            # at least two disconnected components:
            # upper dot and horizontal stroke.
            first = characters[i]

            if len(first.shape) == 3:
                first_gray = cv2.cvtColor(
                    first,
                    cv2.COLOR_BGR2GRAY
                )
            else:
                first_gray = first

            _, first_binary = cv2.threshold(
                first_gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            component_count, _ = cv2.connectedComponents(
                first_binary,
                connectivity=8
            )

            has_multiple_parts = (
                component_count - 1 >= 2
            )

            if (
                small_dot
                and horizontally_aligned
                and below_first
                and reasonably_close
                and has_multiple_parts
            ):

                # Calculate bounding box containing both.
                new_x = min(x1, x2)
                new_y = min(y1, y2)

                new_right = max(
                    x1 + w1,
                    x2 + w2
                )

                new_bottom = max(
                    y1 + h1,
                    y2 + h2
                )

                new_w = new_right - new_x
                new_h = new_bottom - new_y

                # Create a black canvas for the merged symbol.
                canvas = np.zeros(
                    (new_h, new_w),
                    dtype=np.uint8
                )

                # Convert character images to grayscale.
                first_image = first_gray

                second = characters[i + 1]

                if len(second.shape) == 3:
                    second_image = cv2.cvtColor(
                        second,
                        cv2.COLOR_BGR2GRAY
                    )
                else:
                    second_image = second

                # Place both regions at their original positions.
                offset_x1 = x1 - new_x
                offset_y1 = y1 - new_y

                offset_x2 = x2 - new_x
                offset_y2 = y2 - new_y

                # Use actual image dimensions for placement.
                h_first, w_first = first_image.shape
                h_second, w_second = second_image.shape

                canvas[
                    offset_y1:offset_y1 + h_first,
                    offset_x1:offset_x1 + w_first
                ] = first_image

                canvas[
                    offset_y2:offset_y2 + h_second,
                    offset_x2:offset_x2 + w_second
                ] = second_image

                merged_characters.append(canvas)

                merged_boxes.append(
                    (new_x, new_y, new_w, new_h)
                )

                print(
                    f"  Merged division fragments "
                    f"{i + 1} and {i + 2}"
                )

                # Skip both original regions.
                i += 2
                continue

        # Keep ordinary characters unchanged.
        merged_characters.append(
            characters[i]
        )

        merged_boxes.append(
            boxes[i]
        )

        i += 1

    return merged_characters, merged_boxes