
import cv2
import numpy as np


MIN_COMPONENT_AREA = 8
MIN_SPLIT_WIDTH = 6
MAX_SPLIT_DEPTH_RATIO = 0.55
MIN_WIDTH_RATIO = 1.65


def ensure_binary(image):

    if image is None:
        raise ValueError("Image cannot be None")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Image must be 2D or 3D")

    if gray.size == 0:
        raise ValueError("Image is empty")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Assume foreground occupies less than half the image.
    if np.count_nonzero(binary) > binary.size / 2:
        binary = cv2.bitwise_not(binary)

    return binary


def remove_small_noise(image, min_area=MIN_COMPONENT_AREA):
    

    binary = ensure_binary(image)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    cleaned = np.zeros_like(binary)

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def sort_boxes(boxes):
    """Sort bounding boxes from left to right."""

    return sorted(
        boxes,
        key=lambda box: (box[0], box[1])
    )


def segment_contours(image):
    """Detect character regions using contours."""

    binary = ensure_binary(image)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        
        mask = np.zeros((h, w), dtype=np.uint8)

        shifted = contour.copy()
        shifted[:, :, 0] -= x
        shifted[:, :, 1] -= y

        cv2.drawContours(
            mask, [shifted], -1, 255, thickness=cv2.FILLED
        )

        area = cv2.countNonZero(mask)

        if area >= MIN_COMPONENT_AREA:
            boxes.append((x, y, w, h))

    return sort_boxes(boxes)


def segment_connected_components(image):
    """Detect connected foreground components."""

    binary = ensure_binary(image)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    boxes = []

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        if area >= MIN_COMPONENT_AREA:
            boxes.append((
                int(x), int(y), int(w), int(h)
            ))

    return sort_boxes(boxes)


def segment_vertical_projection(image):
    """Find character regions separated by empty columns."""

    binary = ensure_binary(image)

    projection = np.count_nonzero(binary, axis=0)

    boxes = []
    start = None

    for x, count in enumerate(projection):

        if count > 0 and start is None:
            start = x

        if count == 0 and start is not None:
            region = binary[:, start:x]

            ys, xs = np.where(region > 0)

            if len(xs) >= MIN_COMPONENT_AREA:
                left = start + int(xs.min())
                right = start + int(xs.max()) + 1
                top = int(ys.min())
                bottom = int(ys.max()) + 1

                boxes.append((
                    left,
                    top,
                    right - left,
                    bottom - top
                ))

            start = None

    if start is not None:
        region = binary[:, start:]

        ys, xs = np.where(region > 0)

        if len(xs) >= MIN_COMPONENT_AREA:
            left = start + int(xs.min())
            right = start + int(xs.max()) + 1
            top = int(ys.min())
            bottom = int(ys.max()) + 1

            boxes.append((
                left,
                top,
                right - left,
                bottom - top
            ))

    return sort_boxes(boxes)


def estimate_reference_width(boxes):
    """
    Estimate typical character width using existing boxes.

    This is a heuristic, not a guaranteed digit-width estimate.
    """

    if len(boxes) < 2:
        return None

    widths = [
        w for x, y, w, h in boxes
        if w >= 3 and h >= 5
    ]

    if len(widths) < 2:
        return None

    return float(np.median(widths))


def find_split_position(region):
    """
    Find a possible split using a vertical projection valley.

    A split is accepted only if:
    - Both sides contain enough pixels.
    - Both sides are sufficiently wide.
    - The valley is substantially lower than nearby peaks.
    """

    height, width = region.shape

    if width < 2 * MIN_SPLIT_WIDTH:
        return None

    projection = np.count_nonzero(region, axis=0)

    # Smooth the projection slightly.
    smooth = np.convolve(
        projection.astype(np.float32),
        np.ones(3, dtype=np.float32) / 3,
        mode="same"
    )

    left_limit = max(
        MIN_SPLIT_WIDTH,
        int(width * 0.25)
    )

    right_limit = min(
        width - MIN_SPLIT_WIDTH,
        int(width * 0.75)
    )

    if right_limit <= left_limit:
        return None

    best_position = None
    best_score = float("inf")

    for position in range(left_limit, right_limit):

        left = region[:, :position]
        right = region[:, position:]

        left_pixels = cv2.countNonZero(left)
        right_pixels = cv2.countNonZero(right)

        if (
            left_pixels < MIN_COMPONENT_AREA or
            right_pixels < MIN_COMPONENT_AREA
        ):
            continue

        left_peak = float(
            np.max(smooth[:position])
        )

        right_peak = float(
            np.max(smooth[position:])
        )

        peak = min(left_peak, right_peak)

        if peak <= 0:
            continue

        valley = float(smooth[position])

        depth_ratio = valley / peak

        if depth_ratio > MAX_SPLIT_DEPTH_RATIO:
            continue

        # Prefer deep valleys near the centre.
        centre_penalty = (
            abs(position - width / 2) / width
        )

        score = depth_ratio + centre_penalty * 0.2

        if score < best_score:
            best_score = score
            best_position = position

    return best_position


def split_wide_region(binary, box, reference_width):
    """
    Attempt to split an unusually wide region.

    Returns original box when no reliable split is found.
    """

    x, y, w, h = box

    if reference_width is None:
        return [box]

    if w < reference_width * MIN_WIDTH_RATIO:
        return [box]

    region = binary[y:y + h, x:x + w]

    split_x = find_split_position(region)

    if split_x is None:
        return [box]

    left_region = region[:, :split_x]
    right_region = region[:, split_x:]

    result = []

    for offset, part in [
        (0, left_region),
        (split_x, right_region)
    ]:

        ys, xs = np.where(part > 0)

        if len(xs) < MIN_COMPONENT_AREA:
            return [box]

        local_x = int(xs.min())
        local_y = int(ys.min())

        local_w = int(xs.max() - xs.min() + 1)
        local_h = int(ys.max() - ys.min() + 1)

        result.append((
            x + offset + local_x,
            y + local_y,
            local_w,
            local_h
        ))

    return result


def segment_hybrid(image, reference_width=None):
    """
    Hybrid segmentation:
    1. Detect contours.
    2. Estimate typical character width.
    3. Identify unusually wide regions.
    4. Attempt projection-based splitting.
    """

    binary = ensure_binary(image)

    boxes = segment_contours(binary)

    if not boxes:
        return []

    if reference_width is None:
        reference_width = estimate_reference_width(boxes)

    if reference_width is None:
        return boxes

    result = []

    for box in boxes:
        split_boxes = split_wide_region(
            binary,
            box,
            reference_width
        )

        result.extend(split_boxes)

    return sort_boxes(result)


def extract_digit_regions(image, boxes):
    """Extract individual character crops."""

    binary = ensure_binary(image)

    height, width = binary.shape

    regions = []

    for x, y, w, h in boxes:

        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x2 > x1 and y2 > y1:
            regions.append(
                binary[y1:y2, x1:x2].copy()
            )

    return regions



def resize_digit(image, size=(28, 28)):
    """
    Resize an already-segmented character to 28x28.

    Preserves aspect ratio and centres the character
    on a black background.

    Input must have white foreground on black background.
    """

    if image is None or image.size == 0:
        raise ValueError("Invalid digit image")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Invalid image dimensions")

    # IMPORTANT:
    # Do not use ensure_binary() here.
    # The image is already segmented with white foreground.
    binary = np.where(gray > 127, 255, 0).astype(np.uint8)

    target_w, target_h = size

    canvas = np.zeros(
        (target_h, target_w),
        dtype=np.uint8
    )

    ys, xs = np.where(binary > 0)

    if len(xs) == 0:
        return canvas

    # Crop unnecessary background.
    x1 = xs.min()
    x2 = xs.max() + 1
    y1 = ys.min()
    y2 = ys.max() + 1

    cropped = binary[y1:y2, x1:x2]

    h, w = cropped.shape

    # Keep padding around the character.
    max_w = max(1, target_w - 8)
    max_h = max(1, target_h - 8)

    scale = min(
        max_w / w,
        max_h / h
    )

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=(
            cv2.INTER_AREA
            if scale < 1
            else cv2.INTER_LINEAR
        )
    )

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas


def normalize_image(image):
    """Convert pixel values to 0-1."""

    return image.astype(np.float32) / 255.0


def remove_border_artifacts(image):
    """
    Remove rectangular frame artifacts that touch
    the image boundary.

    Does not remove interior loops such as digit 0.
    """
    binary = ensure_binary(image).copy()
    height, width = binary.shape

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        touches_border = (
            x == 0 or y == 0 or
            x + w == width or y + h == height
        )

        # Only remove very large boundary components.
        # Small strokes touching the edge are preserved.
        frame_like = (
            w >= 0.85 * width and
            h >= 0.85 * height
        )

        if touches_border and frame_like:
            binary[labels == i] = 0

    return binary


def boxes_overlap_horizontally(a, b):
    """Calculate horizontal overlap between two boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    overlap = max(
        0,
        min(ax + aw, bx + bw) - max(ax, bx)
    )

    return overlap


def group_disconnected_components(boxes):
    """
    Group small vertically aligned components.

    Useful for disconnected strokes and division dots.

    This is a conservative heuristic and may need
    adjustment for individual handwriting styles.
    """
    boxes = sort_boxes(boxes)

    if not boxes:
        return []

    widths = [b[2] for b in boxes]
    heights = [b[3] for b in boxes]

    typical_width = float(np.median(widths))
    typical_height = float(np.median(heights))

    used = set()
    grouped = []

    for i, box in enumerate(boxes):
        if i in used:
            continue

        x, y, w, h = box

        group = [box]
        used.add(i)

        for j, other in enumerate(boxes):
            if j in used:
                continue

            ox, oy, ow, oh = other

            overlap = boxes_overlap_horizontally(
                box, other
            )

            horizontal_alignment = (
                overlap > 0 or
                abs((x + w / 2) - (ox + ow / 2))
                < typical_width * 0.25
            )

            vertical_gap = max(
                0,
                max(y, oy) - min(y + h, oy + oh)
            )

            small_component = (
                min(w * h, ow * oh)
                < typical_width * typical_height * 0.20
            )

            if (
                horizontal_alignment and
                vertical_gap < typical_height * 0.45 and
                small_component
            ):
                group.append(other)
                used.add(j)

        gx = min(b[0] for b in group)
        gy = min(b[1] for b in group)

        right = max(b[0] + b[2] for b in group)
        bottom = max(b[1] + b[3] for b in group)

        grouped.append((
            gx,
            gy,
            right - gx,
            bottom - gy
        ))

    return sorted(
        grouped,
        key=lambda b: (b[0] + b[2] / 2, b[1])
    )


def segment_improved(image):
    """
    Experimental component-grouping method.

    1. Remove boundary artifacts.
    2. Detect connected components.
    3. Group likely disconnected character strokes.
    """
    binary = remove_border_artifacts(image)

    boxes = segment_connected_components(binary)

    grouped_boxes = group_disconnected_components(
        boxes
    )

    return grouped_boxes


def group_division_symbols(image, boxes):
    """
    Group the three components of a handwritten division
    symbol into one bounding box.

    Expected arrangement:
           dot
         -------
           dot

    Returns updated bounding boxes.
    """

    binary = ensure_binary(image)
    boxes = sort_boxes(boxes)

    if len(boxes) < 3:
        return boxes

    used = set()
    result = []

    # Identify possible horizontal division bars.
    for i, (x, y, w, h) in enumerate(boxes):

        if i in used:
            continue

        # Division bars are generally wider than tall.
        if w < max(5, h * 2.0):
            continue

        bar_cx = x + w / 2
        bar_cy = y + h / 2

        upper_candidates = []
        lower_candidates = []

        for j, (dx, dy, dw, dh) in enumerate(boxes):

            if j == i or j in used:
                continue

            dot_cx = dx + dw / 2
            dot_cy = dy + dh / 2

            # Dots should be relatively small.
            if dw > max(6, w * 0.5):
                continue

            if dh > max(6, w * 0.5):
                continue

            # Dots must be horizontally aligned with the bar.
            horizontal_distance = abs(dot_cx - bar_cx)

            if horizontal_distance > max(5, w * 0.45):
                continue

            # Vertical distance must be reasonably close.
            vertical_distance = abs(dot_cy - bar_cy)

            if vertical_distance > max(12, w * 1.2):
                continue

            if dy + dh <= y:
                upper_candidates.append(
                    (vertical_distance, j)
                )

            elif dy >= y + h:
                lower_candidates.append(
                    (vertical_distance, j)
                )

        if not upper_candidates or not lower_candidates:
            continue

        upper_candidates.sort()
        lower_candidates.sort()

        upper_index = upper_candidates[0][1]
        lower_index = lower_candidates[0][1]

        selected = [
            boxes[i],
            boxes[upper_index],
            boxes[lower_index]
        ]

        # Combine the three bounding boxes.
        left = min(b[0] for b in selected)
        top = min(b[1] for b in selected)

        right = max(
            b[0] + b[2] for b in selected
        )

        bottom = max(
            b[1] + b[3] for b in selected
        )

        result.append((
            left,
            top,
            right - left,
            bottom - top
        ))

        used.update([
            i,
            upper_index,
            lower_index
        ])

    # Keep all components not used in a division symbol.
    for i, box in enumerate(boxes):
        if i not in used:
            result.append(box)

    return sort_boxes(result)



def segment_digits(
    image,
    method="contours",
    reference_width=None,
    group_division=True
):
    """
    Segment handwritten characters and group division signs.

    Returns:
        digit_images: list of cropped character images
        bounding_boxes: corresponding bounding boxes
    """

    binary = ensure_binary(image)

    if method == "contours":
        boxes = segment_contours(binary)

    elif method == "connected_components":
        boxes = segment_connected_components(binary)

    elif method == "vertical_projection":
        boxes = segment_vertical_projection(binary)

    elif method == "hybrid":
        boxes = segment_hybrid(
            binary,
            reference_width=reference_width
        )

    elif method == "improved":
        binary = remove_border_artifacts(binary)
        boxes = segment_improved(binary)

    else:
        raise ValueError(
            f"Unknown segmentation method: {method}"
        )

    # Group disconnected parts of division symbols.
    if group_division:
        boxes = group_division_symbols(binary, boxes)

    boxes = sort_boxes(boxes)

    digit_images = extract_digit_regions(
        binary,
        boxes
    )

    return digit_images, boxes
