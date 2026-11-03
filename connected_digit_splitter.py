"""Conservative post-segmentation splitter for unusually wide connected digits."""
import cv2
import numpy as np


def _foreground(character):
    if character is None or character.size == 0:
        return None
    gray = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY) if character.ndim == 3 else character
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def split_connected_digits(characters, boxes, min_width_ratio=1.65, return_split_flags=False):
    """Split strong wide candidates.

    By default returns (characters, boxes), preserving compatibility.
    If return_split_flags=True, also returns one boolean per output character.
    True means that character came from a successfully split wide component
    and should be classified by the digit CNN rather than operator rules.
    """
    if len(characters) != len(boxes):
        raise ValueError("characters and boxes must have the same length")

    def result(chars, bxs, flags):
        return (chars, bxs, flags) if return_split_flags else (chars, bxs)

    if len(characters) < 3:
        return result(list(characters), list(boxes), [False] * len(characters))

    widths = [w for (_, _, w, h) in boxes if h >= 12 and w >= 5
              and 0.18 <= w / max(h, 1) <= 1.30]
    if len(widths) < 2:
        return result(list(characters), list(boxes), [False] * len(characters))
    reference_width = float(np.median(widths))

    from Segmentation import find_split_position

    out_characters, out_boxes, flags = [], [], []
    for character, (x, y, w, h) in zip(characters, boxes):
        binary = _foreground(character)
        if (binary is None or h < 20 or w < reference_width * min_width_ratio
                or w / max(h, 1) < 1.25 or w / max(h, 1) > 3.5):
            out_characters.append(character)
            out_boxes.append((x, y, w, h))
            flags.append(False)
            continue

        ch, cw = binary.shape
        if (ch, cw) != (h, w):
            out_characters.append(character)
            out_boxes.append((x, y, w, h))
            flags.append(False)
            continue

        split = find_split_position(binary)
        if split is None or split < 6 or cw - split < 6:
            out_characters.append(character)
            out_boxes.append((x, y, w, h))
            flags.append(False)
            continue

        pieces = []
        for offset, part in ((0, binary[:, :split]), (split, binary[:, split:])):
            points = cv2.findNonZero(part)
            if points is None or cv2.countNonZero(part) < 20:
                break
            px, py, pw, ph = cv2.boundingRect(points)
            if pw < 5 or ph < 12 or not (0.22 <= pw / ph <= 1.85):
                break
            pieces.append((part[py:py + ph, px:px + pw].copy(),
                           (x + offset + px, y + py, pw, ph)))

        if len(pieces) == 2:
            for part, box in pieces:
                out_characters.append(part)
                out_boxes.append(box)
                flags.append(True)
        else:
            out_characters.append(character)
            out_boxes.append((x, y, w, h))
            flags.append(False)

    return result(out_characters, out_boxes, flags)
