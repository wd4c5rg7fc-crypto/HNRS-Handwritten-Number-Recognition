"""Experimental V5 mask-based cleanup. Does not modify teammate segmentation.

Detect a parenthesis component wrapped around a second character in an
already-extracted bounding-box crop. The second character is generally already
present in the neighbouring crop, so retain only the parenthesis mask.
Conservative and experimental: validate against the full labelled dataset.
"""
import cv2
import numpy as np


def _parts(character):
    gray = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY) if character.ndim == 3 else character
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    parts = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if area < 50:
            continue
        mask = np.zeros_like(binary)
        mask[labels == i] = 255
        parts.append((mask, (x, y, w, h, area)))
    return parts


def _tight(mask):
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    return mask[y:y+h, x:x+w].copy()


def isolate_parentheses(characters, boxes, return_flags=False):
    """Replace mixed parenthesis/digit crops with masked parenthesis crops.

    Returns same number of crops and bounding boxes. Optional flags contain
    '(' or ')' when an isolated parenthesis is confidently detected, else None.
    This is *not* a general-purpose touching-character splitter.
    """
    if len(characters) != len(boxes):
        raise ValueError('characters and boxes lengths differ')
    output = list(characters)
    flags = [None] * len(characters)
    for index, (char, box) in enumerate(zip(characters, boxes)):
        parts = _parts(char)
        if len(parts) != 2:
            continue
        ch, cw = char.shape[:2]
        if ch < 45 or cw < 35:
            continue
        # One component spans nearly the full crop height and forms a tall,
        # narrow outside arc; the other is an interior digit-shaped component.
        for arc_mask, (ax, ay, aw, ah, aa) in parts:
            if ah < 0.88 * ch or aw < 0.60 * cw:
                continue
            if not (0.30 <= aw / max(ah, 1) <= 1.05):
                continue
            other = next(p for p in parts if p[0] is not arc_mask)
            _, (bx, by, bw, bh, ba) = other
            if bh >= 0.85 * ch or ba < 0.20 * aa:
                continue
            if not (ay <= by and by + bh <= ay + ah):
                continue
            # Arc should occupy one side of the other component.
            # Confirm from leftmost and rightmost 15% of arc pixels.
            ys, xs = np.where(arc_mask > 0)
            if len(xs) < 80:
                continue
            upper = xs[ys < ch * 0.25]
            lower = xs[ys > ch * 0.75]
            mid = xs[(ys >= ch * 0.4) & (ys <= ch * 0.6)]
            if min(len(upper), len(lower), len(mid)) < 10:
                continue
            topx, midx, botx = np.median(upper), np.median(mid), np.median(lower)
            left_arc = (topx - midx > cw * 0.10 and botx - midx > cw * 0.10)
            right_arc = (midx - topx > cw * 0.10 and midx - botx > cw * 0.10)
            if not (left_arc or right_arc):
                # Alternative: outer arc sits clearly to the left or right of digit.
                left_arc = ax < bx - cw * 0.15 and (ax + aw) >= bx
                right_arc = (ax + aw) > bx + bw + cw * 0.15 and ax <= bx
            if not (left_arc or right_arc):
                continue
            # The companion crop must already cover the digit; otherwise
            # masking it would silently delete an actual character.
            other_abs = (box[0] + bx, box[1] + by, bw, bh)
            duplicated = False
            for j, (xx, yy, ww, hh) in enumerate(boxes):
                if j == index:
                    continue
                overlap_w = max(0, min(xx + ww, other_abs[0] + bw) - max(xx, other_abs[0]))
                overlap_h = max(0, min(yy + hh, other_abs[1] + bh) - max(yy, other_abs[1]))
                if overlap_w * overlap_h >= 0.70 * bw * bh:
                    duplicated = True
                    break
            if not duplicated:
                continue
            output[index] = _tight(arc_mask)
            flags[index] = '(' if left_arc else ')'
            break
    return (output, list(boxes), flags) if return_flags else (output, list(boxes))
