"""Read-only V5.2 comparison against V5.1.

Run:
    python compare_narrow_paren_v52.py --new-dir test_images/new_tests

Requires compare_narrow_paren_v51.py and its dependencies in project root.
Only writes test_results/v52_narrow_paren_comparison.csv.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import compare_narrow_paren_v51 as v51

base = v51.base
OUT = Path(__file__).resolve().parent / "test_results" / "v52_narrow_paren_comparison.csv"


def narrow_open_paren_v52(char):
    """Accept both conventional curved '(' and a narrow upper-hook variant."""
    accepted, old_detail = v51.narrow_open_paren_geometry(char)
    if accepted:
        return True, "V5.1 geometry: " + old_detail

    mask = v51.v50.v49.v48.v47.experiment.mask_of(char)
    if mask is None:
        return False, "empty"
    h, w = mask.shape
    if h < 25 or w < 5 or not 0.12 <= w / h <= 0.55:
        return False, f"size={w}x{h}, aspect={w/h:.2f}"

    centers = []
    for lo, hi in ((0.08, 0.28), (0.42, 0.58), (0.72, 0.92)):
        band = mask[int(lo*h):max(int(hi*h), int(lo*h)+1)]
        _, xs = np.nonzero(band)
        if len(xs) < 3:
            return False, f"sparse-band size={w}x{h}"
        centers.append(float(np.median(xs)))
    top, mid, bottom = centers

    # Some handwritten '(' shapes have a pronounced upper hook and an
    # almost vertical lower half. A '1' can also be hooked, so geometry
    # alone is insufficient; recognise_v52 requires strong syntax context.
    top_shift = (top - mid) / w
    lower_shift = abs(bottom - mid) / w
    ink_density = float(np.count_nonzero(mask)) / (h*w)
    ok = (top_shift >= 0.30 and lower_shift <= 0.18
          and 0.04 <= ink_density <= 0.65)
    return ok, (f"upper-hook size={w}x{h} aspect={w/h:.2f} "
                f"top-shift={top_shift:.2f} lower-shift={lower_shift:.2f} "
                f"density={ink_density:.2f} centers={[round(x, 1) for x in centers]}")


def has_matching_close(symbols, j):
    """Require an available ')' after this position, accounting for nesting."""
    depth = 1
    for symbol in symbols[j+1:]:
        if symbol == "(":
            depth += 1
        elif symbol == ")":
            depth -= 1
            if depth == 0:
                return True
    return False


def recognise_v52(chars, boxes, flags, model, parens):
    old, diagnostics = v51.recognise_v51(chars, boxes, flags, model, parens)
    diagnostics = list(diagnostics)
    revised = v51.v50.v49.v48.v47.experiment.revised
    _, _, removed, _ = revised.recognise_revised(chars, boxes, flags, model, parens)
    keep = [i for i in range(len(chars)) if i not in removed]
    if (v51.v50.v49.v48.v47.experiment.v45.edge_fragment_candidate(
            0, chars, boxes, flags, parens) and 0 in keep):
        keep.remove(0)
    if len(old) != len(keep):
        return old, diagnostics + ["V5.2 skipped: alignment mismatch"]

    symbols = list(old)
    for j in range(1, len(symbols)-1):
        i = keep[j]
        if (symbols[j] != "1" or symbols[j-1] != "/"
                or not symbols[j+1].isdigit() or flags[i] or parens[i]):
            continue
        if not has_matching_close(symbols, j):
            diagnostics.append(f"V5.2 candidate {i}: rejected (no matching closing parenthesis)")
            continue
        accepted, detail = narrow_open_paren_v52(chars[i])
        diagnostics.append(f"V5.2 candidate {i}: {accepted} ({detail})")
        if accepted:
            symbols[j] = "("
            diagnostics.append(f"V5.2 REPLACED segment {i}: 1 -> (")
    return "".join(symbols), diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-dir", type=Path)
    args = parser.parse_args()
    if not base.MODEL.is_file() or not base.IMAGES.is_dir():
        raise FileNotFoundError("Missing model or original testcase image directory")
    model = base.load_model(str(base.MODEL))
    originals = sorted(
        (p for p in base.IMAGES.glob("testcase*.*")
         if p.suffix.lower() in (".png", ".jpg", ".jpeg")),
        key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    items = [
        ("original", p, base.EXPECTED.get(
            int("".join(c for c in p.stem if c.isdigit()) or 0), ""))
        for p in originals
    ]
    if args.new_dir:
        if not args.new_dir.is_dir():
            parser.error(f"New image directory not found: {args.new_dir}")
        known = {
            "new_test_01": "18*(12-7)+36/9",
            "new_test_02": "200-(48/6)*7",
            "new_test_03": "(100-25)/(5+10)*3",
        }
        for path in sorted(args.new_dir.iterdir()):
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            stem = path.name
            while Path(stem).suffix.lower() in (".png", ".jpg", ".jpeg"):
                stem = Path(stem).stem
            items.append(("new", path, known.get(stem, "")))

    rows = []
    for group, path, expected in items:
        row = dict(group=group, image=path.name, expected=expected,
                   v51="", v52="", diagnostics="", error="")
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Cannot read image")
            chars, boxes, flags, parens = v51.v50.segment(image)
            row["v51"], _ = v51.recognise_v51(chars, boxes, flags, model, parens)
            row["v52"], diag = recognise_v52(chars, boxes, flags, model, parens)
            row["diagnostics"] = "; ".join(diag)
        except Exception as exc:
            row["error"] = repr(exc)
        rows.append(row)
        print(f'{group:8} {path.name:27} expected={expected or "?":22} '
              f'V5.1={row["v51"]:22} V5.2={row["v52"]:22} '
              f'{row["error"] or row["diagnostics"]}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for group in ("original", "new"):
        subset = [r for r in rows if r["group"] == group and r["expected"] and not r["error"]]
        before = sum(r["v51"] == r["expected"] for r in subset)
        after = sum(r["v52"] == r["expected"] for r in subset)
        regressions = [r["image"] for r in subset
                       if r["v51"] == r["expected"] and r["v52"] != r["expected"]]
        print(f"{group}: V5.1 {before}/{len(subset)} | "
              f"V5.2 {after}/{len(subset)} | regressions={regressions}")
    print("Changed:", [r["image"] for r in rows
                       if r["v51"] != r["v52"] and not r["error"]])
    print("Errors:", [(r["image"], r["error"]) for r in rows if r["error"]])
    print("CSV:", OUT)
    print("Read-only: no GUI, model, or input images modified.")


if __name__ == "__main__":
    main()
