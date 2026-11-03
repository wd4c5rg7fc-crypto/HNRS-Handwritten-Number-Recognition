"""Modern HNRS desktop GUI. Place in project root as app.py; run python app.py."""
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

from hnrs.image_acquisition import generate_number_image
from operator_recognition import recognise_operator, recognise_asterisk_v43, recognise_division_v44
from division_grouping import merge_division_parts
from connected_digit_splitter import split_connected_digits
from parenthesis_mask_splitter_v5 import isolate_parentheses

PROJECT_DIR = Path(__file__).resolve().parent
MEMBER_DIR = PROJECT_DIR / "member_sources" / "member_2"
sys.path.insert(0, str(MEMBER_DIR))
from Segmentation import segment_digits
from expression_evaluator import calculate_expression

BG = "#0B1020"
PANEL = "#151D32"
PANEL2 = "#1B2740"
TEXT = "#F1F5FF"
MUTED = "#9CAAC4"
ACCENT = "#728BFF"
GREEN = "#51D5B2"
BORDER = "#2A3855"


def preprocess_digit(character):
    """White-on-black segmented character -> centered 28x28 float32 image."""
    gray = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY) if character.ndim == 3 else character.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    if not w or not h:
        return None
    cropped = binary[y:y + h, x:x + w]
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    ox, oy = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    moments = cv2.moments(canvas)
    if moments["m00"]:
        cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        dx, dy = round(13.5 - cx) + 1, round(13.5 - cy)
        canvas = cv2.warpAffine(canvas, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28))
    return canvas.astype(np.float32) / 255.0


def predict_digit_adaptive(processed, model, threshold=0.80):
    """V4.2: try left shifts for uncertain CNN predictions only."""
    probabilities = model.predict(processed.reshape(1, 28, 28, 1), verbose=0)[0]
    best_digit = int(np.argmax(probabilities))
    best_confidence = float(probabilities[best_digit])
    best_shift = 0
    if best_confidence < threshold:
        for dx in (-1, -2):
            shifted = cv2.warpAffine(
                processed, np.float32([[1, 0, dx], [0, 1, 0]]), (28, 28),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            shifted_probs = model.predict(shifted.reshape(1, 28, 28, 1), verbose=0)[0]
            candidate = int(np.argmax(shifted_probs))
            confidence = float(shifted_probs[candidate])
            if confidence > best_confidence:
                best_digit, best_confidence, best_shift = candidate, confidence, dx
    return str(best_digit), best_confidence, best_shift


def filter_noise(characters, boxes):
    """Digit-only heuristic: discard boxes below 10% of largest box area."""
    if not characters:
        return [], []
    largest = max(c.shape[0] * c.shape[1] for c in characters)
    kept = [(c, b) for c, b in zip(characters, boxes)
            if c.shape[0] * c.shape[1] >= largest * 0.10]
    return [c for c, _ in kept], [b for _, b in kept]


def _binary_crop_v45(character):
    """Tight white-on-black binary crop used by the V4.5 geometric rules."""
    gray = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY) if character.ndim == 3 else character.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    return binary[y:y + h, x:x + w]

def radial_star_candidate(character):
    """Experimental single-piece radial asterisk detector, not a trained classifier."""
    crop = _binary_crop_v45(character)
    if crop is None:
        return False
    h, w = crop.shape
    if min(h, w) < 35 or not (0.80 <= w / h <= 1.12):
        return False
    n, _, _, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    if n - 1 != 1:
        return False
    mask = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_NEAREST) > 0
    ys, xs = np.where(mask)
    if len(xs) < 150:
        return False
    cx, cy = float(np.median(xs)), float(np.median(ys))
    yy, xx = np.indices(mask.shape)
    radius = np.hypot(xx - cx, yy - cy)
    angles = (np.arctan2(yy - cy, xx - cx) + 2 * np.pi) % (2 * np.pi)
    hub_density = float(np.mean(mask[radius <= 7]))
    occupied = sum(np.count_nonzero(mask & (radius >= 15) & (radius <= 29)
                                   & (angles >= i * np.pi / 6)
                                   & (angles < (i + 1) * np.pi / 6)) >= 15
                   for i in range(12))
    return hub_density >= 0.70 and occupied >= 10

def edge_fragment_candidate(index, chars, boxes, flags, parens):
    """Flag substantial but small marks far left and vertically outside text.

    Uses image geometry, never testcase IDs or expected expressions.
    Deliberately avoids ultra-tiny specks pending review of unlabelled images.
    """
    if index != 0 or len(chars) < 3 or flags[0] or parens[0]:
        return False
    first = _binary_crop_v45(chars[0])
    if first is None:
        return False
    x0, y0, w0, h0 = map(int, boxes[0])
    area0 = cv2.countNonZero(first)
    other = [tuple(map(int, b)) for b in boxes[1:]]
    if not other:
        return False
    main_left = min(b[0] for b in other)
    main_top = min(b[1] for b in other)
    main_bottom = max(b[1] + b[3] for b in other)
    heights = sorted(b[3] for b in other)
    typical_height = heights[len(heights) // 2]
    gap = main_left - (x0 + w0)
    vertical_separation = max(main_top - (y0 + h0), y0 - main_bottom, 0)
    return (
        80 <= area0 <= 0.30 * max(cv2.countNonZero(_binary_crop_v45(c))
                                    for c in chars[1:] if _binary_crop_v45(c) is not None)
        and 15 <= h0 <= 0.55 * typical_height
        and w0 <= 0.55 * typical_height
        and x0 <= 0.05 * max(b[0] + b[2] for b in boxes)
        and gap >= 2.0 * typical_height
        and vertical_separation >= 0.75 * typical_height
    )


def plus_geometry_v46(character):
    """Cross-shaped plus candidate, used only with an adjacent parenthesis candidate."""
    crop = _binary_crop_v45(character)
    if crop is None:
        return False
    mask = (crop > 0).astype(np.uint8)
    h, w = mask.shape
    if min(h, w) < 25 or not 0.65 <= w / h <= 1.4:
        return False
    midrows = mask[int(h * .30):max(int(h * .70), int(h * .30) + 1)]
    midcols = mask[:, int(w * .30):max(int(w * .70), int(w * .30) + 1)]
    horiz = max((np.count_nonzero(row) / w for row in midrows), default=0)
    vert = max((np.count_nonzero(col) / h for col in midcols.T), default=0)
    return horiz >= .60 and vert >= .58


def open_paren_geometry_v46(character):
    """Recognise a left-bulging opening parenthesis in a tightly gated '+(' pair."""
    crop = _binary_crop_v45(character)
    if crop is None:
        return False
    mask = (crop > 0).astype(np.uint8)
    h, w = mask.shape
    if h < 45 or w < 20 or not 0.35 <= w / h <= 1.05:
        return False
    centers = []
    for lo, hi in ((.10, .25), (.42, .58), (.75, .90)):
        band = mask[int(h * lo):max(int(h * hi), int(h * lo) + 1)]
        _, xs = np.nonzero(band)
        if len(xs) < 10:
            return False
        centers.append(float(np.median(xs)) / w)
    top, middle, bottom = centers
    bend = middle - (top + bottom) / 2
    return bend <= -.12 and top - middle >= .18 and abs(top - bottom) <= .38


def apply_plus_paren_v46(predictions, details, source_indices, characters,
                         split_flags, paren_flags):
    """Correct adjacent CNN '4','6' only when both V4.6 geometric rules pass."""
    for j in range(len(predictions) - 1):
        i, k = source_indices[j], source_indices[j + 1]
        if k != i + 1 or split_flags[i] or split_flags[k] or paren_flags[i] or paren_flags[k]:
            continue
        if predictions[j:j + 2] != ['4', '6']:
            continue
        if plus_geometry_v46(characters[i]) and open_paren_geometry_v46(characters[k]):
            predictions[j:j + 2] = ['+', '(']
            details[j] = f"{i + 1:02d}   +     paired plus rule (V4.6)"
            details[j + 1] = f"{k + 1:02d}   (     paired opening parenthesis (V4.6)"
    return predictions, details


class App:
    def __init__(self, root):
        self.root = root
        root.title("HNRS Studio | Handwriting Recognition")
        root.geometry("1120x790")
        root.minsize(900, 680)
        root.configure(bg=BG)

        self.model_name = tk.StringVar(value="CNN")
        self.mode = tk.StringVar(value="Calculator")
        self.number = tk.StringVar(value="12345")
        self.models = {}
        self.image = None
        self.photo = None
        self.image_path = ""
        self.result = tk.StringVar(value="Ready to recognise")
        self.answer = tk.StringVar(value="Load an image to begin")
        self.status = tk.StringVar(value="Ready")

        self._setup_styles()
        self._build()

    def _setup_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("HNRS.TCombobox", fieldbackground=PANEL2, background=PANEL2,
                        foreground=TEXT, bordercolor=BORDER, lightcolor=BORDER,
                        darkcolor=BORDER, arrowcolor=TEXT, padding=7)
        style.map("HNRS.TCombobox", fieldbackground=[("readonly", PANEL2)],
                  foreground=[("readonly", TEXT)], selectbackground=[("readonly", PANEL2)],
                  selectforeground=[("readonly", TEXT)])
        self.root.option_add("*TCombobox*Listbox.background", PANEL2)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)

    def _label(self, parent, text, size=11, color=TEXT, weight="normal", **kwargs):
        return tk.Label(parent, text=text, font=("Segoe UI", size, weight),
                        fg=color, bg=parent.cget("bg"), **kwargs)

    def _button(self, parent, text, command, primary=False):
        color = ACCENT if primary else PANEL2
        btn = tk.Button(parent, text=text, command=command, bg=color, fg="white",
                        activebackground="#91A3FF" if primary else "#334467",
                        activeforeground="white", relief="flat", bd=0,
                        font=("Segoe UI", 10, "bold"), padx=18, pady=11, cursor="hand2")
        return btn

    def _card(self, parent):
        return tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)

    def _build(self):
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=28, pady=(22, 14))
        self._label(header, "✦  HNRS STUDIO", 23, TEXT, "bold").pack(side="left")
        self._label(header, "Handwritten Number & Expression Recognition", 11, MUTED).pack(
            side="left", padx=20, pady=(10, 0))

        toolbar = self._card(self.root)
        toolbar.pack(fill="x", padx=28, pady=(0, 15))
        controls = tk.Frame(toolbar, bg=PANEL)
        controls.pack(fill="x", padx=16, pady=13)
        self._label(controls, "MODE", 9, MUTED, "bold").pack(side="left", padx=(0, 8))
        ttk.Combobox(controls, textvariable=self.mode, values=["Calculator", "Number"],
                     state="readonly", width=13, style="HNRS.TCombobox").pack(side="left", padx=(0, 22))
        self._label(controls, "MODEL", 9, MUTED, "bold").pack(side="left", padx=(0, 8))
        ttk.Combobox(controls, textvariable=self.model_name, values=["CNN", "MLP"],
                     state="readonly", width=8, style="HNRS.TCombobox").pack(side="left", padx=(0, 18))
        self._button(controls, "Load image", self.load_image).pack(side="left", padx=(0, 9))
        self._button(controls, "Recognise  →", self.recognise, True).pack(side="left")
        self._button(controls, "Clear", self.clear).pack(side="right")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=28)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        left = self._card(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self._label(left, "IMAGE PREVIEW", 10, MUTED, "bold").pack(anchor="w", padx=20, pady=(18, 8))
        self.preview_holder = tk.Frame(left, bg="#0F1729")
        self.preview_holder.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        self.preview = tk.Label(self.preview_holder, text="✎\n\nLoad a handwritten image\nto see it here",
                                bg="#0F1729", fg=MUTED, font=("Segoe UI", 15), justify="center")
        self.preview.pack(fill="both", expand=True, padx=10, pady=10)
        self.filename = self._label(left, "No image selected", 10, MUTED, anchor="w")
        self.filename.pack(fill="x", padx=20, pady=(0, 16))
        self.preview_holder.bind("<Configure>", self._refresh_preview)

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        result_card = self._card(right)
        result_card.pack(fill="x", pady=(0, 12))
        self._label(result_card, "RECOGNISED INPUT", 10, MUTED, "bold").pack(
            anchor="w", padx=20, pady=(20, 8))
        self._label(result_card, "", 22).destroy()  # keep layout helper simple
        tk.Label(result_card, textvariable=self.result, font=("Segoe UI", 24, "bold"),
                 fg=TEXT, bg=PANEL, anchor="w", wraplength=380, justify="left").pack(
            fill="x", padx=20, pady=(0, 14))
        self._label(result_card, "CALCULATED ANSWER", 10, MUTED, "bold").pack(
            anchor="w", padx=20, pady=(5, 8))
        tk.Label(result_card, textvariable=self.answer, font=("Segoe UI", 20, "bold"),
                 fg=GREEN, bg=PANEL, anchor="w", wraplength=380, justify="left").pack(
            fill="x", padx=20, pady=(0, 20))

        details = self._card(right)
        details.pack(fill="both", expand=True)
        self._label(details, "RECOGNITION DETAILS", 10, MUTED, "bold").pack(
            anchor="w", padx=20, pady=(17, 10))
        self.log = tk.Text(details, height=10, bg=PANEL, fg=TEXT, insertbackground=TEXT,
                           relief="flat", bd=0, font=("Consolas", 10), wrap="word",
                           highlightthickness=0, state="disabled")
        self.log.pack(fill="both", expand=True, padx=20, pady=(0, 15))

        bottom = self._card(self.root)
        bottom.pack(fill="x", padx=28, pady=(15, 8))
        compose = tk.Frame(bottom, bg=PANEL)
        compose.pack(fill="x", padx=16, pady=12)
        self._label(compose, "COMPOSE DIGITS", 10, MUTED, "bold").pack(side="left", padx=(0, 12))
        entry = tk.Entry(compose, textvariable=self.number, bg=PANEL2, fg=TEXT,
                         insertbackground=TEXT, relief="flat", font=("Segoe UI", 12), width=15)
        entry.pack(side="left", ipady=9, padx=(0, 12))
        self._button(compose, "Choose digit folder", self.compose).pack(side="left")
        self._label(compose, "Requires 0.png to 9.png", 9, MUTED).pack(side="right")

        tk.Label(self.root, textvariable=self.status, bg=BG, fg=MUTED,
                 font=("Segoe UI", 9), anchor="w").pack(fill="x", padx=30, pady=(0, 10))
        self.write("Ready. Choose Number or Calculator mode, then load an image.")

    def write(self, message):
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear(self):
        self.image = None
        self.photo = None
        self.image_path = ""
        self.preview.configure(image="", text="✎\n\nLoad a handwritten image\nto see it here")
        self.filename.configure(text="No image selected")
        self.result.set("Ready to recognise")
        self.answer.set("Load an image to begin")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.status.set("Ready")

    def _refresh_preview(self, event=None):
        if self.image is None:
            return
        width = max(200, self.preview_holder.winfo_width() - 35)
        height = max(180, self.preview_holder.winfo_height() - 35)
        preview = self.image.copy()
        preview.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(preview)
        self.preview.configure(image=self.photo, text="")

    def show_image(self, image, name="Composed image"):
        self.image = image.convert("RGB")
        self.filename.configure(text=name if len(name) < 65 else "..." + name[-62:])
        self.result.set("Image ready")
        self.answer.set("Click Recognise")
        self._refresh_preview()
        self.status.set("Image loaded")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not path:
            return
        try:
            with Image.open(path) as img:
                self.show_image(img.copy(), Path(path).name)
            self.image_path = path
            self.write(f"Loaded: {path}")
        except Exception as exc:
            messagebox.showerror("Cannot load image", str(exc))

    def compose(self):
        digits = self.number.get().strip()
        if not digits or not digits.isdigit():
            messagebox.showwarning("Invalid number", "Enter digits only, such as 12345.")
            return
        folder = filedialog.askdirectory(title="Select folder containing 0.png to 9.png")
        if not folder:
            return
        try:
            self.show_image(generate_number_image(folder, digits), f"Composed: {digits}")
            self.write(f"Composed digits: {digits}")
        except Exception as exc:
            messagebox.showerror("Composition error", str(exc))

    def _get_model(self):
        kind = self.model_name.get()
        path = PROJECT_DIR / "models" / ("best_cnn_model.keras" if kind == "CNN" else "mlp.keras")
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        if kind not in self.models:
            self.status.set(f"Loading {kind} model...")
            self.root.update_idletasks()
            self.models[kind] = load_model(str(path))
        return self.models[kind]

    def recognise(self):
        if self.image is None:
            messagebox.showinfo("Image required", "Please load or compose an image first.")
            return
        self.status.set("Recognising handwriting...")
        self.root.update_idletasks()
        try:
            model = self._get_model()
            image_bgr = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2BGR)
            characters, boxes = segment_digits(image_bgr, method="improved", group_division=True)
            original_count = len(characters)
            mode = self.mode.get()
            if mode == "Calculator":
                # Group BEFORE filtering; small division dots must be retained.
                characters, boxes = merge_division_parts(characters, boxes)
                # V4.3: isolate duplicate-covered parentheses BEFORE digit splitting.
                characters, boxes, original_parens = isolate_parentheses(
                    characters, boxes, return_flags=True
                )
                # Only successfully split digit pieces bypass operator rules.
                paren_by_box = {
                    tuple(int(v) for v in box): flag
                    for box, flag in zip(boxes, original_parens) if flag
                }
                characters, boxes, split_flags = split_connected_digits(
                    characters, boxes, return_split_flags=True
                )
                paren_flags = [
                    None if from_split else paren_by_box.get(tuple(int(v) for v in box))
                    for box, from_split in zip(boxes, split_flags)
                ]
            else:
                characters, boxes = filter_noise(characters, boxes)
                split_flags = [False] * len(characters)
                paren_flags = [None] * len(characters)
            if not characters:
                raise ValueError("No handwriting components detected.")

            debug_dir = PROJECT_DIR / "test_images" / "gui_segments"
            debug_dir.mkdir(parents=True, exist_ok=True)
            for old in debug_dir.glob("component_*.png"):
                old.unlink()
            for index, character in enumerate(characters):
                cv2.imwrite(str(debug_dir / f"component_{index}.png"), character)

            # V4.5: preserve all debug crops, but omit a confirmed edge artifact
            # from calculator/CNN recognition. The rule uses geometry only.
            removed_v45 = set()
            if mode == "Calculator" and self.model_name.get() == "CNN":
                if edge_fragment_candidate(0, characters, boxes, split_flags, paren_flags):
                    removed_v45.add(0)

            predictions = []
            details = []
            source_indices = []
            for index, (character, from_split, paren_flag) in enumerate(
                zip(characters, split_flags, paren_flags), 1
            ):
                if index - 1 in removed_v45:
                    continue
                source_indices.append(index - 1)
                # Parenthesis flags take priority; they were checked for duplicate digit crops.
                if mode == "Calculator" and paren_flag:
                    predictions.append(paren_flag)
                    details.append(f"{index:02d}   {paren_flag:<3}   isolated parenthesis (V4.3)")
                    continue
                # V4.4: detect dot/bar/dot division with tiny disconnected fragments.
                # Keep the same precedence as the validated V4.4 comparison script.
                if mode == "Calculator" and not from_split and recognise_division_v44(character):
                    predictions.append("/")
                    details.append(f"{index:02d}   /     division rule (V4.4)")
                    continue
                # V4.3 star rule takes priority over plus recognition, as in the experiment.
                if mode == "Calculator" and not from_split and recognise_asterisk_v43(character):
                    predictions.append("*")
                    details.append(f"{index:02d}   *     asterisk rule (V4.3)")
                    continue
                symbol = recognise_operator(character) if mode == "Calculator" and not from_split else None
                if symbol is not None:
                    predictions.append(symbol)
                    details.append(f"{index:02d}   {symbol:<3}   rule-based operator")
                    continue
                processed = preprocess_digit(character)
                if processed is None:
                    raise ValueError(f"Character {index} has no usable pixels.")
                if mode == "Calculator" and self.model_name.get() == "CNN":
                    digit, confidence, shift = predict_digit_adaptive(processed, model)
                else:
                    # Preserve original behavior for MLP and Number mode.
                    probabilities = model.predict(processed.reshape(1, 28, 28, 1), verbose=0)[0]
                    digit = str(int(np.argmax(probabilities)))
                    confidence = float(np.max(probabilities))
                    shift = 0
                # V4.5: only reinterpret CNN-predicted '4' as a radial star.
                # This avoids the V4.5 regressions where '3'/'8' became '*'.
                if (mode == "Calculator" and self.model_name.get() == "CNN"
                        and not from_split and digit == "4"
                        and radial_star_candidate(character)):
                    predictions.append("*")
                    details.append(f"{index:02d}   *     radial asterisk (V4.5; CNN predicted 4)")
                    continue
                predictions.append(digit)
                origin = " (split digit)" if from_split else ""
                shift_note = f" [shift {shift:+d}]" if shift else ""
                details.append(f"{index:02d}   {digit:<3}   {confidence:.1%} digit confidence{origin}{shift_note}")

            if mode == "Calculator" and self.model_name.get() == "CNN":
                predictions, details = apply_plus_paren_v46(
                    predictions, details, source_indices, characters,
                    split_flags, paren_flags)
            for removed_index in sorted(removed_v45):
                details.insert(removed_index, f"{removed_index + 1:02d}   -     isolated edge artifact removed (V4.5)")

            expression = "".join(predictions)
            self.result.set(expression)
            if mode == "Calculator":
                answer = calculate_expression(expression)
                self.answer.set(str(answer))
            else:
                self.answer.set("Number recognition mode")

            self.log.configure(state="normal")
            self.log.delete("1.0", "end")
            self.log.configure(state="disabled")
            self.write(f"Mode: {mode} | Model: {self.model_name.get()}")
            self.write(f"Segments: {original_count} → {len(characters)}")
            self.write("\n".join(details))
            self.write(f"\nDebug crops: {debug_dir}")
            if mode == "Calculator":
                self.write("Note: a valid calculation does not prove correct recognition.")
            self.status.set(f"Completed: {len(characters) - len(removed_v45)} characters processed")
        except Exception as exc:
            self.status.set("Recognition failed")
            messagebox.showerror("Recognition error", str(exc))
            self.write(f"ERROR: {exc}")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
