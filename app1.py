"""HNRS desktop interface. Start with python app.py."""

import sys
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk

from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

from hnrs.image_acquisition import generate_number_image


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

# Import teammate's improved segmentation module
sys.path.insert(
    0,
    str(PROJECT_DIR / "member_sources" / "member_2")
)

from Segmentation import segment_digits


# --------------------------------------------------
# DIGIT PREPROCESSING
# --------------------------------------------------

def preprocess_digit(character):
    """
    Convert a segmented handwritten digit into a
    28x28 grayscale image suitable for MNIST-trained models.
    """

    # Convert to grayscale if necessary
    if len(character.shape) == 3:
        character = cv2.cvtColor(
            character,
            cv2.COLOR_BGR2GRAY
        )

    # Preserve white handwriting on a black background
    _, binary = cv2.threshold(
        character,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find handwriting pixels
    points = cv2.findNonZero(binary)

    if points is None:
        return None

    # Find bounding box around the digit
    x, y, w, h = cv2.boundingRect(points)

    # Crop the digit
    digit = binary[y:y + h, x:x + w]

    # Resize while preserving aspect ratio
    scale = 20 / max(w, h)

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    digit = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Create a black 28x28 canvas
    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    # Centre the resized digit
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = digit

    # Calculate centre of mass
    moments = cv2.moments(canvas)

    if moments["m00"] != 0:

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        # Calculate required shifts
        shift_x = int(round(13.5 - cx))
        shift_y = int(round(13.5 - cy))

        # Create transformation matrix
        matrix = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y]
        ])

        # Shift the digit towards the centre
        canvas = cv2.warpAffine(
            canvas,
            matrix,
            (28, 28),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    # Normalise pixel values to 0-1
    return canvas.astype("float32") / 255.0


# --------------------------------------------------
# NOISE FILTERING
# --------------------------------------------------

def filter_noise(characters, boxes):
    """
    Remove tiny components caused by background noise.

    Components smaller than 10% of the largest
    component's bounding rectangle area are removed.

    This is a heuristic and may need adjustment for
    images containing digits of very different sizes.
    """

    if len(characters) == 0:
        return [], []

    # Calculate the area of each component
    areas = [
        character.shape[0] * character.shape[1]
        for character in characters
    ]

    largest_area = max(areas)

    # Keep components at least 10% as large as the largest
    filtered = [
        (character, box)
        for character, box, area in zip(
            characters,
            boxes,
            areas
        )
        if area >= largest_area * 0.10
    ]

    # Separate filtered characters and boxes
    filtered_characters = [
        character
        for character, box in filtered
    ]

    filtered_boxes = [
        box
        for character, box in filtered
    ]

    return filtered_characters, filtered_boxes


# --------------------------------------------------
# MAIN GUI
# --------------------------------------------------

class App:

    def __init__(self, root):

        self.root = root

        root.title(
            "HNRS | Handwritten Number Recognition"
        )

        root.geometry("850x650")

        # Model selection
        self.model_name = tk.StringVar(value="CNN")

        # Store loaded models
        self.models = {}

        # Store current image
        self.image = None
        self.photo = None

        # Top controls
        top = ttk.Frame(
            root,
            padding=12
        )

        top.pack(fill="x")

        ttk.Label(
            top,
            text="Model:"
        ).pack(side="left")

        ttk.Combobox(
            top,
            textvariable=self.model_name,
            values=["CNN", "MLP"],
            state="readonly",
            width=8
        ).pack(
            side="left",
            padx=8
        )

        ttk.Button(
            top,
            text="Load image",
            command=self.load_image
        ).pack(
            side="left",
            padx=6
        )

        ttk.Button(
            top,
            text="Recognise",
            command=self.recognise
        ).pack(
            side="left",
            padx=6
        )

        # Image preview
        self.preview = ttk.Label(
            root,
            text="Load an image or compose digits below.",
            anchor="center"
        )

        self.preview.pack(
            fill="both",
            expand=True,
            padx=12,
            pady=10
        )

        # Number composition section
        compose = ttk.LabelFrame(
            root,
            text="Compose number from folder containing 0.png to 9.png",
            padding=10
        )

        compose.pack(
            fill="x",
            padx=12
        )

        self.number = tk.StringVar(value="12345")

        ttk.Entry(
            compose,
            textvariable=self.number,
            width=15
        ).pack(
            side="left",
            padx=6
        )

        ttk.Button(
            compose,
            text="Choose folder and compose",
            command=self.compose
        ).pack(side="left")

        # Output area
        self.output = tk.Text(
            root,
            height=8,
            wrap="word"
        )

        self.output.pack(
            fill="x",
            padx=12,
            pady=12
        )

        self.write(
            "HNRS ready. Load an image and click Recognise."
        )

    def write(self, msg):
        """Display a message in the output area."""

        self.output.insert(
            "end",
            msg + "\n"
        )

        self.output.see("end")

    def show_image(self, image):
        """Display the selected image."""

        self.image = image.convert("RGB")

        preview = self.image.copy()
        preview.thumbnail((780, 320))

        self.photo = ImageTk.PhotoImage(preview)

        self.preview.configure(
            image=self.photo,
            text=""
        )

    def load_image(self):
        """Allow the user to select an image."""

        path = filedialog.askopenfilename(
            filetypes=[
                (
                    "Image files",
                    "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"
                )
            ]
        )

        if path:

            try:

                self.show_image(
                    Image.open(path)
                )

                self.write(
                    "Loaded " + path
                )

            except Exception as exc:

                messagebox.showerror(
                    "Cannot load image",
                    str(exc)
                )

    def compose(self):
        """Generate an image from digit files."""

        folder = filedialog.askdirectory(
            title="Select folder containing 0.png ... 9.png"
        )

        if folder:

            try:

                generated_image = generate_number_image(
                    folder,
                    self.number.get()
                )

                self.show_image(
                    generated_image
                )

                self.write(
                    "Composed " + self.number.get()
                )

            except Exception as exc:

                messagebox.showerror(
                    "Composition error",
                    str(exc)
                )

    def recognise(self):
        """Segment and recognise handwritten digits."""

        # Check whether an image has been loaded
        if self.image is None:

            messagebox.showinfo(
                "Image required",
                "Please load or compose an image first."
            )

            return

        # Get selected model
        kind = self.model_name.get()

        # Select trained model
        if kind == "CNN":

            model_path = (
                PROJECT_DIR / "models" / "best_cnn_model.keras"
            )

        else:

            model_path = (
                PROJECT_DIR / "models" / "mlp.keras"
            )

        # Check model file
        if not model_path.exists():

            messagebox.showerror(
                "Model missing",
                f"Model file not found: {model_path}"
            )

            return

        try:

            # Load model only once
            if kind not in self.models:

                self.models[kind] = load_model(
                    model_path
                )

            model = self.models[kind]

            # Convert PIL image to OpenCV BGR format
            image_array = np.array(
                self.image
            )

            image_bgr = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2BGR
            )

            # ----------------------------------------
            # SEGMENT HANDWRITTEN CHARACTERS
            # ----------------------------------------

            characters, boxes = segment_digits(
                image_bgr,
                method="improved",
                group_division=True
            )

            if len(characters) == 0:

                self.write(
                    "No characters detected."
                )

                return

            # Record original number of components
            original_count = len(characters)

            # ----------------------------------------
            # FILTER OUT BACKGROUND NOISE
            # ----------------------------------------

            characters, boxes = filter_noise(
                characters,
                boxes
            )

            removed_count = (
                original_count - len(characters)
            )

            print(
                "Original components:",
                original_count
            )

            print(
                "Components after filtering:",
                len(characters)
            )

            print(
                "Noise components removed:",
                removed_count
            )

            self.write(
                f"Noise components removed: {removed_count}"
            )

            if len(characters) == 0:

                self.write(
                    "No valid characters remain after filtering."
                )

                return

            # ----------------------------------------
            # SAVE FILTERED SEGMENTED COMPONENTS
            # ----------------------------------------

            debug_folder = (
                PROJECT_DIR
                / "test_images"
                / "gui_segments"
            )

            debug_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            # Remove old debug images
            for old_file in debug_folder.glob(
                "component_*.png"
            ):
                old_file.unlink()

            # Save each remaining component
            for i, character in enumerate(characters):

                image_path = (
                    debug_folder
                    / f"component_{i}.png"
                )

                cv2.imwrite(
                    str(image_path),
                    character
                )

            print(
                "GUI detected components:",
                len(characters)
            )

            print(
                "GUI bounding boxes:",
                boxes
            )

            self.write(
                f"Debug images saved to: {debug_folder}"
            )

            # ----------------------------------------
            # PREPROCESS AND RECOGNISE DIGITS
            # ----------------------------------------

            predictions = []

            for i, character in enumerate(characters):

                # Preprocess the segmented character
                processed = preprocess_digit(
                    character
                )

                if processed is None:

                    self.write(
                        f"Component {i}: No valid digit pixels."
                    )

                    continue

                # Prepare input for CNN or MLP
                input_image = processed.reshape(
                    1, 28, 28, 1
                )

                # Predict digit
                probabilities = model.predict(
                    input_image,
                    verbose=0
                )[0]

                predicted_digit = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    np.max(probabilities)
                )

                predictions.append(
                    (
                        predicted_digit,
                        confidence
                    )
                )

                print(
                    f"Component {i}: "
                    f"Predicted {predicted_digit}, "
                    f"Confidence {confidence:.2%}"
                )

            # Check whether predictions were made
            if not predictions:

                self.write(
                    "No valid digits detected."
                )

                return

            # Extract predicted digits
            labels = [
                digit
                for digit, score in predictions
            ]

            confidences = [
                score
                for digit, score in predictions
            ]

            # Combine predictions into a number
            recognised_number = "".join(
                str(x)
                for x in labels
            )

            # Display results
            self.write(
                f"{kind} recognised: {recognised_number}"
            )

            self.write(
                "Digit confidences: "
                + ", ".join(
                    f"{x:.1%}"
                    for x in confidences
                )
            )

            self.write(
                f"Segmented components: {len(boxes)}"
            )

        except Exception as exc:

            messagebox.showerror(
                "Recognition error",
                str(exc)
            )


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    root = tk.Tk()

    App(root)

    root.mainloop()