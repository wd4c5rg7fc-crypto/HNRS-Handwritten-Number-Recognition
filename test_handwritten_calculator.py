import sys
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from operator_recognition import recognise_operator


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

MEMBER_2_DIR = (
    PROJECT_DIR / "member_sources" / "member_2"
)

MODEL_PATH = (
    PROJECT_DIR / "models" / "best_cnn_model.keras"
)

# Test the second handwritten mathematical expression.
IMAGE_PATH = (
    MEMBER_2_DIR / "testimages" / "testcase2.jpeg"
)


# ==================================================
# IMPORT TEAMMATE'S MODULES
# ==================================================

sys.path.insert(0, str(MEMBER_2_DIR))

from Segmentation import segment_digits
from expression_evaluator import calculate_expression


# ==================================================
# PREPROCESS DIGIT FOR CNN
# ==================================================

def preprocess_digit(character):
    """
    Convert a segmented handwritten digit into the
    28x28 grayscale format expected by the MNIST CNN.
    """

    # Step 1: Convert to grayscale.
    if len(character.shape) == 3:
        gray = cv2.cvtColor(
            character,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = character.copy()

    # Step 2: Convert to binary.
    # Segmented characters are white on black.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 3: Find the digit boundaries.
    points = cv2.findNonZero(binary)

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)

    crop = binary[
        y:y + h,
        x:x + w
    ]

    # Step 4: Resize while maintaining aspect ratio.
    scale = 20 / max(w, h)

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Step 5: Place the digit on a 28x28 black canvas.
    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    # Step 6: Centre the digit using image moments.
    moments = cv2.moments(canvas)

    if moments["m00"] != 0:

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        shift_x = int(round(13.5 - cx))
        shift_y = int(round(13.5 - cy))

        matrix = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y]
        ])

        canvas = cv2.warpAffine(
            canvas,
            matrix,
            (28, 28)
        )

    # Step 7: Normalise pixel values to 0-1.
    return canvas.astype(np.float32) / 255.0


# ==================================================
# RECOGNISE A SINGLE CHARACTER
# ==================================================

def recognise_character(character, model):
    """
    First attempt operator recognition.

    If no operator is detected, use the CNN
    to recognise the handwritten digit.
    """

    # Step 1: Check for + or -.
    operator = recognise_operator(character)

    if operator is not None:
        return operator, "rule-based"

    # Step 2: Preprocess for CNN.
    processed = preprocess_digit(character)

    if processed is None:
        return None, "empty image"

    # Step 3: Prepare CNN input.
    model_input = processed.reshape(
        1, 28, 28, 1
    )

    # Step 4: Predict digit.
    probabilities = model.predict(
        model_input,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    predicted_digit = str(predicted_index)

    return predicted_digit, f"{confidence * 100:.2f}%"


# ==================================================
# PROCESS HANDWRITTEN EXPRESSION
# ==================================================

def process_handwritten_expression(image_path, model):
    """
    Load image, segment characters, recognise
    digits/operators, and calculate the result.
    """

    # Step 1: Load the handwritten image.
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"ERROR: Could not load {image_path}")
        return

    print("\n================================")
    print("HANDWRITTEN MATHEMATICAL CALCULATOR")
    print("================================")

    print(f"\nImage: {image_path.name}")

    # Step 2: Segment handwritten characters.
    characters, boxes = segment_digits(
        image,
        method="improved",
        group_division=True
    )

    print(f"Segmented characters: {len(characters)}")

    if not characters:
        print("No characters detected.")
        return

    # Step 3: Recognise each character.
    recognised_characters = []

    print("\n--- CHARACTER RECOGNITION ---")

    for index, character in enumerate(
        characters,
        start=1
    ):

        prediction, confidence = recognise_character(
            character,
            model
        )

        if prediction is None:
            print(
                f"Character {index}: "
                "Could not recognise"
            )
            return

        recognised_characters.append(prediction)

        print(
            f"Character {index}: "
            f"{prediction} ({confidence})"
        )

    # Step 4: Construct the mathematical expression.
    expression = "".join(recognised_characters)

    # Step 5: Calculate the answer.
    answer = calculate_expression(expression)

    # Step 6: Display the final results.
    print("\n================================")
    print(f"Recognised expression: {expression}")
    print(f"Calculated answer: {answer}")
    print("================================")


# ==================================================
# MAIN PROGRAM
# ==================================================

def main():

    # Check that the CNN model exists.
    if not MODEL_PATH.exists():
        print(f"ERROR: CNN model not found: {MODEL_PATH}")
        return

    # Check that the test image exists.
    if not IMAGE_PATH.exists():
        print(f"ERROR: Test image not found: {IMAGE_PATH}")
        return

    # Load the trained CNN.
    print("Loading CNN model...")

    model = load_model(str(MODEL_PATH))

    # Process the handwritten expression.
    process_handwritten_expression(
        IMAGE_PATH,
        model
    )


if __name__ == "__main__":
    main()