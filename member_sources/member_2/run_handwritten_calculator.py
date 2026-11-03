
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

from Segmentation import (
    ensure_binary,
    segment_digits,
    resize_digit
)

from expression_evaluator import calculate_expression


# ==================================================
# CONFIGURATION
# ==================================================

INPUT_FOLDER = Path("testimages")

MODEL_PATH = Path("math_character_model.keras")

SEGMENTATION_METHOD = "improved"

# IMPORTANT:
# This order must exactly match the class indices
# used when the recognition model was trained.
CLASS_NAMES = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9",
    "+", "-", "*", "/", "(", ")"
]


# ==================================================
# LOAD RECOGNITION MODEL
# ==================================================

def load_recognition_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Recognition model not found: {MODEL_PATH}\n"
            "You need a trained character recognition model "
            "before automatic prediction can work."
        )

    model = load_model(str(MODEL_PATH))

    return model


# ==================================================
# RECOGNISE ONE CHARACTER
# ==================================================

def recognise_character(digit_image, model):

    digit_28 = resize_digit(
        digit_image,
        size=(28, 28)
    )

    normalized = (
        digit_28.astype(np.float32) / 255.0
    )

    # Expected CNN input: (batch, height, width, channels).
    model_input = normalized.reshape(
        1, 28, 28, 1
    )

    predictions = model.predict(
        model_input,
        verbose=0
    )[0]

    if len(predictions) != len(CLASS_NAMES):
        raise ValueError(
            "Model output classes do not match CLASS_NAMES."
        )

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_character = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return predicted_character, confidence


# ==================================================
# PROCESS A HANDWRITTEN IMAGE
# ==================================================

def process_handwritten_expression(image_path, model):

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Cannot open {image_path}")
        return

    # Step 1: Preprocess.
    processed = ensure_binary(image)

    # Step 2: Segment characters.
    digit_images, boxes = segment_digits(
        processed,
        method=SEGMENTATION_METHOD,
        group_division=True
    )

    if not digit_images:
        print("No characters detected.")
        return

    # Step 3: Recognise characters.
    characters = []

    print(f"\nImage: {image_path.name}")
    print(f"Detected characters: {len(digit_images)}")

    for index, digit in enumerate(digit_images, start=1):

        character, confidence = recognise_character(
            digit,
            model
        )

        characters.append(character)

        print(
            f"Character {index}: {character} "
            f"({confidence * 100:.1f}%)"
        )

    # Step 4: Construct expression.
    expression = "".join(characters)

    # Step 5: Calculate answer.
    answer = calculate_expression(expression)

    print(f"Recognised expression: {expression}")
    print(f"Calculated answer: {answer}")

    # Step 6: Show original image.
    display = image.copy()

    cv2.putText(
        display,
        f"Expression: {expression}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Answer: {answer}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.namedWindow(
        "Handwritten Calculator",
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        "Handwritten Calculator",
        display
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==================================================
# MAIN
# ==================================================

def main():

    model = load_recognition_model()

    image_files = sorted(
        [
            file for file in INPUT_FOLDER.iterdir()
            if file.suffix.lower() in {
                ".png", ".jpg", ".jpeg"
            }
        ],
        key=lambda p: p.name
    )

    for image_path in image_files:

        process_handwritten_expression(
            image_path,
            model
        )


if __name__ == "__main__":
    main()
