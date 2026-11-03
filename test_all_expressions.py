import sys
import csv
import re
from pathlib import Path

import cv2
import numpy as np

from tensorflow.keras.models import load_model

from operator_recognition import recognise_operator
from division_grouping import merge_division_parts


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

IMAGE_FOLDER = (
    MEMBER_2_DIR / "testimages"
)

OUTPUT_FOLDER = (
    PROJECT_DIR / "test_results"
)

CSV_PATH = (
    OUTPUT_FOLDER / "expression_test_results.csv"
)

DEBUG_TESTCASE = "testcase7"

DEBUG_FOLDER = (
    OUTPUT_FOLDER / "debug_testcase7"
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
    Prepare a segmented digit for the MNIST CNN.

    Returns:
        28x28 float32 image with values from 0 to 1.
    """

    # Convert to grayscale.
    if len(character.shape) == 3:
        gray = cv2.cvtColor(
            character,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = character.copy()

    # Convert to binary.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Find digit boundaries.
    points = cv2.findNonZero(binary)

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)

    if w == 0 or h == 0:
        return None

    crop = binary[
        y:y + h,
        x:x + w
    ]

    # Resize while preserving aspect ratio.
    scale = 20 / max(w, h)

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Place digit on a 28x28 black canvas.
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

    # Centre using image moments.
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

    return canvas.astype(np.float32) / 255.0


# ==================================================
# RECOGNISE CHARACTER
# ==================================================

def recognise_character(character, model):
    """
    Recognise an operator using rules.

    If no operator is found, classify the
    character using the trained digit CNN.
    """

    # Try operator recognition first.
    detected_operator = recognise_operator(character)

    if detected_operator is not None:
        return detected_operator, "rule-based"

    # Otherwise classify as a digit.
    processed = preprocess_digit(character)

    if processed is None:
        return None, "empty image"

    model_input = processed.reshape(
        1, 28, 28, 1
    )

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

    return (
        str(predicted_index),
        f"{confidence * 100:.2f}%"
    )


# ==================================================
# SAVE DEBUG CHARACTERS
# ==================================================

def save_debug_characters(image_path, characters, boxes):
    """
    Save the segmented characters after
    division grouping.

    Used to inspect testcase7.
    """

    if image_path.stem != DEBUG_TESTCASE:
        return

    DEBUG_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # Remove old debug character images so that
    # previous results do not cause confusion.
    for old_file in DEBUG_FOLDER.glob("character_*.png"):
        old_file.unlink()

    print("\n  --- SEGMENTATION DEBUG ---")

    print(
        f"  Characters after grouping: {len(characters)}"
    )

    # Save each character.
    for index, character in enumerate(
        characters,
        start=1
    ):

        output_path = (
            DEBUG_FOLDER / f"character_{index:02d}.png"
        )

        cv2.imwrite(
            str(output_path),
            character
        )

        print(
            f"  Saved: {output_path.name}"
        )

    # Save bounding boxes.
    boxes_path = (
        DEBUG_FOLDER / "bounding_boxes.txt"
    )

    with open(
        boxes_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Image: {image_path.name}\n"
        )

        file.write(
            f"Segmented characters: {len(characters)}\n\n"
        )

        for index, box in enumerate(
            boxes,
            start=1
        ):

            file.write(
                f"Character {index}: {box}\n"
            )

    print("  --- END SEGMENTATION DEBUG ---\n")


# ==================================================
# PROCESS ONE EXPRESSION
# ==================================================

def process_expression(image_path, model):
    """
    Segment, recognise and evaluate one
    handwritten mathematical expression.
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        return {
            "filename": image_path.name,
            "characters": 0,
            "expression": "",
            "answer": "",
            "status": "IMAGE ERROR",
            "details": ""
        }

    # ==================================================
    # STEP 1: SEGMENT CHARACTERS
    # ==================================================

    characters, boxes = segment_digits(
        image,
        method="improved",
        group_division=True
    )

    original_count = len(characters)

    # ==================================================
    # STEP 2: MERGE SPLIT DIVISION SYMBOLS
    # ==================================================

    characters, boxes = merge_division_parts(
        characters,
        boxes
    )

    print(
        f"  Segmentation: {original_count} "
        f"-> {len(characters)} characters"
    )

    # Save debug images after grouping.
    save_debug_characters(
        image_path,
        characters,
        boxes
    )

    if not characters:
        return {
            "filename": image_path.name,
            "characters": 0,
            "expression": "",
            "answer": "",
            "status": "NO CHARACTERS",
            "details": ""
        }

    # ==================================================
    # STEP 3: RECOGNISE CHARACTERS
    # ==================================================

    recognised = []
    character_details = []

    for index, character in enumerate(
        characters,
        start=1
    ):

        prediction, confidence = recognise_character(
            character,
            model
        )

        if prediction is None:
            return {
                "filename": image_path.name,
                "characters": len(characters),
                "expression": "".join(recognised),
                "answer": "",
                "status": "RECOGNITION ERROR",
                "details": "; ".join(character_details)
            }

        recognised.append(prediction)

        character_details.append(
            f"{index}:{prediction}({confidence})"
        )

        # Print individual predictions for testcase7.
        if image_path.stem == DEBUG_TESTCASE:

            print(
                f"  Character {index}: "
                f"{prediction} ({confidence})"
            )

    # ==================================================
    # STEP 4: CONSTRUCT EXPRESSION
    # ==================================================

    expression = "".join(recognised)

    # ==================================================
    # STEP 5: CALCULATE ANSWER
    # ==================================================

    try:

        answer = calculate_expression(
            expression
        )

        if (
            isinstance(answer, str)
            and answer.startswith("Error:")
        ):
            status = "INVALID EXPRESSION"

        else:
            status = "EVALUATED"

    except Exception as error:

        answer = str(error)
        status = "EVALUATION ERROR"

    # ==================================================
    # STEP 6: RETURN RESULT
    # ==================================================

    return {
        "filename": image_path.name,
        "characters": len(characters),
        "expression": expression,
        "answer": answer,
        "status": status,
        "details": "; ".join(character_details)
    }


# ==================================================
# NATURAL FILE SORTING
# ==================================================

def natural_sort_key(path):
    """
    Sort testcase2 before testcase10.
    """

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(
            r"(\d+)",
            path.name
        )
    ]


# ==================================================
# MAIN BATCH TEST
# ==================================================

def main():

    print("\n========================================")
    print("HNRS HANDWRITTEN EXPRESSION BATCH TEST")
    print("========================================")

    # Check required files.
    if not MODEL_PATH.exists():
        print(
            f"ERROR: CNN model not found:\n{MODEL_PATH}"
        )
        return

    if not IMAGE_FOLDER.exists():
        print(
            f"ERROR: Test image folder not found:\n{IMAGE_FOLDER}"
        )
        return

    # Find test images.
    image_files = sorted(
        [
            path
            for path in IMAGE_FOLDER.iterdir()
            if path.suffix.lower() in {
                ".png", ".jpg", ".jpeg"
            }
        ],
        key=natural_sort_key
    )

    if not image_files:
        print("No test images found.")
        return

    print(
        f"\nFound {len(image_files)} test images."
    )

    # Load trained CNN once.
    print("\nLoading trained CNN model...")

    model = load_model(
        str(MODEL_PATH)
    )

    # Create output directory.
    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    # ==================================================
    # PROCESS ALL TEST IMAGES
    # ==================================================

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"\n[{index}/{len(image_files)}] "
            f"Processing {image_path.name}"
        )

        try:

            result = process_expression(
                image_path,
                model
            )

        except Exception as error:

            result = {
                "filename": image_path.name,
                "characters": 0,
                "expression": "",
                "answer": "",
                "status": "PROCESSING ERROR",
                "details": str(error)
            }

        results.append(result)

        print(
            f"  Characters: {result['characters']}"
        )

        print(
            f"  Expression: {result['expression']}"
        )

        print(
            f"  Answer: {result['answer']}"
        )

        print(
            f"  Status: {result['status']}"
        )

    # ==================================================
    # SAVE CSV RESULTS
    # ==================================================

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "characters",
                "expression",
                "answer",
                "status",
                "details"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    # ==================================================
    # DISPLAY SUMMARY
    # ==================================================

    evaluated = sum(
        result["status"] == "EVALUATED"
        for result in results
    )

    invalid = sum(
        result["status"] == "INVALID EXPRESSION"
        for result in results
    )

    errors = (
        len(results) - evaluated - invalid
    )

    print("\n========================================")
    print("BATCH TEST SUMMARY")
    print("========================================")

    print(
        f"Total images: {len(results)}"
    )

    print(
        f"Expressions evaluated: {evaluated}"
    )

    print(
        f"Invalid expressions: {invalid}"
    )

    print(
        f"Other errors: {errors}"
    )

    print(
        f"\nResults saved to:\n{CSV_PATH}"
    )

    print(
        f"\nDebug images saved to:\n{DEBUG_FOLDER}"
    )

    print("\nIMPORTANT:")
    print(
        "EVALUATED means the recognised expression "
        "was mathematically valid."
    )

    print(
        "It does not necessarily mean the handwritten "
        "expression was recognised correctly."
    )

    print("========================================")


# ==================================================
# RUN PROGRAM
# ==================================================

if __name__ == "__main__":
    main()