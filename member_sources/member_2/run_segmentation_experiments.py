import cv2
import numpy as np
from pathlib import Path

from Segmentation import (
    segment_digits,
    resize_digit,
    ensure_binary
)


INPUT_FOLDER = Path("testimages")
OUTPUT_FOLDER = Path("results")

TESTCASE_NUMBER = None
METHOD = "improved"
GROUP_DIVISION = True
REFERENCE_WIDTH = None
SHOW_DEBUG = True
DISPLAY_SECONDS = 0
BOUNDING_BOX_SCALE = 3
SEGMENT_PREVIEW_SCALE = 4
SAVE_DEBUG_IMAGES = False
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def show_segmentation_boxes(image, boxes):
    """
    Draw bounding boxes around detected characters.

    Green rectangles indicate detected regions.
    Red numbers indicate character order.
    """

    if image.ndim == 2:
        display = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )
    else:
        display = image.copy()

    for index, (x, y, w, h) in enumerate(
        boxes,
        start=1
    ):

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            1
        )

        cv2.putText(
            display,
            str(index),
            (x, max(10, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1
        )

    # Enlarge only the preview.
    # Original image and coordinates are unchanged.
    if BOUNDING_BOX_SCALE > 1:

        display = cv2.resize(
            display,
            None,
            fx=BOUNDING_BOX_SCALE,
            fy=BOUNDING_BOX_SCALE,
            interpolation=cv2.INTER_NEAREST
        )

    return display




def create_segment_preview(digit_images):
    """
    Create a visual preview of all segmented characters.

    Each character is resized to 28x28, enlarged,
    and displayed in left-to-right order.
    """

    if not digit_images:
        return None

    scale = SEGMENT_PREVIEW_SCALE

    cell_size = 28 * scale
    padding = 15
    label_height = 35

    count = len(digit_images)

    preview_width = (
        count * cell_size +
        (count + 1) * padding
    )

    preview_height = (
        cell_size +
        2 * padding +
        label_height
    )

    preview = np.zeros(
        (preview_height, preview_width, 3),
        dtype=np.uint8
    )

    for index, digit in enumerate(digit_images):

        digit_28 = resize_digit(
            digit,
            size=(28, 28)
        )

        enlarged = cv2.resize(
            digit_28,
            (cell_size, cell_size),
            interpolation=cv2.INTER_NEAREST
        )

        x = padding + index * (cell_size + padding)
        y = padding

        preview[
            y:y + cell_size,
            x:x + cell_size
        ] = cv2.cvtColor(
            enlarged,
            cv2.COLOR_GRAY2BGR
        )

        cv2.putText(
            preview,
            f"{index + 1}",
            (x, y + cell_size + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1
        )

    return preview




def display_debug_windows(
    image_name,
    bounding_box_image,
    segment_preview
):


    if not SHOW_DEBUG:
        return

    box_window = f"Bounding Boxes - {image_name}"
    preview_window = f"Segment Preview - {image_name}"

    cv2.namedWindow(
        box_window,
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        box_window,
        bounding_box_image
    )

    if segment_preview is not None:

        cv2.namedWindow(
            preview_window,
            cv2.WINDOW_NORMAL
        )

        cv2.imshow(
            preview_window,
            segment_preview
        )

    if DISPLAY_SECONDS == 0:

        print(
            "\nClick an OpenCV window and press "
            "any key to continue."
        )

        cv2.waitKey(0)

    else:

        print(
            f"\nDisplaying {image_name} for "
            f"{DISPLAY_SECONDS} seconds..."
        )

        cv2.waitKey(
            int(DISPLAY_SECONDS * 1000)
        )

    cv2.destroyAllWindows()





def process_image(image_path):
   

    print("\n" + "=" * 60)
    print(f"PROCESSING: {image_path.name}")
    print("=" * 60)

   

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"ERROR: Cannot read {image_path.name}"
        )

        return False

   

    processed = ensure_binary(image)

   

    digit_images, bounding_boxes = segment_digits(
        processed,
        method=METHOD,
        reference_width=REFERENCE_WIDTH,
        group_division=GROUP_DIVISION
    )

    print(
        f"Detected characters: {len(digit_images)}"
    )

    print("\nBounding boxes:")

    for index, box in enumerate(
        bounding_boxes,
        start=1
    ):

        print(
            f"Character {index}: {box}"
        )

   

    testcase_folder = (
        OUTPUT_FOLDER /
        image_path.stem
    )

    testcase_folder.mkdir(
        parents=True,
        exist_ok=True
    )

   

    for old_file in testcase_folder.glob(
        "segment_*.png"
    ):

        old_file.unlink()


    for index, digit in enumerate(
        digit_images,
        start=1
    ):

        digit_28 = resize_digit(
            digit,
            size=(28, 28)
        )

        output_path = (
            testcase_folder /
            f"segment_{index:02d}.png"
        )

        success = cv2.imwrite(
            str(output_path),
            digit_28
        )

        if success:

            print(
                f"Saved: {output_path}"
            )

        else:

            print(
                f"ERROR saving: {output_path}"
            )

   

    if SHOW_DEBUG or SAVE_DEBUG_IMAGES:

        bounding_box_image = show_segmentation_boxes(
            processed,
            bounding_boxes
        )

        segment_preview = create_segment_preview(
            digit_images
        )

        # Save debug images only if requested.
        if SAVE_DEBUG_IMAGES:

            cv2.imwrite(
                str(
                    testcase_folder /
                    "bounding_boxes.png"
                ),
                bounding_box_image
            )

            if segment_preview is not None:

                cv2.imwrite(
                    str(
                        testcase_folder /
                        "segment_preview.png"
                    ),
                    segment_preview
                )

        # Display the debug windows.
        display_debug_windows(
            image_path.name,
            bounding_box_image,
            segment_preview
        )

    print(
        f"\nCOMPLETED: {image_path.name}"
    )

    return True



def get_test_images():
   

    if not INPUT_FOLDER.exists():

        print(
            "ERROR: testimages folder does not exist."
        )

        return []

    image_files = sorted(
        [
            file
            for file in INPUT_FOLDER.iterdir()
            if (
                file.is_file() and
                file.suffix.lower() in EXTENSIONS
            )
        ],
        key=lambda file: file.name
    )

    if TESTCASE_NUMBER is not None:

        testcase_name = (
            f"testcase{TESTCASE_NUMBER}"
        )

        image_files = [
            file
            for file in image_files
            if (
                file.stem.lower()
                == testcase_name.lower()
            )
        ]

        if not image_files:

            print(
                f"ERROR: {testcase_name} not found."
            )

            return []

    return image_files




def main():

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = get_test_images()

    if not image_files:

        print(
            "No valid test images found."
        )

        return

    print("\n" + "=" * 60)

    print("=" * 60)

    print(f"Segmentation method: {METHOD}")
    print(f"Division grouping: {GROUP_DIVISION}")
    print(f"Selected test case: {TESTCASE_NUMBER}")
    print(f"Debug windows: {SHOW_DEBUG}")
    print(f"Display seconds: {DISPLAY_SECONDS}")
    print(f"Images found: {len(image_files)}")

    successful = 0
    failed = 0

    for image_path in image_files:

        try:

            result = process_image(image_path)

            if result:
                successful += 1
            else:
                failed += 1

        except Exception as error:

            failed += 1

            print(
                f"\nERROR processing "
                f"{image_path.name}: {error}"
            )

            # Continue with the remaining test cases.
            cv2.destroyAllWindows()
            continue

    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("SEGMENTATION EXPERIMENTS COMPLETED")
    print("=" * 60)

    print(f"Total images: {len(image_files)}")
    print(f"Processed successfully: {successful}")
    print(f"Processing errors: {failed}")

    print(
        f"\nResults saved in: "
        f"{OUTPUT_FOLDER.resolve()}"
    )



if __name__ == "__main__":
    main()
