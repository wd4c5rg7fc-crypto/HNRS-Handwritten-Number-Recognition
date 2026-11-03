"""Adapter: Member 1 preprocessing -> Member 2 segmentation -> GUI contract."""
import numpy as np
from hnrs.preprocessing import preprocess_image
from hnrs.member2_segmentation import segment_digits as member2_segment_digits

def segment_digits(image, method="hybrid"):
    """Return (digit crops, bounding boxes) sorted left-to-right."""
    if image is None:
        raise ValueError("No input image")
    arr = np.asarray(image)
    # GUI supplies RGB PIL arrays; member 1 expects BGR OpenCV arrays.
    if arr.ndim == 3 and arr.shape[2] >= 3:
        import cv2
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    binary = preprocess_image(arr)
    return member2_segment_digits(binary, method=method, group_division=False)
