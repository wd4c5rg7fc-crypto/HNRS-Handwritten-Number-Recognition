"""Member 1 image acquisition interface."""
from pathlib import Path
import numpy as np
from PIL import Image
from hnrs.core import compose_number

def load_image(path):
    """Return RGB uint8 numpy array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))

def generate_number_image(folder, digits, spacing=8):
    """Return PIL grayscale image from digit files 0.png ... 9.png."""
    return compose_number(Path(folder), str(digits), spacing=spacing)
