# MEMBER 1: Preprocessing and image acquisition
**Own:** `hnrs/preprocessing.py`, `hnrs/image_acquisition.py`, and your own tests/results.

1. Follow `00_READ_FIRST.md` and verify Python and dependencies.
2. Open the two files and run `python -m unittest tests.test_team_interfaces -v`.
3. Understand `preprocess_image`: RGB/grayscale -> grayscale -> polarity -> Gaussian blur -> Otsu binary. Preserve the original image dimensions so Member 2 can segment it.
4. Understand `prepare_digit`: crop, fit into 20x20 preserving aspect ratio, centre on 28x28 canvas, normalize to [0,1]. **Never resize the whole multi-digit image to 28x28**.
5. Test with at least 10 real images (white background, dark background, different sizes, blur/noise). Compare Gaussian, median, and no blur; global, adaptive, and Otsu thresholding. Save before/after pictures and a comparison table. Keep the chosen production pipeline stable.
6. For image generation, create folder `assets/digits/` with files `0.png` ... `9.png`. Example:
```python
from hnrs.image_acquisition import generate_number_image
image = generate_number_image('assets/digits', '5284', spacing=12)
image.save('assets/generated_5284.png')
```
7. Report failures, explain which method was selected and why. Add tests for invalid files and empty images.
8. Share a PR with Member 4. Do not rename functions or change outputs.

**Deliver:** two Python files, sample images, test results, comparison screenshots, 1-2 pages of genuine observations. **Initial handoff:** Oct 28; **final:** Oct 30.
