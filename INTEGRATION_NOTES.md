# HNRS combined integration package

## Included contributions
- Member 1: `member_sources/member_1/` original image acquisition, preprocessing and tests. `hnrs/preprocessing.py` uses their preprocessing code.
- Member 2: `member_sources/member_2/` original segmentation implementation, experiments, test images and results. `hnrs/member2_segmentation.py` is their original segmentation module; `hnrs/segmentation.py` adapts it to the GUI.
- Member 3: the original starter project ML notebook, MLP model and evaluation results are retained. The previous CNN is backed up as `models/cnn_previous.keras`.
- Your Week 7 CNN: `models/cnn.keras` (primary GUI CNN) and `models/best_cnn_model.keras` (same model copy).
- GUI: existing Tkinter `app.py` is retained, not replaced with Streamlit.

## Start on Windows (VS Code terminal)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```
If Python 3.11 is unavailable, use an installed compatible Python version supported by TensorFlow.

## Integration status
The package has been assembled and Python files checked for syntax. **End-to-end GUI recognition has not been verified on your Windows machine**. Image thresholding, digit segmentation and model preprocessing may need adjustment based on real examples. Member 2's segmentation was written for broader handwritten characters, and the adapter disables special grouping of division symbols for digit recognition.

## Notes
- The GUI's CNN choice now loads your Week 7 model. MLP choice loads the retained starter MLP.
- Keep the `member_sources/` folder for attribution and testing.
- Source projects are preserved, but caches and compiled Python files are omitted.
- `external_digits/` contains sample images without verified ground-truth labels.
