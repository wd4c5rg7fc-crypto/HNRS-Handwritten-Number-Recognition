# SHARED SETUP AND CONTRACT (READ FIRST)

Use ONE Git repository and ONE project folder. Each member edits ONLY assigned files. Do not create separate incompatible apps. All commands below run in the project root (the folder containing app.py).

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```
If Python 3.11 is unavailable, install it first. If activation is blocked, use `.\.venv\Scripts\python.exe` instead of `python` in commands. TensorFlow installation may vary by hardware and Python version.

## Frozen interfaces
1. Member 1 `hnrs/preprocessing.py`: `preprocess_image(image)` -> 2D uint8 array, white foreground on black background; `prepare_digit(crop)` -> float32 `(28,28,1)` values 0..1. `hnrs/image_acquisition.py`: `load_image(path)` -> RGB uint8 array; `generate_number_image(folder,digits,spacing=8)` -> PIL image.
2. Member 2 `hnrs/segmentation.py`: `segment_digits(image)` -> `(crops,boxes)` where crops are 2D uint8 raw digit regions, boxes are `(x,y,w,h)` left-to-right. Do not normalize inside segmentation.
3. Member 3 `hnrs/predict.py`: `load_trained_model(kind)` -> Keras model; `predict_digits(model,digit_tensors)` -> `[(digit:int, confidence:float),...]`. Trained models MUST be saved as `models/cnn.keras` and `models/mlp.keras`.
4. Member 4 `app.py`: GUI integrates functions above. This starter uses **Tkinter**, not Streamlit. Do not change GUI framework mid-project without agreement.

The included modules are working baseline implementations, not evidence of experimental superiority. Tests check contracts, not real-world recognition accuracy. Touching digits, disconnected strokes, and domain shift can cause failures.

Git workflow: each person works on a named branch; submit a pull request; Member 4 merges after running all tests. Never commit `.venv/`, MNIST downloads, or private data. Commit trained model files only if repository file-size rules allow; otherwise distribute through agreed team storage and include them in submission ZIP.
