# HNRS | COS30018 Option B

## Important status
This is a functional **source-code starter implementation**, not a tested or submission-ready group deliverable. No model weights or genuine benchmark results are included until training is run. The report is a draft with placeholders for measured outcomes, screenshots, team details, and references. Do not submit unverified results or claim work you did not do.

## Setup
Use Python 3.10–3.12 with a working Tkinter installation. From this directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --epochs 5 --model both
python app.py
```

Training downloads MNIST if not cached and saves `models/cnn.keras`, `models/mlp.keras`, evaluation JSON files and confusion matrices. It may take several minutes to hours depending on hardware.

## GUI
Load a handwritten number image or use the composition function with a folder of digit PNGs named `0.png` through `9.png`. Select CNN/MLP and click Recognise. Segmentation uses connected components and works best with separated, non-touching dark digits on a bright background.

## Testing
`python -m unittest discover -s tests -v`

## Evaluation still needed
Collect real handwritten single-digit and multi-digit test images not used in training. Document segmentation failures, number-level accuracy, per-digit accuracy, timing, and preprocessing comparisons. Do not use MNIST test images to tune the model. Record measured values and screenshots in the report.

## Submission checklist
- Train both models and include saved weights and measured results.
- Verify GUI on your own computer and collect screenshots.
- Add team names, student IDs, contributions, weekly logs and sprint reports.
- Expand report to the required 20–30 pages with authentic experiments and citations.
- Record a genuine 10–12 minute demo video.
- Publish code in Git and documentation on OneDrive; grant staff access.
- Seek tutor approval for any optional research extension.
