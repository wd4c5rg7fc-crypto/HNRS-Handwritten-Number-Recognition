# MEMBER 4: GUI, integration and system evaluation (YOU)
**Own:** `app.py`, integration/system tests, user guide, team merge.

1. Follow `00_READ_FIRST.md`; use Tkinter baseline GUI in `app.py` (no Streamlit required).
2. Test GUI startup:
```powershell
python app.py
```
It should open even before models are trained. Test file upload and image composition (needs `assets/digits/0.png` ... `9.png`).
3. Ask Member 1 for preprocessing and acquisition modules; Member 2 for segmentation; Member 3 for model files and predict module. All names and formats are fixed in `00_READ_FIRST.md`.
4. Before merging any teammate PR, run:
```powershell
python -m unittest discover -s tests -v
```
5. Verify complete flow: load image -> preprocess -> segment -> prepare each digit -> select CNN/MLP -> predict -> display full number and confidences.
6. Check that empty input, corrupt file, no digits, missing models and invalid image formats produce clear messages rather than crashes.
7. Make a labelled evaluation set of at least 20 multi-digit images. Record exact-match accuracy (whole number correct), per-digit accuracy, segmentation success, and failure cases. Do not invent scores.
8. Add GUI screenshots, architecture diagram, installation instructions, evaluation tables and contribution evidence to team report. The assignment also requires Git access, OneDrive documentation, weekly logs, sprint reports and a 10–12 minute video.
9. Run final smoke test on a clean machine/environment and package source + trained models + report + video or OneDrive video link according to course instructions.

**Deliver:** working integrated GUI, tests, 20-image system evaluation, report sections, screenshots and final handoff. **Target:** Sunday Nov 1; **official due:** Tuesday Nov 3 at 11:59 pm.
