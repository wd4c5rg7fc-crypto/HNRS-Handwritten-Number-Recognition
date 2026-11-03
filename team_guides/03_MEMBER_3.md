# MEMBER 3: CNN, MLP and evaluation
**Own:** `hnrs/models.py`, `train.py`, `hnrs/predict.py`, `models/*.keras`, `results/` and your tests.

1. Follow `00_READ_FIRST.md` and confirm TensorFlow imports:
```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```
2. Review `hnrs/models.py`. Both CNN and MLP take `(28,28,1)` and output ten class probabilities. Keep these input/output shapes unchanged.
3. Train a quick baseline, then the final runs:
```powershell
python train.py --model both --epochs 1
python train.py --model both --epochs 10
```
MNIST downloads automatically on first run and may require internet. Training may take considerable time on CPU. The second command overwrites the first run's model/results, so save experimental runs separately before repeating.
4. Verify files `models/cnn.keras`, `models/mlp.keras`, `results/cnn_evaluation.json`, `results/mlp_evaluation.json` and both confusion-matrix CSVs exist.
5. Test prediction contract:
```python
import numpy as np
from hnrs.predict import load_trained_model, predict_digits
model = load_trained_model('CNN')
print(predict_digits(model, [np.zeros((28,28,1),dtype=np.float32)]))
```
This checks the interface only, **not accuracy**.
6. Compare CNN and MLP on held-out MNIST test data; calculate accuracy, precision, recall, F1 and confusion matrices. Save real results. Use training-only validation for hyperparameter selection. Make accuracy/loss plots from JSON history.
7. Test at least 20 independent handwritten digits from group members and record correct/incorrect predictions. Compare external-image accuracy separately from MNIST.
8. Send trained model files to Member 4 early. Never change filenames or expected tensor shape.

**Deliver:** source, two trained models, test metrics, graphs, external tests and limitations. **Initial handoff:** Oct 28; **final:** Oct 30.
