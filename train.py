
"""Train and evaluate both models on MNIST; writes reproducible evaluation files."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from hnrs.models import make_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--model', choices=['CNN', 'MLP', 'both'], default='both')
    args = p.parse_args()

    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)
    random.seed(42)

    (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()
    xtrain = xtrain.astype('float32')[..., None] / 255.0
    xtest = xtest.astype('float32')[..., None] / 255.0

    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)

    for kind in (['CNN', 'MLP'] if args.model == 'both' else [args.model]):
        model = make_model(kind, args.learning_rate)

        # Measure training duration
        start_time = time.perf_counter()

        history = model.fit(
            xtrain,
            ytrain,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.1,
            verbose=2
        )

        training_seconds = time.perf_counter() - start_time
        print(f'{kind} training time: {training_seconds:.2f} seconds')

        loss, accuracy = model.evaluate(xtest, ytest, verbose=0)
        predicted = model.predict(xtest, verbose=0).argmax(axis=1)

        model.save(f'models/{kind.lower()}.keras')

        # Save each model's evaluation results inside the loop
        with open(f'results/{kind.lower()}_evaluation.json', 'w') as f:
            json.dump({
                'test_loss': float(loss),
                'test_accuracy': float(accuracy),
                'training_time_seconds': float(training_seconds),
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'classification_report': classification_report(
                    ytest, predicted, output_dict=True
                ),
                'history': {
                    k: [float(v) for v in values]
                    for k, values in history.history.items()
                }
            }, f, indent=2)

        np.savetxt(
            f'results/{kind.lower()}_confusion_matrix.csv',
            confusion_matrix(ytest, predicted),
            fmt='%d',
            delimiter=','
        )

        print(f'{kind}: test accuracy={accuracy:.4f}, loss={loss:.4f}')


if __name__ == '__main__':
    main()
