from tensorflow.keras.models import load_model
from pathlib import Path

model_dir = Path(r"D:\Intelligent Systems\HNRS_Final_Submission\models")
output_dir = Path("HNRS_Final_Evidence")
output_dir.mkdir(exist_ok=True)

models = {
    "CNN": "best_cnn_model.keras",
    "MLP": "mlp.keras",
    "MathCNN": "math_cnn_16.keras"
}

for name, filename in models.items():
    model = load_model(model_dir / filename, compile=False)

    output_file = output_dir / f"{name}_architecture.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{name} MODEL ARCHITECTURE\n")
        f.write("=" * 85 + "\n\n")
        f.write(
            f"{'Layer':<30}"
            f"{'Output Shape':<30}"
            f"{'Parameters':>15}\n"
        )
        f.write("-" * 85 + "\n")

        for layer in model.layers:
            layer_name = f"{layer.name} ({layer.__class__.__name__})"

            try:
                output_shape = str(layer.output.shape)
            except Exception:
                output_shape = "N/A"

            params = layer.count_params()

            f.write(
                f"{layer_name:<30}"
                f"{output_shape:<30}"
                f"{params:>15,}\n"
            )

        f.write("-" * 85 + "\n")
        f.write(f"Total parameters: {model.count_params():,}\n")

    print(f"Created: {output_file}")

print("\nAll architecture evidence generated successfully!")