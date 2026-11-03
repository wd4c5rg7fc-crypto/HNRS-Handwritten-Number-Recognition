import tensorflow as tf

# Load the trained CNN model
model = tf.keras.models.load_model(
    "models/best_cnn_model.keras"
)

print("\nCNN model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

model.summary()