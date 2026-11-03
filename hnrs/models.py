"""Two independently trainable architectures for comparative evaluation."""

def make_model(kind='CNN', learning_rate=0.001):
    import tensorflow as tf
    if kind == 'CNN':
        model = tf.keras.Sequential([
            tf.keras.layers.Input((28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation='softmax')])
    elif kind == 'MLP':
        model = tf.keras.Sequential([
            tf.keras.layers.Input((28, 28, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')])
    else:
        raise ValueError('Model must be CNN or MLP')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
