import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = 224
NUM_CLASSES = 1  # Binary classification: NSFW vs SFW

def build_model():
    # Input layer: expects images already resized and normalized by data pipeline
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Load MobileNetV2 with pretrained weights, no top, and freeze backbone initially
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                             include_top=False,
                             weights="imagenet")
    base_model.trainable = False

    # Forward pass through base model, batchnorm layers in inference mode
    x = base_model(inputs, training=False)

    # Global average pooling to reduce spatial dims to vector
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout to reduce overfitting
    x = layers.Dropout(0.3)(x)

    # Dense layer with sigmoid for binary classification
    outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_fs_binary")

    # Compile with Adam optimizer and binary cross-entropy loss
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    return model, base_model
if __name__ == "__main__":
    model, base_model = build_model()
    model.summary()
