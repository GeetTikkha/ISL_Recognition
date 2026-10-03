import tensorflow as tf

from model import build_model


# -----------------------------
# 1. Paths
# -----------------------------

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 36


# -----------------------------
# 2. Load training dataset
# -----------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)


# -----------------------------
# 3. Load validation dataset
# -----------------------------

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# -----------------------------
# 4. Normalize pixel values
# -----------------------------

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# -----------------------------
# 5. Build CNN model
# -----------------------------

model = build_model()


# -----------------------------
# 6. Compile model
# -----------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------
# 7. Display model
# -----------------------------

model.summary()


# -----------------------------
# 8. Train model
# -----------------------------

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20
)


# -----------------------------
# 9. Save trained model
# -----------------------------

model.save("models/isl_cnn.keras")

print("\nTraining completed successfully!")
print("Model saved at: models/isl_cnn.keras")