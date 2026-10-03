import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# -----------------------------
# 1. Paths and settings
# -----------------------------

TEST_DIR = "dataset/test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


# -----------------------------
# 2. Load test dataset
# -----------------------------

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# -----------------------------
# 3. Normalize images
# -----------------------------

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# -----------------------------
# 4. Load trained model
# -----------------------------

model = tf.keras.models.load_model(
    "models/isl_cnn.keras"
)


# -----------------------------
# 5. Evaluate model
# -----------------------------

test_loss, test_accuracy = model.evaluate(test_dataset)

print("\n-----------------------------")
print("TEST RESULTS")
print("-----------------------------")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# -----------------------------
# 6. Get predictions
# -----------------------------

y_true = []
y_pred = []


for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(labels.numpy(), axis=1)

    y_pred.extend(predicted_classes)
    y_true.extend(actual_classes)


# -----------------------------
# 7. Class names
# -----------------------------

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = test_dataset.class_names


# -----------------------------
# 8. Classification report
# -----------------------------

print("\n-----------------------------")
print("CLASSIFICATION REPORT")
print("-----------------------------")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


# -----------------------------
# 9. Confusion matrix
# -----------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n-----------------------------")
print("CONFUSION MATRIX")
print("-----------------------------")

print(cm)