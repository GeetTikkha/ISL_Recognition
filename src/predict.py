import tensorflow as tf
import numpy as np
import cv2


# -----------------------------
# 1. Settings
# -----------------------------

MODEL_PATH = "models/isl_cnn.keras"
IMAGE_PATH = "test_image.jpg"

IMG_SIZE = (128, 128)


# -----------------------------
# 2. Class names
# -----------------------------

class_names = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]


# -----------------------------
# 3. Load trained model
# -----------------------------

model = tf.keras.models.load_model(MODEL_PATH)


# -----------------------------
# 4. Read image
# -----------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Error: Image could not be loaded.")
    exit()


# -----------------------------
# 5. Preprocess image
# -----------------------------

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.resize(image, IMG_SIZE)

image = image.astype(np.float32) / 255.0

image = np.expand_dims(image, axis=0)


# -----------------------------
# 6. Prediction
# -----------------------------

prediction = model.predict(image, verbose=0)

predicted_class = np.argmax(prediction[0])

predicted_label = class_names[predicted_class]

confidence = prediction[0][predicted_class] * 100


# -----------------------------
# 7. Display result
# -----------------------------

print("\n-----------------------------")
print("PREDICTION RESULT")
print("-----------------------------")

print("Predicted Sign:", predicted_label)
print(f"Confidence: {confidence:.2f}%")