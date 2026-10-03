import cv2
import tensorflow as tf
import numpy as np


# -----------------------------
# 1. Settings
# -----------------------------

MODEL_PATH = "models/isl_cnn.keras"
IMG_SIZE = (128, 128)

class_names = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z"
]


# -----------------------------
# 2. Load model
# -----------------------------

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# -----------------------------
# 3. Start webcam
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started.")
print("Place your hand inside the green box.")
print("Press 'q' to quit.")


# -----------------------------
# 4. Webcam loop
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # -------------------------
    # Centered ROI
    # -------------------------

    height, width = frame.shape[:2]

    box_size = 350

    x1 = (width - box_size) // 2
    y1 = (height - box_size) // 2

    x2 = x1 + box_size
    y2 = y1 + box_size

    # Draw green box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Extract ROI
    roi = frame[y1:y2, x1:x2]

    # -------------------------
    # Preprocessing
    # -------------------------

    image = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        IMG_SIZE
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    # -------------------------
    # Prediction
    # -------------------------

    prediction = model.predict(
        image,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction[0]
    )

    predicted_label = class_names[
        predicted_class
    ]

    confidence = (
        prediction[0][predicted_class] * 100
    )

    # -------------------------
    # Display prediction
    # -------------------------

    text = (
        f"Sign: {predicted_label} | "
        f"Confidence: {confidence:.2f}%"
    )

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Place hand inside box",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Show webcam
    cv2.imshow(
        "ISL Recognition",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# 5. Close webcam
# -----------------------------

cap.release()

cv2.destroyAllWindows()

print("Webcam closed.")