import cv2
import numpy as np
import os

# One sample image
IMAGE_PATH = "dataset/train/0"

# Get first image from class 0
image_files = os.listdir(IMAGE_PATH)

image_file = image_files[0]

full_path = os.path.join(IMAGE_PATH, image_file)

print("Testing image:", full_path)


# -----------------------------
# 1. Read image
# -----------------------------
image = cv2.imread(full_path)

if image is None:
    print("Error: Image could not be loaded.")
    exit()

print("Original shape:", image.shape)


# -----------------------------
# 2. BGR → RGB
# -----------------------------
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# -----------------------------
# 3. Resize
# -----------------------------
image = cv2.resize(image, (128, 128))

print("After resize:", image.shape)


# -----------------------------
# 4. Normalize
# -----------------------------
image = image.astype(np.float32) / 255.0

print("After normalization:")
print("Minimum pixel value:", image.min())
print("Maximum pixel value:", image.max())


# -----------------------------
# 5. Final result
# -----------------------------
print("\nPreprocessing successful!")
print("Final image shape:", image.shape)
print("Final data type:", image.dtype)