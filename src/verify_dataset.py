import os
from PIL import Image

DATASET_PATH = "dataset/original_images"

# Supported image extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Get class folders
classes = sorted(
    [
        folder
        for folder in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, folder))
    ]
)

print("=" * 50)
print("ISL DATASET VERIFICATION")
print("=" * 50)

print(f"\nTotal classes found: {len(classes)}")
print("Classes:", classes)

print("\nImages per class:")
print("-" * 50)

total_images = 0
image_sizes = set()
corrupt_images = []

for class_name in classes:

    class_path = os.path.join(DATASET_PATH, class_name)

    images = [
        file for file in os.listdir(class_path)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ]

    print(f"{class_name}: {len(images)} images")

    total_images += len(images)

    # Check image dimensions and corrupted images
    for image_name in images:

        image_path = os.path.join(class_path, image_name)

        try:
            with Image.open(image_path) as img:
                image_sizes.add(img.size)

        except Exception:
            corrupt_images.append(image_path)

print("\n" + "=" * 50)
print(f"TOTAL IMAGES: {total_images}")
print(f"UNIQUE IMAGE SIZES: {len(image_sizes)}")
print("IMAGE SIZES:", image_sizes)

print(f"\nCORRUPTED IMAGES: {len(corrupt_images)}")

if corrupt_images:
    print("\nCorrupted files:")
    for file in corrupt_images:
        print(file)

print("=" * 50)