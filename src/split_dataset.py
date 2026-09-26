import os
import random
import shutil

# Original dataset
SOURCE_DIR = "dataset/original_images"

# Where split dataset will be created
OUTPUT_DIR = "dataset"

# Split ratio
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

# For reproducible splitting
random.seed(42)

# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# Get all class folders
classes = sorted(
    [
        folder
        for folder in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, folder))
    ]
)

print(f"Total classes found: {len(classes)}")
print("Classes:", classes)


for class_name in classes:

    source_class_path = os.path.join(SOURCE_DIR, class_name)

    # Get images
    images = [
        file
        for file in os.listdir(source_class_path)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ]

    # Shuffle images
    random.shuffle(images)

    total_images = len(images)

    train_count = int(total_images * TRAIN_RATIO)
    validation_count = int(total_images * VALIDATION_RATIO)

    train_images = images[:train_count]

    validation_images = images[
        train_count:train_count + validation_count
    ]

    test_images = images[
        train_count + validation_count:
    ]

    # Create destination folders
    train_path = os.path.join(OUTPUT_DIR, "train", class_name)
    validation_path = os.path.join(OUTPUT_DIR, "validation", class_name)
    test_path = os.path.join(OUTPUT_DIR, "test", class_name)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Copy training images
    for image in train_images:
        source = os.path.join(source_class_path, image)
        destination = os.path.join(train_path, image)
        shutil.copy2(source, destination)

    # Copy validation images
    for image in validation_images:
        source = os.path.join(source_class_path, image)
        destination = os.path.join(validation_path, image)
        shutil.copy2(source, destination)

    # Copy testing images
    for image in test_images:
        source = os.path.join(source_class_path, image)
        destination = os.path.join(test_path, image)
        shutil.copy2(source, destination)

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Validation={len(validation_images)}, "
        f"Test={len(test_images)}"
    )


print("\nDataset splitting completed successfully!")