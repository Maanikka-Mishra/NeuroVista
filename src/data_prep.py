import os
import cv2
from tqdm import tqdm

from .config import DATA_DIR

"""
data_prep.py
-------------
This script cleans the dataset by:
- Checking for corrupted images
- Removing unreadable images
- Printing class distribution
"""

def is_image_valid(image_path):
    """Check if an image can be loaded using OpenCV."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        return True
    except:
        return False


def clean_dataset():
    print(f"\n🧹 Starting dataset cleaning in: {DATA_DIR}")

    total_removed = 0
    classes = os.listdir(DATA_DIR)

    for cls in classes:
        class_path = os.path.join(DATA_DIR, cls)

        if not os.path.isdir(class_path):
            continue

        print(f"\n🔍 Checking class: {cls}")

        for img_name in tqdm(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)

            if not is_image_valid(img_path):
                print(f"❌ Removing corrupted image: {img_path}")
                os.remove(img_path)
                total_removed += 1

    print(f"\n✔ Cleaning complete. Total corrupted images removed: {total_removed}")


def count_images():
    print(f"\n📊 Dataset Summary in: {DATA_DIR}")

    classes = os.listdir(DATA_DIR)
    for cls in classes:
        class_path = os.path.join(DATA_DIR, cls)

        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            print(f"  📁 {cls}: {num_images} images")


if __name__ == "__main__":
    clean_dataset()
    count_images()
