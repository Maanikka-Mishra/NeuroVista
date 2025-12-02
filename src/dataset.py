import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .config import DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

def create_datasets():
    print(f"\n📁 Loading dataset from: {DATA_DIR}")

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.20,
        rotation_range=15,
        zoom_range=0.10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    train = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    val = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    print("\n📌 Dataset Ready:")
    print(f"  Training batches: {len(train)}")
    print(f"  Validation batches: {len(val)}")

    return train, val
