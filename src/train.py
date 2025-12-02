import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from .dataset import create_datasets
from .config import IMG_HEIGHT, IMG_WIDTH, EPOCHS, MODEL_PATH, MODEL_DIR

def build_resnet_model():
    print("\n🔧 Building ResNet50 Alzheimer Model...")

    base = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )

    base.trainable = False  # Freeze base model layers

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(4, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(model.summary())
    return model


def train():
    # Load datasets
    train_ds, val_ds = create_datasets()

    # Create model directory if not exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss")
    ]

    # Build Model
    model = build_resnet_model()

    print("\n🚀 Starting Training...\n")

    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=300,      # Much faster
    validation_steps=80,
    callbacks=callbacks
    )
    print(f"\n✅ Model saved at: {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()
