import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from .config import MODEL_PATH, IMG_HEIGHT, IMG_WIDTH

# Class labels in the same order as training
STAGE_LABELS = [
    "Mild Dementia",
    "Moderate Dementia",
    "Non Demented",
    "Very mild Dementia"
]

def pick_image_dialog():
    """Open a file picker dialog to choose an image."""
    root = tk.Tk()
    root.withdraw()  # Hide window

    file_path = filedialog.askopenfilename(
        title="Select MRI Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    return file_path

def predict_image(image_path: str):
    print(f"\n📸 Predicting for image: {image_path}")

    # Load trained model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found at: {MODEL_PATH}")
        print("➡ Train the model first using: python -m src.train")
        return

    model = load_model(MODEL_PATH)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ ERROR: Could not read the selected image. Try another file.")
        return

    # Preprocess image
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)[0]  # e.g., [0.1, 0.6, 0.2, 0.1]
    class_idx = int(np.argmax(preds))
    stage = STAGE_LABELS[class_idx]
    confidence = float(preds[class_idx]) * 100

    # Alzheimer: YES / NO
    if stage == "Non Demented":
        alzheimer = "NO"
    else:
        alzheimer = "YES"

    # Print results in your desired order
    print(f"\n🧠 Alzheimer: {alzheimer}")
    print(f"🩺 Stage: {stage}")
    print(f"📊 Confidence: {confidence:.2f}%")

    # Create pie chart
    plt.figure()
    plt.title("Alzheimer Stage Probability Distribution")
    plt.pie(
        preds,
        labels=STAGE_LABELS,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90
    )
    plt.axis("equal")
    plt.tight_layout()

    # Save chart
    output_path = "prediction_pie.png"
    plt.savefig(output_path)
    plt.show()

    print(f"\n📈 Pie chart saved as: {os.path.abspath(output_path)}")

    return {
        "alzheimer": alzheimer,
        "stage": stage,
        "confidence": confidence,
        "probabilities": preds.tolist()
    }


if __name__ == "__main__":
    print("\n🖼️ Please pick an MRI image to predict...")
    image_path = pick_image_dialog()

    if image_path:
        predict_image(image_path)
    else:
        print("\n❌ No image selected. Exiting.")
