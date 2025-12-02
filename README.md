# 🧠 **NeuroVista – Alzheimer's Disease Detection using MRI (Backend)**

### *Deep Learning Powered Medical Diagnosis System*

---

## 📌 **Project Overview**

**NeuroVista** is a deep-learning–based system designed to detect different stages of **Alzheimer’s Disease (AD)** using **MRI Brain Scans**.
This backend provides:

* MRI preprocessing
* Deep Learning model training
* Stage prediction
* Pie-chart visualization of predicted probabilities
* Ready-to-use REST API for frontend integration

This is the **backend-only repository**, containing model scripts, preprocessing code, training pipeline, and prediction tools.

---

## 🎯 **Objective**

The goal is to build an **AI-powered early diagnosis system** that can classify MRIs into:

* **Non Demented**
* **Very Mild Demented**
* **Mild Demented**
* **Moderate Demented**

This helps in **early detection**, **better intervention**, and **accurate monitoring** of Alzheimer’s progression.

---

## 📂 **Project Structure**

```
Alzhiemer/
│
├── data/
│   ├── raw/
│   │   └── dataset/
│   │       └── Data/      # Kaggle MRI images
│
├── models/
│   └── resnet_alzheimer.h5   # saved trained model
│
├── notebooks/
│   └── exploration.ipynb     # (optional) EDA / experiments
│
├── reports/
│   └── prediction_pie.png    # sample output pie chart
│
├── src/
│   ├── config.py             # central configuration
│   ├── data_prep.py          # preprocessing utilities
│   ├── dataset.py            # Data loading pipeline
│   ├── train.py              # training script
│   ├── predict.py            # prediction + pie chart
│   └── __init__.py
│
├── venv/                     # virtual environment
│
├── requirements.txt          # python dependencies
└── README.md                 # this file
```

---

## 🧪 **Tech Stack**

### **Programming Language**

* Python 3.10+

### **Deep Learning**

* TensorFlow / Keras
* Transfer Learning (ResNet50)

### **Data Science**

* NumPy
* Pandas
* Matplotlib

### **Image Processing**

* OpenCV
* PIL (Pillow)

---

## 🖼 **Dataset**

Dataset used:
✅ **Kaggle MRI Dataset for Alzheimer’s Classification**
Contains 4 classes:

| Class Name           | Description         |
| -------------------- | ------------------- |
| **NonDemented**      | Healthy brain       |
| **VeryMildDemented** | Early stage         |
| **MildDemented**     | Noticeable dementia |
| **ModerateDemented** | Advanced dementia   |

Place the dataset here:

```
data/raw/dataset/Data/
```

---

## ⚙️ **Setup Instructions**

### 1️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate:

**Windows**

```
venv\Scripts\activate
```

**Mac/Linux**

```
source venv/bin/activate
```

---

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## 🚀 **Training the Model**

Run the training script:

```
python -m src.train
```

What it does:

* Loads MRI images
* Preprocesses (resize, normalize)
* Builds ResNet50 model
* Trains for defined epochs
* Saves model → `models/resnet_alzheimer.h5`

---

## 🔍 **Making Predictions**

Place an MRI image anywhere, then run:

```
python -m src.predict --image path_to_image
```

This:

* Loads trained model
* Predicts all four Alzheimer classes
* Shows prediction percentages
* Generates a **pie chart** (saved in `/reports/prediction_pie.png`)

---

## 📊 **Sample Output (Pie Chart)**

The prediction script generates a detailed pie chart showing probability of each Alzheimer stage.

```
Non Demented: 72.4%
Very Mild Demented: 15.2%
Mild Demented: 9.7%
Moderate Demented: 2.7%
```

---

## 🧠 **Model Used: ResNet50 (Transfer Learning)**

Why ResNet50?

* High accuracy on medical images
* Deep residual connections prevent vanishing gradients
* Lightweight compared to larger networks
* Works well with limited datasets

Training properties are configured in `config.py`:

```
IMG_SIZE = 224 × 224
BATCH_SIZE = 32
EPOCHS = 5  (modifiable)
```

---

## 🛠 **Configuration File**

All paths/settings are centralized in:

```
src/config.py
```

You can update:

* DATASET path
* EPOCHS
* IMAGE size
* MODEL saving paths

---

## 🔧 **Backend API (Optional)**

If you want to expose prediction as API:

```
POST /predict
Content-Type: multipart/form-data
```

Returns:

```json
{
  "predictions": {
    "NonDemented": 0.72,
    "VeryMildDemented": 0.15,
    "MildDemented": 0.09,
    "ModerateDemented": 0.04
  }
}
```

You already have prediction logic (`predict.py`), so API can be added later without changes.

---

## 📈 **Future Scope**

* Add Flask/FastAPI backend
* Build full frontend dashboard using React(for cognitive games)
* Prepare doctor patient dashboard
* Add explainability (Grad-CAM heatmaps)
* Train using more MRI scans for higher accuracy
* Deploy on cloud / HuggingFace Spaces

---

