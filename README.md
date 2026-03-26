# 🦋 ThyraX: AI System for Thyroid Cancer Analysis

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" alt="Python Badge">
  <img src="https://img.shields.io/badge/FastAPI-0.135+-009688?logo=fastapi" alt="FastAPI Badge">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker" alt="Docker Badge">
  <img src="https://img.shields.io/badge/ONNX-Runtime-blue?logo=onnx" alt="ONNX Badge">
  <img src="https://img.shields.io/badge/OpenCV-Image_Processing-5C3EE8?logo=opencv" alt="OpenCV Badge">
  <img src="https://img.shields.io/badge/Deployed_on-Lightning_AI-792EE5?logo=lightning" alt="Lightning Badge">
</div>

<p align="center">
  <em>An advanced, ultra-fast Medical AI API processing Ultrasound and X-Ray imagery for precise tumor segmentation and malignancy classification.</em>
</p>

---

## ⚕️ Project Overview

**ThyraX** is an advanced AI-powered diagnostic tool developed to assist healthcare professionals in early and accurate detection of Thyroid Cancer. By analyzing medical ultrasound imagery, the system performs precise tumor localization (segmentation) and malignancy assessment (classification) to ultimately estimate the corresponding **TIRADS stage**.

The project places an immense focus on software engineering best practices, deploying heavy deep learning models into ultra-fast, lightweight production environments.

---

## 🚀 Key Technical Achievements

- **Lightweight Edge-Ready AI:** Replaced heavy Deep Learning frameworks (TensorFlow/PyTorch) with **ONNX Runtime**, drastically reducing model sizes and enabling ultra-fast CPU inference.
- **Minimal Docker Footprint:** Fully containerized the application, achieving an image size of **< 400MB** compared to typical multi-gigabyte ML containers.
- **High-Concurrency Backend:** Built an asynchronous, non-blocking API using **FastAPI** to serve clinical predictions instantaneously.
- **Modern Dependency Management:** Leveraged **`uv`**, the extremely fast Rust-based package manager, ensuring reproducible and secure builds over traditional `pip`.
- **Cloud Deployment:** Deployed successfully and efficiently on **Lightning AI** cloud spaces.

---

## 🏗️ Architecture Pipeline

The system processes incoming imagery through a highly-optimized sequence:

1. **Input Image:** Client sends a medical image (Ultrasound/X-Ray) via `multipart/form-data`.
2. **Preprocessing (OpenCV):** Image is dynamically resized, normalized, and transformed to match model input dimensions.
3. **Segmentation (ONNX):** The Segmentation Model extracts a precise spatial mask localizing the thyroid nodule.
4. **Cropping (ROI):** A bounding box is extracted from the segmentation mask, tightly cropping the Region of Interest (ROI).
5. **Classification (ONNX):** The Classification Model evaluates the cropped nodule to predict the probability of malignancy and assign a TIRADS category.
6. **JSON Response:** The backend returns formatted JSON containing the bounding box coordinates, confidence scores, and Base64-encoded visual results.

---

## 🛠️ Tech Stack

- **Backend Framework:** FastAPI, Uvicorn, Python `python-multipart`
- **Deep Learning/Inference:** ONNX Runtime (`onnxruntime`)
- **Image Processing:** OpenCV (`opencv-python-headless`), NumPy
- **Containerization:** Docker, Docker Compose
- **Package Management:** Astral `uv`
- **Cloud Hosting:** Lightning AI Spaces

---

## 💻 Local Setup & Installation

Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/thyrax.git
cd thyrax
```

### Option A: Using Docker (Recommended for Production)
Build and run the highly optimized container:
```bash
docker build -t thyrax-api .
docker run -p 8000:8000 thyrax-api
```

### Option B: Using `uv` (Recommended for Development)
If you wish to test or develop locally without Docker, use `uv`:
```bash
# Install dependencies directly from pyproject.toml / uv.lock
uv sync

# Run the backend server
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The API Swagger UI will now be active at `http://localhost:8000/docs`.

---

## 🌐 API Usage (Frontend Integration)

The primary endpoint handles visual diagnosis via a standard form submission.

**Endpoint:** `POST /predict/image`

### HTTP Request Example (cURL)

```bash
curl -X 'POST' \
  'http://localhost:8000/predict/image' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_ultrasound.jpg;type=image/jpeg'
```

### Expected JSON Response

```json
{
  "status": "success",
  "data": {
    "bounding_box": {
      "x": 120,
      "y": 85,
      "width": 204,
      "height": 190
    },
    "classification": {
      "malignancy_confidence": 0.941,
      "is_malignant": true,
      "tirads_stage": "TIRADS 5"
    },
    "visualizations": {
      "annotated_image_base64": "iVBORw0KGgoAAAANSUhEUgA..."
    }
  }
}
```

---

## 👥 Contributors

- **[Your Name Here]** - Deep Learning / Computer Vision Engineer

*Designed and developed as a Graduation Capstone Project.*
