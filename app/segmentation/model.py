import os
import numpy as np
import cv2
import base64
import uuid
from pathlib import Path
import onnxruntime as ort # البطل الوحيد بتاعنا دلوقتي

current_dir = Path(__file__).resolve().parent
BASE_DIR = current_dir.parent.parent

# مسارات الموديلات الجديدة
SEG_PATH = BASE_DIR / "models" / "compressed" / "segmentation.onnx"
CLS_PATH = BASE_DIR / "models" / "compressed" / "classification_b4.onnx"

# تحميل الموديلات في الرام
seg_session = ort.InferenceSession(str(SEG_PATH))
cls_session = ort.InferenceSession(str(CLS_PATH))

# جلب أسماء المدخلات أوتوماتيكياً (عشان ONNX بيحتاجها)
seg_input_name = seg_session.get_inputs()[0].name
cls_input_name = cls_session.get_inputs()[0].name

def save_image_to_media(image_np, filename: str, base_url: str) -> str:
    """Save image to media directory and return public URL."""
    media_dir = BASE_DIR / "media"
    media_dir.mkdir(exist_ok=True)

    filepath = media_dir / filename
    cv2.imwrite(str(filepath), image_np)

    # Construct full public URL
    return f"{base_url}media/{filename}"

def estimate_tirads(class_idx, confidence):
    if class_idx == 0: 
        return "TR1" if confidence >= 0.90 else "TR2" if confidence >= 0.70 else "TR3"
    return "TR5" if confidence >= 0.85 else "TR4"

def process_full_pipeline(image_bytes: bytes, base_url: str = "http://localhost:8000/", threshold=0.6):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    orig_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # --- 1. Segmentation Phase ---
    img_seg_in = cv2.resize(img_gray, (256, 256)).astype(np.float32) / 255.0
    img_seg_in = np.expand_dims(img_seg_in, axis=(0, -1))

    # تشغيل ONNX للـ Segmentation
    mask_pred = seg_session.run(None, {seg_input_name: img_seg_in})[0][0, :, :, 0]

    mask = (mask_pred > threshold).astype(np.uint8)
    mask_full = cv2.resize(mask, (orig_rgb.shape[1], orig_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(mask_full > 0)
    if len(xs) == 0:
        return {"status": "success", "message": "No tumor detected", "bbox": None}

    x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
    bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
    roi = orig_rgb[y_min:y_max+1, x_min:x_max+1]

    # --- 2. Overlay Phase ---
    overlay = orig_rgb.copy()
    overlay[mask_full > 0] = [0, 255, 0]
    blended = cv2.addWeighted(orig_rgb, 0.7, overlay, 0.3, 0)

    # --- 3. Classification Phase ---
    roi_cls_in = cv2.resize(roi, (224, 224)).astype(np.float32) / 255.0
    roi_cls_in = np.expand_dims(roi_cls_in, axis=0)

    # تشغيل ONNX للـ Classification
    cls_pred = cls_session.run(None, {cls_input_name: roi_cls_in})[0][0]

    prob = float(cls_pred[0])
    class_idx = 1 if prob > 0.5 else 0
    confidence = prob if class_idx == 1 else 1 - prob

    # Generate unique ID for this prediction
    unique_id = str(uuid.uuid4())

    # Save images and get URLs
    mask_url = save_image_to_media(mask_full * 255, f"mask_{unique_id}.png", base_url)
    overlay_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    overlay_url = save_image_to_media(overlay_bgr, f"overlay_{unique_id}.png", base_url)
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    roi_url = save_image_to_media(roi_bgr, f"roi_{unique_id}.png", base_url)

    return {
        "status": "success",
        "bbox": bbox,
        "classification": {
            "prediction": class_idx,
            "label": "malignant" if class_idx == 1 else "benign",
            "confidence": round(confidence * 100, 2),
            "tirads_stage": estimate_tirads(class_idx, confidence)
        },
        "images": {
            "mask_url": mask_url,
            "overlay_url": overlay_url,
            "roi_url": roi_url
        }
    }