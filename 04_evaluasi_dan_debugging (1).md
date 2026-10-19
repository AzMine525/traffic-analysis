# Evaluasi dan Debugging Model

**Durasi Baca**: 18 Menit | **Level**: Menengah

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan mampu:
- Melakukan evaluasi visual dan kuantitatif terhadap model
- Membaca dan menginterpretasi confusion matrix
- Mendiagnosis masalah umum (low recall, low precision, overfitting)
- Menerapkan teknik debugging lanjutan
- Mengimplementasikan active learning loop untuk perbaikan berkelanjutan

---

## 🎬 Pendahuluan

Model sudah ditraining dan mAP-nya 0.85. Bagus kan?

**Tunggu dulu**. Angka mAP tidak menggambarkan keseluruhan cerita:
- Apakah model bisa mendeteksi motor di malam hari?
- Apakah model sering salah antara bus dan truk?
- Apakah model "over-confident" di prediksi yang salah?

Artikel ini membahas **cara mengevaluasi dan memperbaiki** model secara sistematis.

---

## 1. Evaluasi Kuantitatif

### 1.1 Menjalankan Validasi

```python
from ultralytics import YOLO

# Load model terbaik
model = YOLO("runs/detect/traffic_v1/weights/best.pt")

# Jalankan validasi
metrics = model.val(data="traffic_dataset/data.yaml")

# Print metrics
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
```

### 1.2 Memahami Metrik

| Metrik | Formula | Interpretasi |
|--------|---------|--------------|
| **Precision** | TP / (TP + FP) | Seberapa akurat prediksi positif? |
| **Recall** | TP / (TP + FN) | Berapa % objek yang terdeteksi? |
| **mAP50** | Mean AP @ IoU=0.5 | Rata-rata AP dengan threshold 50% |
| **mAP50-95** | Mean AP @ IoU=0.5:0.95 | Rata-rata AP dengan threshold bervariasi |

### 1.3 Target Metrik untuk Traffic Analytics

| Aplikasi | mAP50 Target | Precision Target | Recall Target |
|----------|--------------|------------------|---------------|
| **Demo/Prototype** | > 0.60 | > 0.70 | > 0.60 |
| **Production** | > 0.75 | > 0.80 | > 0.80 |
| **Mission-Critical** | > 0.85 | > 0.90 | > 0.90 |

---

## 2. Evaluasi Visual

### 2.1 Prediksi pada Validation Set

```python
# Jalankan prediksi dengan visualisasi
results = model.predict(
    source="traffic_dataset/images/val/",
    save=True,
    conf=0.5,
    project="runs/debug",
    name="visual_eval"
)

print(f"Predictions saved to: runs/debug/visual_eval/")
```

### 2.2 Checklist Visual

Buka folder hasil dan periksa secara manual:

```
Checklist Evaluasi Visual:

□ Bounding box akurat (pas dengan objek, tidak terlalu besar/kecil)
□ Tidak ada objek yang terlewat (terutama yang kecil)
□ Tidak ada false positive (deteksi pada area kosong)
□ Klasifikasi benar (motor tidak disebut car)
□ Confidence score masuk akal (objek jelas = score tinggi)
□ Konsisten di berbagai kondisi (siang/malam, cerah/hujan)
```

### 2.3 Script untuk Perbandingan GT vs Prediksi

```python
"""
compare_gt_pred.py - Bandingkan Ground Truth vs Prediction
"""
import cv2
import numpy as np
from pathlib import Path

def compare_gt_vs_pred(image_path: str, label_path: str, model, class_names: list):
    """
    Visualisasi ground truth (hijau) vs prediction (merah)
    """
    # Load image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Draw ground truth (HIJAU)
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # Convert normalized to pixel
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Hijau
            cv2.putText(img, f"GT: {class_names[class_id]}", 
                       (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw predictions (MERAH)
    results = model.predict(image_path, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Merah
        cv2.putText(img, f"Pred: {class_names[class_id]} {conf:.2f}", 
                   (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return img

# Usage
from ultralytics import YOLO

model = YOLO("best.pt")
class_names = ['car', 'motorcycle', 'bus', 'truck']

for img_path in Path("traffic_dataset/images/val").glob("*.jpg")[:10]:
    label_path = str(img_path).replace("images", "labels").replace(".jpg", ".txt")
    result_img = compare_gt_vs_pred(str(img_path), label_path, model, class_names)
    cv2.imwrite(f"debug/{img_path.name}", result_img)

print("Comparison images saved to debug/")
```

---

## 3. Confusion Matrix

### 3.1 Apa Itu Confusion Matrix?

Confusion matrix menunjukkan **bagaimana model salah klasifikasi** antar kelas.

![Confusion Matrix Example](assets/confusion_matrix.png)

### 3.2 Cara Membaca

```
              │        PREDICTED
              │   Car    Motor   Bus   Truck   BG
─────────────────────────────────────────────────
         Car  │   92      3       2      1      2
              │   ↑ True Positive (benar)
ACTUAL  Motor │    1     94       0      0      5
              │                               ↑ Missed (FN)
         Bus  │    1      1      91      4      3
         Truck│    1      0       5     90      4
              │                   ↑ Confused (truk → bus)
         BG   │    1      2       1      0     96
              │   ↑ False Positive (FP)
```

### 3.3 Interpretasi

| Cell | Meaning | Problem |
|------|---------|---------|
| **Diagonal Tinggi** | Prediksi benar | ✅ Bagus |
| **Off-diagonal Tinggi** | Sering salah klasifikasi | ❌ Perlu investigasi |
| **Background Row** | False Positive | Model "halusinasi" |
| **Background Column** | False Negative | Model miss objek |

### 3.4 Generate Confusion Matrix

```python
from ultralytics import YOLO

model = YOLO("best.pt")

# Validasi akan generate confusion_matrix.png
metrics = model.val(data="data.yaml", plots=True)

# Lihat di runs/detect/val/confusion_matrix.png
```

---

## 4. Masalah Umum dan Solusinya

### 4.1 Low Recall (Banyak Objek Terlewat)

**Gejala:**
```
Recall: 0.55  ← Rendah!
Artinya 45% objek tidak terdeteksi
```

**Diagnosis:**
```python
# Lihat per-class recall
for i, class_name in enumerate(metrics.names.values()):
    recall = metrics.box.r[i]
    print(f"{class_name}: {recall:.3f}")

# Output:
# car: 0.85
# motorcycle: 0.42  ← Problem!
# bus: 0.78
# truck: 0.68
```

**Penyebab & Solusi:**

| Penyebab | Solusi |
|----------|--------|
| Objek terlalu kecil | Naikkan `imgsz` (640 → 1280) |
| Dataset kurang variasi | Tambah data edge case |
| Confidence threshold tinggi | Turunkan conf saat inference |
| Class imbalance | Tambah data untuk kelas minor |

### 4.2 Low Precision (Banyak False Positive)

**Gejala:**
```
Precision: 0.60  ← Rendah!
Artinya 40% deteksi adalah salah
```

**Diagnosis:**
```python
# Lihat prediksi dengan confidence rendah
results = model.predict("test_images/", conf=0.1)  # Conf rendah

for r in results:
    for box in r.boxes:
        if box.conf < 0.5:
            print(f"Low conf detection: {box.conf:.2f} at {box.xyxy}")
```

**Penyebab & Solusi:**

| Penyebab | Solusi |
|----------|--------|
| Overfit ke pattern | Tambah regularisasi (weight_decay) |
| Kurang negative samples | Tambah gambar tanpa objek |
| Model terlalu besar | Gunakan model lebih kecil |

### 4.3 Class Confusion (Truk Disebut Mobil)

**Gejala:**
```
Confusion matrix menunjukkan:
Truck → Car: 15%
```

**Diagnosis:**
```python
# Lihat kasus class confusion
from collections import defaultdict

confusion_cases = defaultdict(list)

for img_path in val_images:
    results = model.predict(img_path, verbose=False)[0]
    gt_labels = load_labels(img_path)
    
    # Bandingkan
    for pred_box in results.boxes:
        for gt_box in gt_labels:
            if iou(pred_box, gt_box) > 0.5:
                if pred_box.cls != gt_box.cls:
                    confusion_cases[(gt_box.cls, pred_box.cls)].append(img_path)

# Print top confusions
for (gt, pred), images in sorted(confusion_cases.items(), key=lambda x: -len(x[1]))[:5]:
    print(f"{class_names[gt]} → {class_names[pred]}: {len(images)} cases")
```

**Solusi:**

| Penyebab | Solusi |
|----------|--------|
| Definisi kelas ambigu | Pertegas guideline annotator |
| Visual mirip | Tambah data yang kontras |
| Class imbalance | Balance dengan oversample/undersample |

### 4.4 Overfitting

**Gejala:**
```
Train Loss: 0.05  ← Sangat rendah
Val Loss:   0.45  ← Tinggi (gap besar!)
mAP stagnant atau turun setelah epoch tertentu
```

**Diagnosis:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training results
results = pd.read_csv("runs/detect/train/results.csv")

# Plot train vs val loss
plt.figure(figsize=(10, 4))
plt.plot(results['train/box_loss'], label='Train')
plt.plot(results['val/box_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Box Loss')
plt.legend()
plt.title('Overfitting Detection')
plt.savefig('overfitting_check.png')
```

**Solusi:**

| Penyebab | Solusi |
|----------|--------|
| Training terlalu lama | Gunakan early stopping |
| Dataset terlalu kecil | Tambah data atau augmentasi |
| Model terlalu besar | Gunakan model lebih kecil |
| Kurang regularisasi | Naikkan weight_decay |

---

## 5. Teknik Debugging Lanjutan

### 5.1 Per-Condition Analysis

Pisahkan evaluasi berdasarkan kondisi:

```python
import os
from pathlib import Path

# Organisasi validation set berdasarkan kondisi
conditions = {
    'siang': [],
    'malam': [],
    'hujan': [],
    'padat': []
}

# Evaluasi per kondisi
for condition, images in conditions.items():
    results = model.val(data=create_temp_yaml(images))
    print(f"{condition}: mAP50={results.box.map50:.3f}, Recall={results.box.mr:.3f}")

# Output:
# siang: mAP50=0.892, Recall=0.845
# malam: mAP50=0.654, Recall=0.621  ← Problem!
# hujan: mAP50=0.723, Recall=0.689
# padat: mAP50=0.812, Recall=0.778
```

### 5.2 Size-based Analysis

Evaluasi berdasarkan ukuran objek:

```python
def categorize_by_size(box, img_size):
    """Kategorikan objek berdasarkan ukuran"""
    area = (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
    relative_area = area / (img_size[0] * img_size[1])
    
    if relative_area < 0.01:
        return 'small'
    elif relative_area < 0.05:
        return 'medium'
    else:
        return 'large'

# Hitung precision/recall per size
size_metrics = {'small': [], 'medium': [], 'large': []}
# ... accumulate metrics ...

print("Performance by object size:")
print("  Small:  Recall=0.52")  # Biasanya paling rendah
print("  Medium: Recall=0.78")
print("  Large:  Recall=0.91")
```

### 5.3 Error Catalog

Buat katalog kesalahan untuk analisis:

```python
"""
error_catalog.py - Katalog kesalahan untuk analisis
"""
import json
from datetime import datetime

def log_error(image_path: str, error_type: str, details: dict):
    """Log kesalahan ke file JSON"""
    error = {
        "timestamp": datetime.now().isoformat(),
        "image": image_path,
        "type": error_type,
        "details": details
    }
    
    # Append to catalog
    with open("error_catalog.json", "a") as f:
        f.write(json.dumps(error) + "\n")

# Usage
log_error(
    image_path="val/frame_0042.jpg",
    error_type="false_negative",
    details={
        "missed_class": "motorcycle",
        "reason": "occluded by truck",
        "confidence_if_detected": None
    }
)

log_error(
    image_path="val/frame_0089.jpg",
    error_type="class_confusion",
    details={
        "ground_truth": "bus",
        "predicted": "truck",
        "confidence": 0.67
    }
)
```

---

## 6. Active Learning Loop

### 6.1 Konsep

Model improvement adalah proses **iteratif**:

```
┌────────────────────────────────────────────────────────────┐
│                  ACTIVE LEARNING LOOP                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│  │ Train    │ → │ Deploy   │ → │ Collect  │               │
│  │ Model V1 │   │ Evaluate │   │ Failures │               │
│  └──────────┘   └──────────┘   └────┬─────┘               │
│                                      │                     │
│       ┌──────────────────────────────┘                     │
│       ▼                                                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│  │ Train    │ ← │ Add to   │ ← │ Label    │               │
│  │ Model V2 │   │ Dataset  │   │ Errors   │               │
│  └──────────┘   └──────────┘   └──────────┘               │
│       │                                                    │
│       ▼                                                    │
│  Repeat until satisfied                                    │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 6.2 Implementasi

```python
"""
active_learning.py - Active Learning Pipeline
"""
from ultralytics import YOLO
from pathlib import Path
import shutil

def collect_hard_samples(model, new_data_dir: str, output_dir: str, conf_threshold: float = 0.3):
    """
    Kumpulkan sampel yang sulit (low confidence atau missed)
    """
    hard_samples = []
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for img_path in Path(new_data_dir).glob("*.jpg"):
        results = model.predict(str(img_path), verbose=False)[0]
        
        # Check for low confidence detections
        has_low_conf = any(box.conf < conf_threshold for box in results.boxes)
        
        # Check for few/no detections (might be missing objects)
        few_detections = len(results.boxes) < 2
        
        if has_low_conf or few_detections:
            hard_samples.append(img_path)
            shutil.copy(img_path, output_path / img_path.name)
    
    print(f"Found {len(hard_samples)} hard samples")
    print(f"Saved to {output_dir}")
    print("Next: Label these samples and add to training set")
    
    return hard_samples

# Usage
model = YOLO("models/traffic_v1.pt")
hard_samples = collect_hard_samples(
    model=model,
    new_data_dir="new_footage/",
    output_dir="to_label/",
    conf_threshold=0.4
)
```

### 6.3 Workflow Lengkap

```bash
# 1. Deploy model V1
python deploy.py --model models/traffic_v1.pt

# 2. Collect failures (manual atau otomatis)
python active_learning.py --collect

# 3. Label failures (gunakan Roboflow/CVAT)
# Upload to_label/ ke Roboflow, label, download

# 4. Merge dengan dataset lama
python merge_datasets.py \
    --old traffic_dataset_v1/ \
    --new labeled_failures/ \
    --output traffic_dataset_v2/

# 5. Training V2
python train.py \
    --data traffic_dataset_v2/data.yaml \
    --model models/traffic_v1.pt \
    --name traffic_v2

# 6. Compare V1 vs V2
python compare_models.py \
    --model1 models/traffic_v1.pt \
    --model2 models/traffic_v2.pt \
    --test_data test_set/
```

---

## 7. Export dan Deploy

### 7.1 Setelah Puas dengan Model

```python
from ultralytics import YOLO

# Load best model
model = YOLO("runs/detect/traffic_v2/weights/best.pt")

# Validate sekali lagi
metrics = model.val()
print(f"Final mAP50: {metrics.box.map50:.4f}")

# Copy ke folder models/
import shutil
shutil.copy(
    "runs/detect/traffic_v2/weights/best.pt",
    "models/traffic_production.pt"
)

print("Model ready for deployment!")
```

### 7.2 Documentation

```python
# Generate model card
model_info = {
    "name": "Traffic Detection v2",
    "version": "2.0.0",
    "trained_on": "2024-01-15",
    "dataset": "CCTV Traffic Indonesia",
    "classes": ["car", "motorcycle", "bus", "truck"],
    "metrics": {
        "mAP50": 0.87,
        "mAP50-95": 0.62,
        "precision": 0.89,
        "recall": 0.84
    },
    "training": {
        "epochs": 150,
        "imgsz": 640,
        "batch": 16,
        "base_model": "yolov8s.pt"
    },
    "known_limitations": [
        "Performance drops in heavy rain (mAP50 ~0.65)",
        "Small motorcycles (<20px) may be missed",
        "Class confusion possible between minibus and large SUV"
    ]
}

import json
with open("models/traffic_production_card.json", "w") as f:
    json.dump(model_info, f, indent=2)
```

---

## Kesimpulan

| Aspek | Key Point |
|-------|-----------|
| **Evaluasi** | Tidak cukup mAP saja, perlu visual + per-class + per-condition |
| **Confusion Matrix** | Kunci untuk memahami kesalahan klasifikasi |
| **Debugging** | Kategorikan error, analisis pattern |
| **Improvement** | Active learning loop untuk perbaikan berkelanjutan |
| **Documentation** | Dokumentasikan limitations untuk user |

---

## Langkah Selanjutnya

Selamat! Anda telah menyelesaikan **Lab 02: Custom Model Training**.

Model custom Anda sudah siap digunakan. Lanjutkan ke lab berikutnya:

👉 **[Lab 03: Object Detection →](../03_object_detection/)**
