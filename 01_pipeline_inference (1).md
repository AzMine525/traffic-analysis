# Memahami Pipeline Inference

**Durasi Baca**: 15 Menit | **Level**: Pemula-Menengah

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan memahami:
- Tiga tahap utama dalam inference (pre-processing, neural network, post-processing)
- Apa yang terjadi di setiap tahap secara detail
- Bagaimana bounding box dihasilkan dari neural network
- Peran NMS (Non-Maximum Suppression) dalam filtering deteksi
- Cara mengimplementasikan pipeline inference secara manual

---

## 🎬 Pendahuluan

Inference adalah proses menjalankan model yang sudah di-training pada data baru. Meskipun kodenya sangat sederhana dengan Ultralytics:

```python
results = model(frame)  # Satu baris saja!
```

Ada **banyak tahapan kompleks** yang terjadi di balik layar. Memahami pipeline ini penting untuk:
- Debugging ketika hasil tidak sesuai ekspektasi
- Optimasi performa
- Custom deployment (tanpa Ultralytics)

---

## 1. Overview Pipeline

### 1.1 Diagram Pipeline

![Pipeline Inference YOLO](assets/inference_pipeline.png)

### 1.2 Tiga Tahap Utama

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                                    │
├────────────────────────┬────────────────────────┬────────────────────────────┤
│    PRE-PROCESSING      │    NEURAL NETWORK      │     POST-PROCESSING        │
│    (Siapkan Input)     │    (Prediksi)          │     (Bersihkan Output)     │
├────────────────────────┼────────────────────────┼────────────────────────────┤
│ • Resize/Letterbox     │ • Backbone (Extract)   │ • Decode coordinates       │
│ • BGR → RGB            │ • Neck (Aggregate)     │ • Confidence filter        │
│ • Normalize (0-1)      │ • Head (Predict)       │ • Non-Max Suppression      │
│ • HWC → CHW            │                        │ • Scale back to original   │
│ • Add batch dimension  │                        │                            │
├────────────────────────┼────────────────────────┼────────────────────────────┤
│ Input: BGR Image       │ Input: (1,3,640,640)   │ Input: (1,84,8400)         │
│ (H,W,3) uint8          │ Output: (1,84,8400)    │ Output: N bounding boxes   │
└────────────────────────┴────────────────────────┴────────────────────────────┘
```

---

## 2. Pre-processing (Persiapan Input)

Neural network sangat "pilih-pilih" tentang format input. Data harus dipersiapkan dengan tepat.

### 2.1 Resize dan Letterbox

Model YOLO memiliki input size tetap (biasanya 640x640). Gambar dengan ukuran berbeda harus di-resize.

**Letterboxing** adalah teknik resize yang menjaga aspect ratio:

```python
import cv2
import numpy as np

def letterbox(img: np.ndarray, new_shape: int = 640) -> tuple:
    """
    Resize gambar dengan letterbox (menjaga aspect ratio).
    
    Args:
        img: Input image (BGR)
        new_shape: Target size (square)
        
    Returns:
        padded: Gambar yang sudah di-letterbox
        ratio: Rasio scaling
        pad: Padding yang ditambahkan (left, top)
    """
    h, w = img.shape[:2]
    
    # Hitung rasio scaling
    ratio = new_shape / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    
    # Resize gambar
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Hitung padding
    dw = new_shape - new_w  # Delta width
    dh = new_shape - new_h  # Delta height
    
    # Center padding
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    
    # Tambahkan padding (warna abu-abu 114)
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    return padded, ratio, (left, top)

# Contoh penggunaan
img = cv2.imread("test.jpg")  # 1920x1080
letterboxed, ratio, pad = letterbox(img, 640)
print(f"Original: {img.shape}")        # (1080, 1920, 3)
print(f"Letterboxed: {letterboxed.shape}")  # (640, 640, 3)
print(f"Ratio: {ratio}, Padding: {pad}")
```

**Visualisasi Letterbox:**

![Letterbox Preprocessing - Resize dan padding untuk menjaga aspect ratio](assets/letterbox_visualization.png)

### 2.2 Color Space Conversion

OpenCV menggunakan BGR, tapi PyTorch/YOLO menggunakan RGB:

```python
# Convert BGR ke RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

### 2.3 Normalization

Pixel values perlu dinormalisasi dari 0-255 ke 0.0-1.0:

```python
# Normalize
img_normalized = img_rgb.astype(np.float32) / 255.0
```

### 2.4 Tensor Conversion

Terakhir, konversi ke format tensor yang dibutuhkan PyTorch:

```python
import torch

# HWC (Height, Width, Channel) → CHW (Channel, Height, Width)
img_chw = np.transpose(img_normalized, (2, 0, 1))

# NumPy → PyTorch Tensor
tensor = torch.from_numpy(img_chw)

# Add batch dimension: (3, 640, 640) → (1, 3, 640, 640)
batch = tensor.unsqueeze(0)

# Move to GPU
batch = batch.to('cuda')

print(f"Final shape: {batch.shape}")  # torch.Size([1, 3, 640, 640])
```

### 2.5 Summary Pre-processing

```python
def preprocess(img_bgr: np.ndarray, size: int = 640) -> torch.Tensor:
    """Complete pre-processing pipeline"""
    # 1. Letterbox
    img_lb, ratio, pad = letterbox(img_bgr, size)
    
    # 2. BGR → RGB
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    
    # 3. Normalize
    img_norm = img_rgb.astype(np.float32) / 255.0
    
    # 4. HWC → CHW
    img_chw = np.transpose(img_norm, (2, 0, 1))
    
    # 5. To tensor + batch dimension
    tensor = torch.from_numpy(img_chw).unsqueeze(0)
    
    return tensor, ratio, pad
```

---

## 3. Neural Network (Forward Pass)

### 3.1 Arsitektur YOLOv8

```
Input: (1, 3, 640, 640)
            │
            ▼
    ┌───────────────────────────────────────┐
    │           BACKBONE (CSPDarknet)       │
    │  Extract visual features              │
    │  • Edges, textures, shapes            │
    │  • Hierarchical representations       │
    └───────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────┐
    │           NECK (FPN + PAN)            │
    │  Aggregate multi-scale features       │
    │  • Small objects (P3: 80x80)          │
    │  • Medium objects (P4: 40x40)         │
    │  • Large objects (P5: 20x20)          │
    └───────────────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────────────┐
    │           HEAD (Detection)            │
    │  Generate predictions                 │
    │  • Bounding box coordinates           │
    │  • Class probabilities                │
    │  • Objectness score                   │
    └───────────────────────────────────────┘
            │
            ▼
Output: (1, 84, 8400)
```

### 3.2 Memahami Output Shape

Output YOLOv8 memiliki shape `(1, 84, 8400)`:

| Dimension | Nilai | Penjelasan |
|-----------|-------|------------|
| **Batch** | 1 | Jumlah gambar dalam batch |
| **Attributes** | 84 | 4 (box coords) + 80 (COCO classes) |
| **Predictions** | 8400 | Total kandidat deteksi |

**Breakdown 8400 Predictions:**
```
Scale     Grid     Predictions
P3        80x80    = 6400    (small objects)
P4        40x40    = 1600    (medium objects)  
P5        20x20    = 400     (large objects)
────────────────────────────
Total             = 8400 candidates
```

### 3.3 Format Attributes (84)

```
Index 0-3:   Box coordinates (x_center, y_center, width, height)
Index 4-83:  Class probabilities (80 COCO classes)

Untuk custom model dengan 4 kelas:
Index 0-3:   Box coordinates
Index 4-7:   Class probabilities (car, motorcycle, bus, truck)
```

---

## 4. Post-processing (Pembersihan Output)

Dari 8400 kandidat, kebanyakan adalah "noise". Kita perlu menyaringnya.

### 4.1 Decode Output

Konversi format internal ke koordinat yang berguna:

```python
def decode_predictions(output: np.ndarray, ratio: float, pad: tuple):
    """
    Decode raw model output ke bounding boxes
    
    Args:
        output: Model output (84, 8400)
        ratio: Letterbox scaling ratio
        pad: Letterbox padding (left, top)
    """
    # Transpose untuk kemudahan akses: (84, 8400) → (8400, 84)
    predictions = output.T
    
    boxes = []
    scores = []
    class_ids = []
    
    for pred in predictions:
        # Box coordinates (normalized)
        x_center, y_center, width, height = pred[:4]
        
        # Class probabilities
        class_probs = pred[4:]
        class_id = np.argmax(class_probs)
        confidence = class_probs[class_id]
        
        if confidence < 0.25:  # Skip low confidence
            continue
        
        # Convert to corner format (x1, y1, x2, y2)
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Scale back to original image
        x1 = (x1 - pad[0]) / ratio
        y1 = (y1 - pad[1]) / ratio
        x2 = (x2 - pad[0]) / ratio
        y2 = (y2 - pad[1]) / ratio
        
        boxes.append([x1, y1, x2, y2])
        scores.append(confidence)
        class_ids.append(class_id)
    
    return np.array(boxes), np.array(scores), np.array(class_ids)
```

### 4.2 Confidence Filter

Buang prediksi dengan skor rendah:

```python
def confidence_filter(boxes, scores, class_ids, threshold=0.25):
    """Filter predictions by confidence score"""
    mask = scores >= threshold
    return boxes[mask], scores[mask], class_ids[mask]
```

**Efek Threshold:**
```
Threshold 0.1:  [████████████████████] 500 detections (banyak noise)
Threshold 0.25: [██████████████]       350 detections (balanced)
Threshold 0.5:  [████████]             200 detections (bersih)
Threshold 0.8:  [████]                  80 detections (miss beberapa)
```

### 4.3 Non-Maximum Suppression (NMS)

Jika ada beberapa box untuk satu objek, pilih yang terbaik.

**Problem & Solution:**

![NMS Process - Menghilangkan duplikasi bounding box](assets/nms_process.png)

**Algoritma NMS:**

```python
import cv2

def nms(boxes, scores, iou_threshold=0.7):
    """
    Non-Maximum Suppression
    
    1. Urutkan box berdasarkan confidence (tertinggi pertama)
    2. Ambil box teratas, simpan
    3. Hitung IoU dengan semua box lain
    4. Hapus box yang IoU-nya > threshold
    5. Ulangi sampai habis
    """
    # Convert to format yang dibutuhkan cv2.dnn.NMSBoxes
    # boxes format: [[x, y, w, h], ...]
    boxes_xywh = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
    
    # OpenCV NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xywh, 
        scores.tolist(), 
        score_threshold=0.25,
        nms_threshold=iou_threshold
    )
    
    return indices.flatten() if len(indices) > 0 else []
```

### 4.4 Complete Post-processing

```python
def postprocess(output, ratio, pad, conf_threshold=0.25, iou_threshold=0.7):
    """Complete post-processing pipeline"""
    # 1. Decode predictions
    boxes, scores, class_ids = decode_predictions(output, ratio, pad)
    
    if len(boxes) == 0:
        return [], [], []
    
    # 2. Confidence filter
    boxes, scores, class_ids = confidence_filter(boxes, scores, class_ids, conf_threshold)
    
    if len(boxes) == 0:
        return [], [], []
    
    # 3. NMS
    indices = nms(boxes, scores, iou_threshold)
    
    return boxes[indices], scores[indices], class_ids[indices]
```

---

## 5. Implementasi Lengkap

### 5.1 Dengan Ultralytics (Recommended)

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")

# Inference (semua pre/post-processing ditangani otomatis)
img = cv2.imread("test.jpg")
results = model(img, conf=0.25, iou=0.7)

# Akses hasil
for box in results[0].boxes:
    # Koordinat
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # Confidence dan class
    confidence = float(box.conf[0])
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    
    # Visualisasi
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{class_name} {confidence:.2f}"
    cv2.putText(img, label, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)
```

### 5.2 Manual Implementation (untuk pembelajaran)

```python
import torch
import cv2
import numpy as np

class YOLOv8Inference:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = torch.load(model_path)['model'].to(device).float().eval()
    
    def __call__(self, img: np.ndarray, conf: float = 0.25, iou: float = 0.7):
        # Pre-process
        tensor, ratio, pad = preprocess(img)
        tensor = tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(tensor)[0]
        
        # Post-process
        output = output.cpu().numpy()[0]
        boxes, scores, class_ids = postprocess(output, ratio, pad, conf, iou)
        
        return boxes, scores, class_ids
```

---

## 6. Tips dan Debugging

### 6.1 Debug Pre-processing

```python
# Simpan hasil letterbox untuk verifikasi
cv2.imwrite("debug_letterbox.jpg", letterboxed)

# Verifikasi shape
print(f"Tensor shape: {tensor.shape}")  # Should be (1, 3, 640, 640)
print(f"Tensor dtype: {tensor.dtype}")  # Should be float32
print(f"Tensor range: {tensor.min():.2f} - {tensor.max():.2f}")  # Should be 0-1
```

### 6.2 Debug Post-processing

```python
# Lihat raw output shape
print(f"Raw output shape: {output.shape}")  # (1, 84, 8400)

# Cek jumlah kandidat sebelum dan sesudah filtering
print(f"Before confidence filter: {len(all_boxes)}")
print(f"After confidence filter: {len(filtered_boxes)}")
print(f"After NMS: {len(final_boxes)}")
```

### 6.3 Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **No detections** | Empty results | Lower confidence threshold |
| **Wrong coordinates** | Boxes offset | Check letterbox padding |
| **Color mismatch** | Model confused | Ensure BGR→RGB conversion |
| **Slow inference** | Low FPS | Check if using GPU |

---

## Kesimpulan

| Stage | Input | Output | Key Operations |
|-------|-------|--------|----------------|
| **Pre-processing** | BGR image | Tensor (1,3,H,W) | Letterbox, normalize, CHW |
| **Neural Network** | Tensor | Raw predictions | Backbone → Neck → Head |
| **Post-processing** | Raw predictions | Final detections | Decode, filter, NMS |

---

## Langkah Selanjutnya

Sekarang Anda paham apa yang terjadi di balik `model(frame)`. Artikel berikutnya membahas parameter yang mengontrol sensitivitas deteksi:

👉 **[Parameter Confidence dan IoU →](02_confidence_dan_iou.md)**
