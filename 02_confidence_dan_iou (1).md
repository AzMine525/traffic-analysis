# Parameter Confidence dan IoU

**Durasi Baca**: 15 Menit | **Level**: Pemula-Menengah

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan memahami:
- Apa itu confidence score dan bagaimana threshold mempengaruhi deteksi
- Apa itu IoU (Intersection over Union) dan cara menghitungnya
- Bagaimana kedua parameter ini saling berinteraksi
- Best practices untuk Traffic Analytics
- Cara melakukan tuning parameter optimal

---

## 🎬 Pendahuluan

Dua parameter terpenting saat inference adalah **confidence threshold** dan **IoU threshold**. Mengatur keduanya dengan benar menentukan keseimbangan antara:
- Mendeteksi banyak objek (high recall)
- Menghindari false alarm (high precision)

```python
# Dua parameter kritis
results = model(img, conf=0.25, iou=0.7)
#                      ↑        ↑
#                      │        └─── IoU threshold untuk NMS
#                      └──────────── Confidence threshold
```

---

## 1. Confidence Threshold

### 1.1 Definisi

**Confidence score** adalah probabilitas gabungan bahwa:
1. Ada objek di lokasi tersebut (objectness)
2. Objek tersebut adalah kelas tertentu (class probability)

$$
\text{Confidence} = P(\text{Object}) \times P(\text{Class} | \text{Object})
$$

**Range**: 0.0 (tidak yakin) sampai 1.0 (100% yakin)

**Range**: 0.0 (tidak yakin) sampai 1.0 (100% yakin)

### 1.2 Visualisasi Confidence

![Effect of Confidence Thresholds on Object Detection](assets/confidence_comparison.png)

### 1.3 Efek Threshold

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
img = cv2.imread("traffic.jpg")

# Berbagai threshold
thresholds = [0.1, 0.25, 0.5, 0.8]

for conf in thresholds:
    results = model(img, conf=conf, verbose=False)
    num_detections = len(results[0].boxes)
    print(f"Conf {conf}: {num_detections} detections")

# Output:
# Conf 0.1: 85 detections  (banyak noise)
# Conf 0.25: 52 detections (balanced)
# Conf 0.5: 31 detections  (bersih)
# Conf 0.8: 12 detections  (miss beberapa objek)
```

### 1.4 Trade-off Matrix

| Threshold | Precision | Recall | Karakteristik |
|-----------|-----------|--------|---------------|
| **0.1 - 0.2** | Rendah ↓ | Tinggi ↑ | Banyak noise, tapi jarang miss |
| **0.2 - 0.4** | Seimbang | Seimbang | **Sweet spot untuk tracking** |
| **0.4 - 0.6** | Tinggi ↑ | Sedang | General purpose |
| **0.6 - 0.8** | Sangat tinggi | Rendah ↓ | Hanya deteksi sangat jelas |

### 1.5 Rekomendasi untuk Traffic Analytics

**Recommended: `conf=0.25` sampai `conf=0.35`**

**Alasan:**
```
Traffic Analytics Pipeline:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Detector   │ →   │   Tracker   │ →   │  Counter    │
│  (YOLOv8)   │     │  (ByteTrack)│     │             │
└─────────────┘     └─────────────┘     └─────────────┘
      ↑
      │
Confidence rendah OK!
Tracker akan filter noise.
```

Tracker seperti ByteTrack memiliki mekanisme "smoothing" yang dapat:
- Memperkuat deteksi yang konsisten
- Menghilangkan false positive yang muncul sekali
- Menginterpolasi saat deteksi hilang sementara

---

## 2. IoU (Intersection over Union)

### 2.1 Definisi

**IoU** mengukur seberapa banyak dua bounding box overlap.

![Visualisasi IoU](assets/iou_visualization.png)

### 2.2 Rumus IoU

$$
\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}} = \frac{A \cap B}{A \cup B}
$$

Atau dapat ditulis:

$$
\text{IoU} = \frac{\text{Area}(A \cap B)}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}
$$

### 2.3 Implementasi IoU

```python
def calculate_iou(box1, box2):
    """
    Hitung IoU antara dua bounding box
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU value (0.0 - 1.0)
    """
    # Koordinat intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Cek apakah ada intersection
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    
    # Area intersection
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Area masing-masing box
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Area union
    union = area1 + area2 - intersection
    
    # IoU
    iou = intersection / union
    
    return iou

# Contoh
box_a = [100, 100, 200, 200]  # Ground truth
box_b = [120, 110, 210, 210]  # Prediction

iou = calculate_iou(box_a, box_b)
print(f"IoU: {iou:.2f}")  # IoU: 0.65
```

### 2.4 IoU dalam Context

| Nilai IoU | Interpretasi | Contoh Visual |
|-----------|--------------|---------------|
| **0.0** | Tidak overlap sama sekali | `[A] [B]` |
| **0.1 - 0.3** | Sedikit overlap | `[A [B]` |
| **0.4 - 0.6** | Overlap signifikan | `[A[B]` |
| **0.7 - 0.9** | Hampir sama | `[[AB]` |
| **1.0** | Identik | `[AB]` |

---

## 3. IoU untuk NMS (Non-Maximum Suppression)

### 3.1 Mengapa NMS Butuh IoU?

Model sering mengeluarkan **multiple boxes** untuk satu objek:

```
Tanpa NMS:              Dengan NMS (IoU=0.7):
┌──────────────────┐    ┌──────────────────┐
│  ┌─────────┐     │    │                  │
│  │┌───────┐│     │    │    ┌───────┐     │
│  ││ 🚗   ││     │ →  │    │  🚗  │     │
│  │└───────┘│     │    │    └───────┘     │
│  └─────────┘     │    │                  │
│  3 boxes!        │    │  1 box           │
└──────────────────┘    └──────────────────┘
```

### 3.2 Algoritma NMS

```python
def nms_explained(boxes, scores, iou_threshold=0.7):
    """
    Non-Maximum Suppression step by step
    """
    # 1. Urutkan berdasarkan confidence (tertinggi dulu)
    order = scores.argsort()[::-1]
    
    keep = []
    
    while len(order) > 0:
        # 2. Ambil box dengan confidence tertinggi
        best_idx = order[0]
        keep.append(best_idx)
        
        if len(order) == 1:
            break
        
        # 3. Hitung IoU dengan semua box lain
        remaining = order[1:]
        ious = [calculate_iou(boxes[best_idx], boxes[i]) for i in remaining]
        
        # 4. Hapus box yang IoU-nya > threshold
        #    (dianggap mendeteksi objek yang sama)
        mask = [iou <= iou_threshold for iou in ious]
        order = remaining[mask]
    
    return keep
```

### 3.3 Efek IoU Threshold untuk NMS

```python
# Simulasi efek IoU threshold
iou_thresholds = [0.3, 0.5, 0.7, 0.9]

for iou in iou_thresholds:
    results = model(img, conf=0.25, iou=iou, verbose=False)
    num_boxes = len(results[0].boxes)
    print(f"IoU {iou}: {num_boxes} boxes")

# Output untuk frame kemacetan:
# IoU 0.3: 15 boxes  (agresif merge)
# IoU 0.5: 23 boxes  (moderate merge)
# IoU 0.7: 31 boxes  (konservatif, default)
# IoU 0.9: 42 boxes  (banyak duplikat tersisa)
```

| IoU Threshold | Behavior | Use Case |
|---------------|----------|----------|
| **0.3 - 0.5** | Agresif merge | Objek jarang, ingin bersih |
| **0.5 - 0.7** | Seimbang | **General purpose** |
| **0.7 - 0.9** | Konservatif | Kemacetan padat, objek berdekatan |

### 3.4 Rekomendasi untuk Traffic Analytics

**Untuk lalu lintas normal: `iou=0.7`**
**Untuk kemacetan padat: `iou=0.85`**

```python
def get_iou_threshold(density_estimate: int) -> float:
    """
    Adjust IoU berdasarkan kepadatan traffic
    
    Args:
        density_estimate: Estimasi jumlah kendaraan per frame
    """
    if density_estimate > 30:
        return 0.85  # Padat: hindari merge
    elif density_estimate > 15:
        return 0.75  # Sedang
    else:
        return 0.7   # Normal
```

---

## 4. Interaksi Confidence × IoU

### 4.1 Trade-off Matrix

```
                        Confidence Threshold
                    Low (0.1)         High (0.7)
            ┌───────────────────┬───────────────────┐
            │                   │                   │
IoU    Low  │  Banyak deteksi,  │  Sedikit         │
(0.3)       │  banyak merge     │  deteksi,        │
            │  = Coarse         │  bersih tapi     │
            │                   │  miss banyak     │
            ├───────────────────┼───────────────────┤
            │                   │                   │
      High  │  Banyak deteksi,  │  Sedikit         │
      (0.8) │  terpisah         │  deteksi,        │
            │  = Banyak box     │  sangat bersih   │
            │  per objek        │                   │
            │                   │                   │
            └───────────────────┴───────────────────┘
```

### 4.2 Konfigurasi Umum

| Scenario | Conf | IoU | Rationale |
|----------|------|-----|-----------|
| **Development/Debug** | 0.1 | 0.5 | Lihat semua kandidat |
| **General Traffic** | 0.3 | 0.7 | Balance recall/precision |
| **Heavy Traffic** | 0.25 | 0.85 | Hindari merge, high recall |
| **Night Vision** | 0.2 | 0.7 | Lower conf karena image gelap |
| **Alerting System** | 0.6 | 0.5 | Minimize false alarm |

---

## 5. Eksperimen untuk Menemukan Nilai Optimal

### 5.1 Grid Search

```python
from ultralytics import YOLO
import cv2
import itertools

def grid_search_thresholds(model_path: str, image_paths: list):
    """
    Grid search untuk menemukan threshold optimal
    """
    model = YOLO(model_path)
    
    conf_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    iou_values = [0.3, 0.5, 0.7, 0.9]
    
    results_table = []
    
    for conf, iou in itertools.product(conf_values, iou_values):
        total_detections = 0
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            results = model(img, conf=conf, iou=iou, verbose=False)
            total_detections += len(results[0].boxes)
        
        avg_detections = total_detections / len(image_paths)
        results_table.append({
            'conf': conf,
            'iou': iou,
            'avg_detections': avg_detections
        })
        print(f"conf={conf}, iou={iou} → avg {avg_detections:.1f} detections")
    
    return results_table

# Jalankan pada gambar representatif
test_images = [
    "samples/traffic_day.jpg",
    "samples/traffic_night.jpg",
    "samples/traffic_crowded.jpg",
    "samples/traffic_empty.jpg"
]

results = grid_search_thresholds("best.pt", test_images)
```

### 5.2 Visualisasi Perbandingan

```python
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

def visualize_threshold_comparison(img_path: str, model_path: str):
    """
    Visualisasi side-by-side berbagai threshold
    """
    model = YOLO(model_path)
    img = cv2.imread(img_path)
    
    configs = [
        {'conf': 0.1, 'iou': 0.5, 'label': 'Low conf, Low IoU'},
        {'conf': 0.3, 'iou': 0.7, 'label': 'Balanced'},
        {'conf': 0.5, 'iou': 0.7, 'label': 'High conf'},
        {'conf': 0.3, 'iou': 0.9, 'label': 'High IoU'}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, config in zip(axes, configs):
        results = model(img, conf=config['conf'], iou=config['iou'], verbose=False)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        ax.imshow(annotated_rgb)
        ax.set_title(f"{config['label']}\nconf={config['conf']}, iou={config['iou']}\n{len(results[0].boxes)} detections")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('threshold_comparison.png')
    plt.show()

visualize_threshold_comparison("traffic_sample.jpg", "best.pt")
```

---

## 6. Dynamic Thresholds

### 6.1 Adaptasi Berdasarkan Kondisi

Dalam sistem production, threshold bisa dinamis:

```python
class AdaptiveThresholds:
    """
    Adjust thresholds berdasarkan kondisi
    """
    
    def __init__(self):
        self.base_conf = 0.3
        self.base_iou = 0.7
    
    def get_thresholds(self, frame_info: dict) -> tuple:
        """
        Args:
            frame_info: {
                'brightness': 0-255,
                'hour': 0-23,
                'estimated_density': int
            }
        """
        conf = self.base_conf
        iou = self.base_iou
        
        # Adjust for lighting
        if frame_info.get('brightness', 128) < 50:
            # Malam/gelap: turunkan threshold
            conf = max(0.15, conf - 0.1)
        
        # Adjust for time of day
        hour = frame_info.get('hour', 12)
        if 0 <= hour < 6:  # Dini hari
            conf = max(0.2, conf - 0.05)
        
        # Adjust for density
        density = frame_info.get('estimated_density', 10)
        if density > 25:
            # Padat: naikkan IoU untuk hindari merge
            iou = min(0.9, iou + 0.1)
        
        return conf, iou

# Usage
adaptive = AdaptiveThresholds()

# Frame malam, padat
conf, iou = adaptive.get_thresholds({
    'brightness': 40,
    'hour': 22,
    'estimated_density': 35
})
print(f"conf={conf}, iou={iou}")  # conf=0.2, iou=0.8
```

### 6.2 Feedback Loop dengan Tracker

```python
def adaptive_detection(model, tracker, frame, prev_count):
    """
    Adjust threshold berdasarkan tracking quality
    """
    # Default thresholds
    conf = 0.3
    iou = 0.7
    
    # Jika banyak track hilang, turunkan confidence
    if hasattr(tracker, 'lost_tracks') and len(tracker.lost_tracks) > 5:
        conf = 0.2  # Coba tangkap lebih banyak deteksi
    
    # Jika terlalu banyak new tracks, naikkan confidence
    if hasattr(tracker, 'new_tracks') and len(tracker.new_tracks) > 10:
        conf = 0.4  # Kurangi false positive
    
    results = model(frame, conf=conf, iou=iou, verbose=False)
    return results
```

---

## 7. Best Practices Summary

### 7.1 Cheat Sheet

```python
# === TRAFFIC ANALYTICS DEFAULTS ===
CONF_DEFAULT = 0.3
IOU_DEFAULT = 0.7

# === CONDITION-SPECIFIC ===
# Siang cerah
conf_day = 0.35
iou_day = 0.7

# Malam
conf_night = 0.2
iou_night = 0.7

# Hujan/kabut
conf_rain = 0.2
iou_rain = 0.7

# Kemacetan padat
conf_heavy = 0.25
iou_heavy = 0.85

# Lalu lintas sepi
conf_light = 0.4
iou_light = 0.5
```

### 7.2 Quick Decision Tree

```
Banyak false positive?
├── Ya → Naikkan confidence (0.3 → 0.4)
└── Tidak
    └── Banyak miss detection?
        ├── Ya → Turunkan confidence (0.3 → 0.2)
        └── Tidak
            └── Box duplicate/merge?
                ├── Merge → Turunkan IoU (0.7 → 0.5)
                └── Duplicate → Naikkan IoU (0.7 → 0.85)
```

---

## Kesimpulan

| Parameter | Fungsi | Traffic Default | Range |
|-----------|--------|-----------------|-------|
| **Confidence** | Filter prediksi low-quality | 0.25 - 0.35 | 0.0 - 1.0 |
| **IoU** | Threshold untuk NMS | 0.70 | 0.0 - 1.0 |

**Key Insight**: Tidak ada nilai "benar" universal. Parameter harus disesuaikan dengan:
- Kondisi kamera (posisi, kualitas, lighting)
- Karakteristik traffic (kepadatan, jenis kendaraan)
- Kebutuhan aplikasi (counting accuracy vs minimal false alarm)

---

## Langkah Selanjutnya

Sekarang Anda paham cara mengontrol sensitivitas deteksi. Artikel berikutnya membahas optimasi untuk performa real-time:

👉 **[Filtering Kelas dan Optimisasi Performa →](03_filtering_dan_optimisasi.md)**
