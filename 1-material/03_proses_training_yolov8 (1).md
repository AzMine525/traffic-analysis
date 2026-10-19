# Proses Training YOLOv8

**Durasi Baca**: 15 Menit | **Level**: Pemula-Menengah

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan mampu:
- Menjalankan training YOLOv8 dengan Ultralytics
- Memahami apa yang terjadi di balik layar saat training
- Mengatur hyperparameter dengan tepat
- Membaca dan menginterpretasi output training
- Melakukan resume training jika terputus

---

## 🎬 Pendahuluan

Dengan dataset siap, saatnya melatih model. YOLOv8 membuat proses ini sangat sederhana — hanya beberapa baris kode. Namun, memahami apa yang terjadi di balik layar membantu Anda men-debug dan mengoptimasi.

---

## 1. Memulai Training

### 1.1 Instalasi Ultralytics

```bash
# Install package Ultralytics
pip install ultralytics

# Verifikasi instalasi
yolo version
# Output: Ultralytics YOLOv8.x.x ...
```

### 1.2 Pilihan Model

YOLOv8 hadir dalam beberapa ukuran:

| Model | Params | Size | Speed (ms) | mAP50-95 |
|-------|--------|------|------------|----------|
| **YOLOv8n** | 3.2M | 6MB | 2.3 | 37.3 |
| **YOLOv8s** | 11.2M | 22MB | 3.4 | 44.9 |
| **YOLOv8m** | 25.9M | 52MB | 6.3 | 50.2 |
| **YOLOv8l** | 43.7M | 87MB | 9.1 | 52.9 |
| **YOLOv8x** | 68.2M | 137MB | 14.0 | 53.9 |

**Rekomendasi:**
- **Prototyping**: `yolov8n.pt` (tercepat untuk testing)
- **Production**: `yolov8s.pt` atau `yolov8m.pt` (balance speed/accuracy)
- **Akurasi Maksimal**: `yolov8l.pt` atau `yolov8x.pt`

### 1.3 Kode Training Minimal

```python
from ultralytics import YOLO

# Load model pre-trained
model = YOLO("yolov8n.pt")  # Download otomatis jika belum ada

# Mulai training
results = model.train(
    data="path/to/data.yaml",  # Path ke file konfigurasi dataset
    epochs=100,                 # Jumlah iterasi penuh
    imgsz=640,                  # Ukuran input gambar
    batch=16,                   # Batch size (sesuaikan dengan VRAM)
    device=0                    # GPU index (0, 1, 2...) atau 'cpu'
)
```

### 1.4 Training via CLI

Alternatif menggunakan command line:

```bash
# Training dari terminal
yolo detect train \
    data=path/to/data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0
```

---

## 2. Apa yang Terjadi Saat Training?

### 2.1 Training Loop

![Training Loop YOLOv8](assets/training_loop.png)

### 2.2 Penjelasan Detail

```
┌─────────────────────────────────────────────────────────────┐
│                    SATU EPOCH                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌─────────────┐    ┌──────────────┐        │
│  │  Load   │ →  │ Augmentasi  │ →  │  Forward     │        │
│  │  Batch  │    │ (Mosaic,    │    │  Pass        │        │
│  │         │    │  Flip, etc) │    │  (Prediksi)  │        │
│  └─────────┘    └─────────────┘    └──────────────┘        │
│                                           │                 │
│                                           ▼                 │
│  ┌─────────┐    ┌─────────────┐    ┌──────────────┐        │
│  │  Update │ ←  │  Backward   │ ←  │  Calculate   │        │
│  │  Weights│    │  Pass       │    │  Loss        │        │
│  │         │    │  (Gradient) │    │              │        │
│  └─────────┘    └─────────────┘    └──────────────┘        │
│                                                             │
│  ↺ Repeat untuk setiap batch dalam dataset                 │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Setelah epoch selesai:                                     │
│  1. Evaluate pada validation set                            │
│  2. Simpan best.pt jika mAP meningkat                      │
│  3. Log metrics ke tensorboard                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Augmentasi Otomatis

YOLOv8 menerapkan augmentasi secara otomatis:

| Augmentasi | Deskripsi | Default |
|------------|-----------|---------|
| **Mosaic** | Gabung 4 gambar jadi 1 | ✅ On |
| **Mixup** | Blend 2 gambar | ❌ Off |
| **HSV** | Variasi warna (Hue, Saturation, Value) | ✅ On |
| **Flip Horizontal** | Mirror horizontal | ✅ On (50%) |
| **Flip Vertical** | Mirror vertical | ❌ Off |
| **Scale** | Zoom in/out | ✅ On (±50%) |
| **Translate** | Geser posisi | ✅ On (±10%) |

---

## 3. Hyperparameter Penting

### 3.1 Epochs

Berapa kali model melihat **seluruh dataset**.

```
Epochs     Status          Indikator
─────────────────────────────────────────────────
10-20      Underfitting    Loss masih tinggi, mAP <50%
50-100     Normal          Loss stabil, mAP meningkat
100-200    Fine-tuning     mAP naik perlahan
200+       Overfitting?    Val loss naik, train loss turun
```

**Rekomendasi untuk fine-tuning**: 50-150 epochs

### 3.2 Image Size (imgsz)

Resolusi input ke model. Gambar akan di-resize ke ukuran ini.

| Size | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| **320** | Tercepat | Terendah | Objek besar, real-time |
| **640** | Balance | Balance | **Default, recommended** |
| **1280** | 4x lambat | Tertinggi | Objek kecil, high accuracy |

```python
# Untuk objek kecil (motor jauh)
model.train(imgsz=1280, ...)

# Untuk real-time inference
model.train(imgsz=416, ...)
```

### 3.3 Batch Size

Jumlah gambar diproses sekaligus dalam satu iterasi.

| Batch Size | Pro | Con |
|------------|-----|-----|
| **Besar (32-64)** | Training stabil, gradient smooth | Butuh banyak VRAM |
| **Kecil (4-8)** | Hemat VRAM | Noisy gradient, unstable |

**Rule of thumb**: Maksimalkan sampai GPU OOM, lalu turunkan 20%.

```python
# Cek VRAM usage
import torch
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
```

### 3.4 Hyperparameter Lengkap

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    # === Dataset ===
    data="data.yaml",
    
    # === Training Config ===
    epochs=100,
    patience=50,           # Early stopping
    batch=16,
    imgsz=640,
    device=0,              # GPU 0, atau 'cpu', atau [0,1] multi-GPU
    workers=8,             # DataLoader workers
    
    # === Optimizer ===
    optimizer='auto',      # SGD, Adam, AdamW, auto
    lr0=0.01,              # Initial learning rate
    lrf=0.01,              # Final LR = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,     # Warmup epochs
    warmup_momentum=0.8,
    
    # === Augmentation ===
    hsv_h=0.015,           # Hue variation
    hsv_s=0.7,             # Saturation variation
    hsv_v=0.4,             # Value variation
    degrees=0.0,           # Rotation (0 = disabled)
    translate=0.1,         # Translation
    scale=0.5,             # Scale ±50%
    shear=0.0,             # Shear
    perspective=0.0,       # Perspective
    flipud=0.0,            # Vertical flip (0 = disabled)
    fliplr=0.5,            # Horizontal flip
    mosaic=1.0,            # Mosaic probability
    mixup=0.0,             # Mixup probability
    copy_paste=0.0,        # Copy-paste augmentation
    
    # === Output ===
    project="runs/detect",
    name="traffic_v1",
    exist_ok=False,
    save=True,
    save_period=-1,        # Save checkpoint every N epochs (-1=disabled)
    
    # === Validation ===
    val=True,
    split='val',
    
    # === Transfer Learning ===
    freeze=None,           # Freeze N layers (e.g., freeze=10)
    
    # === Logging ===
    verbose=True,
    plots=True,            # Generate training plots
)
```

---

## 4. Memahami Output Training

### 4.1 Live Output

Selama training, Anda akan melihat output seperti:

```
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/100      5.2G      1.892      1.234      1.567         45        640
        2/100      5.2G      1.654      1.089      1.423         52        640
        3/100      5.2G      1.432      0.956      1.312         48        640
        ...
       99/100      5.2G      0.324      0.156      0.289         51        640
      100/100      5.2G      0.312      0.148      0.278         49        640
```

### 4.2 Penjelasan Kolom

| Kolom | Deskripsi | Target |
|-------|-----------|--------|
| **Epoch** | Progress training | - |
| **GPU_mem** | VRAM usage | Stabil |
| **box_loss** | Error lokasi bounding box | ↓ Turun |
| **cls_loss** | Error klasifikasi | ↓ Turun |
| **dfl_loss** | Distribution Focal Loss | ↓ Turun |
| **Instances** | Objek dalam batch | - |
| **Size** | Input image size | - |

### 4.3 Interpretasi Loss

```
Loss Trend yang BAGUS:
─────────────────────────────────
Epoch 1:   ████████████ 1.5
Epoch 25:  ████████     1.0
Epoch 50:  █████        0.6
Epoch 75:  ███          0.4
Epoch 100: ██           0.3

Loss Trend yang BERMASALAH:
─────────────────────────────────
Epoch 1:   ████████████ 1.5
Epoch 25:  █████████    1.2  ← Turun sedikit
Epoch 50:  █████████    1.1  ← Plateau
Epoch 75:  █████████    1.1  ← Tidak improve
→ Kemungkinan: Data kurang, model terlalu kecil
```

### 4.4 Validation Metrics

Setelah setiap epoch:

```
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all        100        500      0.892      0.834      0.871      0.562
                   car        100        350      0.923      0.878      0.912      0.643
            motorcycle        100        100      0.856      0.790      0.823      0.478
                   bus        100         30      0.912      0.833      0.889      0.567
                 truck        100         20      0.878      0.835      0.861      0.523
```

### 4.5 Penjelasan Metrik

| Metrik | Rumus | Interpretasi |
|--------|-------|--------------|
| **Precision (P)** | TP / (TP + FP) | Dari yang diprediksi positif, berapa yang benar? |
| **Recall (R)** | TP / (TP + FN) | Dari yang seharusnya terdeteksi, berapa yang terdeteksi? |
| **mAP50** | Mean AP @ IoU=0.5 | Akurasi dengan threshold IoU 50% |
| **mAP50-95** | Mean AP @ IoU=0.5:0.95 | Akurasi dengan threshold IoU bervariasi |

**Target yang baik untuk Traffic Analytics:**
- ✅ mAP50 > 0.75
- ✅ mAP50-95 > 0.50
- ✅ Semua kelas memiliki recall > 0.70

---

## 5. Early Stopping

Untuk mencegah overfitting dan menghemat waktu:

```python
model.train(
    data="data.yaml",
    epochs=300,           # Set tinggi
    patience=50           # Berhenti jika 50 epoch tidak ada improvement
)
```

**Cara kerja:**
1. Monitor validation mAP setiap epoch
2. Simpan `best.pt` ketika mAP tertinggi
3. Jika 50 epoch berturut-turut tidak ada improvement → STOP
4. Gunakan `best.pt`, bukan `last.pt`

---

## 6. Resume Training

### 6.1 Jika Training Terputus

```python
# Load dari checkpoint terakhir
model = YOLO("runs/detect/traffic_v1/weights/last.pt")

# Lanjutkan training
model.train(resume=True)
```

### 6.2 Melanjutkan dengan Modifikasi

```python
# Load checkpoint
model = YOLO("runs/detect/traffic_v1/weights/last.pt")

# Lanjutkan dengan parameter berbeda (tidak bisa resume=True)
model.train(
    data="data.yaml",
    epochs=50,        # Tambah 50 epoch lagi
    lr0=0.001         # Learning rate lebih kecil
)
```

---

## 7. Hasil Training

### 7.1 Struktur Output

Setelah training selesai, cek folder output:

```
runs/detect/traffic_v1/
├── weights/
│   ├── best.pt          # Model dengan mAP tertinggi ← GUNAKAN INI
│   └── last.pt          # Model dari epoch terakhir
├── args.yaml            # Hyperparameters yang digunakan
├── results.csv          # Metrics per epoch
├── results.png          # Grafik training
├── confusion_matrix.png # Matrix confusion
├── F1_curve.png         # F1 score curve
├── P_curve.png          # Precision curve
├── R_curve.png          # Recall curve
├── PR_curve.png         # Precision-Recall curve
├── labels.jpg           # Visualisasi distribusi label
├── labels_correlogram.jpg
├── train_batch*.jpg     # Sample training batches
└── val_batch*_pred.jpg  # Sample validation predictions
```

### 7.2 Membaca results.png

```python
from IPython.display import Image, display

# Tampilkan grafik training
display(Image("runs/detect/traffic_v1/results.png"))
```

**Yang harus dilihat:**
1. **Loss graphs**: Semua loss harus turun dan stabil
2. **mAP graphs**: mAP harus naik dan plateau
3. **Learning rate**: Harus turun sesuai schedule

### 7.3 Copy Model ke Folder Shared

```python
import shutil

# Copy model terbaik ke folder models/
shutil.copy(
    "runs/detect/traffic_v1/weights/best.pt",
    "models/traffic_v1.pt"
)
print("Model saved to models/traffic_v1.pt")
```

---

## 8. Tips dan Troubleshooting

### 8.1 Out of Memory (OOM)

```python
# Error: CUDA out of memory

# Solusi 1: Kurangi batch size
model.train(batch=8)

# Solusi 2: Kurangi image size
model.train(imgsz=480)

# Solusi 3: Gunakan model lebih kecil
model = YOLO("yolov8n.pt")  # Bukan yolov8m.pt
```

### 8.2 Training Lambat

```python
# Pastikan GPU aktif
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Tingkatkan workers
model.train(workers=8)

# Gunakan mixed precision (otomatis di YOLOv8)
model.train(amp=True)  # Default: True
```

### 8.3 mAP Tidak Naik

```
Kemungkinan penyebab:
1. Data terlalu sedikit → Tambah data atau augmentasi
2. Label quality buruk → Review dan perbaiki label
3. Hyperparameter salah → Coba lr0 berbeda
4. Model terlalu kecil/besar → Coba ukuran lain
5. Epoch kurang → Tambah epochs
```

### 8.4 Overfitting

```
Gejala: Train loss turun, val loss naik

Solusi:
1. Tambah augmentasi
2. Kurangi epochs (gunakan early stopping)
3. Tambah lebih banyak data
4. Gunakan weight decay lebih besar
5. Gunakan model lebih kecil
```

---

## 9. Script Training Lengkap

```python
"""
train.py - Script training YOLOv8 untuk Traffic Analytics
"""
from ultralytics import YOLO
import shutil
from pathlib import Path

# Configuration
CONFIG = {
    "data": "traffic_dataset/data.yaml",
    "model": "yolov8s.pt",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "project": "runs/traffic",
    "name": "v1"
}

def train():
    """Run training with configured parameters"""
    print("🚀 Starting YOLOv8 Training")
    print(f"   Dataset: {CONFIG['data']}")
    print(f"   Model: {CONFIG['model']}")
    print(f"   Epochs: {CONFIG['epochs']}")
    
    # Load model
    model = YOLO(CONFIG["model"])
    
    # Train
    results = model.train(
        data=CONFIG["data"],
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        device=CONFIG["device"],
        project=CONFIG["project"],
        name=CONFIG["name"],
        patience=50,
        verbose=True
    )
    
    # Copy best model to models/
    best_pt = Path(CONFIG["project"]) / CONFIG["name"] / "weights" / "best.pt"
    output_path = Path("models") / f"traffic_{CONFIG['name']}.pt"
    output_path.parent.mkdir(exist_ok=True)
    shutil.copy(best_pt, output_path)
    
    print(f"\n✅ Training complete!")
    print(f"   Best model saved to: {output_path}")
    print(f"   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
    
    return results

if __name__ == "__main__":
    train()
```

---

## Kesimpulan

| Aspek | Key Point |
|-------|-----------|
| **Setup** | Cukup 3-5 baris kode untuk mulai training |
| **Hyperparameters** | epochs, imgsz, batch adalah yang paling penting |
| **Monitoring** | Perhatikan loss turun dan mAP naik |
| **Output** | Gunakan `best.pt`, bukan `last.pt` |
| **Troubleshooting** | OOM? Kurangi batch. mAP rendah? Tambah data. |

---

## Langkah Selanjutnya

Training selesai? Saatnya evaluasi model:

👉 **[Evaluasi dan Debugging Model →](04_evaluasi_dan_debugging.md)**

Atau jika ingin training di cloud:

👉 **[Training di Google Colab →](03a_training_di_google_colab.md)**
