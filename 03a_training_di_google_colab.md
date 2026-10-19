# Training YOLOv8 di Google Colab

**Durasi Baca**: 15 Menit | **Level**: Pemula-Menengah

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan mampu:
- Setup Google Colab dengan GPU gratis
- Mount Google Drive untuk menyimpan model
- Download dataset dari Roboflow
- Menjalankan training YOLOv8
- Monitor progress dan download hasil

---

## 🎬 Mengapa Google Colab?

| Aspek | Lokal (CPU) | Lokal (GPU) | Google Colab |
|-------|-------------|-------------|--------------|
| **Harga** | Gratis | GPU mahal ($500+) | **Gratis** |
| **Speed** | Sangat lambat | Cepat | **Cepat** |
| **Setup** | Rumit | Rumit (CUDA) | **Mudah** |
| **GPU** | ❌ | Tergantung | ✅ T4/V100 |
| **Limit** | Tidak ada | Tidak ada | 12 jam/session |

> 💡 **Tip**: Untuk training serius, upgrade ke **Colab Pro** ($10/bulan) untuk GPU lebih kencang dan waktu lebih lama.

---

## 1. Setup Google Colab

### 1.1 Buka Colab

1. Buka [colab.research.google.com](https://colab.research.google.com)
2. Login dengan akun Google
3. Klik **"New Notebook"**

### 1.2 Aktifkan GPU

1. Klik **Runtime** > **Change runtime type**
2. Hardware accelerator: **T4 GPU** (atau GPU lain yang tersedia)
3. Klik **Save**

### 1.3 Cek GPU

```python
!nvidia-smi
```

Output yang diharapkan:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   38C    P8     9W /  70W |      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

---

## 2. Mount Google Drive

Simpan model di Drive agar tidak hilang saat session berakhir.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Klik link yang muncul, pilih akun Google, dan izinkan akses.

### Buat Folder Kerja

```python
import os

# Buat folder untuk project
project_path = "/content/drive/MyDrive/traffic-analytics"
os.makedirs(project_path, exist_ok=True)
os.chdir(project_path)

print(f"Working directory: {os.getcwd()}")
```

---

## 3. Install Dependencies

```python
# Install Ultralytics (YOLOv8) dan Roboflow
!pip install ultralytics roboflow -q

# Verifikasi instalasi
import ultralytics
ultralytics.checks()
```

---

## 4. Download Dataset dari Roboflow

### 4.1 Dapatkan API Key

1. Buka [Roboflow](https://roboflow.com) > **Settings** > **API Key**
2. Copy API key Anda

### 4.2 Download Dataset

```python
from roboflow import Roboflow

# Ganti dengan API key dan project Anda
API_KEY = "YOUR_API_KEY"  # ⚠️ Jangan share ini!
WORKSPACE = "your-workspace"
PROJECT = "traffic-detection"
VERSION = 1

# Download
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
dataset = project.version(VERSION).download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")
```

### 4.3 Cek Struktur Dataset

```python
!ls -la {dataset.location}
!cat {dataset.location}/data.yaml
```

---

## 5. Training YOLOv8

### 5.1 Training Dasar

```python
from ultralytics import YOLO

# Load model pre-trained
model = YOLO("yolov8n.pt")  # nano (tercepat untuk testing)

# Training
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU
    project=f"{project_path}/runs",
    name="traffic_v1"
)
```

### 5.2 Parameter Rekomendasi

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `epochs` | 100-150 | Jumlah iterasi |
| `imgsz` | 640 | Resolusi input |
| `batch` | 16 | Sesuaikan dengan VRAM (turunkan jika OOM) |
| `patience` | 50 | Early stopping |
| `lr0` | 0.01 | Learning rate awal |
| `lrf` | 0.01 | Learning rate akhir |

### 5.3 Training dengan Parameter Lengkap

```python
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    
    # Hyperparameters
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    
    # Augmentasi
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    
    # Output
    project=f"{project_path}/runs",
    name="traffic_v1",
    exist_ok=True,
    
    # Performance
    workers=2,
    patience=50,
    save=True,
    save_period=10,
    
    # Logging
    verbose=True
)
```

---

## 6. Monitor Training

### 6.1 Lihat Progress Real-time

Training akan menampilkan output seperti:
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/100      5.2G      1.234      0.876      1.567         45        640
  2/100      5.2G      1.100      0.754      1.456         52        640
...
```

### 6.2 Visualisasi dengan TensorBoard

```python
# Load TensorBoard
%load_ext tensorboard

# Jalankan TensorBoard
%tensorboard --logdir {project_path}/runs
```

### 6.3 Cek Hasil Training

```python
from IPython.display import Image, display

# Lihat hasil training
result_dir = f"{project_path}/runs/traffic_v1"

# Confusion Matrix
display(Image(f"{result_dir}/confusion_matrix.png"))

# Training Curves
display(Image(f"{result_dir}/results.png"))

# Sample Predictions
display(Image(f"{result_dir}/val_batch0_pred.jpg"))
```

---

## 7. Evaluasi Model

### 7.1 Validasi

```python
# Load model terbaik
best_model = YOLO(f"{result_dir}/weights/best.pt")

# Validasi
metrics = best_model.val()

print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
```

### 7.2 Test pada Gambar

```python
# Predict pada gambar test
results = best_model.predict(
    source=f"{dataset.location}/test/images",
    save=True,
    conf=0.5
)
```

### 7.3 Test pada Video

```python
# Upload video test
from google.colab import files
uploaded = files.upload()

# Predict
video_name = list(uploaded.keys())[0]
results = best_model.predict(
    source=video_name,
    save=True,
    conf=0.5
)
```

---

## 8. Download Model

### 8.1 Model sudah di Drive

Karena kita mount Drive, model sudah tersimpan di:
```
Google Drive/traffic-analytics/runs/traffic_v1/weights/
├── best.pt   # Model terbaik
└── last.pt   # Model terakhir
```

### 8.2 Download Langsung

```python
from google.colab import files

# Download model terbaik
files.download(f"{result_dir}/weights/best.pt")
```

### 8.3 Copy ke Folder Project Lokal

Setelah download, pindahkan ke folder `models/`:
```bash
# Di terminal lokal
mv ~/Downloads/best.pt /path/to/traffic-analytics-opencv/models/traffic_v1.pt
```

---

## 9. Tips dan Troubleshooting

### 9.1 Out of Memory (OOM)

```python
# Kurangi batch size
model.train(batch=8)  # atau batch=4

# Atau gunakan model lebih kecil
model = YOLO("yolov8n.pt")  # bukan yolov8m.pt
```

### 9.2 Training Lambat

```python
# Pastikan GPU aktif
!nvidia-smi

# Cek apakah PyTorch menggunakan CUDA
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### 9.3 Session Terputus

- Gunakan **Google Drive** untuk menyimpan progress
- Set `resume=True` untuk melanjutkan training

```python
# Jika training terputus
model = YOLO(f"{project_path}/runs/traffic_v1/weights/last.pt")
model.train(resume=True)
```

### 9.4 Hasil Buruk (mAP Rendah)

1. **Tambah data** - 50 gambar tidak cukup, targetkan 300+
2. **Cek labeling** - Bounding box harus akurat
3. **Tambah epochs** - 100 mungkin belum cukup
4. **Tune hyperparameters** - Coba learning rate berbeda

---

## 10. Notebook Siap Pakai

Untuk kemudahan, gunakan notebook yang sudah saya siapkan:

📓 **[training_yolov8_colab.ipynb](notebooks/training_yolov8_colab.ipynb)**

Notebook ini sudah berisi:
- Setup GPU otomatis
- Mount Drive
- Download dataset dari Roboflow
- Training dengan parameter optimal
- Evaluasi dan visualisasi
- Download model

---

## Kesimpulan

Google Colab adalah solusi gratis terbaik untuk training model deep learning. Dengan integrasi Roboflow, workflow dari labeling hingga model jadi bisa diselesaikan dalam hitungan jam.

**Langkah Selanjutnya**: [Evaluasi dan Debugging Model →](04_evaluasi_dan_debugging.md)
