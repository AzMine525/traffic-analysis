# Menggunakan Roboflow untuk Dataset

**Durasi Baca**: 15 Menit | **Level**: Pemula

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan mampu:
- Membuat project di Roboflow
- Upload dan annotate gambar traffic
- Menerapkan augmentasi data otomatis
- Export dataset dalam format YOLOv8
- Download dataset via API untuk training

---

## 🎬 Apa Itu Roboflow?

**Roboflow** adalah platform cloud untuk mengelola dataset Computer Vision. Keunggulannya:

| Fitur | Deskripsi |
|-------|-----------|
| **Labeling Tool** | Annotate langsung di browser |
| **Augmentasi Otomatis** | Flip, rotate, brightness, dll. |
| **Dataset Versioning** | Track perubahan seperti Git |
| **Export Multi-Format** | YOLO, COCO, Pascal VOC, dll. |
| **API Download** | Integrasi langsung ke Colab/script |
| **Free Tier** | 10.000 gambar/bulan gratis! |

> 💡 **Alternatif**: CVAT (self-hosted), LabelImg (desktop), Label Studio

---

## 1. Membuat Akun dan Project

### 1.1 Daftar Akun

1. Buka [roboflow.com](https://roboflow.com)
2. Klik **"Get Started Free"**
3. Daftar dengan Google/GitHub/Email
4. Pilih plan **"Free"** (cukup untuk learning)

### 1.2 Buat Project Baru

1. Klik **"Create New Project"**
2. Isi detail:
   - **Project Name**: `traffic-detection`
   - **Project Type**: `Object Detection`
   - **Annotation Group**: `traffic-analytics` (opsional)
3. Klik **"Create Project"**

---

## 2. Upload Gambar

### 2.1 Cara Mendapatkan Gambar

**Opsi A: Ekstrak dari Video CCTV**
```python
import cv2
import os

cap = cv2.VideoCapture("traffic_video.mp4")
os.makedirs("frames", exist_ok=True)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Ambil 1 frame setiap 30 frame (1 detik jika 30fps)
    if frame_count % 30 == 0:
        cv2.imwrite(f"frames/frame_{frame_count:04d}.jpg", frame)
    
    frame_count += 1

cap.release()
print(f"Extracted {len(os.listdir('frames'))} frames")
```

**Opsi B: Download Dataset Publik**
- [Roboflow Universe](https://universe.roboflow.com/) — Ribuan dataset gratis
- Search: "traffic detection", "vehicle detection"
- Clone langsung ke workspace Anda

### 2.2 Upload ke Roboflow

1. Buka project Anda
2. Klik **"Upload"** > **"Select Files"**
3. Pilih gambar (drag & drop juga bisa)
4. Klik **"Save and Continue"**

> ⚠️ **Tips**: Upload 50-100 gambar dulu untuk testing, tambah lebih banyak setelah model bekerja.

---

## 3. Annotate Gambar

### 3.1 Buat Kelas

1. Klik gambar pertama untuk masuk ke annotation editor
2. Di panel kiri, klik **"+ Add Class"**
3. Tambahkan kelas:
   - `car`
   - `motorcycle`
   - `bus`
   - `truck`

### 3.2 Cara Menggambar Bounding Box

1. Pilih tool **"Bounding Box"** (shortcut: `B`)
2. Klik dan drag di sekitar objek
3. Pilih kelas dari dropdown
4. Tekan **Enter** atau klik di luar untuk konfirmasi

**Shortcut Berguna:**
| Key | Fungsi |
|-----|--------|
| `B` | Bounding box tool |
| `V` | Select/move tool |
| `←` `→` | Gambar sebelum/berikutnya |
| `D` | Hapus annotation terpilih |
| `Ctrl+S` | Save |

### 3.3 Tips Labeling yang Baik

```
✅ DO:
- Box harus pas mengelilingi objek
- Label semua objek (jangan skip yang kecil)
- Konsisten dengan nama kelas

❌ DON'T:
- Box terlalu besar (banyak background)
- Campur bahasa (car vs mobil)
- Skip objek yang terhalang sebagian (label jika >50% terlihat)
```

---

## 4. Generate Dataset Version

### 4.1 Preprocessing

Setelah selesai labeling, klik **"Generate"** untuk membuat versi dataset.

**Preprocessing Options:**
| Option | Rekomendasi | Penjelasan |
|--------|-------------|------------|
| Auto-Orient | ✅ On | Fix rotasi dari metadata EXIF |
| Resize | **640x640** | Match dengan YOLOv8 input |
| Grayscale | ❌ Off | Warna penting untuk klasifikasi |

### 4.2 Augmentasi (Sangat Penting!)

Augmentasi membuat model lebih robust. Rekomendasi untuk traffic:

| Augmentasi | Setting | Alasan |
|------------|---------|--------|
| **Flip Horizontal** | ✅ | Mobil bisa dari kiri/kanan |
| **Flip Vertical** | ❌ | Mobil tidak terbalik |
| **90° Rotate** | ❌ | CCTV tidak rotasi 90° |
| **Brightness** | ±25% | Variasi cahaya siang/malam |
| **Blur** | Up to 2px | Simulasi motion blur |
| **Noise** | Up to 2% | Simulasi kamera low-quality |

> 💡 **Hasil**: Dari 100 gambar asli, augmentasi bisa menghasilkan 300-500 gambar training!

### 4.3 Split Dataset

Roboflow otomatis membagi dataset:
- **Train**: 70% (untuk learning)
- **Valid**: 20% (untuk evaluasi saat training)
- **Test**: 10% (untuk evaluasi akhir)

Klik **"Generate"** dan tunggu proses selesai.

---

## 5. Export Dataset

### 5.1 Export Format

1. Klik **"Versions"** di sidebar
2. Pilih versi yang baru dibuat
3. Klik **"Export Dataset"**
4. Pilih format: **"YOLOv8"**
5. Pilih **"Show download code"**

### 5.2 Download Code (Python)

Roboflow akan memberikan kode seperti ini:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("traffic-detection")
version = project.version(1)
dataset = version.download("yolov8")
```

> ⚠️ **Simpan API Key**: Buka Account Settings > API Key

### 5.3 Struktur Dataset yang Didownload

```
traffic-detection-1/
├── data.yaml           # Konfigurasi untuk YOLO
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## 6. Integrasi dengan Training

### 6.1 Training Lokal

```python
from ultralytics import YOLO
from roboflow import Roboflow

# Download dataset
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("your-workspace").project("traffic-detection")
dataset = project.version(1).download("yolov8")

# Train
model = YOLO("yolov8n.pt")
model.train(data=f"{dataset.location}/data.yaml", epochs=100, imgsz=640)
```

### 6.2 Training di Google Colab

Lihat artikel berikutnya: **[Training di Google Colab](03a_training_di_google_colab.md)**

---

## 7. Best Practices

### 7.1 Iterative Improvement

```
Dataset Kecil (50 gambar)
        ↓
    Train Model
        ↓
    Evaluate (mAP rendah?)
        ↓
    Add More Data (focus pada error cases)
        ↓
    Retrain → Repeat
```

### 7.2 Version Control

- **v1**: Dataset awal (100 gambar)
- **v2**: Tambah augmentasi
- **v3**: Tambah 100 gambar lagi
- **v4**: Fix labeling errors

Roboflow menyimpan semua versi, Anda bisa rollback kapan saja.

### 7.3 Kolaborasi Tim

1. Invite anggota tim ke workspace
2. Assign gambar untuk di-label
3. Review annotations sebelum generate

---

## Kesimpulan

Roboflow mempermudah workflow dataset dari upload hingga training. Fitur augmentasi otomatis sangat membantu meningkatkan performa model dengan data terbatas.

**Langkah Selanjutnya**: [Training di Google Colab →](03a_training_di_google_colab.md)
