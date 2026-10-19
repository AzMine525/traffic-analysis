# Mengapa Model Pre-trained Tidak Cukup

**Durasi Baca**: 12 Menit | **Level**: Pemula

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan memahami:
- Apa itu dataset COCO dan karakteristiknya
- Konsep **Domain Gap** dan dampaknya pada performa model
- Mengapa fine-tuning adalah keharusan untuk Traffic Analytics
- Kapan harus training custom vs. menggunakan pre-trained

---

## 🎬 Pendahuluan

YOLOv8 sudah di-training pada dataset COCO yang berisi 80 kelas objek termasuk `car`, `bus`, `truck`, dan `motorcycle`. 

**Pertanyaan**: Jika model sudah bisa mendeteksi mobil, mengapa kita perlu melatih ulang?

**Jawaban**: Karena ada **Domain Gap** — perbedaan besar antara kondisi saat training vs. kondisi nyata di CCTV Anda.

---

## 1. Apa Itu Dataset COCO?

**COCO** (Common Objects in Context) adalah dataset raksasa yang menjadi standar benchmark Computer Vision.

### 1.1 Statistik COCO

| Metrik | Jumlah |
|--------|--------|
| Total Gambar | **330.000+** |
| Kategori Objek | **80 kelas** |
| Instance per Gambar | ~7 objek |
| Sumber | Flickr, internet |

### 1.2 80 Kelas COCO

```python
# 80 kelas dalam COCO dataset
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Kelas relevan untuk Traffic Analytics
TRAFFIC_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'bicycle', 'person']
```

### 1.3 Karakteristik Gambar COCO

Gambar COCO memiliki karakteristik khas:

| Aspek | Deskripsi |
|-------|-----------|
| **Fotografer** | Manusia dengan kamera tangan |
| **Sudut Pandang** | Eye-level (sejajar mata) |
| **Jarak Objek** | Dekat, objek besar dan jelas |
| **Pencahayaan** | Bagus (foto siang hari) |
| **Kualitas** | HD/4K, fokus tajam |
| **Komposisi** | Objek biasanya terpusat |

---

## 2. Karakteristik CCTV Traffic

Sekarang bandingkan dengan kondisi kamera CCTV lalu lintas di Indonesia:

### 2.1 Perbandingan COCO vs CCTV

| Aspek | COCO Dataset | CCTV Traffic |
|-------|--------------|--------------|
| **Sudut Pandang** | Eye-level (0-15°) | Top-down (45-90°) |
| **Jarak Objek** | 2-10 meter | 20-100 meter |
| **Ukuran Objek** | 100x100 piksel+ | 20x20 piksel |
| **Pencahayaan** | Bagus | Bervariasi (malam, hujan) |
| **Kualitas** | HD/4K | 720p atau lebih rendah |
| **Lensa** | Normal | Wide-angle/fisheye |
| **Motion Blur** | Jarang | Sering (kendaraan cepat) |

### 2.2 Visualisasi Domain Gap

![Domain Gap antara COCO dan CCTV Traffic](assets/domain_gap.png)

### 2.3 Kondisi Ekstrem di Indonesia

```
☀️ Siang Terik     → Overexposure, siluet gelap
🌧️ Hujan Deras     → Blur, pantulan di aspal
🌙 Malam           → Noise tinggi, headlight glare
🌫️ Kabut/Asap     → Visibility rendah
🏍️ Motor Banyak   → Occlusion, overlap
```

---

## 3. Domain Gap: Masalah Utama

### 3.1 Definisi

> **Domain Gap** adalah perbedaan distribusi statistik antara data yang digunakan untuk training model dan data yang ditemui saat deployment.

### 3.2 Ilustrasi Sederhana

```
Training (COCO):
┌─────────────────────┐
│   🚗  ← Mobil       │  Model belajar:
│   Tampak samping    │  "Mobil = kotak dengan roda di samping"
│   Besar, jelas      │
└─────────────────────┘

Deployment (CCTV):
┌─────────────────────┐
│        ▪️           │  Model melihat:
│     ← Mobil?        │  "Titik kecil dari atas = ???"
│   Tampak dari atas  │
└─────────────────────┘

Hasil: Model "bingung" → Confidence turun drastis
```

### 3.3 Mengapa Model Gagal?

Neural network belajar dengan mengenali **pola visual**:

1. **Bentuk**: Mobil dari samping = persegi panjang dengan roda
2. **Tekstur**: Kaca depan reflektif, bodi mengkilap
3. **Proporsi**: Lebar > tinggi

Saat melihat mobil dari atas:
- Bentuk berubah menjadi kotak/elips
- Tidak ada roda yang terlihat
- Proporsi berbeda

**Model tidak pernah melihat pola ini saat training**, sehingga gagal mengenali.

---

## 4. Bukti Empiris

### 4.1 Eksperimen: Pre-trained vs Custom

Mari buktikan dengan kode:

```python
from ultralytics import YOLO
import cv2

# Load model pre-trained COCO
model = YOLO("yolov8n.pt")

# Test pada gambar eye-level (seperti COCO)
results_eyelevel = model("test_images/car_eyelevel.jpg")
print(f"Eye-level confidence: {results_eyelevel[0].boxes.conf.mean():.2f}")
# Output: Eye-level confidence: 0.92

# Test pada gambar CCTV top-down
results_cctv = model("test_images/car_cctv_topdown.jpg")
print(f"CCTV top-down confidence: {results_cctv[0].boxes.conf.mean():.2f}")
# Output: CCTV top-down confidence: 0.34  ← Drastis turun!
```

### 4.2 Hasil Eksperimen

| Skenario | mAP50 | Deteksi Rate |
|----------|-------|--------------|
| Video YouTube HD | ~90% | 98% |
| CCTV Siang Cerah | ~55% | 75% |
| CCTV Malam | ~35% | 50% |
| CCTV Hujan | ~25% | 40% |

**Kesimpulan**: Perbedaan hingga **65% accuracy** hanya karena domain berbeda!

### 4.3 Visualisasi Kegagalan

```
Pre-trained pada CCTV Traffic:

Frame 1: 🚗 Detected (0.45) - Low confidence
Frame 2: 🚗 Missed!         - Tidak terdeteksi  
Frame 3: 🚗 Detected (0.38) - Low confidence
Frame 4: 🚗 Detected (0.52) - Flickering detection

Masalah:
- ID tidak stabil (tracking gagal)
- Counting jadi tidak akurat
- False positive meningkat
```

---

## 5. Solusi: Transfer Learning

### 5.1 Konsep Transfer Learning

Daripada training dari nol (butuh jutaan gambar), kita **memanfaatkan pengetahuan** yang sudah dipelajari model pre-trained.

```
┌─────────────────────────────────────────────┐
│           YOLOv8 Architecture               │
├─────────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐    │
│  │         BACKBONE (CSPDarknet)       │    │ ← FREEZE
│  │  Sudah belajar: edges, textures,    │    │   (Pertahankan)
│  │  shapes, general features           │    │
│  └─────────────────────────────────────┘    │
│                    │                        │
│                    ▼                        │
│  ┌─────────────────────────────────────┐    │
│  │         NECK (FPN + PAN)            │    │ ← FINE-TUNE
│  │  Feature aggregation                │    │   (Sedikit adjust)
│  └─────────────────────────────────────┘    │
│                    │                        │
│                    ▼                        │
│  ┌─────────────────────────────────────┐    │
│  │         HEAD (Detection)            │    │ ← FINE-TUNE
│  │  Belajar: "mobil dari atas = mobil" │    │   (Training utama)
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

### 5.2 Keuntungan Transfer Learning

| Aspek | Training dari Nol | Transfer Learning |
|-------|-------------------|-------------------|
| **Data Dibutuhkan** | 10.000+ gambar | 50-300 gambar |
| **Waktu Training** | Hari-minggu | Jam |
| **GPU Resource** | Mahal | Terjangkau |
| **Hasil** | Tidak pasti | Lebih stabil |

### 5.3 Kode Fine-tuning

```python
from ultralytics import YOLO

# 1. Load model pre-trained (pengetahuan COCO)
model = YOLO("yolov8n.pt")

# 2. Fine-tune pada dataset CCTV Anda
results = model.train(
    data="cctv_dataset/data.yaml",  # Dataset kamera Anda
    epochs=100,
    imgsz=640,
    batch=16,
    
    # Transfer learning settings
    freeze=10,      # Freeze 10 layer pertama (backbone)
    lr0=0.001,      # Learning rate lebih kecil (fine-tune)
)

# 3. Model sekarang "paham" mobil dari sudut CCTV
```

---

## 6. Kapan Harus Training Custom?

### 6.1 WAJIB Training Custom

✅ **Sudut kamera sangat berbeda**
- CCTV top-down (45-90°)
- Drone footage
- Kamera di tiang tinggi

✅ **Ada objek baru**
- Becak, bajaj (tidak ada di COCO)
- Truk trailer khusus Indonesia
- Jenis motor tertentu

✅ **Lingkungan ekstrem**
- Malam tanpa penerangan
- Hujan deras
- Kabut/asap
- Backlight (matahari terbit/terbenam)

✅ **Akurasi kritis**
- Sistem pembayaran otomatis (e-toll)
- Penegakan hukum (e-tilang)
- Counting untuk traffic management

### 6.2 Bisa Pakai Pre-trained

❌ **Cukup pre-trained jika:**
- Kamera eye-level (dashcam, roadside camera setinggi mata)
- Jarak dekat ke objek
- Pencahayaan bagus dan konsisten
- Hanya butuh deteksi kasar (bukan counting akurat)
- Prototipe/demo cepat

---

## 7. Berapa Banyak Data yang Dibutuhkan?

### 7.1 Panduan Jumlah Data

| Kompleksitas | Jumlah Gambar | Keterangan |
|--------------|---------------|------------|
| **Minimal** | 50-100 | Proof of concept |
| **Cukup** | 200-500 | Akurasi reasonable |
| **Bagus** | 500-1000 | Produksi |
| **Excellent** | 1000+ | Mission-critical |

### 7.2 Tips Mengumpulkan Data

```python
# Script untuk ekstrak frame dari video CCTV
import cv2
import os

def extract_diverse_frames(video_path, output_dir, interval=30):
    """
    Ekstrak frame dengan interval tertentu.
    interval=30 berarti 1 frame per detik (untuk 30fps video)
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            filename = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Saved {saved_count} frames to {output_dir}")
    return saved_count

# Ekstrak dari berbagai kondisi
extract_diverse_frames("cctv_siang.mp4", "dataset/siang", interval=60)
extract_diverse_frames("cctv_malam.mp4", "dataset/malam", interval=60)
extract_diverse_frames("cctv_hujan.mp4", "dataset/hujan", interval=60)
```

### 7.3 Prioritas Data

```
Prioritas pengumpulan data:

1. VARIASI PENCAHAYAAN (paling penting)
   - Siang cerah
   - Siang mendung
   - Sore (golden hour)
   - Malam dengan lampu
   - Malam minim lampu

2. VARIASI CUACA
   - Cerah
   - Hujan ringan
   - Hujan deras
   - Berkabut

3. VARIASI TRAFFIC
   - Sepi
   - Normal
   - Padat/macet
   - Dengan kejadian khusus (kecelakaan, dll)

4. VARIASI OBJEK
   - Semua jenis kendaraan yang ingin dideteksi
   - Berbagai ukuran (motor kecil vs truk besar)
   - Berbagai warna
```

---

## 8. Demo: Perbandingan Langsung

### 8.1 Script Perbandingan

```python
from ultralytics import YOLO
import cv2

def compare_models(image_path):
    """
    Bandingkan hasil pre-trained vs custom model
    """
    # Load models
    pretrained = YOLO("yolov8n.pt")
    custom = YOLO("models/traffic_custom.pt")  # Model setelah fine-tune
    
    # Run inference
    img = cv2.imread(image_path)
    
    results_pre = pretrained(img, verbose=False)[0]
    results_cust = custom(img, verbose=False)[0]
    
    # Count detections
    pre_count = len(results_pre.boxes)
    cust_count = len(results_cust.boxes)
    
    # Average confidence
    pre_conf = results_pre.boxes.conf.mean().item() if pre_count > 0 else 0
    cust_conf = results_cust.boxes.conf.mean().item() if cust_count > 0 else 0
    
    print(f"\n📊 Comparison on: {image_path}")
    print(f"{'Metric':<20} {'Pre-trained':<15} {'Custom':<15}")
    print("-" * 50)
    print(f"{'Detections':<20} {pre_count:<15} {cust_count:<15}")
    print(f"{'Avg Confidence':<20} {pre_conf:.2f}{'':<13} {cust_conf:.2f}")
    
    return pre_count, cust_count, pre_conf, cust_conf

# Test
compare_models("test_images/cctv_frame_001.jpg")
```

### 8.2 Output yang Diharapkan

```
📊 Comparison on: test_images/cctv_frame_001.jpg
Metric               Pre-trained     Custom         
--------------------------------------------------
Detections           3               8              
Avg Confidence       0.42            0.87           

↑ Custom model mendeteksi lebih banyak dengan confidence lebih tinggi
```

---

## Kesimpulan

| Poin | Penjelasan |
|------|------------|
| **Pre-trained ≠ Solusi Akhir** | Model COCO didesain untuk foto biasa, bukan CCTV |
| **Domain Gap Nyata** | Perbedaan sudut pandang dapat menurunkan akurasi 50%+ |
| **Transfer Learning Efisien** | Hanya butuh 50-300 gambar untuk fine-tune |
| **Data Berkualitas > Kuantitas** | 100 gambar beragam > 1000 gambar mirip |

---

## Langkah Selanjutnya

Sekarang Anda paham mengapa training custom diperlukan. Artikel berikutnya membahas cara menyiapkan dataset:

👉 **[Struktur Dataset YOLO →](02_struktur_dataset_yolo.md)**
