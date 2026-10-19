# Anatomi Video Digital

**Durasi Baca**: 15 Menit | **Level**: Pemula

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan memahami:
- Struktur internal file video (container, codec, frame)
- Perbedaan format video dan kapan menggunakannya
- Konsep FPS dan resolusi serta dampaknya pada traffic analytics
- Protokol streaming untuk IP Camera

---

## 🎬 Pendahuluan

Sebelum kita memproses video dengan OpenCV, penting untuk memahami **apa sebenarnya "video"** itu. Video bukan sekadar kumpulan gambar — ia memiliki struktur kompleks yang melibatkan **kompresi**, **encoding**, dan **protokol transportasi**.

Pemahaman ini krusial karena:
- Codec yang salah bisa membuat CPU Anda 100% bahkan sebelum AI mulai bekerja
- Resolusi yang terlalu rendah membuat objek tidak terdeteksi
- FPS yang tidak sesuai menyebabkan tracking gagal

---

## 1. Arsitektur Video Digital

Video digital terdiri dari **tiga lapisan utama** yang bekerja bersama:

![Diagram Arsitektur Video](images/video_anatomy.png)

### 1.1 Container Layer (Wadah)

**Container** adalah "kotak" yang **membungkus** semua data video, audio, subtitle, dan metadata.

| Container | Ekstensi | Karakteristik | Use Case |
|-----------|----------|---------------|----------|
| **MP4** | `.mp4` | Paling universal, kompatibel hampir semua device | Recording, sharing |
| **AVI** | `.avi` | Format lama Microsoft, ukuran cenderung besar | Legacy systems |
| **MKV** | `.mkv` | Sangat fleksibel, support banyak track | High-quality video |
| **FLV** | `.flv` | Streaming web (Flash era) | Live streaming lama |
| **MOV** | `.mov` | Format Apple QuickTime | macOS, iOS |

> ⚠️ **Penting**: Container **TIDAK** menentukan bagaimana video dikompresi! Container hanya "membungkus" data yang sudah dikompresi oleh codec.

**Analogi**: Bayangkan container seperti **kotak kardus**. Anda bisa memasukkan baju, buku, atau elektronik ke dalam kardus yang sama. Kardus tidak menentukan isi — ia hanya membungkus.

### 1.2 Codec Layer (Coder-Decoder)

**Codec** adalah algoritma yang:
- **Encode**: Mengompresi video mentah menjadi file kecil
- **Decode**: Mengembalikan file terkompresi menjadi frame yang bisa ditampilkan

```
                    ENCODING                      DECODING
Raw Video ──────────────────────> Compressed ──────────────────────> Playable
(Sangat besar)      [Codec]       (File kecil)     [Codec]           (Frame)
```

#### Codec Populer untuk Traffic Analytics

| Codec | Nama Lengkap | Kompresi | CPU Load | Keterangan |
|-------|-------------|----------|----------|------------|
| **H.264** | AVC (Advanced Video Coding) | Bagus | Medium | Standar industri, paling umum |
| **H.265** | HEVC (High Efficiency Video Coding) | Sangat bagus (50% lebih kecil) | Tinggi | IP Camera modern, butuh hardware decode |
| **MJPEG** | Motion JPEG | Rendah (file besar) | Rendah | Setiap frame = JPEG terpisah, mudah di-decode |
| **VP9** | - | Bagus | Medium-Tinggi | YouTube, Google |
| **AV1** | - | Terbaik | Sangat tinggi | Generasi terbaru, belum banyak hardware support |

#### Kapan Memilih Codec?

| Kondisi | Rekomendasi Codec | Alasan |
|---------|-------------------|--------|
| Bandwidth terbatas | **H.265/HEVC** | Kompresi 50% lebih baik dari H.264 |
| CPU terbatas | **MJPEG** | Decode sangat ringan |
| Balanced / Default | **H.264** | Standar industri, kompatibel luas |
| GPU tersedia (NVIDIA) | **H.265 + NVDEC** | Hardware decode, CPU bebas |

> ⚠️ Jika menggunakan H.265, pastikan GPU/hardware mendukung decode!

### 1.3 Raw Frames Layer

Setelah di-decode, video menjadi **sequence of frames** — kumpulan gambar statis yang ditampilkan berurutan.

**Struktur satu frame:**

| Komponen | Penjelasan |
|----------|------------|
| **Dimensi** | Height × Width × Channels (contoh: 1080 × 1920 × 3) |
| **Pixel** | Unit terkecil, posisi (x, y) |
| **Warna** | BGR format: (Blue, Green, Red) |
| **Range** | 0-255 per channel (uint8) |

```
Frame = Matrix 3D
┌─────────────────────────────────┐
│  (0,0)   (1,0)   (2,0)  ...     │  ← Baris 0
│  (0,1)   (1,1)   (2,1)  ...     │  ← Baris 1
│  (0,2)   (1,2)   (2,2)  ...     │  ← Baris 2
│   ...     ...     ...   ...     │
└─────────────────────────────────┘
Setiap cell = [B, G, R] (3 nilai 0-255)
```

**Representasi dalam Python (OpenCV):**
```python
import cv2
import numpy as np

# Baca satu frame dari video
cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()

# Frame adalah NumPy array
print(type(frame))       # <class 'numpy.ndarray'>
print(frame.shape)       # (1080, 1920, 3) = (Height, Width, Channels)
print(frame.dtype)       # uint8 (0-255)

# Akses satu pixel
pixel = frame[540, 960]  # Pixel di tengah
print(pixel)             # [120, 150, 200] = [Blue, Green, Red]
```

---

## 2. Frame Rate (FPS)

**FPS (Frames Per Second)** adalah jumlah frame yang ditampilkan per detik.

![Perbandingan FPS](images/fps_comparison.png)

### 2.1 Standar FPS

| FPS | Nama | Use Case | Keterangan |
|-----|------|----------|------------|
| **24** | Cinema | Film, bioskop | Standar sinematik, "cinematic look" |
| **25** | PAL | TV Eropa, Indonesia | Standar broadcast |
| **30** | NTSC | TV Amerika, CCTV | Standar CCTV paling umum |
| **60** | High Frame Rate | Gaming, sports, slow-mo | Detail gerakan tinggi |

### 2.2 Dampak FPS pada Traffic Analytics

![Dampak FPS pada Tracking](images/fps_tracking_impact.png)


### 2.3 Menghitung Perpindahan Objek

Rumus sederhana untuk estimasi:
```
Perpindahan per frame (pixel) = Kecepatan / FPS

Contoh:
- Mobil bergerak 50 km/h = 13.9 m/s
- Jika 1 meter = 10 pixel (tergantung zoom kamera)
- Kecepatan dalam pixel: 139 px/s

Pada 30 FPS: 139 / 30 = 4.6 pixel per frame  ✅ (mudah di-track)
Pada 5 FPS:  139 / 5  = 27.8 pixel per frame ⚠️ (tracking sulit)
```

---

## 3. Resolusi Video

**Resolusi** adalah jumlah piksel dalam frame (Lebar × Tinggi).

![Perbandingan Resolusi](images/resolution_comparison.png)

### 3.1 Resolusi Umum

| Resolusi | Nama | Total Pixel | Keterangan |
|----------|------|-------------|------------|
| 640×480 | VGA/SD | 307,200 | Rendah, processing cepat |
| 1280×720 | HD/720p | 921,600 | Keseimbangan baik |
| 1920×1080 | Full HD/1080p | 2,073,600 | Detail tinggi |
| 2560×1440 | QHD/2K | 3,686,400 | Profesional |
| 3840×2160 | UHD/4K | 8,294,400 | Sangat detail, berat |

### 3.2 Trade-off Resolusi

| Aspek | Resolusi Tinggi (4K) | Resolusi Rendah (480p) |
|-------|---------------------|------------------------|
| **Detail** | ✅ Sangat tinggi, objek jauh terlihat jelas | ❌ Rendah, detail hilang |
| **Plat Nomor** | ✅ Bisa dibaca | ❌ Tidak terbaca |
| **Small Object** | ✅ Terdeteksi | ❌ Sering miss |
| **Processing** | ❌ Berat, butuh GPU kuat | ✅ Cepat, CPU cukup |
| **Bandwidth** | ❌ Tinggi (~300 Mbps untuk 4K) | ✅ Rendah (~5 Mbps) |
| **Storage** | ❌ Besar (~50 GB/jam) | ✅ Kecil (~2 GB/jam) |

> 🎯 **Sweetspot untuk Traffic Analytics**: **720p atau 1080p** — keseimbangan terbaik antara detail dan performa.


### 3.3 Kalkulasi Kebutuhan Bandwidth

```python
def calculate_bandwidth(width, height, fps, bits_per_pixel=24, compression_ratio=20):
    """
    Estimasi bandwidth video
    
    Args:
        width: Lebar video (pixel)
        height: Tinggi video (pixel)
        fps: Frame per second
        bits_per_pixel: 24 untuk RGB (8 bit x 3 channel)
        compression_ratio: Rasio kompresi (H.264 ~ 20:1)
    
    Returns:
        Bandwidth dalam Mbps
    """
    raw_bitrate = width * height * fps * bits_per_pixel
    compressed_bitrate = raw_bitrate / compression_ratio
    mbps = compressed_bitrate / 1_000_000
    
    return mbps

# Contoh
print(f"720p @ 30fps: {calculate_bandwidth(1280, 720, 30):.1f} Mbps")
print(f"1080p @ 30fps: {calculate_bandwidth(1920, 1080, 30):.1f} Mbps")
print(f"4K @ 30fps: {calculate_bandwidth(3840, 2160, 30):.1f} Mbps")

# Output:
# 720p @ 30fps: 33.2 Mbps
# 1080p @ 30fps: 74.6 Mbps
# 4K @ 30fps: 298.6 Mbps
```

---

## 4. Protokol Transport Video

Bagaimana video sampai ke komputer Anda?

![Protokol Streaming](images/streaming_protocols.png)

### 4.1 File Lokal

```python
# Baca dari file di disk
cap = cv2.VideoCapture("C:/videos/traffic.mp4")
cap = cv2.VideoCapture("./data/sample.avi")
```

**Karakteristik:**
- ✅ Latency: Nol
- ✅ Reliability: Sempurna
- ❌ Live: Tidak bisa

### 4.2 Webcam / USB Camera

```python
# Index 0 = webcam pertama, 1 = kedua, dst.
cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)
```

**Karakteristik:**
- ✅ Latency: Sangat rendah (<50ms)
- ✅ Reliability: Stabil
- ⚠️ Jarak: Terbatas panjang kabel USB

### 4.3 RTSP (Real Time Streaming Protocol)

```python
# Format URL RTSP
rtsp_url = "rtsp://username:password@192.168.1.100:554/stream1"
cap = cv2.VideoCapture(rtsp_url)
```

**Format URL:**
```
rtsp://[username:password@]<IP_ADDRESS>:<PORT>/<PATH>

Contoh:
rtsp://admin:12345@192.168.1.108:554/Streaming/channels/101
rtsp://192.168.1.50/live/ch00_0
```

**Karakteristik:**
- ✅ Jarak: Unlimited (network)
- ✅ Live: Yes
- ⚠️ Latency: 100ms - 5s (tergantung buffer)
- ⚠️ Reliability: Bisa disconnect

### 4.4 HTTP / HLS (HTTP Live Streaming)

```python
# Stream dari web
cap = cv2.VideoCapture("http://example.com/stream.m3u8")
cap = cv2.VideoCapture("https://stream-server.com/live/camera1")
```

**Karakteristik:**
- ✅ Firewall friendly (port 80/443)
- ✅ CDN compatible
- ⚠️ Latency: Tinggi (2-30 detik)

### 4.5 Tabel Perbandingan

| Sumber | Latency | Reliability | Live | Jarak |
|--------|---------|-------------|------|-------|
| File Lokal | 0 | ⭐⭐⭐⭐⭐ | ❌ | N/A |
| Webcam USB | <50ms | ⭐⭐⭐⭐⭐ | ✅ | <5m |
| RTSP | 100ms-5s | ⭐⭐⭐ | ✅ | Unlimited |
| HTTP/HLS | 2-30s | ⭐⭐⭐⭐ | ✅ | Unlimited |

---

## 5. Dampak pada Traffic Analytics

### 5.1 Codec → CPU Load

```python
# ⚠️ Masalah: H.265 stream bisa membuat CPU 100%!
import cv2
import psutil

cap = cv2.VideoCapture("rtsp://192.168.1.100/h265_stream")

while True:
    ret, frame = cap.read()
    
    # Cek CPU usage
    cpu_percent = psutil.cpu_percent()
    print(f"CPU: {cpu_percent}%")  # Bisa 80-100% hanya untuk decode!
    
    # Belum ada AI, tapi CPU sudah habis!
```

**Solusi:**
1. Gunakan GPU decode (CUDA, NVDEC)
2. Request H.264 dari kamera (jika support)
3. Gunakan hardware decoder

### 5.2 Resolusi → Detection Quality

| Resolusi | Ukuran Mobil Jauh | Deteksi YOLO | Keterangan |
|----------|------------------|--------------|------------|
| **480p** | ~10×10 pixel | ❌ Tidak terdeteksi | Terlalu kecil untuk YOLO |
| **720p** | ~20×20 pixel | ⚠️ Kadang terdeteksi | Batas minimum |
| **1080p** | ~30×30 pixel | ✅ Terdeteksi baik | Rekomendasi |
| **4K** | ~60×60 pixel | ✅ Sangat jelas | Untuk objek sangat jauh |

> 💡 **Minimum object size untuk YOLO**: ~20×20 pixel

### 5.3 FPS → Tracking Accuracy

| FPS | Tracking Quality | Counting Accuracy |
|-----|------------------|-------------------|
| 5 | Sering lost, ID switch | ±30% error |
| 15 | Cukup stabil | ±10% error |
| 30 | Stabil | <5% error |
| 60 | Sangat stabil | <2% error |

---

## 6. Checklist Kesiapan Video

Sebelum memulai traffic analytics, pastikan:

- [ ] **Format video** sesuai kebutuhan (H.264 untuk balanced)
- [ ] **Resolusi** minimal 720p untuk general traffic
- [ ] **FPS** minimal 15, idealnya 25-30
- [ ] **Protokol** sudah ditest konektivitasnya (RTSP accessible)
- [ ] **CPU/GPU** mampu decode codec yang digunakan
- [ ] **Bandwidth** cukup untuk streaming

---

## 📝 Quiz Singkat

1. Apa perbedaan container dan codec?
2. Mengapa H.265 lebih berat di-decode daripada MJPEG?
3. Jika kamera Anda 5 FPS dan mobil bergerak cepat, apa yang terjadi pada tracking?
4. Kapan Anda memilih resolusi 4K vs 720p?

<details>
<summary>Lihat Jawaban</summary>

1. **Container** adalah wadah/pembungkus (MP4, AVI), sedangkan **codec** adalah algoritma kompresi (H.264, H.265).
2. H.265 menggunakan algoritma kompresi lebih kompleks (temporal prediction, larger block sizes), sehingga butuh lebih banyak komputasi untuk decode.
3. Mobil akan "melompat" jauh antar frame, tracker kehilangan objek, dan terjadi ID switch.
4. 4K untuk: objek jauh, license plate reading, detail tinggi. 720p untuk: real-time, banyak kamera, resource terbatas.

</details>

---

## 🔗 Artikel Berikutnya

Sekarang Anda memahami struktur video. Selanjutnya, kita akan mempelajari:
→ [02. Cara Kerja OpenCV VideoCapture](02_cara_kerja_videocapture.md)
