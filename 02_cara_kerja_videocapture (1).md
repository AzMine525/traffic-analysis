# Cara Kerja OpenCV VideoCapture

**Durasi Baca**: 15 Menit | **Level**: Pemula-Menengah

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan memahami:
- Arsitektur internal `cv2.VideoCapture`
- Backend yang digunakan OpenCV untuk decode video
- Cara membaca properti video (FPS, resolusi, codec)
- Penanganan berbagai sumber video (file, webcam, RTSP)
- Perbedaan BGR vs RGB dan konversi warna

---

## 🎬 Pendahuluan

`cv2.VideoCapture` adalah kelas utama OpenCV untuk membaca video. Meskipun penggunaannya sederhana (`cap.read()`), ada banyak hal yang terjadi di balik layar. Memahami mekanisme internalnya membantu Anda men-debug masalah dan mengoptimalkan performa.

---

## 1. Arsitektur VideoCapture

![Pipeline VideoCapture](images/videocapture_pipeline.png)

### 1.1 Backend System

> 💡 **Fakta Penting**: OpenCV sendiri **TIDAK** bisa decode video! Ia bergantung pada library eksternal.

| Backend | Platform | Use Case | Keterangan |
|---------|----------|----------|------------|
| **FFmpeg** | Cross-platform | File, RTSP, HTTP | Paling umum, sudah termasuk di `opencv-python` |
| **GStreamer** | Linux/Windows | Pipeline kompleks | Sangat fleksibel, populer di embedded |
| **DirectShow** | Windows | Webcam | Native Windows API |
| **V4L2** | Linux | Webcam | Native Linux API |
| **AVFoundation** | macOS/iOS | Webcam | Native Apple API |

### 1.2 Cara Kerja di Balik Layar

Ketika Anda menulis `cap = cv2.VideoCapture("video.mp4")`:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PROSES cap.read()                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. BUKA SUMBER                                                         │
│     cv2.VideoCapture("video.mp4")                                       │
│     └── OpenCV meminta FFmpeg membuka file                             │
│                                                                         │
│  2. BACA DATA TERKOMPRESI                                               │
│     FFmpeg mengambil packet data dari file/stream                       │
│     └── Data masih dalam format H.264/H.265/MJPEG                       │
│                                                                         │
│  3. DECODE KE RAW PIXELS                                                │
│     FFmpeg decode packet menjadi bitmap                                 │
│     └── Hasilnya: array pixel mentah (biasanya YUV)                     │
│                                                                         │
│  4. KONVERSI KE BGR                                                     │
│     OpenCV mengkonversi dari YUV ke BGR                                 │
│     └── Hasilnya: numpy.ndarray shape (H, W, 3)                         │
│                                                                         │
│  5. RETURN KE USER                                                      │
│     ret, frame = cap.read()                                             │
│     └── ret = True/False, frame = NumPy array BGR                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Mengecek Backend yang Digunakan

```python
import cv2

# Lihat backend yang tersedia
print("Backends tersedia:")
backends = [cv2.videoio_registry.getBackendName(b) 
            for b in cv2.videoio_registry.getBackends()]
print(backends)
# Output: ['FFMPEG', 'GSTREAMER', 'INTEL_MFX', 'OPENCV_MJPEG', ...]

# Cek backend yang dipakai untuk video tertentu
cap = cv2.VideoCapture("video.mp4")
backend_id = cap.getBackendName()
print(f"Backend digunakan: {backend_id}")  # Biasanya: FFMPEG
```

---

## 2. Kode Dasar VideoCapture

### 2.1 Template Standar

```python
import cv2
import sys

def main():
    # ========================================
    # 1. INISIALISASI
    # ========================================
    source = "video.mp4"  # Bisa juga: 0, "rtsp://...", dll
    cap = cv2.VideoCapture(source)
    
    # Selalu cek apakah berhasil dibuka!
    if not cap.isOpened():
        print(f"ERROR: Tidak bisa membuka '{source}'")
        sys.exit(1)
    
    # ========================================
    # 2. BACA PROPERTI VIDEO
    # ========================================
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} FPS")
    print(f"Total frames: {total_frames}")
    
    # ========================================
    # 3. LOOP PEMBACAAN
    # ========================================
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        # ret = False berarti video habis atau error
        if not ret:
            print("Video selesai atau terjadi error")
            break
        
        frame_count += 1
        
        # ========================================
        # 4. PROSES FRAME DI SINI
        # ========================================
        # Contoh: tambahkan text
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # ========================================
        # 5. TAMPILKAN (optional)
        # ========================================
        cv2.imshow("Video", frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # ========================================
    # 6. CLEANUP - SANGAT PENTING!
    # ========================================
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Selesai. Total frame diproses: {frame_count}")

if __name__ == "__main__":
    main()
```

### 2.2 Penjelasan Setiap Bagian

| Bagian | Fungsi | Catatan Penting |
|--------|--------|-----------------|
| `VideoCapture(source)` | Buka sumber video | Belum tentu berhasil, harus cek `isOpened()` |
| `isOpened()` | Cek koneksi berhasil | Wajib dicek sebelum melanjutkan |
| `cap.get(PROP)` | Baca properti video | Beberapa properti mungkin return 0 jika tidak tersedia |
| `cap.read()` | Ambil frame berikutnya | Return tuple `(success, frame)` |
| `cap.release()` | Tutup koneksi | **Wajib dipanggil** untuk menghindari resource leak |

---

## 3. Properti Video (CAP_PROP)

![Properti VideoCapture](images/videocapture_properties.png)

### 3.1 Properti Umum (Read-Only)

```python
import cv2

cap = cv2.VideoCapture("video.mp4")

# Properti dasar
props = {
    "Width": cv2.CAP_PROP_FRAME_WIDTH,
    "Height": cv2.CAP_PROP_FRAME_HEIGHT,
    "FPS": cv2.CAP_PROP_FPS,
    "Frame Count": cv2.CAP_PROP_FRAME_COUNT,
    "FOURCC": cv2.CAP_PROP_FOURCC,
    "Format": cv2.CAP_PROP_FORMAT,
}

for name, prop_id in props.items():
    value = cap.get(prop_id)
    print(f"{name}: {value}")
```

### 3.2 Tabel Properti Lengkap

| Properti | Konstanta | Deskripsi | Contoh Nilai |
|----------|-----------|-----------|--------------|
| Lebar Frame | `CAP_PROP_FRAME_WIDTH` | Lebar dalam pixel | 1920 |
| Tinggi Frame | `CAP_PROP_FRAME_HEIGHT` | Tinggi dalam pixel | 1080 |
| FPS | `CAP_PROP_FPS` | Frame per second | 30.0 |
| Total Frame | `CAP_PROP_FRAME_COUNT` | Jumlah frame (file only) | 9000 |
| Codec (FOURCC) | `CAP_PROP_FOURCC` | Kode 4 karakter codec | H264, MJPG |
| Posisi Frame | `CAP_PROP_POS_FRAMES` | Indeks frame saat ini | 150 |
| Posisi Waktu (ms) | `CAP_PROP_POS_MSEC` | Waktu dalam milidetik | 5000.0 |
| Brightness | `CAP_PROP_BRIGHTNESS` | Kecerahan (webcam) | 0.5 |
| Contrast | `CAP_PROP_CONTRAST` | Kontras (webcam) | 0.5 |
| Saturation | `CAP_PROP_SATURATION` | Saturasi (webcam) | 0.5 |

### 3.3 Decode FOURCC Code

```python
def decode_fourcc(fourcc_int):
    """Konversi integer FOURCC ke string 4 karakter"""
    fourcc_int = int(fourcc_int)
    chars = []
    for _ in range(4):
        chars.append(chr(fourcc_int & 0xFF))
        fourcc_int >>= 8
    return ''.join(chars)

cap = cv2.VideoCapture("video.mp4")
fourcc = cap.get(cv2.CAP_PROP_FOURCC)
codec_name = decode_fourcc(fourcc)
print(f"Codec: {codec_name}")  # Contoh: avc1, MJPG, XVID
```

### 3.4 Mengatur Properti (Webcam)

```python
cap = cv2.VideoCapture(0)  # Webcam

# Set resolusi (tidak semua webcam support semua resolusi)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set FPS
cap.set(cv2.CAP_PROP_FPS, 30)

# Set exposure (jika didukung)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode
cap.set(cv2.CAP_PROP_EXPOSURE, -5)

# Verifikasi apakah berhasil
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolusi aktual: {actual_width}x{actual_height}")
```

> ⚠️ **Penting**: `cap.set()` bisa gagal tanpa error! Selalu verifikasi dengan `cap.get()`.

---

## 4. Berbagai Sumber Video

### 4.1 File Video Lokal

```python
# Format yang didukung tergantung backend (FFmpeg mendukung hampir semua)
cap = cv2.VideoCapture("video.mp4")
cap = cv2.VideoCapture("video.avi")
cap = cv2.VideoCapture("video.mkv")
cap = cv2.VideoCapture(r"C:\Videos\traffic.mp4")  # Path absolut
```

### 4.2 Webcam / USB Camera

```python
# Index dimulai dari 0
cap = cv2.VideoCapture(0)  # Kamera pertama
cap = cv2.VideoCapture(1)  # Kamera kedua

# Dengan backend spesifik (Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Dengan backend spesifik (Linux)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
```

### 4.3 RTSP (IP Camera)

```python
# Format URL RTSP
rtsp_url = "rtsp://username:password@192.168.1.100:554/stream1"
cap = cv2.VideoCapture(rtsp_url)

# Dengan parameter timeout (OpenCV 4.x)
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)   # 5 detik timeout
cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
```

### 4.4 HTTP / MJPEG Stream

```python
# MJPEG stream (umum di webcam network)
cap = cv2.VideoCapture("http://192.168.1.100:8080/video")

# HLS stream
cap = cv2.VideoCapture("http://example.com/stream.m3u8")
```

### 4.5 Tabel Perbandingan Sumber

| Sumber | Sintaks | Latency | Reliability | Notes |
|--------|---------|---------|-------------|-------|
| File | `"video.mp4"` | 0 | ⭐⭐⭐⭐⭐ | Paling mudah untuk testing |
| Webcam | `0` atau `1` | <50ms | ⭐⭐⭐⭐⭐ | Butuh izin akses kamera |
| RTSP | `"rtsp://..."` | 100ms-5s | ⭐⭐⭐ | Perlu handle reconnect |
| HTTP | `"http://..."` | 2-30s | ⭐⭐⭐⭐ | Firewall friendly |

---

## 5. Navigasi Video (Seeking)

### 5.1 Loncat ke Frame Tertentu

```python
cap = cv2.VideoCapture("video.mp4")

# Loncat ke frame ke-500
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
ret, frame = cap.read()  # Membaca frame ke-500

# Loncat ke waktu tertentu (dalam milidetik)
cap.set(cv2.CAP_PROP_POS_MSEC, 10000)  # Ke detik ke-10
ret, frame = cap.read()

# Loncat ke posisi relatif (0.0 - 1.0)
cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.5)  # Tengah video
ret, frame = cap.read()
```

### 5.2 Mengetahui Posisi Saat Ini

```python
current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
current_time_sec = current_time_ms / 1000

print(f"Posisi: Frame {current_frame}, Waktu: {current_time_sec:.2f}s")
```

> ⚠️ **Catatan**: Seeking tidak bekerja pada live stream (RTSP, webcam).

---

## 6. Format Warna: BGR vs RGB

### 6.1 Mengapa OpenCV Menggunakan BGR?

OpenCV menggunakan **BGR** karena alasan historis dimana **BGR** ini adalah format yang digunakan oleh library awal Windows dan kamera.

```
┌─────────────────────────────────────────────────────────────┐
│                   URUTAN CHANNEL                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   OpenCV (BGR):    [Blue, Green, Red]  = frame[:,:,0/1/2]  │
│   Matplotlib/PIL:  [Red, Green, Blue]  = frame[:,:,0/1/2]  │
│                                                             │
│   Jika tidak dikonversi:                                    │
│   - Biru tampak merah                                       │
│   - Merah tampak biru                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Konversi Warna

```python
import cv2

cap = cv2.VideoCapture("video.mp4")
ret, frame_bgr = cap.read()

# BGR ke RGB (untuk matplotlib, PIL, Tkinter)
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

# BGR ke Grayscale
frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

# BGR ke HSV (untuk color detection)
frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
```

### 6.3 Contoh Perbedaan BGR vs RGB

```python
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# SALAH: Langsung tampilkan BGR dengan matplotlib
axes[0].imshow(frame)
axes[0].set_title("❌ BGR (warna terbalik)")

# BENAR: Konversi ke RGB dulu
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
axes[1].imshow(frame_rgb)
axes[1].set_title("✅ RGB (warna benar)")

plt.tight_layout()
plt.show()
```

---

## 7. Error Handling Best Practices

### 7.1 Cek Setiap Tahap

```python
import cv2
import sys

def safe_video_capture(source):
    """VideoCapture dengan error handling lengkap"""
    
    # Tahap 1: Buka sumber
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise ValueError(f"Tidak bisa membuka sumber: {source}")
    
    # Tahap 2: Verifikasi properti
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    if width == 0 or height == 0:
        cap.release()
        raise ValueError(f"Video tidak memiliki dimensi valid: {width}x{height}")
    
    # Tahap 3: Coba baca frame pertama
    ret, frame = cap.read()
    
    if not ret or frame is None:
        cap.release()
        raise ValueError("Tidak bisa membaca frame pertama")
    
    # Reset posisi ke awal
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    return cap

# Penggunaan
try:
    cap = safe_video_capture("video.mp4")
    print("Video berhasil dibuka!")
    # ... proses video ...
    cap.release()
    
except ValueError as e:
    print(f"ERROR: {e}")
    sys.exit(1)
```

### 7.2 Tabel Penyebab Error Umum

| Error | Kemungkinan Penyebab | Solusi |
|-------|---------------------|--------|
| `isOpened() = False` | File tidak ada, path salah | Cek path, gunakan path absolut |
| `ret = False` segera | Codec tidak didukung | Install FFmpeg, cek format |
| `ret = False` setelah beberapa frame | Video habis atau stream terputus | Loop ulang atau reconnect |
| Warna tertukar | Lupa konversi BGR ke RGB | Gunakan `cv2.cvtColor()` |
| Video lambat | FPS tidak dihandle | Gunakan `cv2.waitKey(delay)` yang tepat |

---

## 8. Tips Optimasi Performa

### 8.1 Gunakan Threading untuk RTSP

```python
import cv2
import threading
from queue import Queue

class ThreadedVideoCapture:
    def __init__(self, source, queue_size=2):
        self.cap = cv2.VideoCapture(source)
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        
        # Mulai thread pembacaan
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
    
    def _read_frames(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.queue.put(frame)
    
    def read(self):
        return self.queue.get()
    
    def release(self):
        self.stopped = True
        self.cap.release()

# Penggunaan
cap = ThreadedVideoCapture("rtsp://...")
while True:
    frame = cap.read()
    # Proses frame...
```

### 8.2 Skip Frame untuk Performa

```python
cap = cv2.VideoCapture("video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

# Proses hanya 10 FPS (skip frame)
target_fps = 10
skip_ratio = int(fps / target_fps)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Skip frame yang tidak perlu
    if frame_count % skip_ratio != 0:
        continue
    
    # Proses frame di sini...
```

---

## 9. Checklist VideoCapture

Sebelum deploy aplikasi video Anda:

- [ ] Selalu cek `isOpened()` setelah membuat `VideoCapture`
- [ ] Handle `ret = False` di dalam loop
- [ ] Panggil `release()` setelah selesai
- [ ] Konversi BGR ke RGB jika menggunakan matplotlib/PIL
- [ ] Gunakan threading untuk RTSP/stream
- [ ] Set timeout untuk network stream
- [ ] Log properti video untuk debugging

---

## 📝 Quiz Singkat

1. Apa backend default yang digunakan OpenCV untuk decode video?
2. Apa perbedaan BGR dan RGB? Library mana yang menggunakan masing-masing?
3. Mengapa `cap.set()` bisa "gagal diam-diam"?
4. Bagaimana cara loncat ke frame tertentu dalam video?

<details>
<summary>Lihat Jawaban</summary>

1. **FFmpeg** — sudah termasuk dalam paket `opencv-python`.
2. **BGR** = Blue-Green-Red (OpenCV), **RGB** = Red-Green-Blue (Matplotlib, PIL). Urutannya berbeda!
3. Karena tidak semua properti didukung oleh semua device/codec. `cap.set()` return boolean tapi sering diabaikan.
4. Gunakan `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)` atau `cap.set(cv2.CAP_PROP_POS_MSEC, milliseconds)`.

</details>

---

## 🔗 Artikel Berikutnya

Sekarang Anda memahami cara kerja `VideoCapture`. Namun, untuk aplikasi real-time seperti Traffic Analytics, ada masalah besar: **Buffer Latency**. 

Artikel berikutnya membahas:
→ [03. Masalah Latency Buffer dan Solusinya](03_masalah_latency_buffer.md)
