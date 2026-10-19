# Filtering Kelas dan Optimisasi Performa

**Durasi Baca**: 15 Menit | **Level**: Menengah

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan mampu:
- Memfilter hanya kelas yang relevan untuk Traffic Analytics
- Mengoptimalkan inference untuk mencapai real-time performance
- Memilih model size yang tepat untuk hardware Anda
- Mengukur dan monitoring FPS
- Menerapkan teknik optimasi lanjutan

---

## 🎬 Pendahuluan

Untuk Traffic Analytics, kita biasanya hanya peduli pada **kendaraan** — bukan orang, anjing, atau payung. Memfilter kelas yang tidak relevan tidak hanya memperbersih output, tapi juga meningkatkan performa keseluruhan sistem.

```python
# Tanpa filtering: 50+ detections (termasuk pejalan kaki, tas, dll)
results = model(frame)

# Dengan filtering: hanya kendaraan
results = model(frame, classes=[2, 3, 5, 7])  # car, motorcycle, bus, truck
```

---

## 1. Class Filtering

### 1.1 COCO Class IDs untuk Traffic

Model YOLOv8 pre-trained (COCO) mendeteksi 80 kelas. Yang relevan untuk traffic:

```python
# COCO class IDs untuk kendaraan
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    1: 'bicycle',  # opsional
}

# Class IDs yang biasanya TIDAK dibutuhkan
IRRELEVANT_CLASSES = {
    0: 'person',     # Banyak di pinggir jalan
    16: 'dog',       # Kadang muncul
    25: 'umbrella',  # Sering false positive
    # ... dan 70+ kelas lainnya
}
```

### 1.2 Filter di Ultralytics

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Hanya deteksi kelas tertentu (by ID)
results = model(
    frame,
    classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
    verbose=False
)

# Cara alternatif: by name
results = model(
    frame,
    classes=['car', 'motorcycle', 'bus', 'truck']
)
```

> ⚠️ **Catatan**: YOLO tetap menjalankan inference penuh. Filter terjadi di post-processing. Speedup minimal, tapi output lebih bersih.

### 1.3 Filter Manual dengan Supervision

```python
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model(frame, verbose=False)

# Convert ke Supervision Detections
detections = sv.Detections.from_ultralytics(results[0])

# Filter manual
vehicle_class_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck
mask = [c in vehicle_class_ids for c in detections.class_id]
filtered_detections = detections[mask]

print(f"Before filter: {len(detections)}")
print(f"After filter: {len(filtered_detections)}")
```

### 1.4 Custom Model: Built-in Filtering

Jika menggunakan model custom (hanya ditraining pada kendaraan), filtering tidak diperlukan:

```python
# Model custom hanya punya 4 kelas
# 0: car, 1: motorcycle, 2: bus, 3: truck
custom_model = YOLO("traffic_custom.pt")

# Tidak perlu parameter classes
results = custom_model(frame)  # Output sudah bersih
```

---

## 2. Memilih Model yang Tepat

### 2.1 Perbandingan Model YOLOv8

| Model | Params | Size | GPU Speed | CPU Speed | mAP50 |
|-------|--------|------|-----------|-----------|-------|
| **YOLOv8n** | 3.2M | 6MB | ~2.3ms | ~100ms | 37.3 |
| **YOLOv8s** | 11.2M | 22MB | ~3.4ms | ~180ms | 44.9 |
| **YOLOv8m** | 25.9M | 52MB | ~6.3ms | ~350ms | 50.2 |
| **YOLOv8l** | 43.7M | 87MB | ~9.1ms | ~500ms | 52.9 |
| **YOLOv8x** | 68.2M | 137MB | ~14ms | ~800ms | 53.9 |

### 2.2 Cara Memilih

```
Hardware                    Recommended Model
─────────────────────────────────────────────────
Edge Device (Jetson Nano)   → yolov8n
Laptop/PC tanpa GPU         → yolov8n / yolov8s
PC dengan GPU mid-range     → yolov8s / yolov8m
Server dengan GPU pro       → yolov8m / yolov8l
Cloud GPU (A100, etc)       → yolov8l / yolov8x
```

### 2.3 Trade-off Speed vs Accuracy

![YOLOv8 Model Comparison - Size, Speed, and Accuracy Trade-offs](assets/yolo_model_sizes.png)

---

## 3. Teknik Optimasi Performa

### 3.1 Half Precision (FP16)

Gunakan 16-bit floating point alih-alih 32-bit:

```python
# Enable half precision
results = model(frame, half=True)  # Butuh GPU dengan FP16 support
```

**Keuntungan:**
- ✅ ~2x lebih cepat
- ✅ ~50% VRAM lebih sedikit
- ✅ Minimal accuracy loss (<1%)

**Requirement:**
- NVIDIA GPU dengan compute capability ≥ 5.3
- CUDA support

### 3.2 Resize Input

Jika objek cukup besar, tidak perlu input 640x640:

```python
# Default
results = model(frame, imgsz=640)  # Standard

# Untuk objek besar, cukup 320
results = model(frame, imgsz=320)  # ~4x lebih cepat!

# Untuk objek kecil (motor jauh)
results = model(frame, imgsz=1280)  # Lebih akurat, tapi lambat
```

**Trade-off:**

| Input Size | Speed | Small Objects | Large Objects |
|------------|-------|---------------|---------------|
| **320** | Tercepat | ❌ Hilang | ✅ OK |
| **480** | Cepat | ⚠️ Kadang miss | ✅ OK |
| **640** | Balance | ✅ OK | ✅ OK |
| **1280** | Lambat | ✅ Excellent | ✅ OK |

### 3.3 Batch Processing

Jika memproses beberapa stream/gambar sekaligus:

```python
# ❌ Satu per satu (lambat)
for img in images:
    result = model(img)

# ✅ Batch processing (efisien)
results = model(images, batch=4)  # Process 4 gambar sekaligus
```

**GPU menunjukkan performa terbaik saat memproses batch:**
- Batch 1: 100 images/sec
- Batch 4: 300 images/sec (3x improvement!)
- Batch 8: 400 images/sec

### 3.4 Disable Verbose Output

```python
# Matikan print untuk production
results = model(frame, verbose=False)
```

Print ke console memakan waktu. Matikan untuk performa optimal.

### 3.5 GPU Stream Processing

Untuk multi-stream scenario:

```python
import torch
from concurrent.futures import ThreadPoolExecutor

# Gunakan CUDA streams untuk parallel inference
streams = [torch.cuda.Stream() for _ in range(4)]

def process_with_stream(frame, stream_id):
    with torch.cuda.stream(streams[stream_id]):
        return model(frame, verbose=False)

# Process 4 camera feeds in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_with_stream, frame, i) 
               for i, frame in enumerate(frames)]
    results = [f.result() for f in futures]
```

---

## 4. Mengukur Performa

### 4.1 FPS Counter Sederhana

```python
import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("video.mp4")

# Metrics
frame_count = 0
start_time = time.time()
fps_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Inference
    t1 = time.time()
    results = model(frame, verbose=False)
    t2 = time.time()
    
    # Calculate FPS
    inference_time = t2 - t1
    instant_fps = 1.0 / inference_time if inference_time > 0 else 0
    fps_history.append(instant_fps)
    
    frame_count += 1
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    
    # Display
    cv2.putText(frame, f"FPS: {avg_fps:.1f} (instant: {instant_fps:.1f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Detection", results[0].plot())
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Summary
print(f"\n=== Performance Summary ===")
print(f"Total frames: {frame_count}")
print(f"Total time: {elapsed:.1f}s")
print(f"Average FPS: {avg_fps:.1f}")
print(f"Min FPS: {min(fps_history):.1f}")
print(f"Max FPS: {max(fps_history):.1f}")
```

### 4.2 Target FPS

| FPS | Status | Tracking Quality | Use Case |
|-----|--------|------------------|----------|
| **<5** | ❌ Tidak acceptable | Gagal | Tidak untuk real-time |
| **5-10** | ⚠️ Minimum | Buruk | Edge case, low priority |
| **10-20** | ✅ Acceptable | OK | Standard traffic monitoring |
| **20-30** | ✅ Good | Bagus | Smooth tracking |
| **30-60** | ✅ Excellent | Excellent | Overhead untuk analytics |
| **60+** | ✅ Overkill | Perfect | Research, high-speed |

### 4.3 Profiling Detail

```python
import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Warm-up (penting untuk GPU)
for _ in range(10):
    _ = model(frame, verbose=False)

# Profile dengan PyTorch
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    for _ in range(100):
        _ = model(frame, verbose=False)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## 5. Konfigurasi Optimal untuk Traffic Analytics

### 5.1 Basic Configuration

```python
from ultralytics import YOLO

# Load model yang sesuai dengan hardware
model = YOLO("yolov8s.pt")  # Balanced untuk kebanyakan case

def detect_vehicles(frame):
    """Optimized detection for traffic analytics"""
    results = model(
        frame,
        classes=[2, 3, 5, 7],  # Hanya kendaraan
        conf=0.3,              # Confidence threshold
        iou=0.7,               # IoU threshold
        half=True,             # FP16 untuk speed
        verbose=False,         # No print
        imgsz=640              # Standard size
    )
    return results[0]
```

### 5.2 Adaptive Configuration

```python
class AdaptiveDetector:
    """Detector dengan konfigurasi adaptif"""
    
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.fps_history = []
        self.target_fps = 20
        
        # Adaptive parameters
        self.current_imgsz = 640
        self.current_conf = 0.3
    
    def detect(self, frame):
        # Measure FPS
        t1 = time.time()
        
        results = self.model(
            frame,
            classes=[2, 3, 5, 7],
            conf=self.current_conf,
            imgsz=self.current_imgsz,
            half=True,
            verbose=False
        )
        
        # Calculate FPS
        elapsed = time.time() - t1
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.fps_history.append(fps)
        
        # Adaptive adjustment setiap 100 frames
        if len(self.fps_history) >= 100:
            avg_fps = sum(self.fps_history[-100:]) / 100
            self._adjust_parameters(avg_fps)
        
        return results[0]
    
    def _adjust_parameters(self, current_fps):
        """Adjust parameters untuk maintain target FPS"""
        if current_fps < self.target_fps * 0.8:  # Terlalu lambat
            if self.current_imgsz > 320:
                self.current_imgsz -= 64
                print(f"⚠️ Reducing imgsz to {self.current_imgsz}")
        elif current_fps > self.target_fps * 1.5:  # Ada headroom
            if self.current_imgsz < 640:
                self.current_imgsz += 64
                print(f"✓ Increasing imgsz to {self.current_imgsz}")
```

### 5.3 Multi-Stream Configuration

```python
class MultiStreamDetector:
    """
    Detector untuk multiple camera streams
    """
    
    def __init__(self, model_path: str, num_streams: int):
        self.model = YOLO(model_path)
        self.num_streams = num_streams
        
        # Adjust batch size based on available VRAM
        self.batch_size = self._calculate_batch_size()
    
    def _calculate_batch_size(self):
        """Estimate optimal batch size based on GPU memory"""
        if not torch.cuda.is_available():
            return 1
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        memory_gb = total_memory / (1024**3)
        
        # Rule of thumb: 1 batch per 2GB VRAM
        return min(self.num_streams, max(1, int(memory_gb / 2)))
    
    def detect_batch(self, frames: list):
        """
        Process multiple frames efficiently
        """
        # Pad to batch size
        while len(frames) < self.batch_size:
            frames.append(frames[-1])  # Duplicate last frame
        
        # Batch inference
        results = self.model(
            frames[:self.batch_size],
            classes=[2, 3, 5, 7],
            conf=0.3,
            half=True,
            verbose=False
        )
        
        return results[:len(frames)]  # Return only valid results

# Usage
multi_detector = MultiStreamDetector("yolov8n.pt", num_streams=4)

# Collect frames from all cameras
frames = [cam.read()[1] for cam in cameras]

# Process in one batch
all_results = multi_detector.detect_batch(frames)
```

---

## 6. Benchmarking Script

```python
"""
benchmark.py - Script untuk benchmark detection performance
"""
from ultralytics import YOLO
import cv2
import time
import numpy as np
import torch

def benchmark_model(model_path: str, video_path: str, num_frames: int = 100):
    """
    Benchmark model performance
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    # Collect sample frames
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        frames.append(frame)
    cap.release()
    
    # Warm-up
    for _ in range(10):
        _ = model(frames[0], verbose=False)
    
    # Benchmark different configurations
    configs = [
        {'imgsz': 320, 'half': True},
        {'imgsz': 640, 'half': True},
        {'imgsz': 640, 'half': False},
        {'imgsz': 1280, 'half': True},
    ]
    
    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")
    
    for config in configs:
        times = []
        
        for frame in frames:
            t1 = time.time()
            _ = model(frame, verbose=False, **config)
            t2 = time.time()
            times.append(t2 - t1)
        
        avg_time = np.mean(times) * 1000  # ms
        fps = 1000 / avg_time
        
        print(f"imgsz={config['imgsz']}, half={config['half']}: "
              f"{avg_time:.1f}ms, {fps:.1f} FPS")
    
    # GPU memory usage
    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nPeak GPU memory: {memory:.0f} MB")

if __name__ == "__main__":
    benchmark_model("yolov8n.pt", "traffic_sample.mp4")
    benchmark_model("yolov8s.pt", "traffic_sample.mp4")
```

---

## Kesimpulan

| Teknik | Speedup | Implementation |
|--------|---------|----------------|
| **Class Filtering** | 10-20% | `classes=[2,3,5,7]` |
| **Model Selection** | 2-4x | Use yolov8n/s |
| **Half Precision** | ~2x | `half=True` |
| **Input Resize** | 2-4x | `imgsz=320/480` |
| **Batch Processing** | 2-3x | `batch=4` |
| **Verbose Off** | 5-10% | `verbose=False` |

**Kombinasi optimal untuk real-time traffic:**
```python
results = model(
    frame,
    classes=[2, 3, 5, 7],
    conf=0.3,
    iou=0.7,
    half=True,
    verbose=False,
    imgsz=640
)
```

---

## Langkah Selanjutnya

Sekarang Anda bisa menjalankan detection dengan performa optimal. Lab selanjutnya membahas cara melacak objek antar frame:

👉 **[Lab 04: Object Tracking →](../04_object_tracking/)**
