# Reconnect Otomatis dan Error Handling

**Durasi Baca**: 15 Menit | **Level**: Menengah

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan memahami:
- Jenis-jenis error yang sering terjadi pada RTSP stream
- Implementasi auto-reconnect dengan exponential backoff
- Penanganan timeout menggunakan threading
- Production-ready reconnecting video capture class
- Logging dan monitoring untuk debugging

---

## 🎬 Pendahuluan

IP Camera di dunia nyata **tidak sempurna**. Koneksi RTSP bisa putus karena:
- Gangguan jaringan (WiFi drop, switch restart)
- Restart kamera (maintenance, power cycle)
- Overload server (terlalu banyak client)
- Firewall/NAT timeout

Sistem analytics yang baik harus bisa **pulih secara otomatis** tanpa intervensi manual.

> 💡 **Prinsip**: Sistem yang andal bukan yang tidak pernah gagal, tapi yang bisa **pulih sendiri** saat gagal.

---

## 1. State Machine Koneksi

![Connection State Machine](images/reconnect_state_machine.png)

### 1.1 States dan Transitions

| State | Deskripsi | Aksi |
|-------|-----------|------|
| **DISCONNECTED** | Tidak ada koneksi | Panggil `connect()` |
| **CONNECTING** | Sedang mencoba konek | Tunggu response |
| **CONNECTED** | Aktif membaca frame | Monitor `ret` value |
| **ERROR** | Terjadi kesalahan | Retry atau abort |

### 1.2 Transitions

| Dari | Ke | Trigger |
|------|----|---------|
| DISCONNECTED | CONNECTING | `connect()` dipanggil |
| CONNECTING | CONNECTED | `isOpened() == True` |
| CONNECTING | ERROR | Timeout atau exception |
| CONNECTED | ERROR | `ret == False` berulang |
| ERROR | CONNECTING | Retry attempt |
| ERROR | DISCONNECTED | Max retries exceeded |

---

## 2. Jenis Error yang Sering Terjadi

### 2.1 Tabel Error Umum

| Error | Gejala | Penyebab | Solusi |
|-------|--------|----------|--------|
| **Connection Failed** | `isOpened() = False` | IP salah, kamera mati, firewall | Cek network, retry |
| **Read Failed** | `ret = False` | Network drop, kamera restart | Reconnect |
| **Timeout/Hang** | `read()` tidak return | Stream stuck, network issue | Threading + timeout |
| **Corrupted Frame** | Frame rusak/artifak | Packet loss, codec error | Skip frame, reconnect |

### 2.2 Mendeteksi Setiap Jenis Error

```python
import cv2
import time

def diagnose_connection(source: str) -> dict:
    """Diagnosa masalah koneksi ke video source"""
    
    result = {
        "source": source,
        "can_open": False,
        "can_read": False,
        "fps": 0,
        "resolution": (0, 0),
        "error": None
    }
    
    try:
        # Test 1: Bisa dibuka?
        cap = cv2.VideoCapture(source)
        result["can_open"] = cap.isOpened()
        
        if not result["can_open"]:
            result["error"] = "Cannot open source"
            return result
        
        # Test 2: Bisa baca frame?
        ret, frame = cap.read()
        result["can_read"] = ret
        
        if not ret:
            result["error"] = "Cannot read frame"
            cap.release()
            return result
        
        # Test 3: Get properties
        result["fps"] = cap.get(cv2.CAP_PROP_FPS)
        result["resolution"] = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        
        cap.release()
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# Penggunaan
diag = diagnose_connection("rtsp://192.168.1.100/stream")
print(diag)
```

---

## 3. Pola Reconnect Sederhana

### 3.1 Implementasi Dasar

```python
import cv2
import time

def create_capture(source: str) -> cv2.VideoCapture:
    """Factory function untuk membuat VideoCapture"""
    cap = cv2.VideoCapture(source)
    return cap

def main():
    source = "rtsp://admin:pass@192.168.1.100/stream"
    cap = create_capture(source)
    
    fail_count = 0
    MAX_FAILS = 30  # Setelah 30 frame gagal, reconnect
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            fail_count += 1
            print(f"⚠️ Frame gagal: {fail_count}/{MAX_FAILS}")
            
            if fail_count >= MAX_FAILS:
                print("🔄 Mencoba reconnect...")
                cap.release()
                time.sleep(2)
                cap = create_capture(source)
                fail_count = 0
            continue
        
        # Reset counter jika sukses
        fail_count = 0
        
        # Proses frame
        cv2.imshow("Stream", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### 3.2 Evaluasi

| Aspek | Nilai | Keterangan |
|-------|-------|------------|
| Kemudahan | ⭐⭐⭐⭐⭐ | Sangat sederhana |
| Reliability | ⭐⭐⭐ | Bekerja untuk kasus umum |
| Efficiency | ⭐⭐ | Menunggu 30 fail = delay |
| Production-ready | ⭐⭐ | Tidak ada logging, backoff |

---

## 4. Reconnect dengan Exponential Backoff

Jika kamera benar-benar mati, jangan spam reconnect terus-menerus. Gunakan **exponential backoff** untuk mengurangi load.

### 4.1 Konsep Exponential Backoff

| Retry | Delay |
|-------|-------|
| 1 | 2s |
| 2 | 4s |
| 3 | 8s |
| 4 | 16s |
| 5 | 32s |
| 6+ | 60s (max) |

### 4.2 Implementasi

```python
import cv2
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ExponentialBackoff:
    """Mengelola delay reconnect dengan exponential backoff"""
    
    def __init__(self, initial: float = 2.0, maximum: float = 60.0, multiplier: float = 2.0):
        self.initial = initial
        self.maximum = maximum
        self.multiplier = multiplier
        self.current = initial
    
    def wait(self):
        """Tunggu sesuai current delay"""
        logger.info(f"⏳ Menunggu {self.current:.1f}s sebelum retry...")
        time.sleep(self.current)
        # Increase untuk next time
        self.current = min(self.current * self.multiplier, self.maximum)
    
    def reset(self):
        """Reset ke initial delay"""
        self.current = self.initial


def main_with_backoff():
    source = "rtsp://admin:pass@192.168.1.100/stream"
    backoff = ExponentialBackoff(initial=2, maximum=60)
    
    while True:
        logger.info(f"🔗 Mencoba koneksi ke {source}...")
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.warning("❌ Gagal konek!")
            backoff.wait()
            continue
        
        # Berhasil konek - reset backoff
        backoff.reset()
        logger.info("✅ Terhubung!")
        
        # Loop pembacaan
        fail_streak = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                fail_streak += 1
                if fail_streak >= 10:
                    logger.warning("❌ Koneksi terputus!")
                    break
                continue
            
            fail_streak = 0
            cv2.imshow("Stream", frame)
            
            if cv2.waitKey(1) == ord('q'):
                cap.release()
                return
        
        cap.release()
        backoff.wait()

if __name__ == "__main__":
    main_with_backoff()
```

---

## 5. Menangani Timeout dengan Threading

`cap.read()` bisa **hang selamanya** jika stream stuck. Ini berbahaya karena program Anda berhenti total.

### 5.1 Masalah

```python
# Ini bisa hang selamanya!
ret, frame = cap.read()  # Tidak pernah return jika stream stuck
```

### 5.2 Solusi: Read dengan Timeout

```python
import cv2
import threading
from typing import Optional, Tuple
import numpy as np

def read_with_timeout(cap: cv2.VideoCapture, timeout: float = 5.0) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Membaca frame dengan timeout.
    Return (False, None) jika timeout.
    """
    result = [False, None]
    
    def reader():
        result[0], result[1] = cap.read()
    
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        # Timeout! Thread masih berjalan
        return False, None
    
    return result[0], result[1]

# Penggunaan
cap = cv2.VideoCapture("rtsp://...")
ret, frame = read_with_timeout(cap, timeout=3.0)

if not ret:
    print("Timeout atau error!")
```

---

## 6. Production-Ready Reconnecting Capture

Kelas lengkap yang menggabungkan semua fitur: threading, auto-reconnect, exponential backoff, dan logging.

```python
import cv2
import threading
import time
import logging
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"

@dataclass
class CameraConfig:
    source: str
    name: str = "Camera"
    read_timeout: float = 5.0
    reconnect_delay_initial: float = 2.0
    reconnect_delay_max: float = 60.0
    max_fail_streak: int = 10

class ReconnectingVideoCapture:
    """
    Production-ready video capture dengan:
    - Auto-reconnect dengan exponential backoff
    - Timeout handling
    - Thread-safe
    - Comprehensive logging
    """
    
    def __init__(self, config: CameraConfig, on_state_change: Optional[Callable] = None):
        self.config = config
        self.logger = logging.getLogger(config.name)
        self.on_state_change = on_state_change
        
        # State
        self._state = ConnectionState.DISCONNECTED
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        
        # Backoff
        self._reconnect_delay = config.reconnect_delay_initial
        
        # Stats
        self._connect_attempts = 0
        self._total_frames = 0
        self._total_errors = 0
        self._last_frame_time = 0.0
    
    @property
    def state(self) -> ConnectionState:
        return self._state
    
    def _set_state(self, new_state: ConnectionState):
        old_state = self._state
        self._state = new_state
        
        if old_state != new_state:
            self.logger.info(f"State: {old_state.value} → {new_state.value}")
            if self.on_state_change:
                self.on_state_change(old_state, new_state)
    
    def connect(self) -> bool:
        """Mencoba konek ke source"""
        self._set_state(ConnectionState.CONNECTING)
        self._connect_attempts += 1
        
        self.logger.info(f"Connecting to {self.config.source} (attempt #{self._connect_attempts})")
        
        try:
            self._cap = cv2.VideoCapture(self.config.source)
            
            if self._cap.isOpened():
                # Verifikasi bisa baca
                ret, _ = self._cap.read()
                if ret:
                    self._set_state(ConnectionState.CONNECTED)
                    self._reconnect_delay = self.config.reconnect_delay_initial
                    
                    # Log properties
                    w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = self._cap.get(cv2.CAP_PROP_FPS)
                    self.logger.info(f"✅ Connected! {w}x{h} @ {fps:.1f} FPS")
                    
                    return True
            
            self._set_state(ConnectionState.ERROR)
            return False
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            self._set_state(ConnectionState.ERROR)
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Membaca frame dengan timeout"""
        if self._state != ConnectionState.CONNECTED or self._cap is None:
            return False, None
        
        result = [False, None]
        
        def reader():
            result[0], result[1] = self._cap.read()
        
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout=self.config.read_timeout)
        
        if thread.is_alive():
            self.logger.warning(f"⏰ Read timeout ({self.config.read_timeout}s)")
            self._total_errors += 1
            return False, None
        
        if result[0]:
            self._total_frames += 1
            self._last_frame_time = time.time()
        else:
            self._total_errors += 1
        
        return result[0], result[1]
    
    def reconnect(self):
        """Disconnect dan reconnect dengan backoff"""
        self.disconnect()
        
        self.logger.info(f"⏳ Waiting {self._reconnect_delay:.1f}s before reconnect...")
        time.sleep(self._reconnect_delay)
        
        # Increase backoff untuk next time
        self._reconnect_delay = min(
            self._reconnect_delay * 2,
            self.config.reconnect_delay_max
        )
        
        self.connect()
    
    def disconnect(self):
        """Tutup koneksi"""
        if self._cap:
            self._cap.release()
            self._cap = None
        self._set_state(ConnectionState.DISCONNECTED)
    
    def get_stats(self) -> dict:
        """Return statistik koneksi"""
        return {
            "state": self._state.value,
            "connect_attempts": self._connect_attempts,
            "total_frames": self._total_frames,
            "total_errors": self._total_errors,
            "last_frame_ago": time.time() - self._last_frame_time if self._last_frame_time else None
        }
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *args):
        self.disconnect()


# ============================================================
# CONTOH PENGGUNAAN
# ============================================================

def main():
    config = CameraConfig(
        source="rtsp://admin:pass@192.168.1.100/stream",
        name="TrafficCam01",
        read_timeout=3.0,
        reconnect_delay_initial=2.0,
        reconnect_delay_max=30.0,
        max_fail_streak=10
    )
    
    def on_state_change(old, new):
        if new == ConnectionState.ERROR:
            # Bisa kirim alert di sini
            print(f"🚨 ALERT: Camera error!")
    
    cam = ReconnectingVideoCapture(config, on_state_change=on_state_change)
    
    fail_streak = 0
    
    while True:
        # Connect jika belum
        if cam.state != ConnectionState.CONNECTED:
            if not cam.connect():
                cam.reconnect()
                continue
        
        # Baca frame
        ret, frame = cam.read()
        
        if not ret:
            fail_streak += 1
            if fail_streak >= config.max_fail_streak:
                print(f"❌ {fail_streak} fails berturut-turut, reconnecting...")
                cam.reconnect()
                fail_streak = 0
            continue
        
        fail_streak = 0
        
        # Proses frame
        cv2.putText(frame, f"Frames: {cam.get_stats()['total_frames']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Reconnecting Capture", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cam.disconnect()
    cv2.destroyAllWindows()
    
    print("\n📊 Final Stats:")
    print(cam.get_stats())

if __name__ == "__main__":
    main()
```

---

## 7. Logging Best Practices

### 7.1 Structured Logging

```python
import logging
import json
from datetime import datetime

class CameraLogger:
    """Logger khusus untuk camera events"""
    
    def __init__(self, camera_name: str, log_file: str = None):
        self.camera_name = camera_name
        self.logger = logging.getLogger(f"camera.{camera_name}")
        
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
    
    def _log_event(self, event_type: str, **kwargs):
        event = {
            "timestamp": datetime.now().isoformat(),
            "camera": self.camera_name,
            "event": event_type,
            **kwargs
        }
        self.logger.info(json.dumps(event))
    
    def log_connect(self, success: bool, attempt: int):
        self._log_event("connect", success=success, attempt=attempt)
    
    def log_disconnect(self, reason: str):
        self._log_event("disconnect", reason=reason)
    
    def log_error(self, error_type: str, message: str):
        self._log_event("error", error_type=error_type, message=message)
    
    def log_stats(self, frames: int, errors: int, uptime: float):
        self._log_event("stats", frames=frames, errors=errors, uptime_seconds=uptime)

# Output contoh (JSON per baris):
# {"timestamp": "2024-01-15T10:30:00", "camera": "Cam01", "event": "connect", "success": true, "attempt": 1}
# {"timestamp": "2024-01-15T11:45:00", "camera": "Cam01", "event": "disconnect", "reason": "network_timeout"}
```

### 7.2 Alerting

```python
import smtplib
from email.message import EmailMessage

def send_alert(camera_name: str, message: str):
    """Kirim email alert"""
    msg = EmailMessage()
    msg['Subject'] = f"🚨 Camera Alert: {camera_name}"
    msg['From'] = "alerts@company.com"
    msg['To'] = "admin@company.com"
    msg.set_content(message)
    
    # Kirim via SMTP (configure sesuai server Anda)
    # with smtplib.SMTP('smtp.company.com') as smtp:
    #     smtp.send_message(msg)
    
    print(f"ALERT: {camera_name} - {message}")

# Trigger alert jika offline lebih dari 5 menit
if time.time() - last_frame_time > 300:
    send_alert("TrafficCam01", "Camera offline lebih dari 5 menit!")
```

---

## 8. Checklist Production

### 8.1 Sebelum Deploy

- [ ] `isOpened()` dicek setelah setiap `VideoCapture()`
- [ ] `ret == False` di-handle dengan reconnect logic
- [ ] Exponential backoff untuk mencegah spam reconnect
- [ ] Timeout untuk mencegah hang
- [ ] Logging semua disconnect dan reconnect
- [ ] Alert jika camera offline > X menit
- [ ] Statistik frame/error terekam

### 8.2 Monitoring Dashboard

| Metric | Threshold | Alert |
|--------|-----------|-------|
| Connection State | != CONNECTED > 5 min | Warning |
| Error Rate | > 10% | Warning |
| Frame Rate | < 50% expected | Critical |
| Last Frame | > 60s ago | Critical |

---

## 📝 Quiz Singkat

1. Mengapa exponential backoff lebih baik daripada fixed delay?
2. Bagaimana cara mendeteksi `cap.read()` yang hang?
3. Kapan state berubah dari CONNECTED ke ERROR?
4. Apa manfaat structured logging (JSON) dibanding plain text?

<details>
<summary>Lihat Jawaban</summary>

1. Exponential backoff mengurangi load pada sistem saat kamera benar-benar down. Fixed delay terus-menerus = spam reconnect.
2. Menggunakan threading dengan timeout. Jalankan `read()` di thread terpisah dan `join(timeout=X)`.
3. Ketika `ret == False` terjadi berturut-turut melebihi threshold (misalnya 10 kali).
4. JSON mudah di-parse oleh tools monitoring (ELK, Grafana). Plain text sulit untuk automated processing.

</details>

---

## 🔗 Lab Berikutnya

Selamat! Anda telah menyelesaikan **Lab 01: Stream Ingestion**. Anda sekarang memahami:
- ✅ Anatomi video digital
- ✅ Cara kerja VideoCapture
- ✅ Masalah latency dan solusinya
- ✅ Auto-reconnect dan error handling

Lanjutkan ke **Lab 02: Model Training** untuk belajar melatih model YOLO kustom:
→ [01. Mengapa Pre-trained Tidak Cukup](../02_model_training/01_mengapa_pretrained_tidak_cukup.md)
