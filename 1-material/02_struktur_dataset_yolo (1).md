# Struktur Dataset YOLO

**Durasi Baca**: 15 Menit | **Level**: Pemula

---

## 📌 Tujuan Pembelajaran

Setelah membaca artikel ini, Anda akan memahami:
- Struktur folder yang benar untuk dataset YOLO
- Format file label (annotation) dan cara menghitungnya
- Cara membuat file `data.yaml`
- Tools untuk labeling dan best practices
- Script validasi dataset

---

## 🎬 Pendahuluan

YOLO memiliki format dataset yang **sangat spesifik**. Jika struktur salah, training akan:
- ❌ Error saat mulai
- ❌ Training berjalan tapi hasil buruk
- ❌ Model tidak bisa menemukan objek

Artikel ini menjelaskan format yang benar secara detail.

---

## 1. Struktur Folder

### 1.1 Hierarki Direktori

```
dataset/
├── data.yaml           # Konfigurasi dataset
├── images/
│   ├── train/          # Gambar untuk training (80%)
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   ├── img_003.png  # Format lain juga OK
│   │   └── ...
│   └── val/            # Gambar untuk validasi (20%)
│       ├── img_100.jpg
│       ├── img_101.jpg
│       └── ...
└── labels/
    ├── train/          # Label untuk training
    │   ├── img_001.txt # Nama HARUS sama dengan gambar
    │   ├── img_002.txt
    │   ├── img_003.txt
    │   └── ...
    └── val/            # Label untuk validasi
        ├── img_100.txt
        ├── img_101.txt
        └── ...
```

### 1.2 Visualisasi Struktur

![Struktur Dataset YOLO](assets/dataset_structure.png)

### 1.3 Aturan Penting

| Aturan | Penjelasan | Contoh |
|--------|------------|--------|
| **Nama file harus SAMA** | Gambar dan label harus punya nama identik | `img_001.jpg` → `img_001.txt` |
| **Ekstensi berbeda OK** | Gambar bisa .jpg/.png, label selalu .txt | `photo.png` → `photo.txt` |
| **Gambar tanpa objek** | Buat file .txt KOSONG (jangan skip!) | File 0 bytes |
| **Case sensitive** | `Image_001.jpg` ≠ `image_001.txt` | Harus exact match |

### 1.4 Split Ratio Standar

```
Total Dataset: 100%
├── Train: 70-80%     # Model belajar dari data ini
├── Validation: 15-20% # Evaluasi selama training
└── Test: 5-10%       # Evaluasi final (opsional)
```

**Rekomendasi untuk project kecil (100-500 gambar):**
- Train: 80%
- Val: 20%
- Test: Tidak perlu (gunakan val sebagai test)

---

## 2. Format Label (Annotation)

### 2.1 Format Dasar

Setiap file `.txt` berisi **satu baris per objek**:

```
<class_id> <x_center> <y_center> <width> <height>
```

### 2.2 Penjelasan Setiap Komponen

| Komponen | Deskripsi | Range |
|----------|-----------|-------|
| `class_id` | Index kelas, mulai dari 0 | 0, 1, 2, ... |
| `x_center` | Koordinat X pusat box (normalized) | 0.0 - 1.0 |
| `y_center` | Koordinat Y pusat box (normalized) | 0.0 - 1.0 |
| `width` | Lebar box (normalized) | 0.0 - 1.0 |
| `height` | Tinggi box (normalized) | 0.0 - 1.0 |

### 2.3 Apa Itu Normalized?

**Normalized** berarti nilai dibagi dengan ukuran gambar:

```
x_normalized = x_pixel / image_width
y_normalized = y_pixel / image_height
```

**Keuntungan normalized:**
- Gambar ukuran berbeda bisa digabung dalam dataset yang sama
- Nilai selalu antara 0 dan 1
- Tidak perlu track resolusi per gambar

### 2.4 Contoh Perhitungan Detail

**Skenario:**
- Gambar: 1920 x 1080 piksel
- Mobil dengan bounding box:
  - Top-left corner: (100, 200)
  - Bottom-right corner: (300, 400)

**Langkah Perhitungan:**

```python
# Koordinat pixel
x1, y1 = 100, 200  # Top-left
x2, y2 = 300, 400  # Bottom-right

# Ukuran gambar
img_width = 1920
img_height = 1080

# Hitung center dan size dalam pixel
x_center_pixel = (x1 + x2) / 2  # = 200
y_center_pixel = (y1 + y2) / 2  # = 300
box_width_pixel = x2 - x1       # = 200
box_height_pixel = y2 - y1      # = 200

# Normalize (bagi dengan ukuran gambar)
x_center = x_center_pixel / img_width   # = 200/1920 = 0.1042
y_center = y_center_pixel / img_height  # = 300/1080 = 0.2778
width = box_width_pixel / img_width     # = 200/1920 = 0.1042
height = box_height_pixel / img_height  # = 200/1080 = 0.1852

# class_id = 0 (mobil)
print(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
# Output: 0 0.104167 0.277778 0.104167 0.185185
```

**Isi file label (`img_001.txt`):**
```
0 0.104167 0.277778 0.104167 0.185185
```

### 2.5 Multiple Objects dalam Satu Gambar

Jika ada banyak objek, tulis satu baris per objek:

```
# img_001.txt
0 0.104167 0.277778 0.104167 0.185185
0 0.520833 0.462963 0.083333 0.148148
1 0.312500 0.555556 0.041667 0.092593
2 0.750000 0.388889 0.156250 0.277778
```

**Penjelasan:**
- Baris 1: Mobil (class 0) di kiri atas
- Baris 2: Mobil (class 0) di tengah
- Baris 3: Motor (class 1)
- Baris 4: Truck (class 2)

### 2.6 Gambar Tanpa Objek

Jika gambar tidak ada objek yang relevan:
- **WAJIB** buat file `.txt` kosong (0 bytes)
- Jangan skip file ini!

```python
# Buat file kosong
open("labels/train/background_001.txt", "w").close()
```

**Mengapa penting?** Model perlu belajar "tidak ada objek" juga (negative samples).

---

## 3. File data.yaml

### 3.1 Struktur Dasar

```yaml
# data.yaml
path: /path/to/dataset  # Path ABSOLUT ke folder dataset
train: images/train     # Path RELATIF dari 'path'
val: images/val         # Path RELATIF dari 'path'

# Jumlah kelas
nc: 4

# Nama kelas (urutan = class_id)
names:
  0: car
  1: motorcycle
  2: bus
  3: truck
```

### 3.2 Penjelasan Setiap Field

| Field | Deskripsi | Contoh |
|-------|-----------|--------|
| `path` | Path absolut ke root dataset | `/home/user/datasets/traffic` |
| `train` | Path relatif ke folder training | `images/train` |
| `val` | Path relatif ke folder validasi | `images/val` |
| `test` | (Opsional) Path ke folder test | `images/test` |
| `nc` | Number of classes | `4` |
| `names` | Mapping class_id ke nama | `{0: car, 1: motorcycle, ...}` |

### 3.3 Format Alternatif untuk Names

```yaml
# Cara 1: Dictionary (recommended)
names:
  0: car
  1: motorcycle
  2: bus
  3: truck

# Cara 2: List
names: ['car', 'motorcycle', 'bus', 'truck']
```

### 3.4 Contoh Lengkap untuk Traffic Analytics

```yaml
# traffic_dataset/data.yaml
path: D:/datasets/traffic_cctv_v1

train: images/train
val: images/val
test: images/test  # opsional

# 6 kelas untuk traffic di Indonesia
nc: 6

names:
  0: car
  1: motorcycle
  2: bus
  3: truck
  4: bicycle
  5: person

# Metadata (opsional, untuk dokumentasi)
# roboflow:
#   workspace: traffic-analytics
#   project: cctv-detection
#   version: 3
```

### 3.5 Catatan Penting

> ⚠️ **CRITICAL**: `nc` harus SAMA dengan jumlah kelas yang Anda label!
> 
> Jika Anda label 4 kelas tapi `nc: 6`, model akan error atau hasilnya buruk.

---

## 4. Tools untuk Labeling

### 4.1 Perbandingan Tools

| Tool | Tipe | Gratis | Augmentasi | Kolaborasi |
|------|------|--------|------------|------------|
| **Roboflow** | Cloud | ✅ (10K/bulan) | ✅ Otomatis | ✅ |
| **CVAT** | Self-hosted | ✅ | ❌ | ✅ |
| **LabelImg** | Desktop | ✅ | ❌ | ❌ |
| **Label Studio** | Self-hosted | ✅ | ❌ | ✅ |

### 4.2 Roboflow (Recommended untuk Pemula)

```
Kelebihan:
+ Interface intuitif (browser-based)
+ Augmentasi otomatis
+ Export langsung ke YOLO format
+ Dataset versioning
+ API untuk download

Kekurangan:
- Limit 10K gambar/bulan di free tier
- Data di cloud (privasi)
```

👉 Tutorial lengkap: [Roboflow untuk Dataset](02a_roboflow_untuk_dataset.md)

### 4.3 CVAT (Production-grade)

```bash
# Install dengan Docker
git clone https://github.com/opencv/cvat
cd cvat
docker compose up -d
# Akses di: http://localhost:8080
```

```
Kelebihan:
+ Open source, self-hosted
+ Support banyak format
+ Cocok untuk tim besar
+ Fitur auto-annotation

Kekurangan:
- Setup lebih rumit
- Butuh server sendiri
```

### 4.4 LabelImg (Quick & Simple)

```bash
# Install
pip install labelImg

# Jalankan
labelImg
```

```
Kelebihan:
+ Sangat ringan
+ Offline
+ Simple untuk project kecil

Kekurangan:
- Tidak ada augmentasi
- File harus di-manage manual
```

---

## 5. Best Practices Labeling

### 5.1 Konsistensi Nama Kelas

```
❌ SALAH:
   - Gambar 1: "car"
   - Gambar 2: "mobil"
   - Gambar 3: "vehicle"

✅ BENAR:
   - Semua gambar: "car"
```

**Tip**: Buat dokumen pedoman sebelum mulai labeling.

### 5.2 Tight Bounding Box

```
❌ Box terlalu BESAR (banyak background):
┌─────────────────────┐
│                     │
│      🚗             │
│                     │
└─────────────────────┘

✅ Box PAS dengan objek:
    ┌─────────┐
    │  🚗    │
    └─────────┘
```

**Aturan**: Box harus menyentuh tepi objek di semua sisi.

### 5.3 Handling Occlusion (Terhalang)

```
Aturan praktis:

> 75% terlihat  → Label penuh seolah tidak terhalang
50-75% terlihat → Label bagian yang terlihat saja
< 50% terlihat  → SKIP (jangan label)
```

**Visualisasi:**
```
🚗 terlihat 80% → ✅ Label
🚗 terlihat 60% → ✅ Label (bagian terlihat)
🚗 terlihat 30% → ❌ Skip
```

### 5.4 Objek di Tepi Frame

```
Aturan:
- Jika objek terpotong > 50%: SKIP
- Jika objek terpotong < 50%: Label bagian yang terlihat
```

### 5.5 Jumlah Minimum per Kelas

| Kelas | Minimum | Ideal | Excellent |
|-------|---------|-------|-----------|
| Per kelas | 50 | 200-300 | 500+ |
| Total dataset | 200 | 1000 | 5000+ |

> 💡 **Rule of thumb**: Jika salah satu kelas punya instance jauh lebih sedikit, model akan bias ke kelas mayoritas.

### 5.6 Class Balance

```python
# Cek distribusi kelas
from collections import Counter
import os

label_dir = "dataset/labels/train"
class_counts = Counter()

for label_file in os.listdir(label_dir):
    with open(os.path.join(label_dir, label_file)) as f:
        for line in f:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

print("Class Distribution:")
for class_id, count in sorted(class_counts.items()):
    print(f"  Class {class_id}: {count} instances")

# Output:
# Class Distribution:
#   Class 0 (car): 1250 instances
#   Class 1 (motorcycle): 890 instances
#   Class 2 (bus): 120 instances  ← Under-represented!
#   Class 3 (truck): 340 instances
```

**Solusi untuk imbalance:**
1. Tambah lebih banyak gambar untuk kelas minoritas
2. Gunakan augmentasi (Roboflow) untuk memperbanyak
3. Gunakan weighted loss saat training

---

## 6. Script Validasi Dataset

### 6.1 Script Validasi Lengkap

```python
"""
validate_dataset.py
Script untuk memvalidasi dataset YOLO sebelum training
"""

import os
from pathlib import Path
from collections import Counter
import yaml

def validate_dataset(dataset_path: str):
    """
    Validasi struktur dan isi dataset YOLO
    """
    dataset_path = Path(dataset_path)
    errors = []
    warnings = []
    
    # 1. Cek data.yaml
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        errors.append("❌ data.yaml tidak ditemukan!")
        return errors, warnings
    
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    nc = config.get('nc', 0)
    names = config.get('names', {})
    
    print(f"📋 Dataset: {dataset_path}")
    print(f"   Classes: {nc}")
    print(f"   Names: {list(names.values()) if isinstance(names, dict) else names}")
    
    # 2. Cek folder structure
    for split in ['train', 'val']:
        img_dir = dataset_path / "images" / split
        lbl_dir = dataset_path / "labels" / split
        
        if not img_dir.exists():
            errors.append(f"❌ Folder {img_dir} tidak ditemukan!")
            continue
        if not lbl_dir.exists():
            errors.append(f"❌ Folder {lbl_dir} tidak ditemukan!")
            continue
        
        # Get files
        images = set(p.stem for p in img_dir.glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png'])
        labels = set(p.stem for p in lbl_dir.glob("*.txt"))
        
        print(f"\n📂 {split}:")
        print(f"   Images: {len(images)}")
        print(f"   Labels: {len(labels)}")
        
        # 3. Cek missing labels
        missing_labels = images - labels
        if missing_labels:
            warnings.append(f"⚠️ {split}: {len(missing_labels)} gambar tanpa label")
            for name in list(missing_labels)[:5]:
                print(f"      Missing: {name}")
        
        # 4. Cek orphan labels (label tanpa gambar)
        orphan_labels = labels - images
        if orphan_labels:
            errors.append(f"❌ {split}: {len(orphan_labels)} label tanpa gambar")
        
        # 5. Validasi format label
        class_counts = Counter()
        for label_file in lbl_dir.glob("*.txt"):
            with open(label_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        errors.append(f"❌ {label_file.name}:{line_num} - Format salah ({len(parts)} parts, expected 5)")
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])
                        
                        # Cek range
                        if class_id >= nc:
                            errors.append(f"❌ {label_file.name}:{line_num} - class_id {class_id} >= nc {nc}")
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            errors.append(f"❌ {label_file.name}:{line_num} - Koordinat di luar range 0-1")
                        
                        class_counts[class_id] += 1
                        
                    except ValueError as e:
                        errors.append(f"❌ {label_file.name}:{line_num} - Parse error: {e}")
        
        # 6. Print class distribution
        print(f"\n   Class distribution ({split}):")
        for class_id in range(nc):
            count = class_counts.get(class_id, 0)
            bar = '█' * min(count // 10, 50)
            name = names.get(class_id, f"class_{class_id}") if isinstance(names, dict) else names[class_id]
            print(f"      {class_id} ({name}): {count:5d} {bar}")
    
    # Summary
    print("\n" + "="*50)
    if errors:
        print(f"❌ {len(errors)} ERRORS found:")
        for e in errors[:10]:
            print(f"   {e}")
    else:
        print("✅ No errors found!")
    
    if warnings:
        print(f"⚠️ {len(warnings)} WARNINGS:")
        for w in warnings:
            print(f"   {w}")
    
    return errors, warnings

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    validate_dataset(path)
```

### 6.2 Cara Menggunakan

```bash
# Validasi dataset
python validate_dataset.py /path/to/dataset

# Output:
# 📋 Dataset: /path/to/dataset
#    Classes: 4
#    Names: ['car', 'motorcycle', 'bus', 'truck']
#
# 📂 train:
#    Images: 800
#    Labels: 800
#
#    Class distribution (train):
#       0 (car): 1250 ████████████████████████████████
#       1 (motorcycle):  890 ██████████████████████
#       2 (bus):  120 ███
#       3 (truck):  340 ████████
#
# ==================================================
# ✅ No errors found!
```

---

## 7. Conversion Scripts

### 7.1 COCO ke YOLO

```python
"""
Konversi COCO format ke YOLO format
"""
import json
from pathlib import Path

def coco_to_yolo(coco_json: str, output_dir: str, class_mapping: dict = None):
    """
    Convert COCO annotations ke YOLO format
    
    Args:
        coco_json: Path ke file annotations.json COCO
        output_dir: Folder output untuk label .txt
        class_mapping: Optional mapping COCO category_id ke YOLO class_id
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(coco_json) as f:
        coco = json.load(f)
    
    # Build image lookup
    images = {img['id']: img for img in coco['images']}
    
    # Build category mapping
    if class_mapping is None:
        class_mapping = {cat['id']: idx for idx, cat in enumerate(coco['categories'])}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Convert each image
    for img_id, img_info in images.items():
        img_width = img_info['width']
        img_height = img_info['height']
        filename = Path(img_info['file_name']).stem + '.txt'
        
        lines = []
        for ann in annotations_by_image.get(img_id, []):
            # COCO bbox = [x, y, width, height] (absolute)
            x, y, w, h = ann['bbox']
            
            # Convert to YOLO format
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            width = w / img_width
            height = h / img_height
            
            class_id = class_mapping.get(ann['category_id'], 0)
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Write label file
        with open(output_dir / filename, 'w') as f:
            f.write('\n'.join(lines))
    
    print(f"Converted {len(images)} images to YOLO format in {output_dir}")

# Usage
coco_to_yolo(
    coco_json="annotations/instances_train.json",
    output_dir="labels/train",
    class_mapping={1: 0, 2: 1, 3: 2, 4: 3}  # COCO car/motorcycle/bus/truck
)
```

### 7.2 Pascal VOC ke YOLO

```python
"""
Konversi Pascal VOC XML ke YOLO format
"""
import xml.etree.ElementTree as ET
from pathlib import Path

def voc_to_yolo(xml_file: str, classes: list) -> str:
    """
    Convert satu file Pascal VOC XML ke format YOLO
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in classes:
            continue
        
        class_id = classes.index(class_name)
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to YOLO
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return '\n'.join(lines)

# Batch conversion
classes = ['car', 'motorcycle', 'bus', 'truck']
xml_dir = Path("annotations_voc")
output_dir = Path("labels/train")
output_dir.mkdir(parents=True, exist_ok=True)

for xml_file in xml_dir.glob("*.xml"):
    yolo_content = voc_to_yolo(xml_file, classes)
    output_file = output_dir / (xml_file.stem + ".txt")
    output_file.write_text(yolo_content)
```

---

## Kesimpulan

| Aspek | Key Point |
|-------|-----------|
| **Struktur** | `images/` dan `labels/` dengan subfolder `train/` dan `val/` |
| **Nama File** | Gambar dan label HARUS nama sama |
| **Format Label** | `class_id x_center y_center width height` (normalized) |
| **data.yaml** | Wajib ada, define path dan class names |
| **Validasi** | Selalu validasi sebelum training! |

---

## Langkah Selanjutnya

Dataset sudah siap? Saatnya training!

👉 **[Proses Training YOLOv8 →](03_proses_training_yolov8.md)**

Atau jika ingin menggunakan Roboflow:

👉 **[Roboflow untuk Dataset →](02a_roboflow_untuk_dataset.md)**
