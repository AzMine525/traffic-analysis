
from pathlib import Path
from collections import Counter

# Lokasi proyek
BASE_DIR = Path(__file__).resolve().parent.parent

# Lokasi dataset
LABELS_DIR = BASE_DIR / "dataset-vehicles" / "labels"

print("=" * 50)
print("PEMERIKSAAN DATASET YOLO")
print("=" * 50)

print(f"Lokasi dataset : {LABELS_DIR}")
print(f"Folder ada     : {LABELS_DIR.exists()}")

# Cari seluruh file anotasi
label_files = list(LABELS_DIR.rglob("*.txt"))

print(f"Jumlah file TXT: {len(label_files)}")

counts = Counter()
invalid = []

for file in label_files:
    for line_no, line in enumerate(
        file.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        parts = line.split()

        if not parts:
            continue

        try:
            if len(parts) != 5:
                raise ValueError("Jumlah kolom bukan 5")

            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            if class_id < 0 or any(
                not 0 <= x <= 1 for x in coords
            ):
                raise ValueError("Nilai di luar rentang")

            counts[class_id] += 1

        except ValueError as e:
            invalid.append((str(file), line_no, str(e)))

print("\nJumlah objek per Class ID:")

if not counts:
    print("Tidak ditemukan anotasi objek.")
else:
    for class_id, count in sorted(counts.items()):
        print(f"Class {class_id}: {count} objek")

print(f"\nTotal objek          : {sum(counts.values())}")
print(f"Anotasi tidak valid : {len(invalid)}")
print(f"Contoh error        : {invalid[:5]}")

print("=" * 50)
