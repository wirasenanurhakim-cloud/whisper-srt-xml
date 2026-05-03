# Whisper Auto Editor

GUI tool untuk:

* Generate subtitle otomatis (SRT)
* Forced alignment (akurasi timing tinggi)
* Export XML timeline (Premiere / Final Cut)
* Translate Korea → Indonesia (opsional)

---

## ⚙️ Requirements

Sebelum menjalankan, pastikan sudah install:

### 1. Python

* Versi **3.10 – 3.11** (disarankan)
* Download: https://www.python.org/downloads/

---

### 2. FFmpeg (WAJIB)

Download dari:
https://ffmpeg.org/download.html

Setelah install:

* Pastikan `ffmpeg` dan `ffprobe` bisa diakses dari terminal

Cek:

```bash
ffmpeg -version
```

Kalau error → berarti belum masuk PATH

---

## 📦 Install Dependencies

Di folder project:

```bash
pip install -r requirements.txt
```

Atau manual (kalau error):

```bash
pip install torch
pip install stable-ts
pip install tkinterdnd2
pip install requests
```

---

## 🚀 Cara Menjalankan

```bash
python main.py
```

---

## 🧠 Catatan Penting

### 🔹 Model Whisper

* Akan **auto download saat pertama kali run**
* Ukuran bisa besar (ratusan MB – GB)
* Pastikan koneksi stabil

---

### 🔹 GPU (Optional)

Jika punya GPU (CUDA), proses akan jauh lebih cepat.

Kalau tidak, akan otomatis pakai CPU (lebih lambat, tapi tetap jalan).

---

### 🔹 Format File

Support:

* Audio: `.mp3`, `.wav`, `.m4a`, `.flac`, dll
* Video: `.mp4`, `.mov`, `.mkv`, dll

---

## ⚠️ Troubleshooting

### ❌ ModuleNotFoundError: stable_whisper

```bash
pip install stable-ts
```

---

### ❌ FFmpeg not found

Pastikan:

* Sudah install FFmpeg
* Sudah masuk PATH

---

### ❌ CUDA error

* Install ulang torch (CPU version), atau
* Gunakan CPU saja

---

## 📁 Output

Aplikasi akan menghasilkan:

* `.srt` (subtitle)
* `_id.srt` (translate Indonesia, opsional)
* `_timeline.xml` (untuk editing timeline)

---

## 🎯 Tujuan Tool Ini

Dibuat untuk mempercepat workflow:

* Editor video
* Konten YouTube
* Subtitle automation

---

## 📌 Catatan

Ini bukan aplikasi installer.
Semua dependency harus di-install manual (atau via `requirements.txt`).

Jika ingin versi tanpa install (portable / .exe), perlu build terpisah.

---
