# Deteksi Bahasa Isyarat (BISINDO)

## 📖  Deskripsi
Deteksi Bahasa Isyarat Indonesia (BISINDO) merupakan aplikasi berbasis **Computer Vision** dan **Machine Learning** yang dikembangkan untuk mengenali gesture bahasa isyarat Indonesia secara **real-time** menggunakan kamera (webcam). Sistem ini memanfaatkan **MediaPipe Hands** untuk mendeteksi tangan dan mengekstraksi 21 titik landmark pada setiap tangan. Landmark tersebut selanjutnya diproses menjadi data numerik melalui proses normalisasi sehingga menghasilkan fitur yang digunakan sebagai input bagi model **Multi-Layer Perceptron (MLP)**.
Model MLP yang telah dilatih menggunakan dataset gesture BISINDO mampu melakukan klasifikasi gesture secara otomatis. Hasil prediksi kemudian dikirimkan oleh backend Flask ke halaman web sehingga pengguna dapat melihat hasil deteksi beserta tingkat keyakinan (confidence) model secara langsung. Aplikasi ini diharapkan dapat menjadi media pembelajaran bahasa isyarat Indonesia sekaligus menjadi implementasi nyata penerapan Computer Vision dan Machine Learning.

## ✨ Fitur
- Deteksi tangan secara real-time menggunakan webcam
- Mendeteksi maksimal dua tangan sekaligus
- Ekstraksi 21 titik landmark tangan menggunakan MediaPipe Hands
- Normalisasi koordinat landmark terhadap titik wrist
- Pembentukan 84 fitur sebagai input model
- Klasifikasi gesture menggunakan algoritma Multi-Layer Perceptron (MLP)
- Menampilkan hasil prediksi gesture secara langsung
- Menampilkan nilai confidence (akurasi prediksi)
- Backend menggunakan Flask
- Antarmuka web menggunakan HTML dan CSS

## Teknologi yang Digunakan

**Bahasa Pemrograman**
- Python 

**Computer Vision**
- OpenCV
- MediaPipe Hands

**Machine Learning**
- Scikit-Learn
- Multi-Layer Perceptron (MLP)

**Framework**
- Flask

**Frontend**
- HTML5
- CSS3
- JavaScript (Fetch API)

**Library**
- NumPy
- Joblib

## 📂 Struktur Proyek

```text
Bisindo_Bahasa_Isyarat/
├── dataset/
│   ├── huruf/
│   ├── angka/
│   └── kata/
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── detect.py
├── train_model.py
└── model.pkl
```

## 📊 Dataset
Dataset (43 kelas: huruf A-Z, angka 0-9, dan 5 kata isyarat) tersedia di google drive https://drive.google.com/drive/folders/1WC7VNRV1xNMGwEqoYjkYqDOTo3Q_jFKR?usp=drive_link

caranya:

1. Download dan extract isi dataset dari link di atas.
2. Letakkan dengan struktur folder berikut:
```
dataset/

├── angka/
│   ├── 1/
│   └── 2/
├── huruf/
│   ├── A/
│   └── B/
├── kata/
│   ├── besok/
│   └── kakak/
```
3. Jalankan train_model.py setelah dataset siap untuk menghasilkan model sendiri.

## Kategori Dataset
**Huruf**
- A–Z (26 kelas)
  
**Angka**
- 0–9 (10 kelas)
  
**Kata BISINDO**
- Kenapa
- Malu
- Kakak
- Besok
- Membaca

## 🚀 Instalasi

Clone repository

```
git clone https://github.com/dinirivani04/Deteksi-Bahasa-Isyarat.git
```

Masuk ke folder project

```
cd Deteksi-Bahasa-Isyarat
```

Install dependency

```
pip install -r requirements.txt
```

## 🧠 Training Model

```
python train_model.py
```

Model hasil training akan disimpan menjadi

```
model.pkl
```

## 📈 Hasil Sistem

Model mampu mengenali gesture bahasa isyarat Indonesia berdasarkan landmark tangan yang diperoleh dari MediaPipe Hands.

Output sistem meliputi:

- Label gesture
- Nilai confidence
- Tampilan hasil secara real-time
- Streaming video melalui browser

## 👩‍💻 Pengembang

**Nama :** Dini Rivani

**Nim :** 2023573010012

**Mahasiswa :** Teknik Informatika

**Jurusan :** Teknologi Informasi dan Komputer

**Politeknik Negeri Lhokseumawe**

**Mata Kuliah:** Computer Vision

**Tahun Akademik:** 2025/2026

Video demo tampilan dapat dilihat pada youtube dibawah ini

https://youtu.be/AYVFJl8Wag0?si=lcFI6sewqzM70JAq 

## 📄 Lisensi

Project ini dibuat untuk keperluan pembelajaran, penelitian, dan tugas akademik pada Program Studi Teknik Informatika Politeknik Negeri Lhokseumawe. Dataset yang digunakan merupakan dataset dari kagle dan dataset pribadi yang dikumpulkan secara mandiri menggunakan webcam.
