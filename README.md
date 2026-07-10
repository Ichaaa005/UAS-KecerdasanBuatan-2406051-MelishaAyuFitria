# UAS-KecerdasanBuatan-2406051-MelishaAyuFitria
## Prediksi Penyakit Jantung Menggunakan Algoritma Decision Tree dan K-Nearest Neighbors

## Deskripsi Proyek
Proyek ini dibuat untuk memenuhi tugas UAS mata kuliah **Kecerdasan Buatan**.  
Topik yang dipilih adalah **Prediksi Penyakit Jantung (Heart Disease Prediction)** menggunakan teknik **machine learning klasifikasi**.

Pada proyek ini digunakan dataset **Heart Disease** yang berisi data klinis pasien seperti usia, tekanan darah, kadar kolesterol, detak jantung maksimum, dan atribut kesehatan lainnya. Tujuan proyek adalah membangun model yang dapat memprediksi apakah seseorang terindikasi memiliki penyakit jantung atau tidak.

Algoritma yang digunakan:
- **Decision Tree**
- **K-Nearest Neighbors (KNN)**

---

## Tujuan Proyek
1. Memprediksi kemungkinan penyakit jantung berdasarkan data klinis pasien.
2. Membandingkan performa algoritma Decision Tree dan K-Nearest Neighbors.
3. Menentukan model terbaik berdasarkan metrik evaluasi klasifikasi.

---

## Dataset
Dataset yang digunakan adalah **Heart Disease Dataset** dengan nama file:

```bash
heart.csv
```

Dataset berisi **303 baris** dan **14 kolom**, dengan target klasifikasi:
- **0** = tidak terindikasi penyakit jantung
- **1** = terindikasi penyakit jantung

Lokasi dataset dalam repository:
```bash
data/dataset/heart.csv
```

---

## Struktur Repository
```bash
UAS-KecerdasanBuatan/
├── README.md
├── Laporan_uas.md
├── uas_model.ipynb
└── data/
    ├── dataset/
    │   └── heart.csv
    └── Jurnal/
        ├── jurnal_1.pdf
        ├── jurnal_2.pdf
        ├── jurnal_3.pdf
        ├── jurnal_4.pdf
        └── jurnal_5.pdf
```

---

## Langkah Pengerjaan Proyek

### 1. Menentukan topik proyek
Topik yang dipilih adalah **Prediksi Penyakit Jantung** karena termasuk permasalahan klasifikasi yang umum digunakan dalam machine learning serta memiliki dataset yang cukup lengkap.

### 2. Mengumpulkan dataset
Dataset yang digunakan adalah **Heart Disease Dataset** yang berisi atribut medis pasien, seperti:
- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal
- target

### 3. Data Understanding
Tahap ini dilakukan untuk memahami isi dataset, meliputi:
- melihat jumlah baris dan kolom
- melihat tipe data
- mengecek missing value
- mengecek duplikasi data
- memahami arti setiap fitur
- melihat distribusi target

### 4. Exploratory Data Analysis (EDA)
Pada tahap ini dilakukan eksplorasi data dengan visualisasi, seperti:
- countplot distribusi target
- histogram fitur numerik
- heatmap korelasi
- boxplot fitur terhadap target
- countplot fitur kategorikal terhadap target

Tujuan EDA adalah untuk memahami pola data dan hubungan antar fitur.

### 5. Data Preparation
Tahap persiapan data meliputi:
- memisahkan fitur (`X`) dan target (`y`)
- membagi data menjadi data latih dan data uji
- melakukan standardisasi data untuk algoritma KNN

### 6. Modeling
Dua algoritma yang digunakan:
- **Decision Tree**
- **K-Nearest Neighbors (KNN)**

Model dilatih menggunakan data latih, lalu dilakukan prediksi pada data uji.

### 7. Evaluation
Evaluasi model dilakukan menggunakan:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

Hasil evaluasi kedua model kemudian dibandingkan untuk menentukan model terbaik.

### 8. Kesimpulan
Dari hasil evaluasi, diperoleh model dengan performa terbaik untuk memprediksi penyakit jantung berdasarkan dataset yang digunakan.

---

## Tools dan Library
Project ini dikerjakan menggunakan:
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

---

## Cara Menjalankan Project
1. Buka file `uas_model.ipynb`
2. Pastikan file dataset `heart.csv` berada pada folder:
   ```bash
   data/dataset/
   ```
3. Jalankan notebook dari atas ke bawah.
4. Hasil evaluasi model akan muncul pada bagian akhir notebook.

---

## Hasil Singkat
Berdasarkan pengujian awal, model **K-Nearest Neighbors (KNN)** memperoleh performa yang lebih baik dibandingkan Decision Tree pada dataset ini.

---

## Penulis
- Nama: **Melisha Ayu Fitria**
- Mata Kuliah: **Kecerdasan Buatan**
- Kelas: **B-Informatika**
