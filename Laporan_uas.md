# Laporan Proyek Machine Learning  
# Prediksi Penyakit Jantung Menggunakan Algoritma Decision Tree dan K-Nearest Neighbors

## 1. Domain Proyek / Latar Belakang
Penyakit jantung merupakan salah satu penyebab kematian tertinggi di dunia. Deteksi dini terhadap kemungkinan penyakit jantung sangat penting agar pasien dapat memperoleh penanganan lebih cepat dan tepat. Dengan memanfaatkan data klinis pasien seperti usia, tekanan darah, kadar kolesterol, detak jantung maksimum, dan indikator medis lainnya, machine learning dapat digunakan untuk membantu memprediksi apakah seseorang berisiko mengalami penyakit jantung.

Pada proyek ini dilakukan pembangunan model machine learning untuk **memprediksi penyakit jantung** menggunakan dataset medis pasien. Model yang digunakan adalah **Decision Tree** dan **K-Nearest Neighbors (KNN)**, kemudian kinerjanya dibandingkan menggunakan metrik evaluasi klasifikasi.

---

## 2. Business Understanding

### 2.1 Problem Statement
Berdasarkan latar belakang di atas, permasalahan yang ingin diselesaikan pada proyek ini adalah:
1. Bagaimana membangun model machine learning yang dapat memprediksi kemungkinan seseorang mengalami penyakit jantung berdasarkan data klinis?
2. Bagaimana performa algoritma **Decision Tree** dan **K-Nearest Neighbors** dalam melakukan prediksi penyakit jantung?
3. Algoritma mana yang memberikan performa terbaik berdasarkan metrik evaluasi klasifikasi?

### 2.2 Goals
Tujuan dari proyek ini adalah:
1. Membangun model klasifikasi untuk memprediksi penyakit jantung berdasarkan fitur-fitur medis pada dataset.
2. Membandingkan performa algoritma **Decision Tree** dan **K-Nearest Neighbors**.
3. Menentukan model terbaik berdasarkan nilai **Accuracy, Precision, Recall, dan F1-score**.

### 2.3 Solution Statement
Solusi yang diajukan dalam proyek ini adalah:
1. Menggunakan dataset **Heart Disease** yang berisi atribut medis pasien.
2. Melakukan tahap **data understanding**, **data preparation**, **modeling**, dan **evaluation**.
3. Membangun dua model klasifikasi, yaitu:
   - **Decision Tree**
   - **K-Nearest Neighbors (KNN)**
4. Membandingkan performa kedua model menggunakan confusion matrix dan metrik evaluasi klasifikasi.

---

## 3. Data Understanding

### 3.1 Sumber Dataset
Dataset yang digunakan adalah **Heart Disease Dataset** dalam file `heart.csv`, yang berisi data klinis pasien untuk memprediksi kemungkinan penyakit jantung.

### 3.2 Jumlah Data
Dataset terdiri dari:
- **303 baris data**
- **14 kolom**

### 3.3 Deskripsi Fitur
Berikut penjelasan masing-masing fitur pada dataset:

| No | Nama Fitur | Deskripsi |
|---|---|---|
| 1 | age | Usia pasien |
| 2 | sex | Jenis kelamin pasien (1 = laki-laki, 0 = perempuan) |
| 3 | cp | Tipe nyeri dada (chest pain type) |
| 4 | trestbps | Tekanan darah saat istirahat |
| 5 | chol | Kadar kolesterol serum |
| 6 | fbs | Gula darah puasa > 120 mg/dl |
| 7 | restecg | Hasil elektrokardiografi saat istirahat |
| 8 | thalach | Detak jantung maksimum yang dicapai |
| 9 | exang | Angina yang dipicu olahraga |
| 10 | oldpeak | Depresi ST akibat olahraga relatif terhadap istirahat |
| 11 | slope | Kemiringan segmen ST saat olahraga puncak |
| 12 | ca | Jumlah pembuluh darah utama yang diwarnai fluoroskopi |
| 13 | thal | Kondisi thalassemia |
| 14 | target | Target klasifikasi (1 = memiliki penyakit jantung, 0 = tidak) |

### 3.4 Kondisi Data
Hasil eksplorasi awal menunjukkan bahwa:
- dataset memiliki **303 data** dan **14 atribut**
- **tidak terdapat missing value**
- target klasifikasi adalah **`target`**
- distribusi kelas cukup seimbang, dengan jumlah data pasien yang terindikasi penyakit jantung sedikit lebih banyak dibanding pasien yang tidak terindikasi

Distribusi target pada dataset:
- **1 (Heart Disease)** = 165 data
- **0 (No Heart Disease)** = 138 data

### 3.5 Exploratory Data Analysis
Pada tahap EDA dilakukan beberapa visualisasi seperti:
1. distribusi target penyakit jantung,
2. distribusi fitur numerik,
3. heatmap korelasi antar fitur,
4. boxplot umur, kolesterol, dan detak jantung maksimum terhadap target,
5. distribusi tipe nyeri dada terhadap target.

EDA bertujuan untuk memahami pola data, melihat persebaran fitur, serta mengetahui fitur-fitur yang berpotensi berpengaruh terhadap target.

---

## 4. Data Preparation

### 4.1 Pengecekan Missing Value
Dataset diperiksa untuk mengetahui apakah terdapat data kosong. Hasil pengecekan menunjukkan bahwa **tidak terdapat missing value** pada seluruh atribut.

### 4.2 Pengecekan Duplikasi
Dataset juga diperiksa untuk mengetahui adanya data duplikat. Jika ditemukan data duplikat, data tersebut dapat dihapus agar tidak memengaruhi proses pelatihan model.

### 4.3 Pemisahan Fitur dan Target
Kolom **`target`** digunakan sebagai variabel target, sedangkan kolom lainnya digunakan sebagai fitur input model.

- **Fitur (X)**: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
- **Target (y)**: `target`

### 4.4 Pembagian Data
Dataset dibagi menjadi:
- **80% data latih**
- **20% data uji**

Pembagian dilakukan menggunakan `train_test_split()` dengan parameter `stratify=y` agar proporsi kelas pada data latih dan data uji tetap seimbang.

### 4.5 Standardisasi Data
Standardisasi dilakukan khusus untuk algoritma **K-Nearest Neighbors (KNN)** menggunakan **StandardScaler**. Hal ini dilakukan karena KNN menghitung jarak antar data, sehingga skala fitur yang berbeda dapat memengaruhi hasil prediksi.

Decision Tree tidak memerlukan standardisasi karena model ini tidak berbasis perhitungan jarak.

---

## 5. Modeling

### 5.1 Algoritma yang Digunakan
Pada proyek ini digunakan dua algoritma klasifikasi, yaitu:

#### 1. Decision Tree
Decision Tree adalah algoritma klasifikasi yang bekerja dengan membentuk struktur pohon keputusan berdasarkan fitur-fitur yang paling informatif. Model ini mudah dipahami karena dapat menunjukkan alur keputusan dari akar hingga daun.

#### 2. K-Nearest Neighbors (KNN)
KNN adalah algoritma klasifikasi yang menentukan kelas suatu data berdasarkan mayoritas kelas dari sejumlah tetangga terdekat. Pada proyek ini digunakan nilai awal **K = 7**, dan model dapat dituning untuk mencari nilai K terbaik.

### 5.2 Proses Pelatihan Model
- **Decision Tree** dilatih menggunakan data latih tanpa standardisasi.
- **KNN** dilatih menggunakan data latih yang telah distandardisasi.

---

## 6. Evaluation

### 6.1 Metrik Evaluasi
Untuk mengevaluasi performa model, digunakan metrik:
1. **Accuracy** → proporsi prediksi benar dari seluruh data
2. **Precision** → ketepatan model saat memprediksi kelas positif
3. **Recall** → kemampuan model menemukan seluruh data positif
4. **F1-score** → rata-rata harmonik precision dan recall
5. **Confusion Matrix** → tabel untuk melihat jumlah prediksi benar dan salah per kelas

### 6.2 Hasil Evaluasi Model
Berdasarkan pengujian awal pada dataset `heart.csv`, diperoleh hasil evaluasi sebagai berikut:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Decision Tree | 0.7869 | 0.7632 | 0.8788 | 0.8169 |
| KNN | 0.8197 | 0.7895 | 0.9091 | 0.8451 |

### 6.3 Analisis Hasil
Berdasarkan hasil evaluasi, model **K-Nearest Neighbors (KNN)** memperoleh performa yang lebih baik dibandingkan Decision Tree. Hal ini ditunjukkan oleh nilai **accuracy**, **recall**, dan **F1-score** yang lebih tinggi.

Nilai **recall** yang tinggi pada model KNN menunjukkan bahwa model lebih baik dalam mendeteksi pasien yang terindikasi memiliki penyakit jantung. Dalam konteks prediksi penyakit, recall menjadi metrik yang penting karena kesalahan dalam tidak mendeteksi pasien yang sebenarnya sakit dapat berdampak besar.

Sementara itu, Decision Tree juga memberikan hasil yang cukup baik dan memiliki keunggulan dari sisi interpretasi model karena struktur pohonnya lebih mudah dipahami. Namun, berdasarkan hasil evaluasi kuantitatif, KNN lebih unggul untuk dataset ini.

---

## 7. Kesimpulan
Berdasarkan proyek yang telah dilakukan, model machine learning dapat digunakan untuk memprediksi penyakit jantung berdasarkan data klinis pasien. Dua algoritma yang digunakan, yaitu **Decision Tree** dan **K-Nearest Neighbors**, sama-sama mampu melakukan klasifikasi dengan cukup baik.

Berdasarkan hasil evaluasi menggunakan accuracy, precision, recall, dan F1-score, model dengan performa terbaik adalah **K-Nearest Neighbors (KNN)** dengan nilai:
- **Accuracy = 0.8197**
- **Precision = 0.7895**
- **Recall = 0.9091**
- **F1-score = 0.8451**

Dengan demikian, model **KNN** lebih direkomendasikan untuk digunakan pada dataset ini dalam memprediksi kemungkinan penyakit jantung.

---

## 8. Saran Pengembangan
Untuk pengembangan selanjutnya, model dapat ditingkatkan dengan:
1. menambahkan algoritma lain seperti **Random Forest**, **Logistic Regression**, atau **Support Vector Machine**,
2. melakukan **hyperparameter tuning** agar performa model lebih optimal,
3. melakukan **feature selection** untuk mengetahui atribut yang paling berpengaruh,
4. menggunakan dataset yang lebih besar agar model memiliki kemampuan generalisasi yang lebih baik.

---

## 9. Referensi
1. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.
2. Han, J., Kamber, M., & Pei, J. (2012). *Data Mining: Concepts and Techniques*. Morgan Kaufmann.
3. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O’Reilly Media.
4. Dokumentasi Scikit-learn: https://scikit-learn.org/
