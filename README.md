# Predictive Machine Learning Model for Term Deposit Target Marketing Optimization

Repositori ini berisi proyek Machine Learning end-to-end untuk mengoptimalkan kampanye pemasaran deposito berjangka (*term deposit*) sebuah bank. Fokus utama proyek ini adalah meminimalkan *opportunity loss* (nasabah potensial yang terlewat) menggunakan pendekatan klasifikasi dengan matriks evaluasi **F2-Score** dan analisis bisnis berbasis finansial.

---

## 📌 1. Latar Belakang & Pernyataan Masalah

* **Latar Belakang:** Kampanye pemasaran langsung (telemarketing) membutuhkan alokasi sumber daya yang efisien. Menghubungi nasabah secara acak mengakibatkan membengkaknya biaya operasional dan potensi *contact fatigue* (nasabah merasa terganggu).
* **Pernyataan Masalah:** Bank mengalami kesulitan dalam memprediksi nasabah mana yang benar-benar berminat melakukan deposito berjangka. Kesalahan prediksi berupa **False Negative** (Model memprediksi nasabah tidak berminat, padahal aslinya berminat) menyebabkan kehilangan potensi dana segar (*lost opportunity cost*) sebesar **$250** per nasabah.
* **Solusi Bisnis:** Membangun model prediktif klasifikasi biner dan menggeser batas keputusan (*decision threshold*) secara optimal untuk memaksimalkan total manfaat bersih (*Net Financial Benefit*).

---

## 🎯 2. Metrik Evaluasi & Analytical Framing

* **Fokus Utama:** Mengurangi **Type 2 Error / False Negative (FN)**.
* **Metrik Utama:** 
  1. **Recall (Class 1):** Mengukur persentase nasabah berminat yang berhasil ditangkap model.
  2. **F2-Score:** Memberikan bobot 2x lebih penting pada *Recall* dibandingkan *Precision*.

| Aspek | Penjelasan |
| --- | --- |
| **Jenis Machine Learning** | Supervised Learning (Klasifikasi Biner) |
| **Target Variabel** | `0` = Tidak Deposit, `1` = Deposit |
| **Model Terpilih** | Random Forest Classifier (Optimized) |

---

## 📈 3. Temuan Utama Exploratory Data Analysis (EDA)

1. **Kondisi Finansial (`balance`):** Saldo rekening nasabah merupakan faktor penentu utama. Nasabah dengan saldo tinggi cenderung jauh lebih siap untuk melakukan investasi deposito.
2. **Kelompok Umur (`age_group`):** Nasabah lanjut usia (**Older Adults 61-75 tahun & Seniors >75 tahun**) dan kelompok mahasiswa muda (**Young Adults 18-25 tahun**) menunjukkan persentase konversi (*conversion rate*) tertinggi.
3. **Riwayat Kontak Sebelumnya (`pdays` & `poutcome`):** Nasabah yang sukses dikonversi di masa lalu memiliki probabilitas konversi ulang sebesar **91.5%**.
4. **Saluran Komunikasi (`contact`):** Panggilan seluler (*cellular*) terbukti jauh lebih efektif (55.11% konversi) dibandingkan panggilan telepon rumah (*telephone*).

---

## ⚙️ 4. Pipeline Preprocessing & Model Selection

Kami mengimplementasikan pipeline preprocessing yang kuat untuk mencegah kebocoran data (*data leakage*):
* **Fitur Kategorikal Nominal:** One-Hot Encoding.
* **Fitur Kategorikal Multiclass (`job`):** Binary Encoding.
* **Fitur Numerik:** Median Imputation & Robust Standard Scaling.

### Perbandingan Performa Model Benchmarking (5-Fold Stratified Cross-Validation):

| Model | Mean F2-Score | Std F2-Score |
| :--- | :---: | :---: |
| **Random Forest** | **0.6476** | **0.0122** |
| XGBoost | 0.6386 | 0.0144 |
| LightGBM | 0.6339 | 0.0105 |
| K-Nearest Neighbors | 0.6276 | 0.0117 |
| Logistic Regression | 0.6167 | 0.0155 |
| Decision Tree | 0.6108 | 0.0181 |

---

## 🚀 5. Optimalisasi Hyperparameter & Threshold Tuning

Setelah memilih **Random Forest**, kami melakukan Hyperparameter Tuning menggunakan `RandomizedSearchCV` diikuti dengan penyesuaian *Decision Threshold* berdasarkan kurva **Precision-Recall (PR-Curve)** untuk mengoptimalkan F2-Score:

* **Threshold Default:** `0.50`
* **Threshold Optimal (Tuned):** **`0.1048`**

### Hasil Perbandingan Sebelum vs Setelah Tuning (Data Test):

| Metrik Evaluasi | Model Default (Threshold 0.50) | Model Tuned (Threshold 0.1048) | Dampak Perubahan |
| :--- | :---: | :---: | :---: |
| **F2-Score Test** | 0.6381 | **0.8156** | **+27.81%** (Peningkatan Akurasi F2) |
| **Recall Test** | 0.6193 | **0.9866** | **+59.31%** (Hampir semua target ditangkap) |
| **Precision Test** | 0.7264 | 0.4817 | -33.68% (Kompensasi penargetan agresif) |
| **Jumlah False Negatives (FN)** | 284 nasabah | **10 nasabah** | **-96.47%** (Menekan peluang hilangnya nasabah) |

---

## 💵 6. Cost-Benefit Analysis (CBA) - Dampak Bisnis

Asumsi parameter finansial nyata dari bank:
* **Benefit True Positive (TP):** +$250 (Pendapatan deposito baru)
* **Cost False Positive (FP):** -$130 (Biaya operasional telemarketing & waktu sales)
* **Cost False Negative (FN):** -$250 (Opportunity cost/kehilangan profit)

### Tabel Perbandingan Finansial Riil pada Data Test:

| Komponen Evaluasi | Kondisi Default (Threshold 0.50) | Kondisi Optimized (Threshold 0.1048) | Dampak Finansial |
| :--- | :---: | :---: | :---: |
| **True Positives (TP)** | 462 nasabah ($115,500) | 736 nasabah ($184,000) | +$68,500 |
| **False Positives (FP)** | 174 nasabah (-$22,620) | 792 nasabah (-$102,960) | -$80,340 |
| **False Negatives (FN)** | 284 nasabah (-$71,000) | 10 nasabah (-$2,500) | +$68,500 |
| **Total Manfaat Bersih** | **$21,880** | **$78,540** | **+$56,660** |

> 🎯 **Kesimpulan Akhir Bisnis:** Penerapan model **Random Forest Tuned dengan Threshold 0.1048** berhasil meningkatkan keuntungan bersih dari kampanye pemasaran sebesar **+$56,660 (Kenaikan profit sebesar 258.9%)** pada data uji.

---

## 🛠️ 7. Cara Menjalankan Project Secara Lokal

### Prasyarat
Pastikan Anda sudah meng-install Python dan library berikut:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm category_encoders seaborn matplotlib plotly
```

### Load Model Menggunakan Pickle
Model final telah diekspor dengan aman menggunakan format `pickle`:
```python
import pickle
import pandas as pd

# Load model package
with open('random_forest_term_deposit_model.pkl', 'rb') as f:
    model_pack = pickle.load(f)
```

Terima kasih!
