# Laporan Proyek Machine Learning Terapan (Membuat Model Sistem Rekomendasi) - Kendrick Filbert

## Project Overview

Industri penerbitan dan literasi digital terus berkembang pesat. Platform seperti Goodreads, Amazon Books, dan Google Books kini menyediakan jutaan judul buku dari berbagai genre dan penulis. Namun, banyaknya pilihan justru sering membuat pengguna kesulitan menemukan buku yang sesuai dengan minat mereka — fenomena ini dikenal sebagai *information overload* [[1]](https://dl.acm.org/doi/10.1145/3285029).

Sistem rekomendasi hadir sebagai solusi atas permasalahan tersebut. Berdasarkan penelitian yang diterbitkan dalam *ACM Computing Surveys*, sistem rekomendasi terbukti meningkatkan kepuasan pengguna dan keterlibatan mereka dalam platform digital secara signifikan [[2]](https://dl.acm.org/doi/10.1145/2827872). Sistem rekomendasi bekerja dengan menganalisis data historis pengguna — seperti buku yang pernah dibaca, rating yang diberikan, serta preferensi konten — untuk memprediksi buku apa yang kemungkinan besar akan disukai pengguna tersebut.

Proyek ini membangun sistem rekomendasi buku menggunakan **Book-Crossing Dataset**, sebuah dataset publik yang kaya dengan lebih dari 270.000 judul buku dan lebih dari 1 juta data rating dari ratusan ribu pengguna. Dua pendekatan utama diterapkan:
1. **Content-Based Filtering** — merekomendasikan buku berdasarkan kemiripan fitur konten (penulis dan penerbit) menggunakan TF-IDF dan Cosine Similarity.
2. **Collaborative Filtering** — merekomendasikan buku berdasarkan pola rating pengguna menggunakan arsitektur Neural Network (RecommenderNet dengan Embedding Layer).

Dengan menggabungkan kedua pendekatan ini, sistem rekomendasi yang dibangun diharapkan mampu memberikan rekomendasi yang relevan, personal, dan berguna bagi setiap pengguna platform.

---

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, berikut adalah permasalahan yang ingin diselesaikan:

1. Bagaimana cara membangun sistem yang dapat merekomendasikan buku berdasarkan kemiripan fitur konten seperti penulis dan penerbit?
2. Bagaimana cara membangun sistem yang dapat merekomendasikan buku secara personal berdasarkan pola rating dari pengguna-pengguna lain yang memiliki selera serupa?
3. Bagaimana mengukur kualitas dan akurasi dari masing-masing sistem rekomendasi yang dibangun?

### Goals

Tujuan yang ingin dicapai dari proyek ini adalah:

1. Menghasilkan sistem rekomendasi berbasis konten (*Content-Based Filtering*) yang mampu merekomendasikan buku dengan penulis atau penerbit yang sama/serupa.
2. Menghasilkan sistem rekomendasi berbasis kolaborasi (*Collaborative Filtering*) menggunakan Neural Network yang mampu memberikan rekomendasi personal berdasarkan riwayat rating pengguna.
3. Mengevaluasi performa kedua model menggunakan metrik yang sesuai — **Precision@K** untuk Content-Based Filtering dan **RMSE** serta **MAE** untuk Collaborative Filtering.

### Solution Approach

Untuk mencapai tujuan di atas, dua solusi pendekatan diterapkan:

**Solusi 1: Content-Based Filtering**
- Menggunakan **TF-IDF Vectorizer** untuk mengubah fitur teks (nama penulis dan penerbit) menjadi representasi vektor numerik sebagai tahap *feature engineering*.
- Mengukur kemiripan antar buku menggunakan **Cosine Similarity**.
- Buku dengan nilai cosine similarity tertinggi terhadap buku acuan akan direkomendasikan.
- **Kelebihan:** Tidak memerlukan data pengguna lain (*user-independent*), mampu merekomendasikan buku baru yang belum pernah dirating (*cold start* untuk item baru), hasil mudah dijelaskan.
- **Kekurangan:** Terbatas pada fitur yang tersedia (hanya author dan publisher), cenderung menghasilkan rekomendasi yang terlalu serupa (*overspecialization*), tidak dapat menangkap preferensi personal.

**Solusi 2: Collaborative Filtering dengan Neural Network**
- Menggunakan arsitektur **RecommenderNet** berbasis *Embedding Layer* untuk merepresentasikan user dan buku dalam ruang vektor berdimensi rendah.
- Model mempelajari pola interaksi user-buku dari data rating historis melalui proses dot product antar embedding.
- **Kelebihan:** Mampu menemukan pola preferensi yang kompleks dan non-linear, menghasilkan rekomendasi yang lebih personal dan beragam.
- **Kekurangan:** Mengalami masalah *cold start* untuk pengguna atau buku baru, membutuhkan data rating dalam jumlah signifikan, lebih lambat dalam pelatihan dibanding model sederhana.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **Book-Crossing Dataset** yang tersedia secara publik di Kaggle.

| Jenis | Keterangan |
|---|---|
| Sumber | [Book-Crossing Dataset](https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset) |
| Pemilik | SHINIGAMI (Kaggle) |
| Lisensi | Open Database License (ODbL) |
| Jenis Berkas | CSV (zip ~25MB) |

**Tabel 1. Informasi Dataset**

Dataset terdiri dari 3 file CSV:

### Deskripsi Variabel

**1. Books.csv** — Berisi informasi metadata buku

| Variabel | Tipe | Deskripsi |
|---|---|---|
| `ISBN` | object | Kode unik internasional untuk setiap buku |
| `Book-Title` | object | Judul buku |
| `Book-Author` | object | Nama penulis buku |
| `Year-Of-Publication` | object | Tahun buku diterbitkan |
| `Publisher` | object | Nama penerbit buku |
| `Image-URL-S/M/L` | object | URL gambar sampul buku (small/medium/large) |

- **Jumlah data:** 271.360 buku
- **Missing values:** `Book-Author` (2), `Publisher` (2)

**2. Users.csv** — Berisi informasi demografis pengguna

| Variabel | Tipe | Deskripsi |
|---|---|---|
| `User-ID` | int64 | ID unik pengguna |
| `Location` | object | Lokasi pengguna (kota, negara bagian, negara) |
| `Age` | float64 | Usia pengguna |

- **Jumlah data:** 278.858 pengguna
- **Missing values:** `Age` (~39% data)

**3. Ratings.csv** — Berisi data rating buku oleh pengguna

| Variabel | Tipe | Deskripsi |
|---|---|---|
| `User-ID` | int64 | ID pengguna yang memberikan rating |
| `ISBN` | object | ISBN buku yang dirating |
| `Book-Rating` | int64 | Nilai rating (0 = implicit, 1–10 = explicit) |

- **Jumlah data:** 1.149.780 rating
- **Catatan:** Rating 0 merepresentasikan *implicit feedback* (buku pernah diakses namun tidak dinilai secara eksplisit).

### Exploratory Data Analysis

#### Distribusi Rating

![Distribusi Rating](https://raw.githubusercontent.com/kendrickassignment/book-recommendation-system/main/images/rating_distribution.png)

**Gambar 1. Distribusi Rating Buku**

Insight:
- Terdapat **716.109 implicit rating (0)** dan **433.671 explicit rating (1–10)**.
- Rating eksplisit paling sering diberikan adalah nilai **8**, menunjukkan pengguna cenderung memberi nilai positif pada buku yang mereka nilai.
- Distribusi rating eksplisit bersifat *left-skewed* (condong ke nilai tinggi), mengindikasikan pengguna umumnya hanya memberikan rating pada buku yang mereka sukai.

#### Top 10 Penulis Paling Produktif

![Top Authors](https://raw.githubusercontent.com/kendrickassignment/book-recommendation-system/main/images/top_authors.png)

**Gambar 2. Top 10 Penulis dengan Jumlah Buku Terbanyak**

Insight:
- **Agatha Christie** merupakan penulis dengan jumlah judul terbanyak dalam dataset (632 buku), diikuti William Shakespeare dan Stephen King.
- Dominasi penulis bergenre fiksi, misteri, dan thriller mencerminkan tren preferensi pembaca secara umum.

#### Distribusi Tahun Terbit

![Year Distribution](https://raw.githubusercontent.com/kendrickassignment/book-recommendation-system/main/images/year_distribution.png)

**Gambar 3. Distribusi Tahun Terbit Buku**

Insight:
- Sebagian besar buku diterbitkan antara **tahun 1980–2005**, dengan puncak sekitar tahun **2000**.
- Terdapat anomali data di mana beberapa buku memiliki tahun terbit yang tidak valid (tahun 0 atau > 2023), yang ditangani pada tahap Data Preparation.

#### Distribusi Usia Pengguna

![Age Distribution](https://raw.githubusercontent.com/kendrickassignment/book-recommendation-system/main/images/age_distribution.png)

**Gambar 4. Distribusi Usia Pengguna**

Insight:
- Usia rata-rata pengguna adalah **34,7 tahun** dengan median **32,0 tahun**.
- Pengguna terbanyak berada pada rentang usia **25–45 tahun**, menunjukkan platform ini dominan digunakan oleh kalangan dewasa muda.
- Terdapat nilai usia yang tidak valid (< 5 atau > 100) yang perlu dibersihkan.

#### Top 10 Publisher

![Top Publishers](https://raw.githubusercontent.com/kendrickassignment/book-recommendation-system/main/images/top_publishers.png)

**Gambar 5. Top 10 Publisher dengan Buku Terbanyak**

Insight:
- **Harlequin** mendominasi sebagai publisher dengan koleksi terbanyak, diikuti Silhouette dan Pocket.
- Keberadaan berbagai publisher besar memberikan variasi konten yang cukup untuk mendukung fitur Content-Based Filtering berbasis penerbit.

---

## Data Preparation

Berikut adalah tahapan-tahapan data preparation yang dilakukan beserta alasannya:

### 1. Menangani Missing Values pada Books.csv

```python
books['Book-Author'] = books['Book-Author'].fillna('Unknown Author')
books['Publisher']   = books['Publisher'].fillna('Unknown Publisher')
books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
books['Year-Of-Publication'] = books['Year-Of-Publication'].fillna(
    books['Year-Of-Publication'].median()
)
```

**Alasan:** Kolom `Book-Author` dan `Publisher` adalah fitur utama Content-Based Filtering. Nilai kosong diisi dengan string 'Unknown' agar buku tetap dapat diproses oleh TF-IDF Vectorizer tanpa kehilangan data lainnya. Untuk `Year-Of-Publication`, nilai tidak valid dikonversi ke NaN lalu diisi dengan nilai median agar distribusi tahun tidak terdistorsi.

### 2. Seleksi dan Standardisasi Kolom

```python
books_clean = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']].copy()
books_clean.columns = ['isbn', 'title', 'author', 'year', 'publisher']
books_clean = books_clean.drop_duplicates(subset='isbn').reset_index(drop=True)
```

**Alasan:** Kolom URL gambar tidak diperlukan untuk pemodelan sehingga dihapus untuk efisiensi memori. Penamaan kolom distandardisasi ke *lowercase* untuk kemudahan pemrosesan. Duplikasi berdasarkan ISBN dihapus untuk menghindari inkonsistensi.

### 3. Filter Rating Eksplisit

```python
ratings_explicit = ratings[ratings['rating'] > 0].copy()
# Hasil: 433.671 dari 1.149.780 total rating (37,7%)
```

**Alasan:** Hanya rating eksplisit (1–10) yang digunakan untuk Collaborative Filtering karena rating 0 merepresentasikan *implicit feedback* dengan semantik berbeda. Mencampurkan keduanya akan mengacaukan proses pembelajaran model.

### 4. Filter User dan Buku Aktif (Minimum Interaksi)

```python
active_users  = user_counts[user_counts >= 5].index   # minimal 5 rating
popular_books = book_counts[book_counts >= 5].index   # minimal 5 rating
# Hasil: 152.280 rating | 13.305 user | 14.513 buku unik
```

**Alasan:** User dengan kurang dari 5 rating tidak memberikan informasi yang cukup bagi model untuk mempelajari preferensinya. Buku yang sangat jarang dirating akan memiliki representasi embedding yang tidak andal. Filter ini sekaligus mengurangi ukuran data secara signifikan untuk efisiensi komputasi.

### 5. Encoding Label (User ID & ISBN)

```python
ratings_filtered['user_encoded'] = user_encoder.fit_transform(ratings_filtered['user_id'])
ratings_filtered['book_encoded'] = book_encoder.fit_transform(ratings_filtered['isbn'])
# Hasil: 13.305 encoded users | 14.513 encoded books
```

**Alasan:** *Embedding Layer* pada Keras membutuhkan input berupa integer berurutan mulai dari 0 sebagai indeks. `LabelEncoder` mengubah User-ID dan ISBN yang tidak berurutan menjadi integer sekuensial yang sesuai kebutuhan arsitektur model.

### 6. Normalisasi Rating ke Rentang [0, 1]

```python
ratings_filtered['rating_normalized'] = (ratings_filtered['rating'] - min_rating) / (max_rating - min_rating)
# Range asli: 1–10  →  Range ternormalisasi: 0.00–1.00
```

**Alasan:** Output layer model menggunakan fungsi aktivasi **sigmoid** yang menghasilkan nilai antara [0, 1]. Agar target prediksi sesuai dengan rentang output tersebut, rating harus dinormalisasi. Tanpa normalisasi, model tidak akan mampu belajar secara efektif.

### 7. Train-Test Split (80:20)

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Data training: 121.824 (80%) | Data testing: 30.456 (20%)
```

**Alasan:** Data dibagi menjadi 80% training dan 20% testing untuk mengevaluasi performa model pada data yang belum pernah dilihat. `random_state=42` digunakan untuk memastikan *reproducibility* eksperimen.

### 8. Pembuatan Fitur Gabungan (Feature Engineering)

```python
books_clean['content_features'] = (
    books_clean['author'].fillna('') + ' ' + books_clean['publisher'].fillna('')
)
# Filter buku yang ada di ratings_filtered → 13.776 buku untuk CBF
```

**Alasan:** Langkah ini menggabungkan kolom `author` dan `publisher` menjadi satu string teks sebagai fitur input untuk proses vektorisasi. Penggabungan dua fitur ini memungkinkan model Content-Based Filtering menangkap kemiripan buku dari dua dimensi sekaligus.

### 9. TF-IDF Vectorization (Feature Extraction)

```python
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(books_cbf['content_features'])
# Shape: (13.776 buku × 5.000 fitur teks)
```

**Alasan:** TF-IDF (*Term Frequency–Inverse Document Frequency*) merupakan tahap *feature extraction* yang mengubah data teks menjadi representasi numerik berbentuk matriks sebelum dapat digunakan oleh model. Proses ini termasuk dalam Data Preparation karena tujuannya adalah mempersiapkan data (mengubah format teks → numerik), bukan membangun model. Parameter `max_features=5000` digunakan untuk membatasi dimensi vektor agar komputasi tetap efisien. `stop_words='english'` menghapus kata-kata umum yang tidak bermakna (seperti "the", "and") sehingga hanya token yang informatif (nama penulis, nama penerbit) yang dipertahankan.

---

## Modeling and Result

### Model 1: Content-Based Filtering

**Cara Kerja:**

Content-Based Filtering merekomendasikan buku berdasarkan kemiripan fitur konten. Setelah data direpresentasikan sebagai vektor TF-IDF pada tahap Data Preparation, model menghitung kemiripan antar semua pasangan buku menggunakan Cosine Similarity, lalu merekomendasikan buku dengan skor kemiripan tertinggi terhadap buku acuan.

**Tahapan Modeling:**

**Cosine Similarity**

```python
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Shape: (13.776 × 13.776) — matriks simetris
```

Cosine similarity mengukur sudut antar dua vektor di ruang berdimensi tinggi. Nilai mendekati 1.0 berarti sangat mirip, mendekati 0.0 berarti sangat berbeda. Hasil perhitungan ini disimpan sebagai matriks simetris berukuran (n_buku × n_buku).

**Fungsi Rekomendasi**

Fungsi `get_content_based_recommendations()` mengambil buku acuan, mencari baris cosine similarity-nya, mengurutkan dari nilai tertinggi, dan mengembalikan top-N buku teratas (mengecualikan buku itu sendiri).

**Kelebihan:**
- Tidak memerlukan data pengguna lain (*user-independent*)
- Dapat merekomendasikan buku baru yang belum pernah dirating (*cold start* untuk item baru)
- Hasil mudah dijelaskan dan diinterpretasikan

**Kekurangan:**
- Terbatas pada fitur yang tersedia; tanpa fitur genre/sinopsis, kemiripan antar buku dari penulis berbeda sulit terdeteksi
- Cenderung menghasilkan rekomendasi dari penulis/penerbit yang sama (*overspecialization*)
- Tidak dapat menangkap preferensi personal pengguna

**Hasil Rekomendasi (Content-Based Filtering):**

Buku acuan: **Clara Callan** — *Richard Bruce Wright*, penerbit *HarperFlamingo Canada* (2001)

| No | Judul Buku | Penulis | Penerbit | Similarity Score |
|----|-----------|---------|----------|:---:|
| 1 | Native Son (Perennial Classics) | Richard A. Wright | Perennial | 0.5004 |
| 2 | Native Son | Richard Wright | Perennial | 0.5004 |
| 3 | Black Boy (American Hunger...) | Richard Wright | Harpercollins | 0.5000 |
| 4 | Native son | Richard Wright | Perennial Library | 0.4404 |
| 5 | Spy Catcher: The Candid Autobiography... | Peter Wright | Penguin USA | 0.3206 |
| 6 | Why I Hate Canadians | Will Ferguson | Harpercollins Canada | 0.3185 |
| 7 | Jade Peony | Wayson Choy | Harpercollins Canada | 0.2924 |
| 8 | Letters for Emily | Camron Wright | Pocket | 0.2907 |
| 9 | Dropped Threads 2: More of What We Aren't Told | Carol Shields | Vintage Books Canada | 0.2804 |
| 10 | Letters for Emily | Camron Wright | Atria | 0.2738 |

**Tabel 2. Hasil Top-10 Rekomendasi Content-Based Filtering**

Model berhasil menangkap kemiripan berbasis token teks pada fitur author dan publisher. Buku-buku dengan kata "Wright" pada nama penulis dan penerbit berafiliasi "Harper" mendapat similarity score tertinggi — menunjukkan TF-IDF dan Cosine Similarity bekerja sesuai desain.

---

### Model 2: Collaborative Filtering (Neural Network)

**Cara Kerja:**

Collaborative Filtering berbasis Neural Network menggunakan teknik *matrix factorization* yang diimplementasikan dengan *Embedding Layer*. Model mempelajari representasi vektor laten untuk setiap user dan buku, kemudian menggunakan dot product dari kedua embedding tersebut untuk memprediksi rating.

**Arsitektur RecommenderNet:**

```
Input: [user_encoded, book_encoded]
  ├── User Embedding Layer (50 dimensi) + User Bias
  └── Book Embedding Layer (50 dimensi) + Book Bias
       ↓
  Dot Product (user_vector · book_vector)
       ↓
  + User Bias + Book Bias
       ↓
  Sigmoid Activation → Output ∈ [0, 1]
```

**Parameter Training:**

| Parameter | Nilai |
|---|---|
| Loss Function | Binary Crossentropy |
| Optimizer | Adam (lr = 0.001) |
| Batch Size | 512 |
| Max Epochs | 50 |
| Early Stopping | patience = 5 (monitor: val_RMSE) |
| ReduceLROnPlateau | factor = 0.5, patience = 3 |
| Regularisasi | L2 (λ = 1e-6) |
| Embedding Size | 50 |

Model berhenti pada **epoch ke-26** karena Early Stopping aktif.

**Visualisasi Training:**

![Training History](https://raw.githubusercontent.com/kendrickassignment/book-recommendation-system/main/images/training_history.png)

**Gambar 6. Kurva Training Loss dan RMSE**

Dari grafik training dapat dilihat bahwa *training loss* menurun secara konsisten, sementara *validation loss* menunjukkan sedikit fluktuasi yang mengindikasikan *mild overfitting* — hal yang umum terjadi pada dataset dengan *sparsity* tinggi.

**Kelebihan:**
- Mampu menangkap pola preferensi yang kompleks dan non-linear
- Memberikan rekomendasi yang personal dan beragam antar pengguna
- Dapat ditingkatkan dengan menambah data interaksi

**Kekurangan:**
- Mengalami masalah *cold start* — tidak dapat merekomendasikan untuk pengguna baru atau buku baru
- Membutuhkan data rating yang cukup banyak (minimal interaksi)
- Lebih lambat dalam pelatihan dibanding Content-Based Filtering

**Hasil Rekomendasi (Collaborative Filtering):**

Riwayat buku dengan rating tertinggi dari **User 11676:**

| Judul | Penulis | Rating |
|-------|---------|:------:|
| Lirael: Daughter of the Clayr | Garth Nix | 10 |
| Sleeping Beauty | Phillip Margolin | 10 |
| Relato de un náufrago | Gabriel Garcia Marquez | 10 |
| La balsa de piedra | Jose Saramago | 10 |

Top-10 rekomendasi yang dihasilkan model untuk User 11676:

| No | Judul Buku | Penulis | Penerbit | Predicted Rating |
|----|-----------|---------|----------|:---:|
| 1 | Girl with a Pearl Earring | Tracy Chevalier | Plume Books | 7.19 |
| 2 | Little Altars Everywhere: A Novel | Rebecca Wells | Perennial | 7.18 |
| 3 | Nickel and Dimed: On (Not) Getting By in America | Barbara Ehrenreich | Owl Books | 7.17 |
| 4 | Harry Potter and the Prisoner of Azkaban (Book 3) | J. K. Rowling | Scholastic | 7.16 |
| 5 | Tears of the Moon (Irish Trilogy) | Nora Roberts | Jove Books | 7.15 |
| 6 | The Hobbit : The Enchanting Prelude to The Lord... | J.R.R. TOLKIEN | Del Rey | 7.14 |
| 7 | Survival of the Fittest: An Alex Delaware Novel | Jonathan Kellerman | Bantam | 7.13 |
| 8 | It | Stephen King | Signet Book | 7.13 |
| 9 | Nathaniel | John Saul | Bantam Books | 7.11 |

**Tabel 3. Hasil Top-10 Rekomendasi Collaborative Filtering untuk User 11676**

Model merekomendasikan buku-buku populer berkualitas tinggi (Harry Potter, The Hobbit, karya Stephen King) yang sejalan dengan profil pembaca fiksi literatur tinggi seperti User 11676.

---

## Evaluation

### Metrik Evaluasi 1: Precision@K (Content-Based Filtering)

**Definisi dan Formula:**

Precision@K mengukur proporsi item yang relevan dari K item teratas yang direkomendasikan:

$$\text{Precision@K} = \frac{\text{Jumlah item relevan dalam top-K rekomendasi}}{K}$$

**Cara kerja:** Dalam konteks proyek ini, sebuah rekomendasi dianggap **relevan** jika buku yang direkomendasikan memiliki **penulis yang sama** dengan buku acuan. Definisi relevansi ini dipilih karena fitur utama model adalah nama penulis — sehingga ukuran keberhasilan yang paling tepat adalah seberapa konsisten model merekomendasikan karya dari penulis yang sama.

**Hasil Evaluasi Precision@10:**

| Buku Acuan | Penulis | Relevan/K | Precision@10 |
|-----------|---------|:---------:|:---:|
| The Kitchen God's Wife | Amy Tan | 10/10 | **1.00** |
| The Testament | John Grisham | 4/10 | **0.40** |
| Beloved (Plume Contemporary Fiction) | Toni Morrison | 8/10 | **0.80** |
| Airframe | Michael Crichton | 5/10 | **0.50** |
| Timeline | Michael Crichton | 5/10 | **0.50** |

**Tabel 4. Hasil Evaluasi Precision@10 Content-Based Filtering**

**Rata-rata Precision@10 = 0.64 (64%)**

**Interpretasi:** Dari setiap 10 buku yang direkomendasikan, rata-rata **6–7 buku** berasal dari penulis yang sama. Variasi yang cukup besar antar buku uji (0.40–1.00) disebabkan oleh perbedaan jumlah karya per penulis dalam dataset — penulis produktif seperti Amy Tan mendapat skor sempurna, sementara penulis dengan karya lebih terbatas mendapat skor lebih rendah karena model terpaksa mengisi slot dengan buku dari penulis lain. Nilai Precision@10 sebesar **64%** ini merupakan hasil yang wajar mengingat model hanya menggunakan dua fitur (author dan publisher) tanpa informasi genre atau sinopsis.

---

### Metrik Evaluasi 2: RMSE dan MAE (Collaborative Filtering)

**1. Root Mean Squared Error (RMSE)**

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{r}_i - r_i)^2}$$

Di mana $\hat{r}_i$ adalah rating prediksi dan $r_i$ adalah rating aktual. RMSE memberikan penalti lebih besar pada kesalahan yang besar karena menggunakan kuadrat selisih, sehingga lebih sensitif terhadap *outlier*.

**2. Mean Absolute Error (MAE)**

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\hat{r}_i - r_i|$$

MAE menghitung rata-rata nilai absolut selisih prediksi dan aktual. Lebih mudah diinterpretasikan dan lebih *robust* terhadap *outlier* dibanding RMSE.

**Hasil Evaluasi:**

| Metrik | Nilai |
|--------|:---:|
| RMSE | **2.6180** |
| MAE | **2.2840** |

**Tabel 5. Hasil Evaluasi Metrik Collaborative Filtering**

**Visualisasi Prediksi vs Aktual:**

![Prediction vs Actual](https://raw.githubusercontent.com/kendrickassignment/book-recommendation-system/main/images/prediction_vs_actual.png)

**Gambar 7. Scatter Plot Rating Aktual vs Rating Prediksi**

**Interpretasi Hasil:**

Nilai RMSE **2.62** dan MAE **2.28** pada skala 1–10 menunjukkan model rata-rata meleset sekitar 2 poin dari rating aktual. Hasil ini perlu dianalisis dalam konteks karakteristik dataset:

- **Sparsity ekstrem:** Setelah filtering, tersisa 152.280 rating dari potensi 13.305 × 14.513 = ~193 juta interaksi — artinya **kepadatan data hanya ~0.079%**. Ini adalah salah satu tantangan terbesar dalam sistem rekomendasi berbasis Collaborative Filtering.
- **Distribusi rating tidak merata:** Mayoritas pengguna hanya memberikan 5–10 rating, sehingga model kesulitan mempelajari preferensi yang beragam dari data yang sangat sedikit per pengguna.
- **Konteks perbandingan:** Berdasarkan literatur sistem rekomendasi, RMSE di kisaran 1.5–3.0 pada skala 1–10 adalah nilai yang **wajar dan umum** untuk dataset buku dengan *sparsity* ekstrem seperti Book-Crossing [[2]](https://dl.acm.org/doi/10.1145/2827872).
- **Kualitas rekomendasi tetap baik:** Meski RMSE relatif tinggi, model terbukti mampu menghasilkan rekomendasi yang bermakna — top-10 untuk User 11676 menampilkan buku-buku populer dan berkualitas yang selaras dengan profil selera pengguna tersebut.
- **Arah perbaikan:** Performa dapat ditingkatkan dengan menambah fitur kontekstual (genre, sinopsis), menggunakan arsitektur yang lebih dalam, atau menerapkan teknik *hybrid filtering* yang menggabungkan kedua pendekatan.

---

## Conclusion

Proyek ini berhasil membangun dua sistem rekomendasi buku menggunakan pendekatan yang berbeda:

- **Content-Based Filtering** dengan TF-IDF + Cosine Similarity menghasilkan **Precision@10 rata-rata 64%**. Model efektif merekomendasikan buku dari penulis yang sama, namun terbatas pada fitur yang tersedia. Performa dapat ditingkatkan dengan menambahkan fitur genre dan sinopsis.

- **Collaborative Filtering** dengan RecommenderNet menghasilkan **RMSE 2.62 dan MAE 2.28** pada skala 1–10. Nilai ini wajar mengingat *sparsity* data yang sangat tinggi (~99.9%). Model tetap mampu menghasilkan rekomendasi yang relevan dan personal terbukti dari kualitas rekomendasi untuk pengguna uji.

Kedua pendekatan bersifat saling melengkapi: Content-Based Filtering unggul dalam menangani *cold start* untuk item baru dan mudah dijelaskan, sementara Collaborative Filtering lebih personal dan mampu menemukan pola preferensi tersembunyi yang tidak dapat ditangkap dari fitur konten semata.

---

## Referensi

[[1]](https://dl.acm.org/doi/10.1145/3285029) Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.

[[2]](https://dl.acm.org/doi/10.1145/2827872) Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer*, 42(8), 30–37.

[[3]](https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset) Ziegler, C. N., McNee, S. M., Konstan, J. A., & Lausen, G. (2005). Improving Recommendation Lists Through Topic Diversification. *Proceedings of WWW '05*.

[[4]](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

[[5]](https://www.tensorflow.org/api_docs/python/tf/keras) Chollet, F. et al. (2015). *Keras: Deep Learning for Humans*. https://keras.io.
