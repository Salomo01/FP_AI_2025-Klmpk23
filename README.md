# FP_AI_2025-Klmpk23

# Aplikasi Web Face Similarity 🔍

Sebuah aplikasi web sederhana untuk **membandingkan dua gambar wajah** dan mengukur tingkat kemiripannya menggunakan deep learning. Proyek ini memanfaatkan [DeepFace](https://github.com/serengil/deepface) — sebuah framework Python yang ringan untuk pengenalan wajah dan analisis atribut wajah.

## 🧠 Konsep dan Model

Aplikasi ini menggunakan library **DeepFace**, yang mendukung berbagai model pengenalan wajah seperti:

- VGG-Face
- Facenet
- OpenFace
- DeepFace (default)
- Dlib
- ArcFace
- SFace

Secara default, aplikasi ini menggunakan **model VGG-Face**, yaitu sebuah jaringan saraf convolutional (CNN) yang sudah dilatih (pre-trained) pada dataset wajah yang besar. Model ini mengekstrak **embedding** (fitur wajah dalam bentuk vektor) dari setiap gambar, lalu menghitung **jarak cosine** antar dua embedding tersebut.

Semakin kecil nilai jaraknya, semakin mirip wajah yang dibandingkan.

## 📦 Teknologi yang Digunakan

- **Python**
- **Flask** – backend web
- **DeepFace** – pengenalan wajah dan perhitungan jarak
- **HTML/CSS** – antarmuka pengguna
- **Werkzeug** – untuk menangani file upload

## 📁 Struktur Proyek

```
face_similarity_app/
├── app.py # Aplikasi utama Flask
├── templates/
│ └── index.html # Antarmuka pengguna
├── uploads/ # Folder sementara untuk gambar upload
├── env/ # Virtual environment Python (gitignore)
└── requirements.txt # Daftar dependensi Python
```


## 🚀 Cara Kerja

1. Pengguna mengunggah **dua gambar wajah**.
2. Aplikasi akan menggunakan **DeepFace** untuk:
   - Mendeteksi wajah dalam gambar
   - Mengekstrak fitur (embedding)
   - Menghitung jarak antar fitur
3. Hasil ditampilkan:
   - Apakah wajah cocok (**Match**) atau tidak (**Not Match**)
   - Nilai **distance (jarak kemiripan)**

## ✅ Threshold (Ambang Batas)

DeepFace secara default menggunakan threshold **0.4** untuk kebanyakan model:
- Jika jarak ≤ 0.4 → Wajah dianggap sama (Match)
- Jika jarak > 0.4 → Wajah dianggap berbeda (Not Match)

Threshold ini bisa disesuaikan sesuai kebutuhan. Sedangkan untuk project ini threshold yg digunakan adalah **"6,8"**.

```
@app.route('/upload', methods=['POST'])
def upload():
   
    res = DeepFace.verify(path1, path2)

    # Cetak semua hasil ke console server
    print("DeepFace.verify result:", res)

    return jsonify(res)

```

Hal ini terlihat ketika mengecek log servernya, kemudian ketika file dijalankan terdapat hasil seperti ini:

```
DeepFace.verify result: {
  'verified': True,
  'distance': 0.6357,
  'max_threshold_to_verify': 0.68,
  'model': 'VGG-Face',
}

```

Berikut adalah dokumentasi untuk hasil pengerjaan:
Saya mencoba untuk membandingkan kedua foto berikut,

![Image](https://github.com/user-attachments/assets/c9bb477d-0792-409a-be94-8f7c30c89a8c)

![Image](https://github.com/user-attachments/assets/872f2b7b-a767-4c20-9e06-e3726fcbc747)

Kemudian hasil yg didapatkan seperti ini:

![Image](https://github.com/user-attachments/assets/767b986a-540d-410c-9a43-4c1db60bf834)


Hasil tersebut menyatakan bahwa dua wajah yang diunggah terdeteksi mirip oleh app-nya. Penjelasannya:

**Jarak (distance) = 0.6357**
DeepFace mengubah setiap wajah jadi vektor “fitur” di ruang multidimensi, lalu menghitung jarak (misalnya Euclidean) antara dua vektor itu. Semakin kecil jaraknya, semakin mirip kedua wajah.

**Threshold bawaan ≈ 0.68**
DeepFace yang dipakai menggunakan ambang (threshold) 0.68 untuk memutuskan “match” vs “no match.”

Kalau distance < threshold → verified = true (✅ Faces match).

Kalau distance ≥ threshold → verified = false (❌ Faces do not match).

**0.6357 < threshold**
Karena 0.6357 berada di bawah ambang (~0.68), model menganggap kedua gambar itu merupakan wajah orang yang sama (mirip).


*dokumentasi tambahan:

![Image](https://github.com/user-attachments/assets/44dd4cd0-096c-4185-b9c9-350bdcb3ee48)
