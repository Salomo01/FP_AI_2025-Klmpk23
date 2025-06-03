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

Secara default, aplikasi ini menggunakan **model DeepFace**, yaitu sebuah jaringan saraf convolutional (CNN) yang sudah dilatih (pre-trained) pada dataset wajah yang besar. Model ini mengekstrak **embedding** (fitur wajah dalam bentuk vektor) dari setiap gambar, lalu menghitung **jarak cosine** antar dua embedding tersebut.

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


1. Siapkan VS Code & Ekstensi
Buka VS Code.

Pastikan ekstensi Python dan Pylance sudah terinstal (sudah muncul di panel Installed).

(Optional) Instal ekstensi REST Client jika suka mengetes API langsung dari VS Code.

2. Buat Folder Proyek & Virtual Environment
Di VS Code, pilih File > Open Folder… dan buat folder baru, misalnya face_similarity_app.

Buka Terminal > New Terminal.

Di terminal, buat venv dan aktifkan:
```
python -m venv env
# Windows
.\env\Scripts\activate
# macOS/Linux
source env/bin/activate

```

Pastikan prompt terminal berganti menandakan venv aktif.

3. Instalasi Dependensi
Di terminal yang sama, jalankan:
```
pip install --upgrade pip
pip install flask deepface opencv-python pillow
```
– flask → web server
– deepface → engine face recognition
– opencv-python & pillow → handling gambar

4. Struktur Direktori
Di Explorer VS Code, buat struktur:
```
face_similarity_app/
├── env/                 ← virtual env
├── app.py               ← Flask app
├── uploads/             ← tempat gambar upload
├── templates/
│   └── index.html       ← halaman utama
└── static/
    ├── css/
    │   └── styles.css
    └── js/
        └── scripts.js
```

5. Tulis Backend (app.py)
Di root folder, buat app.py

Copy–paste kode berikut:
```
import os
# Supaya pesan info/warning TensorFlow minimal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_MODELS = [
    "VGG-Face", "Facenet", "Facenet512",
    "OpenFace", "DeepFace", "DeepID",
    "ArcFace", "Dlib", "SFace"
]

@app.route('/')
def index():
    return render_template('index.html', models=ALLOWED_MODELS)

@app.route('/upload', methods=['POST'])
def upload():
    img1 = request.files.get('image1')
    img2 = request.files.get('image2')
    model_name = request.form.get('model', 'VGG-Face')

    if not img1 or not img2:
        return jsonify({'error': 'Upload kedua gambar!'}), 400

    if model_name not in ALLOWED_MODELS:
        return jsonify({'error': f'Model tidak valid. Gunakan salah satu: {ALLOWED_MODELS}'}), 400

    # Simpan file
    path1 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img1.filename))
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img2.filename))
    img1.save(path1)
    img2.save(path2)

    try:
        res = DeepFace.verify(path1, path2, model_name=model_name)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(">>> Starting Flask server...")  # Membantu memastikan kalau bagian ini terpanggil
    app.run(host='127.0.0.1', port=5000, debug=True)

```
6. Tulis Frontend
a. templates/index.html
```
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Face Similarity Checker</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}" />
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>🧠 Face Similarity Checker</h1>
      <form id="upload-form">
        <div class="file-input">
          <label for="image1">Upload Image 1</label>
          <input type="file" id="image1" name="image1" accept="image/*" required />
          <img id="preview1" class="image-preview" alt="Preview Image 1" />
        </div>
        <div class="file-input">
          <label for="image2">Upload Image 2</label>
          <input type="file" id="image2" name="image2" accept="image/*" required />
          <img id="preview2" class="image-preview" alt="Preview Image 2" />
        </div>
        <div class="file-input">
          <label for="model">Select Model</label>
          <select id="model" name="model" required>
            <option value="VGG-Face">VGG-Face</option>
            <option value="Facenet">Facenet</option>
            <option value="Facenet512">Facenet512</option>
            <option value="ArcFace">ArcFace</option>
            <option value="Dlib">Dlib</option>
            <option value="SFace">SFace</option>
            <option value="OpenFace">OpenFace</option>
          </select>
        </div>
        <button type="submit">Compare Faces</button>
      </form>
      <div id="result" class="result-box"></div>
    </div>
  </div>

  <script src="{{ url_for('static', filename='js/scripts.js') }}"></script>
</body>
</html>
```
b. static/css/styles.css
```
body {
  margin: 0;
  padding: 0;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: #f0f2f5;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 100vh;
}

.container {
  padding: 20px;
}

.card {
  background-color: #fff;
  border-radius: 16px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
  padding: 30px 40px;
  max-width: 400px;
  width: 100%;
  text-align: center;
  animation: fadeIn 0.5s ease;
}

.card h1 {
  font-size: 1.8rem;
  margin-bottom: 25px;
  color: #333;
}

.file-input {
  margin-bottom: 20px;
  text-align: left;
}

.file-input label {
  display: block;
  margin-bottom: 6px;
  font-weight: 600;
  color: #555;
}

input[type="file"],
select {
  width: 100%;
  padding: 8px;
  border-radius: 6px;
  border: 1px solid #ccc;
}

.image-preview {
  width: 100%;
  max-height: 200px;
  object-fit: cover;
  border: 1px solid #ccc;
  border-radius: 8px;
  margin-top: 10px;
  display: none;
}

button {
  background-color: #4a90e2;
  color: white;
  border: none;
  padding: 10px 20px;
  font-size: 1rem;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

button:hover {
  background-color: #357abd;
}

.result-box {
  margin-top: 25px;
  font-weight: bold;
  font-size: 1.1rem;
  color: #333;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

```
c. static/js/scripts.js
```
document.getElementById('upload-form').addEventListener('submit', async function(e) {
  e.preventDefault();

  const image1 = document.getElementById('image1').files[0];
  const image2 = document.getElementById('image2').files[0];
  const model = document.getElementById('model').value;
  const resultDiv = document.getElementById('result');

  if (!image1 || !image2) {
    resultDiv.textContent = 'Please select both images.';
    return;
  }

  const formData = new FormData();
  formData.append('image1', image1);
  formData.append('image2', image2);
  formData.append('model', model);

  resultDiv.textContent = 'Predicting…';

  try {
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();

    if (response.ok) {
      if (data.verified) {
        resultDiv.textContent = `✅ Faces match! Distance: ${data.distance.toFixed(4)}`;
      } else {
        resultDiv.textContent = `❌ Faces do not match. Distance: ${data.distance.toFixed(4)}`;
      }
    } else {
      resultDiv.textContent = `Error: ${data.error}`;
    }
  } catch (err) {
    resultDiv.textContent = `Error: ${err.message}`;
  }
});

document.getElementById('image1').addEventListener('change', function () {
  previewImage(this, 'preview1');
});
document.getElementById('image2').addEventListener('change', function () {
  previewImage(this, 'preview2');
});

function previewImage(input, previewId) {
  const preview = document.getElementById(previewId);
  if (input.files && input.files[0]) {
    const reader = new FileReader();
    reader.onload = function (e) {
      preview.src = e.target.result;
      preview.style.display = 'block';
    };
    reader.readAsDataURL(input.files[0]);
  }
}

```
7. Jalankan & Cek Hasil
Di terminal (venv aktif), jalankan:
```
flask run
```
atau

```
python app.py
```
Buka browser ke http://127.0.0.1:5000/.

Coba Load Image 1, Load Image 2, lalu klik Predict.

Hasil verifikasi akan muncul di bawah form.
