from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    img1 = request.files.get('image1')
    img2 = request.files.get('image2')
    if not img1 or not img2:
        return jsonify({'error': 'Upload kedua gambar!'}), 400

    # Simpan file
    path1 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img1.filename))
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img2.filename))
    img1.save(path1)
    img2.save(path2)

    try:
        # Verifikasi similaritas wajah
        res = DeepFace.verify(path1, path2)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
