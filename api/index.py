import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

def load_file(filename):
    path_root = os.path.join(BASE_DIR, filename)
    path_api = os.path.join(CURRENT_DIR, filename)
    if os.path.exists(path_root):
        return pickle.load(open(path_root, 'rb'))
    elif os.path.exists(path_api):
        return pickle.load(open(path_api, 'rb'))
    else:
        raise FileNotFoundError(f"File {filename} tidak ditemukan.")

try:
    scaler = load_file('scaler.pkl')
    model = load_file('model_stunting.pkl')
except Exception as e:
    scaler = None
    model = None
    load_error = str(e)

@app.route('/', methods=['GET'])
def home():
    if model is None or scaler is None:
        return jsonify({"status": "error", "message": f"Gagal memuat model: {load_error}"}), 500
    return jsonify({"status": "API Stunting Ready!"})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"status": "error", "message": f"Model belum siap: {load_error}"}), 500

    try:
        data = request.get_json()

        umur_bulan = float(data['umur_bulan'])
        tb_cm      = float(data['tb_cm'])
        bb_kg      = float(data['bb_kg'])
        lila_cm    = float(data['lila_cm'])

        feature_names = ['umur_bulan', 'tb_cm', 'bb_kg', 'lila_cm']
        input_df = pd.DataFrame([[umur_bulan, tb_cm, bb_kg, lila_cm]], columns=feature_names)
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)

        prediction = int(model.predict(input_scaled)[0])

        try:
            probabilities = model.predict_proba(input_scaled)[0]
            confidence = float(np.max(probabilities))
        except Exception:
            confidence = 1.0

        hasil_text = "Ya" if prediction == 1 else "Tidak"

        return jsonify({
            "status": "success",
            "hasil_stunting": hasil_text,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
