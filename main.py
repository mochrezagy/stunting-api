import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, 'model_stunting.pkl'), 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API Stunting Ready!"})

@app.route('/predict', methods=['POST'])
def predict():
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)