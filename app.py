from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib, json, numpy as np, os

app = Flask(__name__, static_folder='static')
CORS(app)

BASE = os.path.dirname(__file__)
clf = joblib.load(os.path.join(BASE, 'model/rf_classifier.pkl'))
reg = joblib.load(os.path.join(BASE, 'model/rf_regressor.pkl'))
with open(os.path.join(BASE, 'model/meta.json')) as f:
    meta = json.load(f)

FEATURES = meta['feature_cols']

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    row = [float(data.get(f, 0)) for f in FEATURES]
    X = np.array([row])
    prob = float(clf.predict_proba(X)[0][1]) * 100
    risk_pct = float(reg.predict(X)[0])
    at_risk = int(clf.predict(X)[0])

    if risk_pct < 35:
        category = "Low"
        color = "green"
    elif risk_pct < 65:
        category = "Medium"
        color = "amber"
    else:
        category = "High"
        color = "red"

    top_factors = []
    imp = meta['feature_importances']
    for feat, weight in list(imp.items())[:8]:
        val = data.get(feat, 0)
        top_factors.append({
            "name": feat,
            "importance": round(weight * 100, 1),
            "active": bool(float(val) > 0)
        })

    return jsonify({
        "risk_percent": round(risk_pct, 1),
        "probability": round(prob, 1),
        "at_risk": at_risk,
        "category": category,
        "color": color,
        "top_factors": top_factors,
        "model_accuracy": meta['model_accuracy'],
        "model_auc": meta['model_auc']
    })

@app.route('/meta')
def get_meta():
    return jsonify(meta)

if __name__ == '__main__':
    app.run(debug=True, port=5050)
