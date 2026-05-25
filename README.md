# 🧠 NeuralStroke — Stroke Risk Prediction System

A full-stack AI web application that predicts stroke risk from clinical symptoms using a trained Random Forest model.

## 📊 Model Performance
| Metric | Score |
|---|---|
| Accuracy | **94.06%** |
| AUC-ROC | **98.88%** |
| Regressor MAE | **2.86%** |
| Training samples | **56,000** |

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your dataset in the project root
cp stroke_risk_dataset.csv ./

# 3. Train the model
python train.py

# 4. Start the web server
python app.py

# 5. Open browser
open http://localhost:5050
```

## 📁 Project Structure

```
stroke-risk-predictor/
├── app.py                    # Flask API backend
├── train.py                  # Model training script
├── requirements.txt          # Python dependencies
├── stroke_risk_dataset.csv   # Dataset (70,000 records)
├── model/
│   ├── rf_classifier.pkl     # Trained RF classifier
│   ├── rf_regressor.pkl      # Trained RF regressor
│   └── meta.json             # Feature names & importances
└── static/
    └── index.html            # Frontend web app
```

## 🌐 API Endpoints

### POST `/predict`
Input all 16 features as JSON:
```json
{
  "Age": 55,
  "Chest Pain": 1,
  "High Blood Pressure": 1,
  "Shortness of Breath": 0,
  ...
}
```

Response:
```json
{
  "risk_percent": 72.5,
  "probability": 74.2,
  "at_risk": 1,
  "category": "High",
  "color": "red",
  "top_factors": [...],
  "model_accuracy": 94.06,
  "model_auc": 98.88
}
```

### GET `/meta`
Returns feature names and model metadata.

## 🧪 Features (16 inputs)
- 15 binary symptoms (0/1): Chest Pain, Shortness of Breath, Irregular Heartbeat, Fatigue & Weakness, Dizziness, Swelling (Edema), Pain in Neck/Jaw/Shoulder/Back, Excessive Sweating, Persistent Cough, Nausea/Vomiting, High Blood Pressure, Chest Discomfort (Activity), Cold Hands/Feet, Snoring/Sleep Apnea, Anxiety/Feeling of Doom
- 1 continuous: Age (18–90)

## 🏗️ Architecture
- **Backend**: Flask REST API
- **ML Models**: scikit-learn RandomForest (classifier + regressor)
- **Frontend**: Vanilla HTML/CSS/JS — zero framework, zero build step
- **Charts**: Chart.js for the arc risk gauge

## 📈 Upgrade Path
1. **XGBoost**: Replace RF with `xgb.XGBClassifier()` for ~+1% accuracy
2. **LightGBM**: Faster training, similar accuracy
3. **SHAP values**: Replace feature importances with SHAP for better explainability
4. **Deploy**: Use `gunicorn app:app` + Render/Railway/Heroku

## ⚠️ Disclaimer
For educational and portfolio purposes only. Not a substitute for professional medical advice.
