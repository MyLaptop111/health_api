from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# Simple Health API (single-file)
# ---------------------------
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
DATA_FILE = "data.csv"

# If artifacts not present, create synthetic data, train and save model+scaler
def ensure_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        return

    print("Training a demo XGBoost model (this may take a moment)...")
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 7)  # 7 sensors
    y = np.random.randint(0, 2, size=n_samples)  # 0: healthy, 1: disease

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_scaled, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    # save sample data to data.csv for reference
    df = pd.DataFrame(X, columns=[f"sensor{i+1}" for i in range(7)])
    df["label"] = y
    df.to_csv(DATA_FILE, index=False)
    print("Model and scaler saved.")

# Categorization table (every 20%) and advice
def classify_and_advise(prob):
    if prob < 0.20:
        return "سليم جدًا", "استمر على نمط حياتك الصحي."
    elif prob < 0.40:
        return "سليم", "حافظ على التمارين والغذاء الجيد، وراقب صحتك."
    elif prob < 0.60:
        return "معرّض للخطر", "يُفضل إجراء فحوصات دورية للتأكد من سلامتك."
    elif prob < 0.80:
        return "مريض", "استشر الطبيب قريبًا وأجرِ الفحوصات اللازمة."
    else:
        return "خطر شديد", "راجع الطبيب فورًا ولا تؤجل."

# Prepare artifacts
ensure_model()

# Load into memory
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)


try:
    df_sample = pd.read_csv(DATA_FILE)
    X_sample = df_sample.iloc[:, :7].values
    y_sample = df_sample["label"].values
    Xs = scaler.transform(X_sample)
    cv_scores = cross_val_score(model, Xs, y_sample, cv=5)
    cv_accuracy = float(cv_scores.mean())
except Exception:
    cv_accuracy = None

app = Flask("_name_")

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "health-sensor-api",
        "note": "النتائج للتوجيه فقط — القرار النهائي من الطبيب.",
        "model_ready": True
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data or "sensors" not in data:
        return jsonify({"error":"Send JSON: {'sensors':[v1,..,v7]}"}), 400
    sensors = data["sensors"]
    if not isinstance(sensors, (list,tuple)) or len(sensors) != 7:
        return jsonify({"error":"'sensors' must be a list of 7 numeric values"}), 400
    try:
        X = np.asarray(sensors, dtype=float).reshape(1, -1)
    except Exception as e:
        return jsonify({"error":"invalid sensor values", "details": str(e)}), 400

    Xs = scaler.transform(X)
    proba = float(model.predict_proba(Xs)[:,1][0])
    healthy = round(1.0 - proba, 4)
    disease = round(proba, 4)
    category, advice = classify_and_advise(proba)

    # Optionally append input to data.csv (if client asks 'save': true)
    save_flag = bool(data.get("save", False))
    if save_flag:
        row = list(map(float, sensors))
        df_row = pd.DataFrame([row + [int(proba >= 0.5)]])
        header = not os.path.exists(DATA_FILE)
        if header:
            df_row.to_csv(DATA_FILE, index=False, header=[f"sensor{i+1}" for i in range(7)] + ["label"])
        else:
            df_row.to_csv(DATA_FILE, mode='a', index=False, header=False)

    return jsonify({
        "مريض": disease,
        "سليم": healthy,
        "category": category,
        "advice": advice,
        "note": "هذا التصنيف للتوجيه فقط، القرار النهائي من الطبيب."
    })

@app.route("/metrics", methods=["GET"])
def metrics():
    if cv_accuracy is None:
        return jsonify({"note":"No CV metrics available"}), 200
    return jsonify({"cv_accuracy": round(cv_accuracy, 4)})

@app.route("/get-data", methods=["GET"])
def get_data():
    if not os.path.exists(DATA_FILE):
        return jsonify({"rows": [], "total": 0})
    df = pd.read_csv(DATA_FILE)
    max_rows = int(request.args.get("max", 500))
    return jsonify({"total": len(df), "rows": df.head(max_rows).to_dict(orient="records")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))