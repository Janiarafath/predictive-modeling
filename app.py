from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import joblib
import os
import gdown

app = Flask(__name__)

# ========== 🔽 Auto-download model files if missing ========== #
def download_if_missing(filename, gdrive_url):
    if not os.path.exists(filename):
        print(f"Downloading {filename} from Google Drive...")
        gdown.download(gdrive_url, filename, quiet=False)

download_if_missing("churn_model.pkl", "https://drive.google.com/uc?id=10DbKfumGu5Cl4Y2AoRPDisxzdqAIQkkY")
download_if_missing("scaler.pkl", "https://drive.google.com/uc?id=1_eOQQoaR7E54CYBjkTCA7e3JqiwgH8DQ")
download_if_missing("columns.pkl", "https://drive.google.com/uc?id=1sB4OkGTmAyCVp_7TLBrVdKl-htEtaVJ-")

# ========== 🔧 Load Model Components ========== #
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
trained_columns = joblib.load("columns.pkl")
raw_df = pd.read_csv("deepq_ai_assignment1_data.csv")

# Use one sample row
sample_row = raw_df.drop(['UID', 'Target_ChurnFlag'], axis=1).iloc[0:1]
sample_encoded = pd.get_dummies(sample_row)
sample_encoded = sample_encoded.reindex(columns=trained_columns, fill_value=0)
template_input = sample_encoded.iloc[0].copy()

# Top N features based on importance (you can customize)
top_features = ['X16', 'X7', 'X8', 'X4', 'X98', 'X85']
HISTORY_FILE = "prediction_history.csv"

# ========== 🏠 Home ========== #
@app.route('/')
def home():
    options = {}
    for col in top_features:
        if raw_df[col].dtype == object:
            options[col] = sorted(raw_df[col].dropna().unique())
        else:
            options[col] = round(raw_df[col].mean(), 2)

    history = []
    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE).tail(5).to_dict(orient='records')

    return render_template("index.html", options=options, history=history)

# ========== 🔮 Predict ========== #
@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    input_row = template_input.copy()

    for col in top_features:
        val = form.get(col)
        if raw_df[col].dtype == object:
            for c in trained_columns:
                if c.startswith(f"{col}_"):
                    input_row[c] = 1 if c == f"{col}_{val}" else 0
        else:
            input_row[col] = float(val)

    input_df = pd.DataFrame([input_row])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    result_text = "⚠️ Churn Likely" if prediction == 1 else "✅ Customer Likely to Stay"

    # Save history
    history_row = {col: form[col] for col in top_features}
    history_row["Prediction"] = result_text
    history_row["Churn Probability"] = round(prob, 4)

    history_df = pd.DataFrame([history_row])
    if os.path.exists(HISTORY_FILE):
        old = pd.read_csv(HISTORY_FILE)
        history_df = pd.concat([old, history_df], ignore_index=True)
    history_df.to_csv(HISTORY_FILE, index=False)

    return render_template("result.html",
                           prediction=result_text,
                           probability=f"{prob:.2%}",
                           churn_prob=round(prob, 4),
                           not_churn_prob=round(1 - prob, 4))

# ========== 📥 Download History ========== #
@app.route('/download')
def download():
    return send_file(HISTORY_FILE, as_attachment=True)

# ========== 🚀 Run ========== #
if __name__ == '__main__':
    app.run(debug=True)

