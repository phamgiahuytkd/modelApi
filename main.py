import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from flask import Flask, request, jsonify

# === Load scaler, model, threshold, column names ===
scaler = joblib.load("fraud_scaler.pkl")
model = CatBoostClassifier()
model.load_model("fraud_catboost_model.cbm")
best_threshold = joblib.load("fraud_best_threshold.pkl")
columns = joblib.load("fraud_model_columns.pkl")
encoders = joblib.load("fraud_label_encoders.pkl")
valid_categorical_values = joblib.load("fraud_categorical_valids.pkl")
avg_payment_values = joblib.load("fraud_avg_payment_values.pkl")

# === Khai báo lại các biến đặc trưng ===
cat_cols = ['Payment Method', 'Device Used', 'Payment_Device']
num_cols = ['Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days', 'Transaction Hour',
            'Amount_per_day', 'Is_Weekend', 'Hour_Distance', 'Age_Amount_Interact', 'Amount_to_Avg_Payment',
            'Hour_of_Day', 'Amount_Rank', 'Transaction_Frequency', 'Amount_to_Median']

app = Flask(__name__)
# === Hàm chuẩn hóa (reuse từ phần trước) ===
def standardize_input(new_input, scaler, encoders, cat_cols, num_cols,
                      valid_categorical_values, avg_payment_values):
    # 1. Tạo DataFrame từ input
    new_df = pd.DataFrame([new_input])

    # 2. Tiền xử lý cột thời gian
    new_df['Transaction Date'] = pd.to_datetime(new_df['Transaction Date'], errors='coerce')
    new_df["DayOfWeek"] = new_df["Transaction Date"].dt.dayofweek
    new_df["Is_Weekend"] = new_df["DayOfWeek"].apply(lambda x: 1 if x in [5, 6] else 0)
    new_df["Hour_of_Day"] = new_df['Transaction Date'].dt.hour.fillna(new_df["Transaction Hour"])

    # 3. Feature engineering như phần train
    new_df["Amount_per_day"] = new_df["Transaction Amount"] / (new_df["Account Age Days"] + 1)
    new_df["Hour_Distance"] = new_df["Transaction Hour"].apply(lambda x: abs(x - 12))
    new_df["Age_Amount_Interact"] = new_df["Customer Age"] * new_df["Transaction Amount"]

    # 4. Feature dùng giá trị trung bình theo 'Payment Method'
    pm = new_df["Payment Method"].iloc[0]
    avg_payment = avg_payment_values.get(pm, np.mean(list(avg_payment_values.values())))
    new_df["Amount_to_Avg_Payment"] = new_df["Transaction Amount"] / avg_payment

    # 5. Tạo 'Payment_Device'
    new_df["Payment_Device"] = new_df["Payment Method"].astype(str) + "_" + new_df["Device Used"].astype(str)

    # 6. Các feature còn lại mặc định = 0 (không thể tính từ 1 mẫu)
    new_df["Amount_Rank"] = 0
    new_df["Transaction_Frequency"] = 1
    new_df["Amount_to_Median"] = 1

    # 7. Làm sạch categorical: unknown nếu không hợp lệ
    for col in cat_cols:
        new_df[col] = new_df[col].astype(str)
        valid_vals = valid_categorical_values.get(col, [])
        if not isinstance(valid_vals, (list, set)):
            valid_vals = []
        new_df[col] = new_df[col].apply(lambda x: x if x in valid_vals else "unknown")

    # 8. Chuẩn hóa cột số
    new_df[num_cols] = scaler.transform(new_df[num_cols])

    # 9. Trả về DataFrame với đúng thứ tự cột
    return new_df[columns]

# === Tạo dữ liệu đầu vào ===


# === Hàm up lên port ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        new_input = request.get_json()
        print("✅ Received input:", new_input)

        # === Tiền xử lý và chuẩn hóa dữ liệu ===
        standardized_input = standardize_input(
            new_input, scaler, encoders, cat_cols, num_cols,
            valid_categorical_values, avg_payment_values)

        # === Dự đoán ===
        proba = model.predict_proba(standardized_input)[:, 1]
        pred = (proba >= best_threshold).astype(int)

        # === In kết quả ===
        print("✅ Dự đoán:", pred)
        print(f"🔍 Xác suất gian lận: {proba[0]:.4f}")

        return jsonify({
            "prediction": int(pred),
            "probability": float(proba[0])

        })

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 400

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway sẽ cung cấp PORT
    app.run(host="0.0.0.0", port=port)