import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# === 1. Load Model & Kolom ===
model = joblib.load("rf_model.pkl")
columns = joblib.load("columns.pkl")

# === 2. Tampilan Halaman ===
st.title("🎓 Student Overall Performance Predictor")
st.write("Unggah dataset untuk memprediksi nilai *Overall* mahasiswa.")

# === 3. Upload File ===
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data yang diunggah:")
    st.dataframe(df.head())

    # Cek apakah ada kolom "Overall"
    has_overall = "Overall" in df.columns

    # Jika ada kolom Overall → pisahkan target
    if has_overall:
        y_true = df["Overall"]
        df = df.drop(columns=["Overall"])

    # One-hot encoding agar sesuai dengan model
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

    # Prediksi
    y_pred = model.predict(df_encoded)
    df["Prediksi_Overall"] = y_pred

    st.success("✅ Prediksi selesai!")
    st.dataframe(df[["Prediksi_Overall"]])

    # === 4. Evaluasi jika kolom Overall ada ===
    if has_overall:
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        st.subheader("📊 Evaluasi Akurasi Prediksi")
        st.metric("R² (Akurasi)", f"{r2*100:.2f}%")
        st.write(f"**MAE:** {mae:.4f}")
        st.write(f"**RMSE:** {rmse:.4f}")

    # === 5. Download hasil prediksi ===
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download hasil prediksi",
        csv,
        "hasil_prediksi.csv",
        "text/csv",
        key="download-csv"
    )
