import streamlit as st
import pandas as pd
import joblib
import os

# === 1. Load Model & Kolom ===
model = joblib.load(os.path.join(os.path.dirname(__file__), "rf_model.pkl"))
columns = joblib.load(os.path.join(os.path.dirname(__file__), "columns.pkl"))

# === 2. Tampilan Halaman ===
st.title("🎓 Prediksi Nilai Overall Mahasiswa")
st.write("Upload data Anda untuk memprediksi nilai *Overall* menggunakan model Random Forest.")

# === 3. Upload File CSV ===
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    # Baca data yang diupload
    data = pd.read_csv(uploaded_file)
    st.write("📋 Data yang diupload:")
    st.dataframe(data.head())

    # One-hot encoding agar sesuai kolom model
    data_encoded = pd.get_dummies(data, drop_first=True)

    # Pastikan semua kolom sesuai urutan saat training
    for col in columns:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    data_encoded = data_encoded[columns]

    # Prediksi
    preds = model.predict(data_encoded)

    # Tampilkan hasil
    data["Prediksi_Overall"] = preds
    st.success("✅ Prediksi selesai!")
    st.dataframe(data[["Prediksi_Overall"]])

    # Tombol untuk download hasil
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download hasil prediksi", csv, "hasil_prediksi.csv", "text/csv")

else:

    st.info("Silakan upload file CSV terlebih dahulu.")
