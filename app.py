import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    df_raw = df.copy()  # simpan salinan asli untuk EDA
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

    # === 📊 4. Exploratory Data Analysis (EDA) ===
    st.subheader("📈 Exploratory Data Analysis (EDA)")
    st.write("Bagian ini menampilkan ringkasan dan visualisasi dasar dari data yang kamu unggah.")

    # --- Info dasar ---
    st.write("**1️⃣ Informasi Umum Dataset**")
    st.write(f"Jumlah Baris: {df_raw.shape[0]}")
    st.write(f"Jumlah Kolom: {df_raw.shape[1]}")
    st.write("Nama Kolom:", list(df_raw.columns))

    # --- Statistik deskriptif ---
    st.write("**2️⃣ Statistik Deskriptif (Numerik)**")
    st.dataframe(df_raw.describe())

    # --- Cek missing value ---
    st.write("**3️⃣ Cek Nilai Kosong (Missing Values)**")
    st.dataframe(df_raw.isna().sum().reset_index().rename(columns={"index": "Kolom", 0: "Jumlah Missing"}))

    # --- Korelasi antar variabel numerik ---
    num_cols = df_raw.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols) > 1:
        st.write("**4️⃣ Korelasi Antar Variabel Numerik**")
        fig, ax = plt.subplots(figsize=(8, 5))
        corr = df_raw[num_cols].corr()
        sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

    # --- Distribusi nilai Overall (jika ada) ---
    if "Overall" in df_raw.columns:
        st.write("**5️⃣ Distribusi Nilai Overall**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(df_raw["Overall"], bins=10, kde=True, color="skyblue", ax=ax2)
        ax2.set_xlabel("Overall")
        ax2.set_ylabel("Frekuensi")
        st.pyplot(fig2)

    st.markdown("---")

    # === 5. Prediksi ===
    y_pred = model.predict(df_encoded)
    df["Prediksi_Overall"] = y_pred

    st.success("✅ Prediksi selesai!")
    st.dataframe(df[["Prediksi_Overall"]])

    # === 6. Evaluasi jika kolom Overall ada ===
    if has_overall:
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        st.subheader("📊 Evaluasi Akurasi Prediksi")
        st.metric("R² (Akurasi)", f"{r2*100:.2f}%")
        st.write(f"**MAE:** {mae:.4f}")
        st.write(f"**RMSE:** {rmse:.4f}")

        # === 🔍 Keterangan otomatis berdasarkan akurasi ===
        if r2 >= 0.9:
            st.success("Model menunjukkan performa **sangat baik (R² > 90%)**, "
                       "menandakan fitur-fitur memiliki hubungan kuat terhadap nilai keseluruhan siswa.")
        elif r2 >= 0.75:
            st.info("Model menunjukkan performa **baik (R² antara 75–90%)**, "
                    "artinya model cukup akurat namun masih bisa ditingkatkan dengan tuning atau data tambahan.")
        elif r2 >= 0.5:
            st.warning("Model menunjukkan performa **cukup (R² antara 50–75%)**, "
                       "beberapa faktor mungkin belum sepenuhnya terwakili oleh data.")
        else:
            st.error("Model menunjukkan performa **rendah (R² < 50%)**, "
                     "kemungkinan ada noise tinggi atau fitur belum relevan dengan target.")

    # === 7. Download hasil prediksi ===
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download hasil prediksi",
        csv,
        "hasil_prediksi.csv",
        "text/csv",
        key="download-csv"
    )
