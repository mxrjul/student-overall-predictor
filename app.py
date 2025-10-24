import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#1. Load Model & Kolom
model = joblib.load("rf_model.pkl")
columns = joblib.load("columns.pkl")

# 2. Tampilan Halaman 
st.title("🎓 Student Overall Performance Predictor")
st.write("Unggah dataset untuk memprediksi nilai *Overall* mahasiswa.")

# 3. Pilihan Mode
mode = st.radio("Pilih mode input data:", ["🧮 Input Manual", "📂 Upload CSV"])

# MODE 1: INPUT MANUAL UNTUK SATU MAHASISWA
if mode == "🧮 Input Manual":
    st.subheader("🧾 Masukkan Data Mahasiswa")

    # Form Input
    col1, col2 = st.columns(2)
    with col1:
        department = st.selectbox("Jurusan", ["Business Administration", "Computer Science and Engineering", "Economics", "Electrical and Electronic Engineering", "English", "Journalism, Communication and Media Studies", "Law and Human Rights", "Political Science", "Public Health", "Sociology"])
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        hometown = st.selectbox("Daerah asal", ["Village", "City"])
        income = st.number_input("Pendapatan (dalam ribuan ringgit)", 10, 1000, 100)
        ssc = st.number_input("Nilai Akhir SMP", 1.0, 5.0, 4.5)
        hsc = st.number_input("Nilai Akhir SMA", 1.0, 5.0, 4.3)

    with col2:
        computer = st.selectbox("Pengetahuan Komputer", ["1", "2", "3", "4", "5"])
        preparation = st.slider("Waktu Belajar (jam/hari)", 0, 6, 2)
        gaming = st.slider("Waktu Bermain Game (jam/hari)", 0, 6, 1)
        attendance = st.slider("Kehadiran (%)", 0, 100, 85)
        job = st.selectbox("Kerja Paruh Waktu", ["Yes", "No"])
        english = st.selectbox("Kecakapan Berbahasa Inggris", ["Poor", "Average", "Good", "Excellent"])
        extra = st.selectbox("Partisipasi Ekskul", ["Yes", "No"])
        semester = st.number_input("Semester Sekarang", 1, 14, 4)
        last = st.number_input("IPK Terakhir", 1.0, 4.0, 3.2)

    # Buat dataframe 1 baris
    user_data = pd.DataFrame([{
        "Department": department,
        "Gender": gender,
        "Hometown": hometown,
        "Income": income,
        "SSC": ssc,
        "HSC": hsc,
        "Computer": computer,
        "Preparation": preparation,
        "Gaming": gaming,
        "Attendance": attendance,
        "Job": job,
        "English": english,
        "Extra": extra,
        "Semester": semester,
        "Last": last
    }])

    st.write("📋 Data yang kamu masukkan:")
    st.dataframe(user_data)

    # Prediksi ketika tombol ditekan
    if st.button("🚀 Prediksi Nilai Overall"):
        df_encoded = pd.get_dummies(user_data)
        df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

        pred = model.predict(df_encoded)[0]
        st.success(f"🎯 **Prediksi Nilai Overall:** {pred:.2f}")

# MODE 2 : Upload File CSV
elif mode == "📂 Upload CSV":
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

        # === 📊 Exploratory Data Analysis (EDA) ===
        st.subheader("📈 Exploratory Data Analysis (EDA)")

        st.write("Bagian ini menampilkan ringkasan dan visualisasi dasar dari data yang kamu unggah.")

        # --- Info dasar ---
        st.write("**1️⃣ Informasi Umum Dataset**")
        st.write(f"Jumlah Baris: {df.shape[0]}")
        st.write(f"Jumlah Kolom: {df.shape[1]}")
        st.write("Nama Kolom:", list(df.columns))

        # --- Statistik deskriptif ---
        st.write("**2️⃣ Statistik Deskriptif (Numerik)**")
        st.dataframe(df.describe())

        # --- Cek missing value ---
        st.write("**3️⃣ Cek Nilai Kosong (Missing Values)**")
        st.dataframe(df.isna().sum().reset_index().rename(columns={"index": "Kolom", 0: "Jumlah Missing"}))

        # --- Korelasi antar variabel numerik ---
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        if len(num_cols) > 1:
            st.write("**4️⃣ Korelasi Antar Variabel Numerik**")
            import seaborn as sns
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            corr = df[num_cols].corr()
            sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
            st.pyplot(fig)

        # --- Distribusi nilai Overall (jika ada) ---
        if has_overall:
            st.write("**5️⃣ Distribusi Nilai Overall**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(y_true, bins=10, kde=True, color="skyblue", ax=ax2)
            ax2.set_xlabel("Overall")
            ax2.set_ylabel("Frekuensi")
            st.pyplot(fig2)

        st.markdown("---")

        # Prediksi
        y_pred = model.predict(df_encoded)
        df["Prediksi_Overall"] = y_pred

        st.success("✅ Prediksi selesai!")
        st.dataframe(df[["Prediksi_Overall"]])

        # === 5. Evaluasi jika kolom Overall ada ===
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
                st.success("Model menunjukkan performa **sangat baik (R² > 90%)**.")
            elif r2 >= 0.75:
                st.info("Model menunjukkan performa **baik (R² antara 75–90%)**.")
            elif r2 >= 0.5:
                st.warning("Model menunjukkan performa **cukup (R² antara 50–75%)**.")
            else:
                st.error("Model menunjukkan performa **rendah (R² < 50%)**.")

        # === 6. Download hasil prediksi ===
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download hasil prediksi",
            csv,
            "hasil_prediksi.csv",
            "text/csv",
            key="download-csv"
        )








