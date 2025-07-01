import streamlit as st
import pandas as pd
import joblib
import os

#model dan encoder
model = joblib.load('best_random_forest_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

#fitur
full_features = ['ID', 'Nama_Tanaman', 'Warna_Daun',
                 'Bercak_Daun', 'Daun_Layu', 'Batang_Busuk', 'Pertumbuhan_Terhambat']
gejala_features = ['Bercak_Daun', 'Daun_Layu', 'Batang_Busuk', 'Pertumbuhan_Terhambat']

#interface
st.title("🌾 Prediksi Penyakit Tanaman")
st.markdown("Masukkan informasi tanaman dan gejala yang dialami:")

#input
nama_list = sorted(label_encoders['Nama_Tanaman'].classes_)
nama_tanaman = st.selectbox("🪴 Pilih Nama Tanaman", options=nama_list)

#input daun
warna_label = {0: "Hijau", 1: "Kuning", 2: "Coklat"}
warna_encoded = st.selectbox(
    "🌿 Warna Daun",
    options=list(warna_label.keys()),
    format_func=lambda x: warna_label[x]
)

#gejala
st.sidebar.header("🦠 Gejala yang Dialami Tanaman")
gejala_input = []
for feature in gejala_features:
    val = st.sidebar.checkbox(f"{feature.replace('_', ' ').capitalize()}")
    gejala_input.append(1 if val else 0)

#predict
if st.button("🔍 Prediksi Penyakit"):
    # encode nama tanaman
    nama_encoded = label_encoders['Nama_Tanaman'].transform([nama_tanaman])[0]

    #bangun input
    input_dict = {
        'ID': 0,
        'Nama_Tanaman': nama_encoded,
        'Warna_Daun': warna_encoded
    }
    for i, feat in enumerate(gejala_features):
        input_dict[feat] = gejala_input[i]

    input_df = pd.DataFrame([input_dict])[full_features]

    #predict penyakit
    prediction = model.predict(input_df)[0]
    if 'Penyakit' in label_encoders:
        prediction = label_encoders['Penyakit'].inverse_transform([prediction])[0]

    #hasil
    st.success(f"🌱 Nama Tanaman: **{nama_tanaman}**")
    st.success(f"🟢 Warna Daun: **{warna_label[warna_encoded]}**")
    st.success(f"🔬 Prediksi Penyakit: **{prediction}**")

    #save ke csv
    hasil = {
        'Nama_Tanaman': nama_tanaman,
        'Warna_Daun': warna_label[warna_encoded],
        'Bercak_Daun': gejala_input[0],
        'Daun_Layu': gejala_input[1],
        'Batang_Busuk': gejala_input[2],
        'Pertumbuhan_Terhambat': gejala_input[3],
        'Prediksi_Penyakit': prediction
    }

    hasil_df = pd.DataFrame([hasil]).astype(str)  # 🔐 pastikan semua string

    if os.path.exists('riwayat_prediksi.csv'):
        hasil_df.to_csv('riwayat_prediksi.csv', mode='a', header=False, index=False)
    else:
        hasil_df.to_csv('riwayat_prediksi.csv', index=False)

    st.info("📁 Hasil disimpan ke `riwayat_prediksi.csv`")

# === 4. Tampilkan Riwayat & Grafik ===
if os.path.exists('riwayat_prediksi.csv'):
    st.markdown("## 🧾 Riwayat Prediksi")
    riwayat_df = pd.read_csv('riwayat_prediksi.csv')

    # Filter berdasarkan nama tanaman
    tanaman_tersedia = sorted(riwayat_df['Nama_Tanaman'].unique())
    tanaman_terpilih = st.selectbox("🔎 Filter Nama Tanaman", ["-- Semua --"] + tanaman_tersedia)

    if tanaman_terpilih != "-- Semua --":
        riwayat_filtered = riwayat_df[riwayat_df['Nama_Tanaman'] == tanaman_terpilih]
        st.markdown(f"### 📄 Riwayat untuk: **{tanaman_terpilih}**")
    else:
        riwayat_filtered = riwayat_df.copy()
        st.markdown("### 📄 Riwayat Semua Tanaman")

    st.dataframe(riwayat_filtered)

    # Download button
    csv = riwayat_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Unduh CSV", data=csv, file_name="riwayat_prediksi.csv", mime="text/csv")

    # Grafik
    if not riwayat_filtered.empty and 'Prediksi_Penyakit' in riwayat_filtered.columns:
        st.markdown("## 📊 Grafik Frekuensi Penyakit")
        freq = riwayat_filtered['Prediksi_Penyakit'].value_counts().reset_index()
        freq.columns = ['Penyakit', 'Jumlah']
        st.bar_chart(freq.set_index('Penyakit'))

