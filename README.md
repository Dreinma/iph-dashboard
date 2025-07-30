# Dashboard Prediksi IPH Kota Batu

Dashboard interaktif untuk forecasting Indikator Perubahan Harga (IPH) komoditas di Kota Batu menggunakan machine learning.

## 🚀 Fitur Utama

- **Multi-Model Forecasting**: KNN, Random Forest, LightGBM, XGBoost
- **Analisis Interaktif**: Visualisasi tren, fluktuasi, dan kontribusi komoditas
- **Scenario Analysis**: Normal, Optimis, Pesimis
- **Export Data**: CSV dan Excel
- **Alert System**: Peringatan otomatis untuk IPH tinggi/rendah

## 📁 Struktur Project
iph_dashboard/
├── app.py                 # Main dashboard
├── src/                   # Source modules
├── models/               # Model PKL files
├── data/                 # Data files
└── requirements.txt      # Dependencies

## 🛠️ Installation

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Letakkan file model PKL di folder `models/`
4. Letakkan file Excel di folder `data/`
5. Run: `streamlit run app.py`

## 📊 Model Files Required

- `KNN.pkl`
- `LightGBM.pkl` 
- `Random_Forest.pkl`
- `XGBoost_Advanced.pkl`

## 🎯 Usage

1. Pilih sumber data dari sidebar
2. Pilih model forecasting
3. Atur periode prediksi
4. Klik "Buat Prediksi"
5. Analisis hasil dan export data

## 🔧 Configuration

- Alert thresholds: ±2.0%
- Default prediction: 4 weeks
- Confidence interval: 95%
- Update frequency: Weekly

## 📈 Model Features

Input features: `Lag_1`, `Lag_2`, `Lag_3`, `Lag_4`, `MA_3`, `MA_7`
Target: `Indikator_Harga` (IPH %)

## 🚀 Deployment

Deploy to Streamlit Cloud:
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Deploy from repository

---
© 2025 Dashboard IPH Kota Batu