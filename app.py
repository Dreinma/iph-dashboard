import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from model_predictor import ModelPredictor
from visualizations import DashboardViz
from utils import format_metrics, export_data, check_alerts

# Page configuration
st.set_page_config(
    page_title="Dashboard Prediksi IPH Kota Batu",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4A90E2 0%, #7B68EE 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    
    .panel-title {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        margin-bottom: 1rem;
        border-left: 5px solid #4A90E2;
        font-size: 1.1rem;
    }
    
    .sidebar-section {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #dee2e6;
    }
    
    .alert-high {
        background: #ffe6e6;
        border: 2px solid #ff4444;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .alert-low {
        background: #e6f3ff;
        border: 2px solid #4444ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def init_components():
    try:
        return {
            'data_loader': DataLoader(),
            'predictor': ModelPredictor(),
            'viz': DashboardViz()
        }
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Dashboard Prediksi IPH Kota Batu</h1>
        <p>Sistem Forecasting Indikator Perubahan Harga Komoditas</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Data diperbarui per {}</p>
    </div>
    """.format(datetime.now().strftime("%d %B %Y")), unsafe_allow_html=True)
    
    # Initialize components
    components = init_components()
    if not components:
        st.error("❌ Gagal menginisialisasi komponen dashboard")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🎛️ PANEL KONTROL")
        
        # Data source selection
        st.markdown("**Sumber Data:**")
        data_source = st.radio(
            "",
            ["📁 Load File Excel", "📤 Upload File Baru", "🔄 Data Sample"],
            index=0
        )
        
        # Load data based on selection
        df = None
        if data_source == "📁 Load File Excel":
            df = components['data_loader'].load_excel_data()
        elif data_source == "📤 Upload File Baru":
            df = components['data_loader'].upload_data()
        else:
            df = components['data_loader'].load_sample_data()
        
        if df is not None:
            st.success(f"✅ Data berhasil dimuat: {len(df)} records")
            
            # Model selection
            st.markdown("**Pemilihan Model:**")
            available_models = components['predictor'].get_available_models()
            
            if available_models:
                selected_model = st.selectbox(
                    "",
                    available_models,
                    index=0
                )
            else:
                st.error("❌ Tidak ada model yang tersedia")
                return
            
            # Prediction settings
            st.markdown("**Pengaturan Prediksi:**")
            periods = st.slider(
                "Periode Prediksi (minggu):", 
                min_value=1, 
                max_value=12, 
                value=4,
                help="Jumlah minggu ke depan yang akan diprediksi"
            )
            
            # Scenario analysis
            st.markdown("**Analisis Skenario:**")
            scenario = st.selectbox(
                "",
                ["🔵 Normal", "🟢 Optimis", "🔴 Pesimis"],
                help="Pilih skenario untuk analisis what-if"
            )
            
            # Prediction button
            predict_button = st.button(
                "🔮 Buat Prediksi", 
                type="primary",
                use_container_width=True,
                help="Klik untuk membuat prediksi IPH"
            )
            
            if predict_button:
                with st.spinner("🔄 Membuat prediksi... Mohon tunggu..."):
                    try:
                        predictions_result = components['predictor'].predict(
                            selected_model, 
                            df, 
                            periods,
                            scenario
                        )
                        
                        if predictions_result:
                            st.session_state['predictions'] = predictions_result
                            st.session_state['selected_model'] = selected_model
                            st.session_state['periods'] = periods
                            st.success("✅ Prediksi berhasil dibuat!")
                        else:
                            st.error("❌ Gagal membuat prediksi")
                            
                    except Exception as e:
                        st.error(f"❌ Error dalam prediksi: {str(e)}")
            
            # Export functionality
            st.markdown("---")
            st.markdown("**Export Data:**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Export CSV", use_container_width=True):
                    export_data(df, st.session_state.get('predictions'), 'csv')
                    
            with col2:
                if st.button("📈 Export Excel", use_container_width=True):
                    export_data(df, st.session_state.get('predictions'), 'excel')
        
        else:
            st.warning("⚠️ Silakan pilih sumber data")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    if df is not None:
        # Check for alerts
        latest_iph = df['Indikator_Harga'].iloc[-1]
        alert_status = check_alerts(latest_iph)
        
        if alert_status['type'] == 'high':
            st.markdown(f"""
            <div class="alert-high">
                <h4>🚨 PERINGATAN TINGGI</h4>
                <p>IPH saat ini: <strong>{latest_iph:+.2f}%</strong> (di atas threshold +2.0%)</p>
                <p>Diperlukan perhatian khusus terhadap stabilitas harga komoditas</p>
            </div>
            """, unsafe_allow_html=True)
        elif alert_status['type'] == 'low':
            st.markdown(f"""
            <div class="alert-low">
                <h4>📉 PERINGATAN RENDAH</h4>
                <p>IPH saat ini: <strong>{latest_iph:+.2f}%</strong> (di bawah threshold -2.0%)</p>
                <p>Indikasi penurunan harga yang signifikan</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_iph = df['Indikator_Harga'].iloc[-1]
            prev_iph = df['Indikator_Harga'].iloc[-2] if len(df) > 1 else 0
            delta_iph = current_iph - prev_iph
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">IPH Terakhir</div>
                <div class="metric-value" style="color: {'#e74c3c' if current_iph < 0 else '#27ae60'};">
                    {current_iph:+.2f}%
                </div>
                <div class="metric-delta" style="color: {'#27ae60' if delta_iph > 0 else '#e74c3c'};">
                    {delta_iph:+.2f}% dari minggu lalu
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'predictions' in st.session_state:
                next_pred = st.session_state['predictions']['predictions'][0]
                model_mae = st.session_state['predictions']['mae']
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Prediksi Minggu Depan</div>
                    <div class="metric-value" style="color: {'#e74c3c' if next_pred < 0 else '#27ae60'};">
                        {next_pred:+.2f}%
                    </div>
                    <div class="metric-delta" style="color: #3498db;">
                        MAE: {model_mae:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Prediksi Minggu Depan</div>
                    <div class="metric-value" style="color: #666;">--</div>
                    <div class="metric-delta">Buat prediksi terlebih dahulu</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            avg_iph = df['Indikator_Harga'].mean()
            std_iph = df['Indikator_Harga'].std()
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Statistik Historis</div>
                <div class="metric-value" style="color: #3498db;">
                    {avg_iph:+.2f}%
                </div>
                <div class="metric-delta">
                    Rata-rata | Volatilitas: {std_iph:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Main forecasting chart
        if 'predictions' in st.session_state:
            st.markdown(f'<div class="panel-title">📈 Grafik Prediksi IPH - Model: {st.session_state["selected_model"]}</div>', unsafe_allow_html=True)
            
            fig_forecast = components['viz'].create_forecast_chart(
                df, 
                st.session_state['predictions'],
                st.session_state['selected_model']
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Dashboard panels
        st.markdown("## 📊 Panel Analisis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Trend analysis
            st.markdown('<div class="panel-title">📈 Tren IPH Historis</div>', unsafe_allow_html=True)
            trend_fig = components['viz'].create_trend_chart(df)
            st.plotly_chart(trend_fig, use_container_width=True)
            
            # Top commodities contributing to price changes
            st.markdown('<div class="panel-title">🥇 Top Komoditas Penyumbang Perubahan</div>', unsafe_allow_html=True)
            commodity_fig = components['viz'].create_commodity_contribution_chart(df)
            st.plotly_chart(commodity_fig, use_container_width=True)
        
        with col2:
            # Monthly fluctuation
            st.markdown('<div class="panel-title">📊 Fluktuasi Harga Bulanan</div>', unsafe_allow_html=True)
            fluctuation_fig = components['viz'].create_fluctuation_chart(df)
            st.plotly_chart(fluctuation_fig, use_container_width=True)
            
            # High fluctuation commodities
            st.markdown('<div class="panel-title">⚡ Komoditas Fluktuasi Tertinggi</div>', unsafe_allow_html=True)
            high_fluc_fig = components['viz'].create_high_fluctuation_chart(df)
            st.plotly_chart(high_fluc_fig, use_container_width=True)
        
        # Summary table
        st.markdown('<div class="panel-title">📋 Ringkasan 5 Minggu Terakhir</div>', unsafe_allow_html=True)
        
        recent_data = df.tail(5).copy()
        recent_data['Status'] = recent_data['Indikator_Harga'].apply(
            lambda x: '📈 Naik' if x > 0 else '📉 Turun' if x < 0 else '➡️ Stabil'
        )
        recent_data['Perubahan'] = recent_data['Indikator_Harga'].apply(lambda x: f"{x:+.2f}%")
        
        # Display table
        display_columns = ['Tanggal', 'Perubahan', 'Status']
        if 'Komoditas_Utama' in recent_data.columns:
            display_columns.append('Komoditas_Utama')
        
        st.dataframe(
            recent_data[display_columns],
            use_container_width=True,
            hide_index=True
        )
        
        # Summary statistics for recent data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            max_increase = recent_data['Indikator_Harga'].max()
            st.metric("Kenaikan Terbesar", f"+{max_increase:.2f}%", delta="5 minggu terakhir")
        
        with col2:
            max_decrease = recent_data['Indikator_Harga'].min()
            st.metric("Penurunan Terbesar", f"{max_decrease:.2f}%", delta="5 minggu terakhir")
        
        with col3:
            recent_avg = recent_data['Indikator_Harga'].mean()
            st.metric("Rata-rata Terkini", f"{recent_avg:+.2f}%", delta="5 minggu terakhir")
        
        with col4:
            recent_vol = recent_data['Indikator_Harga'].std()
            st.metric("Volatilitas Terkini", f"{recent_vol:.2f}%", delta="Standar deviasi")
    
    else:
        st.info("📁 Silakan pilih sumber data dari sidebar untuk memulai analisis")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <h4>Dashboard Prediksi IPH Kota Batu</h4>
    <p>Dikembangkan dengan ❤️ menggunakan Streamlit | © 2025</p>
    <p style="font-size: 0.8rem;">
        Model: KNN, Random Forest, LightGBM, XGBoost Advanced<br>
        Data terakhir diperbarui: {}</p>
</div>
""".format(datetime.now().strftime("%d %B %Y, %H:%M WIB")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()