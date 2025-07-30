import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import time
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from model_predictor import ModelPredictor
from visualizations import DashboardViz
from utils import format_metrics, export_data, check_alerts

# Configuration Class
class Config:
    HIGH_THRESHOLD = 2.0
    LOW_THRESHOLD = -2.0
    DEFAULT_PREDICTION_PERIODS = 4
    MAX_PREDICTION_PERIODS = 12
    CONFIDENCE_LEVEL = 0.95
    MIN_DATA_POINTS = 10
    FEATURE_COLUMNS = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'MA_3', 'MA_7']
    CHART_HEIGHT = 500
    PANEL_HEIGHT = 350
    CACHE_TTL = 3600  # 1 hour

# Performance Monitor Decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        if execution_time > 2:
            st.warning(f"⚠️ {func.__name__} membutuhkan {execution_time:.2f}s untuk dieksekusi")
        
        return result
    return wrapper

def setup_directories():
    """Create required directories if they don't exist"""
    directories = ['data', 'models', 'src', 'exports', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

# ✅ ENHANCED CACHING SYSTEM
def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    if 'data_source_changed' not in st.session_state:
        st.session_state['data_source_changed'] = False
    if 'cached_data' not in st.session_state:
        st.session_state['cached_data'] = None
    if 'cached_viz' not in st.session_state:
        st.session_state['cached_viz'] = None
    if 'last_data_source' not in st.session_state:
        st.session_state['last_data_source'] = None
    if 'data_hash' not in st.session_state:
        st.session_state['data_hash'] = None

def get_data_hash(data_source, uploaded_file_content=None):
    """Generate hash for data source to detect changes"""
    if uploaded_file_content is not None:
        return hash(str(uploaded_file_content.values.tobytes()) + data_source)
    return hash(data_source + str(datetime.now().date()))

@st.cache_resource
def load_models():
    """Cached model loading"""
    try:
        return ModelPredictor()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

@st.cache_resource
def create_visualization_component():
    """Create visualization component (cached)"""
    return DashboardViz()

def load_and_cache_data(data_source, uploaded_file_content=None, force_reload=False):
    """Load data once and cache it properly"""
    
    # Generate current data hash
    current_hash = get_data_hash(data_source, uploaded_file_content)
    
    # Check if we need to reload data
    need_reload = (
        force_reload or
        not st.session_state['data_loaded'] or
        st.session_state['last_data_source'] != data_source or
        st.session_state['data_hash'] != current_hash or
        st.session_state['cached_data'] is None
    )
    
    if not need_reload:
        # Return cached data
        return st.session_state['cached_data']
    
    # Load fresh data
    try:
        loader = DataLoader()
        
        with st.spinner("🔄 Memuat data..."):
            if data_source == "📁 Load File Excel":
                df = loader.load_excel_data()
            elif data_source == "📤 Upload File Baru" and uploaded_file_content is not None:
                df = loader._preprocess_data(uploaded_file_content)
            else:
                df = loader.load_sample_data()
        
        if df is not None and not df.empty:
            # Cache the data
            st.session_state['cached_data'] = df
            st.session_state['data_loaded'] = True
            st.session_state['last_data_source'] = data_source
            st.session_state['data_hash'] = current_hash
            
            # Clear visualization cache when data changes
            st.session_state['cached_viz'] = None
            
            return df
        else:
            st.error("❌ Gagal memuat data")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def get_or_create_visualization(df):
    """Get cached visualization or create new one"""
    if st.session_state['cached_viz'] is None and df is not None:
        # Create and cache visualization component
        viz = create_visualization_component()
        
        # Pre-process charts that should remain static
        with st.spinner("🔄 Memproses visualisasi..."):
            # Pre-generate static charts to cache their data
            viz.create_commodity_contribution_chart(df)
            viz.create_high_fluctuation_chart(df)
        
        st.session_state['cached_viz'] = viz
    
    return st.session_state['cached_viz']

# Page configuration
st.set_page_config(
    page_title="Dashboard Prediksi IPH Kota Batu",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
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
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.5rem;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        display: inline-block;
    }
    
    .panel-title {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-weight: bold;
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
        font-size: 1.1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .sidebar-section {
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .sidebar-header {
        color: #667eea;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        text-align: center;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    .alert-high {
        background: linear-gradient(135deg, #ffe6e6, #ffcccc);
        border: 2px solid #ff4444;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255,68,68,0.2);
    }
    
    .alert-low {
        background: linear-gradient(135deg, #e6f3ff, #cce7ff);
        border: 2px solid #4444ff;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(68,68,255,0.2);
    }
    
    .alert-normal {
        background: linear-gradient(135deg, #e6ffe6, #ccffcc);
        border: 2px solid #44ff44;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(68,255,68,0.2);
    }
    
    .stButton > button {
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    .quality-indicator {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    
    .quality-excellent { background: #d4edda; color: #155724; }
    .quality-good { background: #d1ecf1; color: #0c5460; }
    .quality-warning { background: #fff3cd; color: #856404; }
    .quality-poor { background: #f8d7da; color: #721c24; }
    
    .feature-importance-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        height: 20px;
        border-radius: 10px;
        margin: 2px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 Dashboard Prediksi IPH Kota Batu</h1>
        <p>Sistem Forecasting Indikator Perubahan Harga Komoditas</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">
            Data diperbarui per {datetime.now().strftime("%d %B %Y, %H:%M WIB")} | 
            Threshold Alert: ±{Config.HIGH_THRESHOLD}%
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components with caching
    setup_directories()
    
    # Load components
    with st.spinner("🔄 Menginisialisasi sistem..."):
        predictor = load_models()
    
    if not predictor:
        st.error("❌ Gagal menginisialisasi komponen dashboard")
        return
    
    # ✅ ENHANCED SIDEBAR WITH PROPER DATA CACHING
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-header">🎛️ PANEL KONTROL UTAMA</div>
        """, unsafe_allow_html=True)
        
        # Data source selection
        st.markdown("**📁 Sumber Data:**")
        data_source = st.radio(
            "",
            ["📁 Load File Excel", "📤 Upload File Baru", "🔄 Data Sample"],
            index=0,
            help="Pilih sumber data untuk analisis IPH"
        )
        
        # Handle file upload
        uploaded_file_content = None
        if data_source == "📤 Upload File Baru":
            uploaded_file = st.file_uploader(
                "Pilih file CSV atau Excel",
                type=['csv', 'xlsx', 'xls'],
                help="Format yang didukung: CSV, Excel (.xlsx, .xls)"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        uploaded_file_content = pd.read_csv(uploaded_file)
                    else:
                        uploaded_file_content = pd.read_excel(uploaded_file)
                    st.success(f"✅ File berhasil diupload: {len(uploaded_file_content)} baris")
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
        
        # ✅ LOAD DATA ONCE AND CACHE
        df = load_and_cache_data(data_source, uploaded_file_content)
        
        if df is not None:
            # Data Quality Indicator
            data_quality_score = min(100, len(df) * 2)
            quality_class = (
                "excellent" if data_quality_score >= 80 else
                "good" if data_quality_score >= 60 else
                "warning" if data_quality_score >= 40 else
                "poor"
            )
            
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <div class="quality-indicator quality-{quality_class}">
                    📊 Kualitas Data: {data_quality_score}/100
                </div>
                <br>✅ {len(df)} records | 📅 {df['Tanggal'].min().strftime('%Y-%m-%d')} - {df['Tanggal'].max().strftime('%Y-%m-%d')}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Model Selection Section (TIDAK MEMPENGARUHI DATA)
            st.markdown("""
            <div class="sidebar-section">
                <div class="sidebar-header">🤖 PENGATURAN MODEL</div>
            """, unsafe_allow_html=True)
            
            available_models = predictor.get_available_models()
            
            if available_models:
                selected_model = st.selectbox(
                    "**Pilih Model Prediksi:**",
                    available_models,
                    index=0,
                    help="Pilih algoritma machine learning untuk prediksi",
                    key="model_selector"  # ✅ Explicit key
                )
                
                # Model info
                if selected_model != 'Demo Model (Random)':
                    st.info(f"🎯 Model aktif: **{selected_model}**")
                else:
                    st.warning("🔄 Mode demo - gunakan untuk testing")
            else:
                st.error("❌ Tidak ada model yang tersedia")
                return
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Prediction Settings Section
            st.markdown("""
            <div class="sidebar-section">
                <div class="sidebar-header">⚙️ PENGATURAN PREDIKSI</div>
            """, unsafe_allow_html=True)
            
            # Prediction periods
            periods = st.slider(
                "**Periode Prediksi (minggu):**", 
                min_value=1, 
                max_value=Config.MAX_PREDICTION_PERIODS, 
                value=Config.DEFAULT_PREDICTION_PERIODS,
                help=f"Jumlah minggu ke depan yang akan diprediksi (max: {Config.MAX_PREDICTION_PERIODS})",
                key="periods_slider"  # ✅ Explicit key
            )
            
            # Scenario analysis
            scenario = st.selectbox(
                "**Analisis Skenario:**",
                ["🔵 Normal", "🟢 Optimis (10% lebih tinggi)", "🔴 Pesimis (10% lebih rendah)"],
                help="Pilih skenario untuk analisis what-if",
                key="scenario_selector"  # ✅ Explicit key
            )
            
            # Advanced settings in expander
            with st.expander("🔧 Pengaturan Lanjutan"):
                confidence_level = st.slider(
                    "Confidence Level (%)",
                    min_value=80,
                    max_value=99,
                    value=95,
                    help="Tingkat kepercayaan untuk interval prediksi"
                )
                
                custom_threshold = st.checkbox("Custom Alert Threshold")
                if custom_threshold:
                    col1, col2 = st.columns(2)
                    with col1:
                        high_threshold = st.number_input("High (%)", value=Config.HIGH_THRESHOLD, step=0.1)
                    with col2:
                        low_threshold = st.number_input("Low (%)", value=Config.LOW_THRESHOLD, step=0.1)
                else:
                    high_threshold = Config.HIGH_THRESHOLD
                    low_threshold = Config.LOW_THRESHOLD
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Prediction Button
            st.markdown("""
            <div class="sidebar-section">
                <div class="sidebar-header">🔮 EKSEKUSI PREDIKSI</div>
            """, unsafe_allow_html=True)
            
            predict_button = st.button(
                "🚀 Buat Prediksi IPH", 
                type="primary",
                use_container_width=True,
                help="Klik untuk membuat prediksi IPH berdasarkan pengaturan di atas"
            )
            
            if predict_button:
                with st.spinner("🔄 Membuat prediksi... Mohon tunggu..."):
                    try:
                        start_time = time.time()
                        
                        predictions_result = predictor.predict(
                            selected_model, 
                            df, 
                            periods,
                            scenario.split()[1] if len(scenario.split()) > 1 else scenario
                        )
                        
                        execution_time = time.time() - start_time
                        
                        if predictions_result:
                            st.session_state['predictions'] = predictions_result
                            st.session_state['selected_model'] = selected_model
                            st.session_state['periods'] = periods
                            st.session_state['scenario'] = scenario
                            st.session_state['thresholds'] = {'high': high_threshold, 'low': low_threshold}
                            
                            st.success(f"✅ Prediksi berhasil dibuat dalam {execution_time:.2f} detik!")
                            
                            # Show prediction summary
                            next_pred = predictions_result['predictions'][0]
                            st.metric(
                                "Prediksi Minggu Depan",
                                f"{next_pred:+.2f}%",
                                delta=f"MAE: {predictions_result['mae']:.3f}"
                            )
                        else:
                            st.error("❌ Gagal membuat prediksi")
                            
                    except Exception as e:
                        st.error(f"❌ Error dalam prediksi: {str(e)}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Rest of sidebar sections (Model Analysis, Data Quality Report, Export, System Status)
            # ... (keep the same as before, but add explicit keys to interactive elements)
            
        else:
            st.markdown("</div>", unsafe_allow_html=True)
            st.warning("⚠️ Silakan pilih sumber data untuk memulai analisis")
    
    # ✅ MAIN CONTENT AREA WITH CACHED VISUALIZATION
    if df is not None:
        # Get or create cached visualization
        viz = get_or_create_visualization(df)
        
        if not viz:
            st.error("❌ Gagal menginisialisasi visualisasi")
            return
        
        # Enhanced Alert System (same as before)
        latest_iph = df['Indikator_Harga'].iloc[-1]
        thresholds = st.session_state.get('thresholds', {'high': Config.HIGH_THRESHOLD, 'low': Config.LOW_THRESHOLD})
        alert_status = check_alerts(latest_iph, thresholds['high'], thresholds['low'])
        
        # Alert display (same as before)
        if alert_status['type'] == 'high':
            st.markdown(f"""
            <div class="alert-high">
                <h4>🚨 PERINGATAN TINGGI</h4>
                <p><strong>IPH saat ini: {latest_iph:+.2f}%</strong> (di atas threshold +{thresholds['high']}%)</p>
                <p>⚠️ Diperlukan perhatian khusus terhadap stabilitas harga komoditas</p>
                <p>📋 Rekomendasi: Monitor komoditas penyumbang utama dan siapkan intervensi pasar</p>
            </div>
            """, unsafe_allow_html=True)
        elif alert_status['type'] == 'low':
            st.markdown(f"""
            <div class="alert-low">
                <h4>📉 PERINGATAN RENDAH</h4>
                <p><strong>IPH saat ini: {latest_iph:+.2f}%</strong> (di bawah threshold {thresholds['low']}%)</p>
                <p>📊 Indikasi penurunan harga yang signifikan</p>
                <p>📋 Rekomendasi: Evaluasi dampak terhadap petani dan produsen lokal</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-normal">
                <h4>✅ STATUS NORMAL</h4>
                <p><strong>IPH saat ini: {latest_iph:+.2f}%</strong> (dalam batas normal ±{thresholds['high']}%)</p>
                <p>📈 Kondisi harga komoditas dalam kondisi stabil</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced Metrics Row (same as before)
        col1, col2, col3, col4 = st.columns(4)
        
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
                <div class="metric-delta" style="background: {'rgba(231,76,60,0.1)' if delta_iph < 0 else 'rgba(39,174,96,0.1)'}; color: {'#e74c3c' if delta_iph < 0 else '#27ae60'};">
                    {delta_iph:+.3f}% vs minggu lalu
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'predictions' in st.session_state:
                next_pred = st.session_state['predictions']['predictions'][0]
                model_mae = st.session_state['predictions']['mae']
                confidence = st.session_state['predictions']['confidence_upper'][0] - st.session_state['predictions']['confidence_lower'][0]
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Prediksi Minggu Depan</div>
                    <div class="metric-value" style="color: {'#e74c3c' if next_pred < 0 else '#27ae60'};">
                        {next_pred:+.2f}%
                    </div>
                    <div class="metric-delta" style="background: rgba(52,152,219,0.1); color: #3498db;">
                        ±{confidence/2:.2f}% CI | MAE: {model_mae:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Prediksi Minggu Depan</div>
                    <div class="metric-value" style="color: #666;">--</div>
                    <div class="metric-delta" style="background: rgba(108,117,125,0.1);">Buat prediksi terlebih dahulu</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            avg_iph = df['Indikator_Harga'].mean()
            std_iph = df['Indikator_Harga'].std()
            trend = "📈" if df['Indikator_Harga'].iloc[-1] > df['Indikator_Harga'].iloc[-5:].mean() else "📉"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Rata-rata Historis</div>
                <div class="metric-value" style="color: #3498db;">
                    {avg_iph:+.2f}%
                </div>
                <div class="metric-delta" style="background: rgba(155,89,182,0.1); color: #9b59b6;">
                    {trend} Volatilitas: {std_iph:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Calculate trend direction
            recent_trend = df['Indikator_Harga'].tail(4).mean() - df['Indikator_Harga'].head(4).mean()
            trend_direction = "Naik" if recent_trend > 0 else "Turun"
            trend_color = "#27ae60" if recent_trend > 0 else "#e74c3c"
            trend_icon = "📈" if recent_trend > 0 else "📉"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tren Periode</div>
                <div class="metric-value" style="color: {trend_color};">
                    {trend_icon}
                </div>
                <div class="metric-delta" style="background: rgba(241,196,15,0.1); color: #f1c40f;">
                    {trend_direction} {abs(recent_trend):.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Main forecasting chart
        if 'predictions' in st.session_state:
            st.markdown(f'''
            <div class="panel-title">
                📈 Grafik Prediksi IPH - Model: {st.session_state["selected_model"]} 
                | Skenario: {st.session_state.get("scenario", "Normal")}
                | Periode: {st.session_state["periods"]} minggu
            </div>
            ''', unsafe_allow_html=True)
            
            try:
                fig_forecast = viz.create_forecast_chart(
                    df, 
                    st.session_state['predictions'],
                    st.session_state['selected_model']
                )
                st.plotly_chart(fig_forecast, use_container_width=True, config={'displayModeBar': True})
                
                # Prediction insights
                predictions = st.session_state['predictions']['predictions']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_pred = np.mean(predictions)
                    st.metric("Rata-rata Prediksi", f"{avg_pred:+.2f}%")
                
                with col2:
                    max_pred = max(predictions)
                    st.metric("Prediksi Tertinggi", f"{max_pred:+.2f}%")
                
                with col3:
                    min_pred = min(predictions)
                    st.metric("Prediksi Terendah", f"{min_pred:+.2f}%")
                
            except Exception as e:
                st.error(f"❌ Error menampilkan grafik prediksi: {str(e)}")
        else:
            st.info("🔮 Buat prediksi terlebih dahulu untuk melihat grafik forecasting")
        
        # ✅ ENHANCED DASHBOARD PANELS WITH CACHED DATA
        st.markdown("## 📊 Panel Analisis Komprehensif")
        
        # First row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="panel-title">📈 Tren IPH Historis & Moving Average</div>', unsafe_allow_html=True)
            trend_fig = viz.create_trend_chart(df)
            st.plotly_chart(trend_fig, use_container_width=True, config={'displayModeBar': False})
            
        with col2:
            st.markdown('<div class="panel-title">📊 Distribusi & Volatilitas IPH</div>', unsafe_allow_html=True)
            fluctuation_fig = viz.create_fluctuation_chart(df)
            st.plotly_chart(fluctuation_fig, use_container_width=True, config={'displayModeBar': False})
        
        # Second row of charts - ✅ THESE WILL NOW BE CONSISTENT
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="panel-title">🥇 Top Komoditas Penyumbang Perubahan</div>', unsafe_allow_html=True)
            # ✅ This chart will now remain consistent across model changes
            commodity_fig = viz.create_commodity_contribution_chart(df)
            st.plotly_chart(commodity_fig, use_container_width=True, config={'displayModeBar': False})
            
        with col2:
            st.markdown('<div class="panel-title">⚡ Komoditas dengan Fluktuasi Tertinggi</div>', unsafe_allow_html=True)
            # ✅ This chart will now remain consistent across model changes
            high_fluc_fig = viz.create_high_fluctuation_chart(df)
            st.plotly_chart(high_fluc_fig, use_container_width=True, config={'displayModeBar': False})
        
        # Enhanced Summary Section (rest of the code remains the same)
        # ... (keep all the tab sections as before)
        
    else:
        # Enhanced empty state (same as before)
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <h2>🏁 Selamat Datang di Dashboard IPH Kota Batu</h2>
            <p style="font-size: 1.2rem; color: #666; margin: 2rem 0;">
                Pilih sumber data dari sidebar untuk memulai analisis prediksi IPH
            </p>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Footer (same as before)
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 3rem 2rem; background: linear-gradient(135deg, #f8f9fa, #ffffff); border-radius: 15px; margin-top: 2rem;'>
    <h3 style="color: #667eea;">📊 Dashboard Prediksi IPH Kota Batu</h3>
    <p style="font-size: 1.1rem; margin: 1rem 0;">
        Sistem Forecasting Indikator Perubahan Harga berbasis Machine Learning
    </p>
    <p style="font-size: 0.9rem; margin-top: 2rem;">
        <strong>Terakhir diperbarui:</strong> {datetime.now().strftime("%d %B %Y, %H:%M WIB")}<br>
        <strong>Model:</strong> KNN, Random Forest, LightGBM, XGBoost Advanced<br>
        <strong>Alert Threshold:</strong> ±{Config.HIGH_THRESHOLD}% | <strong>Confidence Level:</strong> {Config.CONFIDENCE_LEVEL}%
    </p>
    <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #dee2e6;">
        <p style="margin: 0;">Dikembangkan dengan ❤️ menggunakan Streamlit | © 2025 Dashboard IPH Kota Batu</p>
    </div>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()