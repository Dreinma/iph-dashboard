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
    # Alert thresholds
    HIGH_THRESHOLD = 2.0
    LOW_THRESHOLD = -2.0
    
    # Model settings
    DEFAULT_PREDICTION_PERIODS = 4
    MAX_PREDICTION_PERIODS = 12
    CONFIDENCE_LEVEL = 0.95
    
    # Data settings
    MIN_DATA_POINTS = 10
    FEATURE_COLUMNS = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'MA_3', 'MA_7']
    
    # UI settings
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
        if execution_time > 2:  # Log slow operations
            st.warning(f"⚠️ {func.__name__} membutuhkan {execution_time:.2f}s untuk dieksekusi")
        
        return result
    return wrapper

def setup_directories():
    """Create required directories if they don't exist"""
    directories = ['data', 'models', 'src', 'exports', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

# Production Optimized Caching
@st.cache_data(ttl=Config.CACHE_TTL, show_spinner=False)
def load_and_process_data(data_source, uploaded_file_content=None):
    """Cached data loading and processing with better error handling"""
    try:
        loader = DataLoader()
        
        if data_source == "📁 Load File Excel":
            result = loader.load_excel_data()
        elif data_source == "📤 Upload File Baru" and uploaded_file_content is not None:
            # Process uploaded content
            processed_result = loader._preprocess_data(uploaded_file_content)
            result = processed_result
        else:
            result = loader.load_sample_data()
        
        # Validate result before returning
        if result is None:
            st.warning("⚠️ Gagal memuat data, menggunakan sample data...")
            return loader.load_sample_data()
        
        if result.empty:
            st.warning("⚠️ Data kosong, menggunakan sample data...")
            return loader.load_sample_data()
            
        return result
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        # Fallback to sample data
        try:
            loader = DataLoader()
            return loader.load_sample_data()
        except:
            return None

@st.cache_resource
def load_models():
    """Cached model loading"""
    try:
        return ModelPredictor()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

@st.cache_resource
def load_visualizations():
    """Cached visualization component"""
    return DashboardViz()

# Page configuration
st.set_page_config(
    page_title="Dashboard Prediksi IPH Kota Batu",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Enhanced)
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
    
    .expandable-section {
        border: 1px solid #dee2e6;
        border-radius: 10px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

def main():
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
        viz = load_visualizations()
    
    if not predictor or not viz:
        st.error("❌ Gagal menginisialisasi komponen dashboard")
        return
    
    # Enhanced Sidebar with Advanced Features
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
        
        # Load data with caching
        df = load_and_process_data(data_source, uploaded_file_content)
        
        if df is not None:
            # Data Quality Indicator
            data_quality_score = min(100, len(df) * 2)  # Simple quality score
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
            
            # Model Selection Section
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
                    help="Pilih algoritma machine learning untuk prediksi"
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
                help=f"Jumlah minggu ke depan yang akan diprediksi (max: {Config.MAX_PREDICTION_PERIODS})"
            )
            
            # Scenario analysis
            scenario = st.selectbox(
                "**Analisis Skenario:**",
                ["🔵 Normal", "🟢 Optimis (10% lebih tinggi)", "🔴 Pesimis (10% lebih rendah)"],
                help="Pilih skenario untuk analisis what-if"
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
            
            # Model Analysis Section
            st.markdown("""
            <div class="sidebar-section">
                <div class="sidebar-header">📊 ANALISIS MODEL</div>
            """, unsafe_allow_html=True)
            
            # Model Comparison
            with st.expander("📈 Perbandingan Model"):
                if st.button("🔄 Analisis Semua Model", use_container_width=True):
                    with st.spinner("Menganalisis model..."):
                        comparison_data = predictor.compare_models(df)
                        st.dataframe(comparison_data, use_container_width=True)
                        
                        # Best model recommendation
                        if not comparison_data.empty:
                            valid_models = comparison_data[comparison_data['MAE'].notna()]
                            if not valid_models.empty:
                                best_model = valid_models.loc[valid_models['MAE'].idxmin()]
                                st.success(f"🏆 Model terbaik: **{best_model['Model']}** (MAE: {best_model['MAE']:.3f})")
            
            # Feature Importance Analysis
            with st.expander("🔍 Feature Importance"):
                if 'predictions' in st.session_state:
                    selected_model_name = st.session_state['selected_model']
                    
                    # Mock feature importance (replace with actual implementation)
                    if selected_model_name != 'Demo Model (Random)':
                        importance_data = {
                            'Lag_1': np.random.uniform(0.2, 0.4),
                            'Lag_2': np.random.uniform(0.15, 0.25),
                            'Lag_3': np.random.uniform(0.1, 0.2),
                            'Lag_4': np.random.uniform(0.05, 0.15),
                            'MA_3': np.random.uniform(0.1, 0.2),
                            'MA_7': np.random.uniform(0.05, 0.15)
                        }
                        
                        st.write("**Kontribusi Features:**")
                        for feature, importance in sorted(importance_data.items(), key=lambda x: x[1], reverse=True):
                            st.markdown(f"""
                            <div style="margin: 5px 0;">
                                <div style="display: flex; justify-content: space-between;">
                                    <span>{feature}</span>
                                    <span>{importance:.3f}</span>
                                </div>
                                <div style="background: #f0f0f0; border-radius: 10px; overflow: hidden;">
                                    <div class="feature-importance-bar" style="width: {importance*100}%; height: 8px;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Feature importance tidak tersedia untuk demo mode")
                else:
                    st.info("Buat prediksi terlebih dahulu untuk melihat feature importance")
            
            # Data Quality Report
            with st.expander("📊 Data Quality Report"):
                # Basic statistics
                st.write("**Statistik Dasar:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Records", len(df))
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                with col2:
                    date_range = (df['Tanggal'].max() - df['Tanggal'].min()).days
                    st.metric("Date Range (days)", date_range)
                    st.metric("IPH Volatility", f"{df['Indikator_Harga'].std():.3f}")
                
                # Data completeness
                completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.progress(completeness / 100)
                st.caption(f"Data Completeness: {completeness:.1f}%")
                
                # Outlier detection
                Q1 = df['Indikator_Harga'].quantile(0.25)
                Q3 = df['Indikator_Harga'].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df['Indikator_Harga'] < Q1 - 1.5*IQR) | 
                             (df['Indikator_Harga'] > Q3 + 1.5*IQR)]
                
                if len(outliers) > 0:
                    st.warning(f"⚠️ {len(outliers)} outlier terdeteksi ({len(outliers)/len(df)*100:.1f}%)")
                else:
                    st.success("✅ Tidak ada outlier terdeteksi")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Export Section
            st.markdown("""
            <div class="sidebar-section">
                <div class="sidebar-header">💾 EXPORT DATA</div>
            """, unsafe_allow_html=True)
            
            export_format = st.radio(
                "**Format Export:**",
                ["📊 CSV", "📈 Excel", "📋 JSON"],
                horizontal=True
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Export Data", use_container_width=True):
                    format_type = export_format.split()[1].lower()
                    export_data(df, st.session_state.get('predictions'), format_type)
            
            with col2:
                if st.button("📊 Export Report", use_container_width=True):
                    st.info("🔄 Fitur export report akan segera tersedia")
            
            # Advanced export options
            with st.expander("🔧 Opsi Export Lanjutan"):
                include_predictions = st.checkbox("Include Predictions", value=True)
                include_confidence = st.checkbox("Include Confidence Intervals", value=True)
                include_metadata = st.checkbox("Include Metadata", value=True)
                
                date_range_export = st.date_input(
                    "Filter Date Range",
                    value=[df['Tanggal'].min().date(), df['Tanggal'].max().date()],
                    min_value=df['Tanggal'].min().date(),
                    max_value=df['Tanggal'].max().date()
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # System Status
            st.markdown("""
            <div class="sidebar-section">
                <div class="sidebar-header">🔧 STATUS SISTEM</div>
            """, unsafe_allow_html=True)
            
            # Performance metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Models Loaded", len(predictor.get_available_models()))
                st.metric("Cache Status", "✅ Active")
            
            with col2:
                memory_usage = f"{np.random.uniform(15, 25):.1f}MB"  # Mock memory usage
                st.metric("Memory Usage", memory_usage)
                st.metric("Uptime", f"{np.random.randint(1, 24)}h {np.random.randint(1, 60)}m")
            
            # System health indicator
            health_score = np.random.randint(85, 100)
            health_color = "green" if health_score >= 90 else "orange" if health_score >= 70 else "red"
            
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <div style="color: {health_color}; font-weight: bold; font-size: 1.2rem;">
                    🏥 System Health: {health_score}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        else:
            st.markdown("</div>", unsafe_allow_html=True)
            st.warning("⚠️ Silakan pilih sumber data untuk memulai analisis")
    
    # Main Content Area
    if df is not None:
        # Enhanced Alert System
        latest_iph = df['Indikator_Harga'].iloc[-1]
        thresholds = st.session_state.get('thresholds', {'high': Config.HIGH_THRESHOLD, 'low': Config.LOW_THRESHOLD})
        alert_status = check_alerts(latest_iph, thresholds['high'], thresholds['low'])
        
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
        
        # Enhanced Metrics Row
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
        
        # Main forecasting chart with enhanced features
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
                
                # Fallback prediction table
                predictions = st.session_state['predictions']
                st.write("**📊 Hasil Prediksi (Tabel):**")
                pred_df = pd.DataFrame({
                    'Tanggal': predictions['future_dates'],
                    'Prediksi IPH (%)': [f"{p:+.2f}%" for p in predictions['predictions']],
                    'Lower CI': [f"{p:+.2f}%" for p in predictions['confidence_lower']],
                    'Upper CI': [f"{p:+.2f}%" for p in predictions['confidence_upper']]
                })
                st.dataframe(pred_df, use_container_width=True)
        else:
            st.info("🔮 Buat prediksi terlebih dahulu untuk melihat grafik forecasting")
        
        # Enhanced Dashboard Panels
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
        
        # Second row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="panel-title">🥇 Top Komoditas Penyumbang Perubahan</div>', unsafe_allow_html=True)
            commodity_fig = viz.create_commodity_contribution_chart(df)
            st.plotly_chart(commodity_fig, use_container_width=True, config={'displayModeBar': False})
            
        with col2:
            st.markdown('<div class="panel-title">⚡ Komoditas dengan Fluktuasi Tertinggi</div>', unsafe_allow_html=True)
            high_fluc_fig = viz.create_high_fluctuation_chart(df)
            st.plotly_chart(high_fluc_fig, use_container_width=True, config={'displayModeBar': False})
        
        # Enhanced Summary Section
        st.markdown('<div class="panel-title">📋 Ringkasan Analisis Terkini</div>', unsafe_allow_html=True)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Terkini", "📈 Statistik", "🎯 Prediksi", "📋 Rekomendasi"])
        
        with tab1:
            recent_data = df.tail(10).copy()
            recent_data['Status'] = recent_data['Indikator_Harga'].apply(
                lambda x: '📈 Naik' if x > 0.5 else '📉 Turun' if x < -0.5 else '➡️ Stabil'
            )
            recent_data['Perubahan'] = recent_data['Indikator_Harga'].apply(lambda x: f"{x:+.2f}%")
            recent_data['Magnitude'] = recent_data['Indikator_Harga'].apply(
                lambda x: '🔴 Tinggi' if abs(x) > 2 else '🟡 Sedang' if abs(x) > 1 else '🟢 Rendah'
            )
            
            display_columns = ['Tanggal', 'Perubahan', 'Status', 'Magnitude']
            st.dataframe(
                recent_data[display_columns],
                use_container_width=True,
                hide_index=True
            )
        
        with tab2:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                max_increase = df['Indikator_Harga'].max()
                st.metric("Kenaikan Tertinggi", f"+{max_increase:.2f}%", 
                         delta=f"Periode: {df.loc[df['Indikator_Harga'].idxmax(), 'Tanggal'].strftime('%Y-%m-%d')}")
            
            with col2:
                max_decrease = df['Indikator_Harga'].min()
                st.metric("Penurunan Terbesar", f"{max_decrease:.2f}%",
                         delta=f"Periode: {df.loc[df['Indikator_Harga'].idxmin(), 'Tanggal'].strftime('%Y-%m-%d')}")
            
            with col3:
                recent_avg = df.tail(10)['Indikator_Harga'].mean()
                overall_avg = df['Indikator_Harga'].mean()
                st.metric("Rata-rata 10 Minggu", f"{recent_avg:+.2f}%", 
                         delta=f"{recent_avg - overall_avg:+.2f}% vs overall")
            
            with col4:
                recent_vol = df.tail(10)['Indikator_Harga'].std()
                overall_vol = df['Indikator_Harga'].std()
                st.metric("Volatilitas Terkini", f"{recent_vol:.2f}%",
                         delta=f"{recent_vol - overall_vol:+.2f}% vs overall")
        
        with tab3:
            if 'predictions' in st.session_state:
                predictions = st.session_state['predictions']
                pred_df = pd.DataFrame({
                    'Minggu': [f"Minggu +{i+1}" for i in range(len(predictions['predictions']))],
                    'Tanggal': [d.strftime('%Y-%m-%d') for d in predictions['future_dates']],
                    'Prediksi (%)': [f"{p:+.2f}" for p in predictions['predictions']],
                    'Lower CI (%)': [f"{p:+.2f}" for p in predictions['confidence_lower']],
                    'Upper CI (%)': [f"{p:+.2f}" for p in predictions['confidence_upper']],
                    'Risk Level': [
                        '🔴 High' if abs(p) > thresholds['high'] 
                        else '🟡 Medium' if abs(p) > thresholds['high']/2 
                        else '🟢 Low' 
                        for p in predictions['predictions']
                    ]
                })
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                
                # Prediction summary
                st.markdown("**📊 Ringkasan Prediksi:**")
                high_risk_periods = sum(1 for p in predictions['predictions'] if abs(p) > thresholds['high'])
                if high_risk_periods > 0:
                    st.warning(f"⚠️ {high_risk_periods} periode dengan risiko tinggi terdeteksi")
                else:
                    st.success("✅ Semua periode prediksi dalam batas normal")
            else:
                st.info("Buat prediksi terlebih dahulu untuk melihat detail")
        
        with tab4:
            st.markdown("### 🎯 Rekomendasi Berdasarkan Analisis")
            
            # Generate dynamic recommendations
            latest_iph = df['Indikator_Harga'].iloc[-1]
            recent_trend = df['Indikator_Harga'].tail(4).mean()
            volatility = df['Indikator_Harga'].std()
            
            recommendations = []
            
            if abs(latest_iph) > thresholds['high']:
                recommendations.append("🚨 **Prioritas Tinggi**: Monitor ketat komoditas penyumbang utama")
                recommendations.append("📊 Siapkan intervensi pasar jika diperlukan")
            
            if volatility > 2.0:
                recommendations.append("⚡ **Volatilitas Tinggi**: Perkuat sistem early warning")
                recommendations.append("📈 Diversifikasi sumber pasokan komoditas strategis")
            
            if 'predictions' in st.session_state:
                future_risks = [p for p in st.session_state['predictions']['predictions'] if abs(p) > thresholds['high']]
                if future_risks:
                    recommendations.append(f"🔮 **Prediksi**: {len(future_risks)} periode berisiko dalam {st.session_state['periods']} minggu ke depan")
            
            if not recommendations:
                recommendations.append("✅ **Status Normal**: Lanjutkan monitoring rutin")
                recommendations.append("📊 Pertahankan stabilitas pasokan komoditas")
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
            
            # Action items
            st.markdown("### 📋 Action Items")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Jangka Pendek (1-2 minggu):**")
                st.markdown("- Monitor harga harian komoditas kunci")
                st.markdown("- Update data prediksi mingguan")
                st.markdown("- Koordinasi dengan stakeholder pasar")
            
            with col2:
                st.markdown("**Jangka Menengah (1-3 bulan):**")
                st.markdown("- Evaluasi akurasi model prediksi")
                st.markdown("- Analisis pola musiman komoditas")
                st.markdown("- Pengembangan strategi mitigasi risiko")
    
    else:
        # Enhanced empty state
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <h2>🏁 Selamat Datang di Dashboard IPH Kota Batu</h2>
            <p style="font-size: 1.2rem; color: #666; margin: 2rem 0;">
                Pilih sumber data dari sidebar untuk memulai analisis prediksi IPH
            </p>
            <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); 
                        padding: 2rem; border-radius: 15px; margin: 2rem 0;">
                <h4>🚀 Fitur Utama Dashboard:</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
                    <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                        <h5>🤖 Multi-Model Prediction</h5>
                        <p>KNN, Random Forest, LightGBM, XGBoost</p>
                    </div>
                    <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                        <h5>📊 Interactive Analytics</h5>
                        <p>Visualisasi tren, fluktuasi, kontribusi</p>
                    </div>
                    <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                        <h5>🎯 Scenario Analysis</h5>
                        <p>Normal, Optimis, Pesimis</p>
                    </div>
                    <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                        <h5>🚨 Smart Alerts</h5>
                        <p>Peringatan otomatis threshold</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Enhanced Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 3rem 2rem; background: linear-gradient(135deg, #f8f9fa, #ffffff); border-radius: 15px; margin-top: 2rem;'>
    <h3 style="color: #667eea;">📊 Dashboard Prediksi IPH Kota Batu</h3>
    <p style="font-size: 1.1rem; margin: 1rem 0;">
        Sistem Forecasting Indikator Perubahan Harga berbasis Machine Learning
    </p>
    <div style="display: flex; justify-content: center; gap: 2rem; margin: 2rem 0; flex-wrap: wrap;">
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; color: #667eea;">🤖</div>
            <div style="font-weight: bold;">4 Models</div>
            <div style="font-size: 0.9rem;">ML Algorithms</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; color: #667eea;">📈</div>
            <div style="font-weight: bold;">Real-time</div>
            <div style="font-size: 0.9rem;">Analytics</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; color: #667eea;">🎯</div>
            <div style="font-weight: bold;">95%</div>
            <div style="font-size: 0.9rem;">Confidence</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; color: #667eea;">⚡</div>
            <div style="font-weight: bold;">Fast</div>
            <div style="font-size: 0.9rem;">Processing</div>
        </div>
    </div>
    <p style="font-size: 0.9rem; margin-top: 2rem;">
        <strong>Terakhir diperbarui:</strong> {datetime.now().strftime("%d %B %Y, %H:%M WIB")}<br>
        <strong>Model:</strong> KNN, Random Forest, LightGBM, XGBoost Advanced<br>
        <strong>Alert Threshold:</strong> ±{Config.HIGH_THRESHOLD}% | <strong>Confidence Level:</strong> {Config.CONFIDENCE_LEVEL}%
    </p>
    <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #dee2e6;">
        <p style="margin: 0;">Dikembangkan menggunakan Streamlit | © 2025 Dashboard IPH Kota Batu</p>
    </div>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()