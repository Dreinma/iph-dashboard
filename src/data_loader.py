import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os

class DataLoader:
    def __init__(self):
        # More robust path handling
        self.base_path = Path(__file__).parent.parent  # Go up to project root
        self.data_path = self.base_path / "data"
        self.excel_file = "IPH_Kota_Batu.xlsx"
        
        # Create data directory if it doesn't exist
        self.data_path.mkdir(exist_ok=True)
    
    def find_excel_file(self):
        """Find Excel file using multiple strategies"""
        possible_paths = [
            # Current approach
            self.data_path / self.excel_file,
            
            # Relative to current working directory
            Path.cwd() / "data" / self.excel_file,
            
            # Alternative names
            self.data_path / "iph_kota_batu.xlsx",
            self.data_path / "IPH Kota Batu.xlsx",
            self.data_path / "data_iph_modeling.xlsx"
        ]
        
        for path in possible_paths:
            if path.exists():
                st.success(f"✅ File ditemukan di: {path}")
                return path
        
        return None
    
    @st.cache_data
    def load_excel_data(_self):
        """Load data from Excel file with improved error handling"""
        # First try to find Excel file
        excel_path = _self.find_excel_file()
        
        if excel_path is None:
            # Try CSV as fallback
            csv_path = _self.data_path / "data_iph_modeling.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    st.success(f"✅ Data CSV berhasil dimuat: {len(df)} records")
                    return _self._preprocess_data(df)
                except Exception as e:
                    st.error(f"❌ Error loading CSV: {str(e)}")
            
            st.warning(f"⚠️ File {_self.excel_file} tidak ditemukan di folder data/")
            st.info("💡 Silakan upload file atau gunakan data sample")
            return None
        
        try:
            # Try different sheet names
            sheet_names = ['Sheet1', 'Data', 'IPH', 0]
            df = None
            
            for sheet in sheet_names:
                try:
                    df = pd.read_excel(str(excel_path), sheet_name=sheet)
                    st.success(f"✅ Data berhasil dimuat dari sheet: {sheet}")
                    break
                except Exception as sheet_error:
                    continue
            
            if df is None:
                st.error("❌ Gagal membaca file Excel dari semua sheet")
                return None
                
            return _self._preprocess_data(df)
            
        except Exception as e:
            st.error(f"❌ Error loading Excel file: {str(e)}")
            st.info("💡 Coba gunakan data sample atau upload file baru")
            return None
    
    def upload_data(self):
        """Handle file upload with improved error handling"""
        st.markdown("### 📤 Upload File Data IPH")
        
        uploaded_file = st.file_uploader(
            "Pilih file CSV atau Excel",
            type=['csv', 'xlsx', 'xls'],
            help="Format yang didukung: CSV, Excel (.xlsx, .xls)"
        )
        
        if uploaded_file is not None:
            try:
                # Show file info
                st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size} bytes)")
                
                # Read file based on extension
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Check if DataFrame is valid and not empty
                if df is None:
                    st.error("❌ File kosong atau tidak dapat dibaca")
                    return None
                
                if df.empty:
                    st.error("❌ File tidak mengandung data")
                    return None
                
                st.success(f"✅ File berhasil diupload: {len(df)} baris data")
                
                # Show column preview
                st.info(f"📋 Kolom yang ditemukan: {', '.join(df.columns.tolist())}")
                
                # Preprocess the data
                processed_df = self._preprocess_data(df)
                
                # Check if preprocessing was successful
                if processed_df is not None and not processed_df.empty:
                    return processed_df
                else:
                    st.error("❌ Gagal memproses data yang diupload")
                    return None
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("💡 Pastikan format file sesuai dengan template")
                
                # Show detailed error for debugging
                with st.expander("🔍 Detail Error (untuk debugging)"):
                    st.code(str(e))
                
                return None
        
        return None
    
    @st.cache_data
    def load_sample_data(_self):
        """Load sample data for demo"""
        st.info("🔄 Menggunakan data sample...")
        
        try:
            # Create more realistic sample data based on your Excel structure
            np.random.seed(42)
            
            # Generate dates from 2023 to present
            start_date = pd.Timestamp('2023-01-01')
            end_date = pd.Timestamp.now()
            dates = pd.date_range(start=start_date, end=end_date, freq='W')
            n_samples = len(dates)
            
            # Generate realistic IPH data with trends and seasonality
            t = np.arange(n_samples)
            trend = 0.01 * t  # Slight upward trend
            seasonal = 0.5 * np.sin(2 * np.pi * t / 52)  # Annual seasonality
            noise = np.random.normal(0, 1.2, n_samples)
            iph_values = trend + seasonal + noise
            
            # Add some realistic spikes
            spike_indices = np.random.choice(n_samples, size=n_samples//10, replace=False)
            iph_values[spike_indices] += np.random.normal(0, 2, len(spike_indices))
            
            # Commodity lists
            commodities_main = [
                'BERAS', 'CABAI RAWIT', 'CABAI MERAH', 'BAWANG MERAH', 
                'DAGING AYAM RAS', 'TELUR AYAM RAS', 'MINYAK GORENG', 
                'BAWANG PUTIH', 'GULA PASIR', 'DAGING SAPI'
            ]
            
            commodities_fluc = [
                'CABAI RAWIT', 'CABAI MERAH', 'BAWANG MERAH', 'BAWANG PUTIH',
                'TELUR AYAM RAS', 'DAGING AYAM RAS', 'MINYAK GORENG', 
                'STABIL', 'BERAS', 'GULA PASIR'
            ]
            
            # Create sample DataFrame
            sample_data = {
                'Tanggal': dates,
                'Bulan': [d.strftime('%B') for d in dates],
                'Minggu ke-': [f'M{((d.day-1)//7)+1}' for d in dates],
                'Kab/Kota': ['BATU'] * n_samples,
                'Indikator Perubahan Harga (%)': iph_values,
                'Komoditas Andil Perubahan Harga': np.random.choice(commodities_main, n_samples),
                'Komoditas Fluktuasi Harga Tertinggi': np.random.choice(commodities_fluc, n_samples),
                'Fluktuasi Harga': np.random.uniform(0, 0.25, n_samples)
            }
            
            df = pd.DataFrame(sample_data)
            st.success(f"✅ Data sample berhasil dibuat: {len(df)} records")
            
            return _self._preprocess_data(df)
            
        except Exception as e:
            st.error(f"❌ Error creating sample data: {str(e)}")
            return _self._create_minimal_sample()
    
    def _create_minimal_sample(self):
        """Create minimal sample data as fallback"""
        try:
            dates = pd.date_range(start='2024-01-01', periods=50, freq='W')
            iph_values = np.random.normal(0, 1, 50)
            
            df = pd.DataFrame({
                'Tanggal': dates,
                'Indikator_Harga': iph_values
            })
            
            processed_df = self._preprocess_data(df)
            return processed_df
            
        except Exception as e:
            st.error(f"❌ Error creating minimal sample: {str(e)}")
            return None
    
    def _preprocess_data(self, df):
        """Clean and preprocess data with improved error handling"""
        try:
            # Check if df is None or empty
            if df is None:
                st.error("❌ Data kosong (None)")
                return None
                
            if df.empty:
                st.error("❌ DataFrame kosong")
                return None
            
            st.info("🔄 Preprocessing data...")
            
            # Handle different column name variations
            iph_columns = [
                'Indikator Perubahan Harga (%)',
                'Indikator_Harga',
                'IPH',
                'Indikator Perubahan Harga',
                ' Indikator Perubahan Harga (%)',  # With leading space
                'Indikator_Harga'  # From CSV file
            ]
            
            iph_col = None
            for col in iph_columns:
                if col in df.columns:
                    iph_col = col
                    break
            
            if iph_col is None:
                st.error("❌ Kolom Indikator Perubahan Harga tidak ditemukan!")
                st.info("📋 Kolom yang tersedia: " + ", ".join(df.columns.tolist()))
                return None
            
            # Rename to standard column name
            df = df.rename(columns={iph_col: 'Indikator_Harga'})
            
            # Handle date column
            if 'Tanggal' not in df.columns:
                # Look for date-like columns
                date_columns = ['Date', 'Periode', 'Waktu', 'Tgl']
                date_col = None
                
                for col in date_columns:
                    if col in df.columns:
                        df = df.rename(columns={col: 'Tanggal'})
                        date_col = col
                        break
                
                if date_col is None:
                    if 'Bulan' in df.columns and 'Minggu ke-' in df.columns:
                        # Try to create dates from Bulan and Minggu
                        df['Tanggal'] = pd.date_range(
                            start='2023-01-01', 
                            periods=len(df), 
                            freq='W'
                        )
                    else:
                        df['Tanggal'] = pd.date_range(
                            start='2023-01-01', 
                            periods=len(df), 
                            freq='W'
                        )
            
            # Convert to datetime
            df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
            
            # Convert IPH to numeric, handling various formats
            if df['Indikator_Harga'].dtype == 'object':
                # Remove any non-numeric characters except minus and decimal point
                df['Indikator_Harga'] = df['Indikator_Harga'].astype(str).str.replace(r'[^\d.-]', '', regex=True)
            
            df['Indikator_Harga'] = pd.to_numeric(df['Indikator_Harga'], errors='coerce')
            
            # Remove rows with missing critical data
            initial_len = len(df)
            df = df.dropna(subset=['Tanggal', 'Indikator_Harga'])
            dropped_rows = initial_len - len(df)
            
            if dropped_rows > 0:
                st.warning(f"⚠️ {dropped_rows} baris dengan data tidak lengkap telah dihapus")
            
            if len(df) == 0:
                st.error("❌ Tidak ada data valid setelah preprocessing")
                return None
            
            # Sort by date
            df = df.sort_values('Tanggal').reset_index(drop=True)
            
            # Create lag features for model prediction
            df = self._create_features(df)
            
            # Final validation
            if df is None or df.empty:
                st.error("❌ Data kosong setelah feature creation")
                return None
            
            st.success(f"✅ Preprocessing selesai: {len(df)} records valid")
            return df
            
        except Exception as e:
            st.error(f"❌ Error in preprocessing: {str(e)}")
            
            # Show detailed error for debugging
            with st.expander("🔍 Detail Error Preprocessing"):
                st.code(str(e))
                if df is not None and not df.empty:
                    st.write("**DataFrame Info:**")
                    st.write(f"Shape: {df.shape}")
                    st.write(f"Columns: {df.columns.tolist()}")
                    st.write("**Sample Data:**")
                    st.dataframe(df.head())
            
            return None
    
    def _create_features(self, df):
        """Create lag features and moving averages for model prediction"""
        try:
            # Check if df is valid
            if df is None or df.empty:
                st.error("❌ DataFrame kosong dalam feature creation")
                return None
            
            # Create lag features (1-4 periods back)
            for i in range(1, 5):
                df[f'Lag_{i}'] = df['Indikator_Harga'].shift(i)
            
            # Create moving averages
            df['MA_3'] = df['Indikator_Harga'].rolling(window=3, min_periods=1).mean()
            df['MA_7'] = df['Indikator_Harga'].rolling(window=7, min_periods=1).mean()
            
            # Fill NaN values using multiple strategies
            # First, forward fill
            df = df.fillna(method='ffill')
            # Then backward fill for any remaining NaN
            df = df.fillna(method='bfill')
            # Finally, fill any remaining with 0
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            st.warning(f"⚠️ Error creating features: {str(e)}")
            
            # Return df with basic features filled with 0
            try:
                for col in ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'MA_3', 'MA_7']:
                    if col not in df.columns:
                        df[col] = 0
                return df
            except:
                return None
    
    def get_feature_columns(self):
        """Return the feature columns used by models"""
        return ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'MA_3', 'MA_7']
    
    def validate_data_structure(self, df):
        """Validate if data has required structure"""
        if df is None:
            st.error("❌ Data is None")
            return False
            
        if df.empty:
            st.error("❌ Data is empty")
            return False
            
        required_cols = ['Tanggal', 'Indikator_Harga']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Kolom yang diperlukan tidak ditemukan: {missing_cols}")
            return False
        
        return True
    
    def validate_data_quality(self, df):
        """Enhanced data quality validation"""
        if df is None or df.empty:
            return {
                'missing_values': {},
                'data_range': {'start_date': None, 'end_date': None, 'total_periods': 0},
                'iph_statistics': {'min': None, 'max': None, 'mean': None, 'std': None}
            }
            
        quality_report = {
            'missing_values': df.isnull().sum().to_dict(),
            'data_range': {
                'start_date': df['Tanggal'].min(),
                'end_date': df['Tanggal'].max(),
                'total_periods': len(df)
            },
            'iph_statistics': {
                'min': df['Indikator_Harga'].min(),
                'max': df['Indikator_Harga'].max(),
                'mean': df['Indikator_Harga'].mean(),
                'std': df['Indikator_Harga'].std()
            }
        }
        return quality_report