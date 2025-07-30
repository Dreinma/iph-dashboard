import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

class DataLoader:
    def __init__(self):
        self.data_path = Path("data")
        self.excel_file = "IPH_Kota_Batu.xlsx"
    
    @st.cache_data
    def load_excel_data(_self):
        """Load data from Excel file"""
        try:
            file_path = _self.data_path / _self.excel_file
            if file_path.exists():
                df = pd.read_excel(file_path)
                return _self._preprocess_data(df)
            else:
                st.error(f"File {_self.excel_file} tidak ditemukan di folder data/")
                return None
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
            return None
    
    def upload_data(self):
        """Handle file upload"""
        uploaded_file = st.file_uploader(
            "Upload File IPH Data",
            type=['csv', 'xlsx', 'xls'],
            help="Upload file CSV atau Excel dengan data IPH"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                return self._preprocess_data(df)
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")
                return None
        return None
    
    @st.cache_data
    def load_sample_data(_self):
        """Load sample data for demo"""
        # Create sample data based on your Excel structure
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2025-06-30', freq='W')
        n_samples = len(dates)
        
        # Generate realistic IPH data
        trend = np.linspace(-0.5, 0.5, n_samples)
        seasonal = 0.3 * np.sin(2 * np.pi * np.arange(n_samples) / 52)
        noise = np.random.normal(0, 0.8, n_samples)
        iph_values = trend + seasonal + noise
        
        # Create sample commodity data
        commodities = [
            'BERAS', 'CABAI RAWIT', 'CABAI MERAH', 'BAWANG MERAH', 
            'DAGING AYAM RAS', 'TELUR AYAM RAS', 'MINYAK GORENG', 'BAWANG PUTIH'
        ]
        
        sample_data = {
            'Tanggal': dates,
            'Bulan': [d.strftime('%B') for d in dates],
            'Minggu ke-': [f'M{((d.day-1)//7)+1}' for d in dates],
            'Kab/Kota': ['BATU'] * n_samples,
            'Indikator_Harga': iph_values,
            'Komoditas_Utama': np.random.choice(commodities, n_samples),
            'Fluktuasi_Tertinggi': np.random.choice(commodities, n_samples),
            'Nilai_Fluktuasi': np.random.uniform(0, 0.2, n_samples)
        }
        
        df = pd.DataFrame(sample_data)
        return _self._preprocess_data(df)
    
    def _preprocess_data(self, df):
        """Clean and preprocess data"""
        try:
            # Handle different column name variations
            iph_columns = [
                'Indikator Perubahan Harga (%)',
                'Indikator_Harga',
                'IPH',
                'Indikator Perubahan Harga'
            ]
            
            iph_col = None
            for col in iph_columns:
                if col in df.columns:
                    iph_col = col
                    break
            
            if iph_col is None:
                st.error("Kolom Indikator Perubahan Harga tidak ditemukan!")
                return None
            
            # Rename to standard column name
            df = df.rename(columns={iph_col: 'Indikator_Harga'})
            
            # Handle date column
            if 'Tanggal' not in df.columns:
                # Create date from Bulan and Minggu if available
                if 'Bulan' in df.columns and 'Minggu ke-' in df.columns:
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
            
            # Convert IPH to numeric
            df['Indikator_Harga'] = pd.to_numeric(df['Indikator_Harga'], errors='coerce')
            
            # Remove rows with missing critical data
            df = df.dropna(subset=['Tanggal', 'Indikator_Harga'])
            
            # Sort by date
            df = df.sort_values('Tanggal').reset_index(drop=True)
            
            # Create lag features for model prediction
            df = self._create_features(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error in data preprocessing: {str(e)}")
            return None
    
    def _create_features(self, df):
        """Create lag features and moving averages for model prediction"""
        try:
            # Create lag features
            for i in range(1, 5):  # Lag_1 to Lag_4
                df[f'Lag_{i}'] = df['Indikator_Harga'].shift(i)
            
            # Create moving averages
            df['MA_3'] = df['Indikator_Harga'].rolling(window=3, min_periods=1).mean()
            df['MA_7'] = df['Indikator_Harga'].rolling(window=7, min_periods=1).mean()
            
            # Fill NaN values with forward fill method
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            st.error(f"Error creating features: {str(e)}")
            return df
    
    def get_feature_columns(self):
        """Return the feature columns used by models"""
        return ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'MA_3', 'MA_7']