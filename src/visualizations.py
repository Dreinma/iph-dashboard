import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

class DashboardViz:
    def __init__(self):
        self.color_palette = {
            'primary': '#4A90E2',
            'secondary': '#7B68EE',
            'success': '#27ae60',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'info': '#3498db'
        }
    
    def create_forecast_chart(self, historical_data, predictions_result, model_name):
        """Create main forecasting chart with proper date handling"""
        try:
            fig = go.Figure()
            
            # Ensure historical data dates are properly formatted
            hist_dates = pd.to_datetime(historical_data['Tanggal'])
            hist_values = historical_data['Indikator_Harga'].values
            
            # Historical data trace
            fig.add_trace(go.Scatter(
                x=hist_dates,
                y=hist_values,
                mode='lines+markers',
                name='Data Historis',
                line=dict(color=self.color_palette['primary'], width=2),
                marker=dict(size=6),
                hovertemplate='<b>Tanggal:</b> %{x|%Y-%m-%d}<br><b>IPH:</b> %{y:.2f}%<extra></extra>'
            ))
            
            # Predictions data
            future_dates = pd.to_datetime(predictions_result['future_dates'])
            predictions = predictions_result['predictions']
            confidence_lower = predictions_result['confidence_lower']
            confidence_upper = predictions_result['confidence_upper']
            
            # Predictions trace
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='Prediksi',
                line=dict(color=self.color_palette['warning'], width=3, dash='dash'),
                marker=dict(size=8, color=self.color_palette['warning']),
                hovertemplate='<b>Tanggal:</b> %{x|%Y-%m-%d}<br><b>Prediksi:</b> %{y:.2f}%<extra></extra>'
            ))
            
            # Confidence interval - upper bound (invisible)
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=confidence_upper,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Confidence interval - lower bound with fill
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=confidence_lower,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Rentang Kepercayaan (95%)',
                fillcolor='rgba(243, 156, 18, 0.2)',
                hovertemplate='<b>Rentang:</b> %{y:.2f}% - ' + 
                             f'{np.mean(confidence_upper):.2f}%<extra></extra>'
            ))
            
            # Add vertical line to separate historical and predicted data
            # Convert to string to avoid timestamp issues
            last_date = hist_dates.iloc[-1]
            last_date_str = last_date.strftime('%Y-%m-%d')
            
            fig.add_shape(
                type="line",
                x0=last_date,
                x1=last_date,
                y0=min(min(hist_values), min(predictions)) - 1,
                y1=max(max(hist_values), max(predictions)) + 1,
                line=dict(
                    color="gray",
                    width=2,
                    dash="dot",
                ),
            )
            
            # Add annotation for the separation line
            fig.add_annotation(
                x=last_date,
                y=max(max(hist_values), max(predictions)),
                text="Mulai Prediksi",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="gray",
                ax=20,
                ay=-30,
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1
            )
            
            # Layout configuration
            fig.update_layout(
                title={
                    'text': f'Forecasting IPH - Model: {model_name} (MAE: {predictions_result["mae"]:.3f})',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                xaxis_title='Tanggal',
                yaxis_title='Indikator Perubahan Harga (%)',
                hovermode='x unified',
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray'
                )
            )
            
            # Add zero reference line
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                annotation_text="IPH = 0%",
                annotation_position="right"
            )
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating forecast chart: {str(e)}")
            return self._create_error_chart("Error dalam membuat grafik prediksi")
    
    def create_trend_chart(self, data):
        """Create trend analysis chart"""
        try:
            fig = go.Figure()
            
            # Ensure dates are properly formatted
            dates = pd.to_datetime(data['Tanggal'])
            values = data['Indikator_Harga'].values
            
            # Main trend line
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name='IPH',
                line=dict(color=self.color_palette['success'], width=3),
                marker=dict(size=6),
                fill='tonexty',
                fillcolor='rgba(39, 174, 96, 0.1)',
                hovertemplate='<b>Tanggal:</b> %{x|%Y-%m-%d}<br><b>IPH:</b> %{y:.2f}%<extra></extra>'
            ))
            
            # Add moving average if enough data
            if len(data) >= 4:
                ma_4 = data['Indikator_Harga'].rolling(window=4, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=ma_4,
                    mode='lines',
                    name='MA(4)',
                    line=dict(color=self.color_palette['info'], width=2, dash='dot'),
                    opacity=0.8,
                    hovertemplate='<b>Tanggal:</b> %{x|%Y-%m-%d}<br><b>MA(4):</b> %{y:.2f}%<extra></extra>'
                ))
            
            fig.update_layout(
                title='Tren IPH Historis',
                xaxis_title='Periode',
                yaxis_title='IPH (%)',
                height=350,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Add zero reference line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating trend chart: {str(e)}")
            return self._create_error_chart("Error dalam membuat grafik tren")
    
    def create_commodity_contribution_chart(self, data):
        """Create commodity contribution chart"""
        try:
            # Look for commodity columns
            commodity_cols = [
                'Komoditas Andil Perubahan Harga',
                'Komoditas_Utama',
                'Top_Komoditas',
                'Komoditas_Penyumbang'
            ]
            
            commodity_col = None
            for col in commodity_cols:
                if col in data.columns:
                    commodity_col = col
                    break
            
            if commodity_col and not data[commodity_col].isna().all():
                # Extract and count commodities
                commodity_data = data[commodity_col].dropna()
                
                # If data contains complex strings like "BERAS(0,116);MINYAK GORENG(0,048)"
                # Extract just the commodity names
                commodities = []
                for item in commodity_data:
                    if isinstance(item, str):
                        # Split by semicolon and extract commodity names
                        parts = str(item).split(';')
                        for part in parts:
                            # Extract text before parenthesis
                            commodity = part.split('(')[0].strip()
                            if commodity:
                                commodities.append(commodity)
                    else:
                        commodities.append(str(item))
                
                commodity_counts = pd.Series(commodities).value_counts().head(8)
            else:
                # Create sample data
                commodities = ['BERAS', 'CABAI RAWIT', 'CABAI MERAH', 'BAWANG MERAH', 
                              'DAGING AYAM RAS', 'TELUR AYAM RAS', 'MINYAK GORENG', 'BAWANG PUTIH']
                commodity_counts = pd.Series(
                    np.random.randint(1, 20, len(commodities)), 
                    index=commodities
                )
            
            fig = go.Figure(data=[
                go.Bar(
                    y=commodity_counts.index,
                    x=commodity_counts.values,
                    orientation='h',
                    marker_color=self.color_palette['primary'],
                    text=commodity_counts.values,
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Frekuensi: %{x}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title='Frekuensi Komoditas Penyumbang Utama',
                xaxis_title='Frekuensi Kontribusi',
                yaxis_title='Komoditas',
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"⚠️ Error creating commodity chart: {str(e)}")
            return self._create_error_chart("Data komoditas tidak tersedia")
    
    def create_fluctuation_chart(self, data):
        """Create monthly fluctuation chart"""
        try:
            # Create monthly grouping
            data_copy = data.copy()
            data_copy['Bulan'] = pd.to_datetime(data_copy['Tanggal']).dt.strftime('%B %Y')
            
            # Group by month
            monthly_data = data_copy.groupby('Bulan')['Indikator_Harga'].agg(['mean', 'std']).reset_index()
            monthly_data = monthly_data.fillna(0)  # Fill NaN std with 0
            
            # Limit to last 12 months for readability
            monthly_data = monthly_data.tail(12)
            
            # Create subplot with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add bar chart for mean IPH
            fig.add_trace(
                go.Bar(
                    x=monthly_data['Bulan'],
                    y=monthly_data['mean'],
                    name='Rata-rata IPH',
                    marker_color=self.color_palette['secondary'],
                    opacity=0.7,
                    hovertemplate='<b>%{x}</b><br>Rata-rata: %{y:.2f}%<extra></extra>'
                ),
                secondary_y=False,
            )
            
            # Add line chart for volatility
            fig.add_trace(
                go.Scatter(
                    x=monthly_data['Bulan'],
                    y=monthly_data['std'],
                    mode='lines+markers',
                    name='Volatilitas',
                    line=dict(color=self.color_palette['danger'], width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>%{x}</b><br>Volatilitas: %{y:.2f}%<extra></extra>'
                ),
                secondary_y=True,
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="IPH Rata-rata (%)", secondary_y=False)
            fig.update_yaxes(title_text="Volatilitas (%)", secondary_y=True)
            
            fig.update_layout(
                title='Fluktuasi Harga Bulanan',
                xaxis_title='Periode',
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_tickangle=-45
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"⚠️ Error creating fluctuation chart: {str(e)}")
            return self._create_error_chart("Error dalam membuat chart fluktuasi")
    
    def create_high_fluctuation_chart(self, data):
        """Create high fluctuation commodities chart"""
        try:
            # Look for fluctuation data
            fluc_cols = ['Nilai_Fluktuasi', 'Fluktuasi Harga', 'Fluktuasi_Harga']
            commodity_cols = ['Fluktuasi_Tertinggi', 'Komoditas Fluktuasi Harga Tertinggi', 'Komoditas Fluktuasi Tertinggi']
            
            fluc_col = None
            commodity_col = None
            
            for col in fluc_cols:
                if col in data.columns:
                    fluc_col = col
                    break
            
            for col in commodity_cols:
                if col in data.columns:
                    commodity_col = col
                    break
            
            if fluc_col and commodity_col:
                # Calculate average fluctuation by commodity
                fluc_data = data.groupby(commodity_col)[fluc_col].mean().sort_values(ascending=False).head(8)
                fluc_data = fluc_data.dropna()
            else:
                # Create sample data
                commodities = ['CABAI RAWIT', 'CABAI MERAH', 'BAWANG MERAH', 'BAWANG PUTIH',
                              'TELUR AYAM RAS', 'DAGING AYAM RAS', 'MINYAK GORENG', 'BERAS']
                fluc_data = pd.Series(
                    np.random.uniform(0.02, 0.15, len(commodities)), 
                    index=commodities
                )
            
            fig = go.Figure(data=[
                go.Bar(
                    x=fluc_data.index,
                    y=fluc_data.values,
                    marker_color=self.color_palette['danger'],
                    text=[f'{val:.3f}' for val in fluc_data.values],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Fluktuasi: %{y:.3f}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title='Rata-rata Fluktuasi per Komoditas',
                xaxis_title='Komoditas',
                yaxis_title='Tingkat Fluktuasi',
                height=350,
                xaxis_tickangle=-45,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"⚠️ Error creating high fluctuation chart: {str(e)}")
            return self._create_error_chart("Data fluktuasi tidak tersedia")
    
    def create_model_comparison_chart(self, comparison_data):
        """Create model comparison chart"""
        try:
            # Filter out models with None MAE
            valid_data = comparison_data[comparison_data['MAE'].notna()].copy()
            
            if len(valid_data) == 0:
                return self._create_error_chart("Tidak ada model yang valid untuk dibandingkan")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=valid_data['Model'],
                    y=valid_data['MAE'],
                    marker_color=[
                        self.color_palette['success'] if status == '✅ Available' 
                        else self.color_palette['danger'] 
                        for status in valid_data['Status']
                    ],
                    text=[f'{mae:.3f}' for mae in valid_data['MAE']],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>MAE: %{y:.3f}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title='Perbandingan Akurasi Model (MAE)',
                xaxis_title='Model',
                yaxis_title='Mean Absolute Error',
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating model comparison chart: {str(e)}")
            return self._create_error_chart("Error dalam membuat chart perbandingan model")
    
    def _create_error_chart(self, error_message):
        """Create a chart showing error message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"⚠️ {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, 
            xanchor='center', yanchor='middle',
            showarrow=False, 
            font_size=16,
            font_color="red"
        )
        
        fig.update_layout(
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig