import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

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
        """Create main forecasting chart"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data['Tanggal'],
            y=historical_data['Indikator_Harga'],
            mode='lines+markers',
            name='Data Historis',
            line=dict(color=self.color_palette['primary'], width=2),
            marker=dict(size=6),
            hovertemplate='<b>Tanggal:</b> %{x}<br><b>IPH:</b> %{y:.2f}%<extra></extra>'
        ))
        
        # Predictions
        future_dates = predictions_result['future_dates']
        predictions = predictions_result['predictions']
        confidence_lower = predictions_result['confidence_lower']
        confidence_upper = predictions_result['confidence_upper']
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Prediksi',
            line=dict(color=self.color_palette['warning'], width=3, dash='dash'),
            marker=dict(size=8, color=self.color_palette['warning']),
            hovertemplate='<b>Tanggal:</b> %{x}<br><b>Prediksi:</b> %{y:.2f}%<extra></extra>'
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=confidence_upper,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=confidence_lower,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Rentang Kepercayaan (95%)',
            fillcolor='rgba(243, 156, 18, 0.2)',
            hovertemplate='<b>Rentang:</b> %{y:.2f}% - ' + 
                         f'{confidence_upper[0]:.2f}%<extra></extra>'
        ))
        
        # Add vertical line to separate historical and predicted data
        last_date = historical_data['Tanggal'].iloc[-1]
        fig.add_vline(
            x=last_date,
            line_dash="dot",
            line_color="gray",
            annotation_text="Prediksi mulai",
            annotation_position="top"
        )
        
        fig.update_layout(
            title=f'Forecasting IPH - Model: {model_name} (MAE: {predictions_result["mae"]:.3f})',
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
            paper_bgcolor='white'
        )
        
        # Add zero line for reference
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="IPH = 0%",
            annotation_position="right"
        )
        
        return fig
    
    def create_trend_chart(self, data):
        """Create trend analysis chart"""
        fig = go.Figure()
        
        # Main trend line
        fig.add_trace(go.Scatter(
            x=data['Tanggal'],
            y=data['Indikator_Harga'],
            mode='lines+markers',
            name='IPH',
            line=dict(color=self.color_palette['success'], width=3),
            marker=dict(size=6),
            fill='tonexty',
            fillcolor='rgba(39, 174, 96, 0.1)'
        ))
        
        # Add moving average
        if len(data) >= 4:
            ma_4 = data['Indikator_Harga'].rolling(window=4, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=data['Tanggal'],
                y=ma_4,
                mode='lines',
                name='MA(4)',
                line=dict(color=self.color_palette['info'], width=2, dash='dot'),
                opacity=0.8
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
    
    def create_commodity_contribution_chart(self, data):
        """Create commodity contribution chart"""
        try:
            # Extract commodity data from the 'Komoditas Andil Perubahan Harga' column if available
            commodity_col = None
            possible_cols = [
                'Komoditas Andil Perubahan Harga',
                'Komoditas_Utama',
                'Top_Komoditas'
            ]
            
            for col in possible_cols:
                if col in data.columns:
                    commodity_col = col
                    break
            
            if commodity_col is None:
                # Create dummy data
                commodities = ['BERAS', 'CABAI RAWIT', 'CABAI MERAH', 'BAWANG MERAH', 
                              'DAGING AYAM RAS', 'TELUR AYAM RAS', 'MINYAK GORENG']
                commodity_counts = pd.Series(np.random.randint(1, 20, len(commodities)), 
                                           index=commodities)
            else:
                commodity_counts = data[commodity_col].value_counts().head(8)
            
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
            # Return empty chart with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Data komoditas tidak tersedia<br>{str(e)[:50]}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=14
            )
            fig.update_layout(height=350, title='Top Komoditas Penyumbang')
            return fig
    
    def create_fluctuation_chart(self, data):
        """Create monthly fluctuation chart"""
        try:
            # Group by month for fluctuation analysis
            if 'Bulan' in data.columns:
                monthly_data = data.groupby('Bulan')['Indikator_Harga'].agg(['mean', 'std']).reset_index()
            else:
                # Create monthly grouping from date
                data_copy = data.copy()
                data_copy['Bulan'] = data_copy['Tanggal'].dt.strftime('%B')
                monthly_data = data_copy.groupby('Bulan')['Indikator_Harga'].agg(['mean', 'std']).reset_index()
            
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
                xaxis_title='Bulan',
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            # Return empty chart with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error membuat chart fluktuasi<br>{str(e)[:50]}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=14
            )
            fig.update_layout(height=350, title='Fluktuasi Harga Bulanan')
            return fig
    
    def create_high_fluctuation_chart(self, data):
        """Create high fluctuation commodities chart"""
        try:
            # Look for fluctuation data
            fluc_cols = ['Nilai_Fluktuasi', 'Fluktuasi Harga', 'Fluktuasi_Harga']
            commodity_cols = ['Fluktuasi_Tertinggi', 'Komoditas Fluktuasi Harga Tertinggi']
            
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
            else:
                # Create dummy data
                commodities = ['CABAI RAWIT', 'CABAI MERAH', 'BAWANG MERAH', 'BAWANG PUTIH',
                              'TELUR AYAM RAS', 'DAGING AYAM RAS', 'MINYAK GORENG', 'BERAS']
                fluc_data = pd.Series(np.random.uniform(0.02, 0.15, len(commodities)), 
                                    index=commodities)
            
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
            # Return empty chart with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Data fluktuasi tidak tersedia<br>{str(e)[:50]}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=14
            )
            fig.update_layout(height=350, title='Komoditas Fluktuasi Tertinggi')
            return fig
    
    def create_model_comparison_chart(self, comparison_data):
        """Create model comparison chart"""
        fig = go.Figure(data=[
            go.Bar(
                x=comparison_data['Model'],
                y=comparison_data['MAE'],
                marker_color=[
                    self.color_palette['success'] if status == '✅ Available' 
                    else self.color_palette['danger'] 
                    for status in comparison_data['Status']
                ],
                text=[f'{mae:.3f}' if mae is not None else 'Error' 
                      for mae in comparison_data['MAE']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>MAE: %{y:.3f}<br>%{text}<extra></extra>'
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