import pandas as pd
import streamlit as st
from datetime import datetime
import io

def format_metrics(value, metric_type='percentage'):
    """Format metrics for display"""
    if metric_type == 'percentage':
        return f"{value:+.2f}%"
    elif metric_type == 'decimal':
        return f"{value:.3f}"
    else:
        return str(value)

def check_alerts(iph_value, high_threshold=2.0, low_threshold=-2.0):
    """Check if IPH value triggers alerts"""
    if iph_value > high_threshold:
        return {
            'type': 'high',
            'message': f'IPH tinggi: {iph_value:+.2f}% (> +{high_threshold}%)',
            'color': 'red'
        }
    elif iph_value < low_threshold:
        return {
            'type': 'low', 
            'message': f'IPH rendah: {iph_value:+.2f}% (< {low_threshold}%)',
            'color': 'blue'
        }
    else:
        return {
            'type': 'normal',
            'message': f'IPH normal: {iph_value:+.2f}%',
            'color': 'green'
        }

def export_data(data, predictions=None, format_type='csv'):
    """Export data and predictions"""
    try:
        # Prepare export data
        export_df = data[['Tanggal', 'Indikator_Harga']].copy()
        export_df['Tipe'] = 'Historis'
        
        if predictions:
            # Add predictions to export
            pred_df = pd.DataFrame({
                'Tanggal': predictions['future_dates'],
                'Indikator_Harga': predictions['predictions'],
                'Tipe': 'Prediksi'
            })
            export_df = pd.concat([export_df, pred_df], ignore_index=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == 'csv':
            filename = f"IPH_Export_{timestamp}.csv"
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_buffer.getvalue(),
                file_name=filename,
                mime="text/csv"
            )
            
        elif format_type == 'excel':
            filename = f"IPH_Export_{timestamp}.xlsx"
            excel_buffer = io.BytesIO()
            
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='IPH Data', index=False)
                
                if predictions:
                    # Add summary sheet
                    summary_df = pd.DataFrame({
                        'Metric': ['Model', 'MAE', 'Prediction Periods', 'Export Date'],
                        'Value': [
                            predictions['model_name'],
                            f"{predictions['mae']:.3f}",
                            len(predictions['predictions']),
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ]
                    })
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            st.download_button(
                label="📥 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        st.success(f"✅ Data berhasil diekspor sebagai {format_type.upper()}")
        
    except Exception as e:
        st.error(f"❌ Error saat export: {str(e)}")

def create_summary_stats(data):
    """Create summary statistics"""
    stats = {
        'Total Data Points': len(data),
        'Rata-rata IPH': data['Indikator_Harga'].mean(),
        'Median IPH': data['Indikator_Harga'].median(),
        'Standar Deviasi': data['Indikator_Harga'].std(),
        'IPH Minimum': data['Indikator_Harga'].min(),
        'IPH Maksimum': data['Indikator_Harga'].max(),
        'Periode Data': f"{data['Tanggal'].min().strftime('%Y-%m-%d')} - {data['Tanggal'].max().strftime('%Y-%m-%d')}"
    }
    return stats

def validate_data(data):
    """Validate data quality"""
    issues = []
    
    # Check for missing values
    missing_iph = data['Indikator_Harga'].isna().sum()
    if missing_iph > 0:
        issues.append(f"⚠️ {missing_iph} nilai IPH yang hilang")
    
    # Check for outliers
    q1 = data['Indikator_Harga'].quantile(0.25)
    q3 = data['Indikator_Harga'].quantile(0.75)
    iqr = q3 - q1
    outliers = data[(data['Indikator_Harga'] < q1 - 1.5*iqr) | 
                   (data['Indikator_Harga'] > q3 + 1.5*iqr)]
    
    if len(outliers) > 0:
        issues.append(f"⚠️ {len(outliers)} outlier terdeteksi")
    
    # Check data continuity
    date_gaps = data['Tanggal'].diff().dt.days
    large_gaps = date_gaps[date_gaps > 14]  # More than 2 weeks
    if len(large_gaps) > 0:
        issues.append(f"⚠️ {len(large_gaps)} gap besar dalam data")
    
    return issues