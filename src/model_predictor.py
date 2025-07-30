import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelPredictor:
    def __init__(self):
        self.models_path = Path("models")
        self.models = {}
        self.feature_columns = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'MA_3', 'MA_7']
        self._load_models()
    
    def _load_models(self):
        """Load all available models with multiple loading methods"""
        model_files = {
            'KNN': 'KNN.pkl',
            'LightGBM': 'LightGBM.pkl',
            'Random Forest': 'Random_Forest.pkl',
            'XGBoost Advanced': 'XGBoost_Advanced.pkl'
        }
        
        loading_methods = [
            ('pickle', pickle.load),
            ('joblib', joblib.load),
        ]
        
        for name, filename in model_files.items():
            model_loaded = False
            model_path = self.models_path / filename
            
            if not model_path.exists():
                st.warning(f"⚠️ File model {filename} tidak ditemukan di folder models/")
                continue
            
            # Try different loading methods
            for method_name, load_func in loading_methods:
                try:
                    if method_name == 'pickle':
                        with open(model_path, 'rb') as f:
                            self.models[name] = load_func(f)
                    else:  # joblib
                        self.models[name] = load_func(model_path)
                    
                    # Test if model has predict method
                    if hasattr(self.models[name], 'predict'):
                        st.success(f"✅ Model {name} berhasil dimuat dengan {method_name}")
                        model_loaded = True
                        break
                    else:
                        st.warning(f"⚠️ Model {name} tidak memiliki method predict")
                        
                except Exception as e:
                    continue
            
            if not model_loaded:
                st.error(f"❌ Gagal memuat model {name}. Coba semua method loading.")
    
    def get_available_models(self):
        """Return list of available models"""
        available = list(self.models.keys())
        if not available:
            st.error("❌ Tidak ada model yang berhasil dimuat!")
            # Return dummy model for demo
            return ['Demo Model (Random)']
        return available
    
    def predict(self, model_name, data, periods=4, scenario='🔵 Normal'):
        """Make predictions using selected model"""
        try:
            # Handle demo mode
            if model_name == 'Demo Model (Random)':
                return self._demo_predict(periods, scenario)
            
            if model_name not in self.models:
                st.error(f"Model {model_name} tidak tersedia")
                return None
            
            model = self.models[model_name]
            
            # Validate data has required features
            missing_features = [col for col in self.feature_columns if col not in data.columns]
            if missing_features:
                st.error(f"Data tidak memiliki kolom: {missing_features}")
                return None
            
            # Prepare features from the latest data
            latest_data = data.tail(10).copy()
            
            # Get the most recent feature values
            try:
                current_features = latest_data[self.feature_columns].iloc[-1:].values
                
                # Check for NaN values
                if np.isnan(current_features).any():
                    st.warning("⚠️ Data mengandung NaN, menggunakan interpolasi...")
                    current_features = np.nan_to_num(current_features, nan=0.0)
                
            except Exception as e:
                st.error(f"Error preparing features: {str(e)}")
                return None
            
            # Make predictions for multiple periods
            predictions = []
            confidence_lower = []
            confidence_upper = []
            
            feature_input = current_features.copy()
            
            for period in range(periods):
                try:
                    # Make prediction
                    pred = model.predict(feature_input)[0]
                    
                    # Apply scenario adjustment
                    scenario_multiplier = self._get_scenario_multiplier(scenario)
                    pred_adjusted = pred * scenario_multiplier
                    
                    predictions.append(pred_adjusted)
                    
                    # Calculate confidence intervals
                    if model_name == 'Random Forest' and hasattr(model, 'estimators_'):
                        try:
                            tree_predictions = [tree.predict(feature_input)[0] for tree in model.estimators_[:10]]  # Use first 10 trees
                            pred_std = np.std(tree_predictions)
                            confidence_lower.append(pred_adjusted - 1.96 * pred_std)
                            confidence_upper.append(pred_adjusted + 1.96 * pred_std)
                        except:
                            confidence_lower.append(pred_adjusted - 0.5)
                            confidence_upper.append(pred_adjusted + 0.5)
                    else:
                        # Use fixed confidence interval for other models
                        confidence_lower.append(pred_adjusted - 0.5)
                        confidence_upper.append(pred_adjusted + 0.5)
                    
                    # Update feature_input for next prediction
                    new_features = feature_input[0].copy()
                    if len(new_features) >= 4:
                        new_features[1:4] = new_features[0:3]  # Shift lags
                        new_features[0] = pred_adjusted  # New Lag_1
                        
                        # Update moving averages
                        if len(new_features) >= 6:
                            new_features[4] = np.mean([pred_adjusted] + [new_features[i] for i in range(min(3, len(new_features)))])  # MA_3
                            new_features[5] = np.mean(predictions[-7:] if len(predictions) >= 7 else predictions)  # MA_7
                    
                    feature_input = new_features.reshape(1, -1)
                    
                except Exception as e:
                    st.error(f"Error in prediction loop: {str(e)}")
                    # Use last valid prediction or zero
                    pred_adjusted = predictions[-1] if predictions else 0.0
                    predictions.append(pred_adjusted)
                    confidence_lower.append(pred_adjusted - 0.5)
                    confidence_upper.append(pred_adjusted + 0.5)
            
            # Calculate model accuracy
            mae = self._calculate_model_accuracy(model, data)
            
            # Create future dates
            last_date = data['Tanggal'].iloc[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1),
                periods=periods,
                freq='W'
            )
            
            return {
                'predictions': predictions,
                'confidence_lower': confidence_lower,
                'confidence_upper': confidence_upper,
                'future_dates': future_dates,
                'mae': mae,
                'model_name': model_name,
                'scenario': scenario
            }
            
        except Exception as e:
            st.error(f"Error dalam prediksi: {str(e)}")
            return self._demo_predict(periods, scenario)
    
    def _demo_predict(self, periods, scenario):
        """Demo prediction when models fail to load"""
        np.random.seed(42)
        
        # Generate realistic predictions
        base_trend = np.random.normal(0, 1, periods)
        scenario_mult = self._get_scenario_multiplier(scenario)
        predictions = (base_trend * scenario_mult).tolist()
        
        confidence_lower = [p - 0.5 for p in predictions]
        confidence_upper = [p + 0.5 for p in predictions]
        
        future_dates = pd.date_range(
            start=pd.Timestamp.now(),
            periods=periods,
            freq='W'
        )
        
        return {
            'predictions': predictions,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'future_dates': future_dates,
            'mae': 0.85,  # Demo MAE
            'model_name': 'Demo Model',
            'scenario': scenario
        }
    
    def _get_scenario_multiplier(self, scenario):
        """Get multiplier based on scenario"""
        multipliers = {
            '🔵 Normal': 1.0,
            '🟢 Optimis': 1.1,
            '🔴 Pesimis': 0.9
        }
        return multipliers.get(scenario, 1.0)
    
    def _calculate_model_accuracy(self, model, data):
        """Calculate model accuracy on recent data"""
        try:
            recent_data = data.tail(20).copy()
            
            if len(recent_data) < 10:
                return 1.0
            
            train_size = len(recent_data) - 5
            train_data = recent_data.iloc[:train_size]
            test_data = recent_data.iloc[train_size:]
            
            X_test = test_data[self.feature_columns].values
            y_test = test_data['Indikator_Harga'].values
            
            # Handle NaN values
            if np.isnan(X_test).any():
                X_test = np.nan_to_num(X_test, nan=0.0)
            
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            return mae
            
        except Exception as e:
            return 1.0
    
    def compare_models(self, data):
        """Compare performance of all available models"""
        results = []
        
        for model_name in self.models.keys():
            try:
                model = self.models[model_name]
                mae = self._calculate_model_accuracy(model, data)
                results.append({
                    'Model': model_name,
                    'MAE': mae,
                    'Status': '✅ Available'
                })
            except Exception as e:
                results.append({
                    'Model': model_name,
                    'MAE': None,
                    'Status': f'❌ Error: {str(e)[:30]}'
                })
        
        return pd.DataFrame(results)
    def get_feature_importance(self, model_name):
        """Get feature importance for interpretability"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return None
        
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
