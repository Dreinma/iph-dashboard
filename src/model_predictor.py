import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

class ModelPredictor:
    def __init__(self):
        self.models_path = Path("models")
        self.models = {}
        self.feature_columns = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'MA_3', 'MA_7']
        self._load_models()
    
    def _load_models(self):
        """Load all available models"""
        model_files = {
            'KNN': 'KNN.pkl',
            'LightGBM': 'LightGBM.pkl',
            'Random Forest': 'Random_Forest.pkl',
            'XGBoost Advanced': 'XGBoost_Advanced.pkl'
        }
        
        for name, filename in model_files.items():
            try:
                model_path = self.models_path / filename
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    # st.success(f"✅ Model {name} berhasil dimuat")
                else:
                    st.warning(f"⚠️ File model {filename} tidak ditemukan")
            except Exception as e:
                st.error(f"❌ Error loading model {name}: {str(e)}")
    
    def get_available_models(self):
        """Return list of available models"""
        return list(self.models.keys())
    
    def predict(self, model_name, data, periods=4, scenario='🔵 Normal'):
        """Make predictions using selected model"""
        try:
            if model_name not in self.models:
                st.error(f"Model {model_name} tidak tersedia")
                return None
            
            model = self.models[model_name]
            
            # Prepare features from the latest data
            latest_data = data.tail(10).copy()  # Use last 10 points for context
            
            # Get the most recent feature values
            current_features = latest_data[self.feature_columns].iloc[-1:].values
            
            # Make predictions for multiple periods
            predictions = []
            confidence_lower = []
            confidence_upper = []
            
            # Use the current features as starting point
            feature_input = current_features.copy()
            
            for period in range(periods):
                # Make prediction
                if hasattr(model, 'predict'):
                    pred = model.predict(feature_input)[0]
                else:
                    st.error(f"Model {model_name} tidak memiliki method predict")
                    return None
                
                # Apply scenario adjustment
                scenario_multiplier = self._get_scenario_multiplier(scenario)
                pred_adjusted = pred * scenario_multiplier
                
                predictions.append(pred_adjusted)
                
                # Calculate confidence intervals based on model type
                if model_name == 'Random Forest':
                    # Use tree variance for confidence interval
                    try:
                        tree_predictions = [tree.predict(feature_input)[0] for tree in model.estimators_]
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
                
                # Update feature_input for next prediction (rolling forecast)
                # Shift the lag features and add the new prediction
                new_features = feature_input[0].copy()
                new_features[1:4] = new_features[0:3]  # Shift Lag_2, Lag_3, Lag_4
                new_features[0] = pred_adjusted  # New Lag_1
                
                # Update moving averages (simplified)
                new_features[4] = np.mean([pred_adjusted, new_features[0], new_features[1]])  # MA_3
                new_features[5] = np.mean(predictions[-7:] if len(predictions) >= 7 else predictions)  # MA_7
                
                feature_input = new_features.reshape(1, -1)
            
            # Calculate model accuracy (MAE) on recent data
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
            return None
    
    def _get_scenario_multiplier(self, scenario):
        """Get multiplier based on scenario"""
        multipliers = {
            '🔵 Normal': 1.0,
            '🟢 Optimis': 1.1,  # 10% more positive
            '🔴 Pesimis': 0.9   # 10% more negative
        }
        return multipliers.get(scenario, 1.0)
    
    def _calculate_model_accuracy(self, model, data):
        """Calculate model accuracy on recent data"""
        try:
            # Use last 20 points for accuracy calculation
            recent_data = data.tail(20).copy()
            
            if len(recent_data) < 10:
                return 1.0  # Default MAE if not enough data
            
            # Split into train and test
            train_size = len(recent_data) - 5
            train_data = recent_data.iloc[:train_size]
            test_data = recent_data.iloc[train_size:]
            
            # Prepare features and targets
            X_train = train_data[self.feature_columns].values
            y_train = train_data['Indikator_Harga'].values
            X_test = test_data[self.feature_columns].values
            y_test = test_data['Indikator_Harga'].values
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate MAE
            mae = mean_absolute_error(y_test, y_pred)
            return mae
            
        except Exception as e:
            # Return default MAE if calculation fails
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
                    'Status': f'❌ Error: {str(e)[:50]}'
                })
        
        return pd.DataFrame(results)