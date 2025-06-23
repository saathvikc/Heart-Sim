"""
Heart Model with Machine Learning Integration
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict

class HeartModel:
    """
    Comprehensive heart model with baseline parameters and ML predictions
    """
    
    def __init__(self):
        # Baseline physiological parameters
        self.baseline_hr = 70  # bpm
        self.baseline_contractility = 1.5  # mmHg/mL
        self.baseline_afterload = 1.0  # mmHg·s/mL
        self.baseline_preload = 120  # mL (EDV)
        self.esv_base = 50  # mL
        self.v0 = 10  # mL
        self.e_min = 0.06  # mmHg/mL
        
        # Initialize ML model for complex interactions
        self._init_ml_model()
    
    def _init_ml_model(self):
        """Initialize ML model for drug interaction predictions"""
        # Generate synthetic training data based on known physiology
        np.random.seed(42)
        n_samples = 1000
        
        # Features: [drug_concentration, baseline_hr, baseline_contractility, time_since_dose]
        X = np.random.rand(n_samples, 4) * [10, 40, 2, 24]  # Scale ranges
        X[:, 1] += 50  # HR range 50-90
        X[:, 2] += 0.8  # Contractility range 0.8-2.8
        
        # Target: cardiac output (simplified model)
        y = self._synthetic_cardiac_output(X)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.ml_model.fit(X_scaled, y)
    
    def _synthetic_cardiac_output(self, X: np.ndarray) -> np.ndarray:
        """
        Generate synthetic cardiac output data
        
        Args:
            X: Feature matrix [concentration, hr, contractility, time]
            
        Returns:
            Cardiac output values
        """
        conc, hr, contractility, time = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        
        # Simplified CO = SV * HR model with drug effects
        base_sv = 70  # mL
        drug_effect = 1 + 0.3 * conc / (conc + 2) - 0.1 * time / 24
        sv = base_sv * contractility * drug_effect / 1.5
        co = sv * hr / 1000  # L/min
        
        # Add some noise
        co += np.random.normal(0, 0.5, len(co))
        return np.clip(co, 2, 12)  # Physiological range
    
    def predict_cardiac_output(self, drug_conc: float, hr: float, contractility: float, time_hrs: float) -> float:
        """
        Use ML model to predict cardiac output
        
        Args:
            drug_conc: Drug concentration
            hr: Heart rate
            contractility: Contractility
            time_hrs: Time since dose in hours
            
        Returns:
            Predicted cardiac output
        """
        features = np.array([[drug_conc, hr, contractility, time_hrs]])
        features_scaled = self.scaler.transform(features)
        return self.ml_model.predict(features_scaled)[0]
    
    def apply_patient_condition(self, condition: str):
        """
        Adjust baseline parameters based on patient condition
        
        Args:
            condition: Patient condition string
        """
        if condition == "Heart Failure":
            self.baseline_contractility *= 0.6
            self.baseline_hr += 10
            self.baseline_preload += 20
        elif condition == "Hypertension":
            self.baseline_afterload *= 1.4
            self.baseline_hr += 5
        elif condition == "Arrhythmia":
            self.baseline_hr += 15
