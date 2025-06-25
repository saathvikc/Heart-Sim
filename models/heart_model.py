"""
Heart Model with Advanced Machine Learning Integration
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict

class HeartModel:
    """
    Advanced heart model with enhanced ML predictions and physiological modeling
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
        
        # Additional physiological parameters for enhanced modeling
        self.sympathetic_tone = 0.5  # 0-1 scale
        self.arterial_compliance = 1.5  # mL/mmHg
        self.baroreflex_sensitivity = 1.0
        
        # Initialize enhanced ML ensemble
        self._init_advanced_ml_model()
    
    def _init_advanced_ml_model(self):
        """Initialize sophisticated ML ensemble for complex physiological interactions"""
        np.random.seed(42)
        n_samples = 5000  # Larger dataset for better training
        
        # Enhanced features: [drug_conc, hr, contractility, time, sympathetic_tone, age, circadian_phase]
        X = self._generate_comprehensive_features(n_samples)
        
        # Multiple targets for ensemble learning
        y_co = self._synthetic_cardiac_output_advanced(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_co, test_size=0.2, random_state=42)
        
        # Feature scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Ensemble of different algorithms
        self.ml_models = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42),
            'nn': MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', 
                              solver='adam', max_iter=500, random_state=42)
        }
        
        # Train ensemble and calculate weights
        self.model_weights = {}
        self.model_scores = {}
        
        for name, model in self.ml_models.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            self.model_scores[name] = score
        
        # Calculate ensemble weights based on performance
        total_score = sum(self.model_scores.values())
        self.model_weights = {name: score/total_score for name, score in self.model_scores.items()}
    
    def _generate_comprehensive_features(self, n_samples: int) -> np.ndarray:
        """Generate comprehensive feature set with physiological interactions"""
        X = np.random.rand(n_samples, 7)
        
        # Scale to physiological ranges
        X[:, 0] *= 15  # drug_concentration (0-15 mg/L)
        X[:, 1] = X[:, 1] * 80 + 40  # hr (40-120 bpm)
        X[:, 2] = X[:, 2] * 2.5 + 0.5  # contractility (0.5-3.0)
        X[:, 3] *= 48  # time (0-48 hours)
        X[:, 4] *= 1.0  # sympathetic_tone (0-1)
        X[:, 5] = X[:, 5] * 60 + 20  # age (20-80 years)
        X[:, 6] *= 24  # circadian_phase (0-24 hours)
        
        return X
    
    def _synthetic_cardiac_output_advanced(self, X: np.ndarray) -> np.ndarray:
        """
        Generate advanced synthetic cardiac output with complex physiological interactions
        
        Args:
            X: Feature matrix [concentration, hr, contractility, time, sympathetic_tone, age, circadian]
            
        Returns:
            Cardiac output values with realistic physiological complexity
        """
        drug_conc, hr, contractility, time, symp_tone, age, circadian = X.T
        
        # Base stroke volume with Frank-Starling mechanism
        base_sv = 70 * (self.baseline_preload / 120) * (contractility / 1.5)
        
        # Drug effects with tolerance and receptor desensitization
        drug_effect = 1 + 0.4 * drug_conc / (drug_conc + 2)
        tolerance = np.exp(-time / 24)  # Tolerance develops over time
        drug_effect *= tolerance
        
        # Autonomic modulation
        autonomic_effect = 1 + 0.3 * symp_tone - 0.15 * (1 - symp_tone)
        
        # Age-related cardiac decline
        age_effect = 1 - 0.2 * ((age - 20) / 60)
        
        # Circadian rhythm effects (peak performance mid-day)
        circadian_effect = 1 + 0.1 * np.sin(2 * np.pi * (circadian - 6) / 24)
        
        # Baroreceptor feedback (simplified)
        baroreceptor_effect = 1 - 0.05 * np.tanh((hr - 70) / 20)
        
        # Calculate final stroke volume and cardiac output
        sv = base_sv * drug_effect * autonomic_effect * age_effect * baroreceptor_effect
        hr_effective = hr * circadian_effect
        co = sv * hr_effective / 1000  # L/min
        
        # Add physiological noise
        noise = np.random.normal(0, 0.3, len(co))
        co += noise
        
        return np.clip(co, 2, 15)  # Physiological range
    
    def predict_cardiac_output(self, drug_conc: float, hr: float, contractility: float, time_hrs: float, **kwargs) -> float:
        """
        Use enhanced ML ensemble to predict cardiac output with additional physiological factors
        
        Args:
            drug_conc: Drug concentration
            hr: Heart rate
            contractility: Contractility
            time_hrs: Time since dose in hours
            **kwargs: Additional parameters (sympathetic_tone, age, circadian_phase)
            
        Returns:
            Predicted cardiac output
        """
        # Extract additional parameters with defaults
        sympathetic_tone = kwargs.get('sympathetic_tone', self.sympathetic_tone)
        age = kwargs.get('age', 65)
        circadian_phase = kwargs.get('circadian_phase', 12)
        
        # Prepare features
        features = np.array([[drug_conc, hr, contractility, time_hrs, 
                            sympathetic_tone, age, circadian_phase]])
        features_scaled = self.scaler.transform(features)
        
        # Ensemble prediction with weighted average
        ensemble_prediction = 0
        for name, model in self.ml_models.items():
            prediction = model.predict(features_scaled)[0]
            ensemble_prediction += self.model_weights[name] * prediction
        
        return ensemble_prediction
    
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
