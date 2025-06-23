# Heart Simulator Configuration
"""
Configuration file for the heart simulator application
"""

# Application settings
APP_TITLE = "Advanced Cardiac Function Simulator"
APP_DESCRIPTION = "Physiologically-based heart simulation with ML-enhanced drug modeling"

# Simulation defaults
DEFAULT_DOSE = 5.0
DEFAULT_DURATION = 8
DEFAULT_PATIENT_CONDITION = "Normal"

# Plot settings
PLOT_COLORS = {
    'concentration': '#1f77b4',
    'heart_rate': '#d62728',
    'contractility': '#2ca02c',
    'cardiac_output': '#ff7f0e'
}

# ML Model settings
ML_N_ESTIMATORS = 100
ML_RANDOM_STATE = 42
ML_N_SAMPLES = 1000
