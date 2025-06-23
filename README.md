# 🫀 Advanced Cardiac Function Simulator

A sophisticated heart simulation application that combines physiologically-based pharmacokinetic/pharmacodynamic models with machine learning for enhanced drug effect predictions.

## 🚀 Features

- **Comprehensive Drug Database**: 6 pre-configured cardiovascular drugs with realistic pharmacological profiles
- **Machine Learning Integration**: Random Forest model for complex drug interaction predictions
- **Advanced Physiological Models**: Suga-Sagawa elastance model for accurate cardiac mechanics
- **Patient Condition Simulation**: Normal, Heart Failure, Hypertension, and Arrhythmia conditions
- **Interactive Visualizations**: Real-time plots and pressure-volume loop comparisons
- **Professional UI**: Clean, medical-grade interface with organized metrics

## 📁 Project Structure

```
Heart-Sim/
├── app.py                     # Main application (clean, refactored)
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── models/                    # Core simulation models
│   ├── __init__.py
│   ├── drug_database.py       # Drug profiles and database
│   ├── pharmacology.py        # PK/PD models
│   ├── cardiac_mechanics.py   # Cardiovascular mechanics
│   └── heart_model.py         # Heart model with ML integration
├── ui/                        # User interface components
│   ├── __init__.py
│   └── components.py          # Streamlit UI components
└── utils/                     # Utility functions
    ├── __init__.py
    └── visualization.py       # Plotting and visualization utilities
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Heart-Sim
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 🎯 Usage

1. **Select Drug**: Choose from 6 cardiovascular drugs in the sidebar
2. **Set Dosage**: Adjust the drug dose using the slider
3. **Choose Patient Condition**: Select from Normal, Heart Failure, Hypertension, or Arrhythmia
4. **Analyze Results**: View real-time plots and metrics
5. **Compare PV Loops**: Select different time points to compare cardiac performance

## 💊 Available Drugs

- **Dobutamine** (β1-agonist) - Increases contractility
- **Epinephrine** (α/β-agonist) - Increases both contractility and heart rate
- **Propranolol** (β-blocker) - Decreases heart rate and contractility
- **Digoxin** (Na/K-ATPase inhibitor) - Positive inotropic effect
- **Milrinone** (PDE3 inhibitor) - Positive inotropic and vasodilator
- **Atenolol** (β1-selective blocker) - Selective heart rate reduction

## 🧠 Machine Learning

The application uses a Random Forest model trained on synthetic physiological data to predict cardiac output based on:
- Drug concentration
- Heart rate
- Contractility
- Time since dose administration

## 📊 Visualizations

- **Time Course Plots**: Drug concentration, heart rate, contractility, and cardiac output over time
- **Pressure-Volume Loops**: Comparison of cardiac performance at different time points
- **Real-time Metrics**: Live calculation of hemodynamic parameters

## 🏥 Educational Use

This simulator is designed for educational purposes to help understand:
- Cardiovascular pharmacology
- Drug-heart interactions
- Cardiac mechanics
- Hemodynamic monitoring

## ⚠️ Disclaimer

This simulator is for educational purposes only and should not be used for clinical decision-making.

## 🛠️ Development

### Adding New Drugs

1. Add drug profile to `models/drug_database.py`
2. Update the `DRUG_DATABASE` dictionary

### Modifying Models

- **Pharmacology**: Edit `models/pharmacology.py`
- **Cardiac Mechanics**: Edit `models/cardiac_mechanics.py`
- **ML Model**: Modify `models/heart_model.py`

### UI Changes

- **Components**: Edit `ui/components.py`
- **Visualizations**: Edit `utils/visualization.py`

## 📈 Performance

The refactored architecture provides:
- Better code organization and maintainability
- Improved modularity for testing and development
- Cleaner separation of concerns
- Easier extension and modification
