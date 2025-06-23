"""
Streamlit UI components for the heart simulator
"""

import streamlit as st
from typing import Dict, Any

from models.drug_database import DRUG_DATABASE, DrugProfile
from models.heart_model import HeartModel
from models.pharmacology import pk_concentration, pd_effect
from models.cardiac_mechanics import calculate_hemodynamics

def render_sidebar() -> Dict[str, Any]:
    """
    Render the sidebar with drug selection and simulation settings
    
    Returns:
        Dictionary with selected parameters
    """
    # Drug selection
    st.sidebar.header("💊 Drug Administration")
    selected_drug = st.sidebar.selectbox("Select Drug", list(DRUG_DATABASE.keys()))
    drug_profile = DRUG_DATABASE[selected_drug]

    st.sidebar.markdown(f"**Mechanism:** {drug_profile.mechanism}")
    dose = st.sidebar.slider("Dose (mg)", 0.1, 50.0, 5.0, 0.1)
    st.sidebar.markdown(f"**Half-life:** {drug_profile.half_life} hours")

    # Simulation settings
    st.sidebar.header("⚙️ Simulation Settings")
    sim_duration = st.sidebar.slider("Duration (hours)", 1, 24, 8)
    patient_condition = st.sidebar.selectbox("Patient Condition", 
        ["Normal", "Heart Failure", "Hypertension", "Arrhythmia"])
    
    return {
        'selected_drug': selected_drug,
        'drug_profile': drug_profile,
        'dose': dose,
        'sim_duration': sim_duration,
        'patient_condition': patient_condition
    }

def render_current_status(heart_model: HeartModel, drug_profile: DrugProfile, 
                         dose: float, sim_duration: int):
    """
    Render the current status sidebar
    
    Args:
        heart_model: Heart model instance
        drug_profile: Drug profile
        dose: Drug dose
        sim_duration: Simulation duration
    """
    st.subheader("📊 Current Status")
    
    # Current time selector
    current_time = st.slider("Current Time (hours)", 0.0, float(sim_duration), 2.0, 0.1)
    
    # Calculate current values
    current_conc = pk_concentration(current_time, dose, drug_profile)
    current_hr = pd_effect(current_conc, heart_model.baseline_hr, drug_profile.emax_hr, 
                          drug_profile.ec50_hr, drug_profile.hill_coefficient)
    current_contractility = pd_effect(current_conc, heart_model.baseline_contractility, 
                                    drug_profile.emax_contractility, 
                                    drug_profile.ec50_contractility, 
                                    drug_profile.hill_coefficient)
    current_co = heart_model.predict_cardiac_output(current_conc, current_hr, current_contractility, current_time)
    
    current_afterload = pd_effect(current_conc, heart_model.baseline_afterload, 
                                drug_profile.emax_afterload/100, 
                                drug_profile.ec50_afterload, 
                                drug_profile.hill_coefficient)
    
    current_hemodynamics = calculate_hemodynamics(current_hr, current_contractility, current_afterload, 
                                                heart_model.baseline_preload, 
                                                heart_model.esv_base, heart_model.v0, 
                                                heart_model.e_min)
    
    # Display metrics with better formatting
    st.markdown("### 💊 Pharmacology")
    st.metric("Drug Concentration", f"{current_conc:.2f} mg/L")
    
    st.markdown("### ❤️ Cardiac Parameters")
    st.metric("Heart Rate", f"{current_hr:.0f} bpm", 
              f"{current_hr - heart_model.baseline_hr:+.0f}")
    st.metric("Contractility", f"{current_contractility:.2f} mmHg/mL", 
              f"{current_contractility - heart_model.baseline_contractility:+.2f}")
    
    st.markdown("### 🔄 Hemodynamics")
    st.metric("Cardiac Output", f"{current_co:.1f} L/min")
    st.metric("Stroke Volume", f"{current_hemodynamics['stroke_volume']:.0f} mL")
    st.metric("Ejection Fraction", f"{current_hemodynamics['ejection_fraction']:.1f}%")
    st.metric("Mean Arterial Pressure", f"{current_hemodynamics['mean_arterial_pressure']:.0f} mmHg")

def render_pv_loop_controls():
    """
    Render pressure-volume loop controls
    
    Returns:
        Tuple of selected time points
    """
    st.markdown("---")
    st.subheader("📈 Pressure-Volume Loop Analysis")
    st.markdown("Compare cardiac performance at different time points to visualize drug effects on ventricular function.")

    # Select time points for PV loop comparison
    col1, col2 = st.columns(2)
    with col1:
        pv_time_1 = st.selectbox("Time Point 1 (hours)", [0, 1, 2, 4, 6, 8], 0)
    with col2:
        pv_time_2 = st.selectbox("Time Point 2 (hours)", [0, 1, 2, 4, 6, 8], 4)
    
    return pv_time_1, pv_time_2

def render_drug_info(drug_profile: DrugProfile):
    """
    Render drug information panel
    
    Args:
        drug_profile: Drug profile to display
    """
    st.markdown("---")
    st.subheader("💊 Drug Information")
    st.markdown(f"**Current Selection:** {drug_profile.name}")

    drug_info_col1, drug_info_col2 = st.columns(2)

    with drug_info_col1:
        st.markdown("**📋 Basic Properties**")
        st.write(f"• **Mechanism:** {drug_profile.mechanism}")
        st.write(f"• **Volume of Distribution:** {drug_profile.vd} L")
        st.write(f"• **Half-life:** {drug_profile.half_life} hours")

    with drug_info_col2:
        st.markdown("**⚡ Pharmacodynamics**")
        st.write(f"• **EC50 (Contractility):** {drug_profile.ec50_contractility} mg/L")
        st.write(f"• **Max Effect (Contractility):** {drug_profile.emax_contractility}")
        st.write(f"• **EC50 (Heart Rate):** {drug_profile.ec50_hr} mg/L")
        st.write(f"• **Max Effect (Heart Rate):** {drug_profile.emax_hr} bpm")

def render_footer():
    """Render the footer"""
    st.markdown("---")
    st.markdown("*This simulator uses physiologically-based pharmacokinetic/pharmacodynamic models " + 
               "combined with machine learning for enhanced predictions. For educational purposes only.*")
