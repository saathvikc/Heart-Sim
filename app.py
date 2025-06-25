"""
Advanced Cardiac Function Simulator
Main application file using refactored modules
"""

import streamlit as st
import numpy as np

# Import custom modules
from models.heart_model import HeartModel
from ui.components import (
    render_sidebar, render_current_status, render_pv_loop_controls, 
    render_drug_info, render_footer
)
from utils.visualization import (
    plot_time_course, plot_pv_loops, calculate_time_course_data, plot_model_internals
)

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(page_title="Advanced Heart Simulator", layout="wide")
    st.title("Cardiac Function Simulator")
    st.markdown("*Physiologically-based heart simulation with ML-enhanced drug modeling*")

    # Initialize heart model
    heart_model = HeartModel()

    # Render sidebar and get parameters
    params = render_sidebar()
    
    # Apply patient condition to heart model
    heart_model.apply_patient_condition(params['patient_condition'])

    # Main display area
    col1, col2 = st.columns([3, 1])  # Make the plots column wider

    with col1:
        # Time course simulation
        time_points = np.linspace(0, params['sim_duration'], 100)
        
        # Calculate physiological effects over time
        concentrations, hr_values, contractility_values, co_values, map_values = calculate_time_course_data(
            time_points, params['dose'], params['drug_profile'], heart_model
        )
        
        # Create and display time course plots
        fig = plot_time_course(time_points, concentrations, hr_values, 
                              contractility_values, co_values, heart_model)
        st.pyplot(fig)
        
        # Pressure-Volume Loop Section
        pv_time_1, pv_time_2 = render_pv_loop_controls()
        
        # Create and display PV loop plot
        fig_pv = plot_pv_loops([pv_time_1, pv_time_2], params['dose'], 
                            params['drug_profile'], heart_model, params['sim_duration'])
        st.pyplot(fig_pv)

    with col2:
        # Render current status panel
        render_current_status(heart_model, params['drug_profile'], 
                             params['dose'], params['sim_duration'])

    

    # Drug information panel
    render_drug_info(params['drug_profile'])

    # Model Internals Section
    st.markdown("---")
    st.subheader("🔬 Model Internals & Computational Details")
    st.markdown("This section shows what the model computed under the hood, including pharmacokinetics, " +
               "receptor binding, tolerance development, autonomic responses, and machine learning predictions.")
    
    # Create expandable section for model internals
    with st.expander("Show Detailed Model Computations", expanded=True):
        st.markdown("### Comprehensive Model Analysis")
        st.markdown("The following visualization shows all the computational steps and intermediate " +
                   "calculations performed by the enhanced physiological models:")
        
        # Generate model internals plot
        fig_internals = plot_model_internals(time_points, params['dose'], 
                                           params['drug_profile'], heart_model)
        st.pyplot(fig_internals)
        
        # Add detailed explanations
        st.markdown("#### Visualization Explanation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Pharmacokinetics (Row 1):**
            - **Simple vs. Complex PK**: Comparison of 1-compartment vs. 2-compartment models
            - **Receptor Occupancy**: Percentage of target receptors bound by drug
            - **Metabolic Rate**: Rate of drug metabolism (Michaelis-Menten kinetics)
            - **Metabolic Saturation**: How close the metabolism is to saturation
            
            **Pharmacodynamics (Row 2):**
            - **HR/Contractility Effects**: Basic Hill equation vs. tolerance-adjusted effects
            - **Tolerance Development**: How drug tolerance develops over time
            - **Circadian Modulation**: How time of day affects drug response
            """)
        
        with col2:
            st.markdown("""
            **Hemodynamics & Energetics (Row 3):**
            - **Stroke Volume**: Calculated using Frank-Starling mechanism
            - **Blood Pressure**: Windkessel model with systolic/diastolic separation
            - **MVO₂ Consumption**: Myocardial oxygen demand based on wall stress
            - **Mechanical Efficiency**: Ratio of external work to energy consumption
            
            **Autonomic & ML (Row 4):**
            - **Autonomic Response**: Baroreflex-mediated autonomic adjustments
            - **HR Adjustment**: Baroreceptor-driven heart rate changes
            - **ML Feature Importance**: Which features matter most for predictions
            - **Ensemble Weights**: How different ML models are weighted
            """)
        
        # Technical details section
        st.markdown("#### Technical Implementation Details")
        
        st.markdown(f"""
        **Current Model Configuration:**
        - **Drug**: {params['drug_profile'].name} ({params['drug_profile'].mechanism})
        - **Dose**: {params['dose']} mg
        - **Patient Condition**: {params['patient_condition']}
        - **ML Models**: {', '.join(heart_model.ml_models.keys())}
        - **Model Weights**: {', '.join([f'{k}: {v:.2f}' for k, v in heart_model.model_weights.items()])}  
        """)

    # Footer
    render_footer()

if __name__ == "__main__":
    main()
