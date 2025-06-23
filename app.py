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
    plot_time_course, plot_pv_loops, calculate_time_course_data
)

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(page_title="Advanced Heart Simulator", layout="wide")
    st.title("🫀 Advanced Cardiac Function Simulator")
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

    with col2:
        # Render current status panel
        render_current_status(heart_model, params['drug_profile'], 
                             params['dose'], params['sim_duration'])

    # Pressure-Volume Loop Section
    pv_time_1, pv_time_2 = render_pv_loop_controls()
    
    # Create and display PV loop plot
    fig_pv = plot_pv_loops([pv_time_1, pv_time_2], params['dose'], 
                          params['drug_profile'], heart_model, params['sim_duration'])
    st.pyplot(fig_pv)

    # Drug information panel
    render_drug_info(params['drug_profile'])

    # Footer
    render_footer()

if __name__ == "__main__":
    main()
