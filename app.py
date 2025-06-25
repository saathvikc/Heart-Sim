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

    # Model Internals Section with Modular Components
    st.markdown("---")
    st.subheader("🔬 Model Internals & Computational Details")
    st.markdown("This section shows what the model computed under the hood, organized by physiological system.")
    
    # Create tabbed interface for different model components
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Pharmacokinetics", 
        "Pharmacodynamics", 
        "Hemodynamics", 
        "Machine Learning", 
        "Autonomic System"
    ])
    
    with tab1:
        st.markdown("### Pharmacokinetic Model Internals")
        st.markdown("Shows drug absorption, distribution, metabolism, and elimination processes.")
        
        from utils.visualization import plot_pk_internals
        fig_pk = plot_pk_internals(time_points, params['dose'], params['drug_profile'])
        st.pyplot(fig_pk)
        
        st.markdown("""
        **Key Insights:**
        - **Compartment Models**: Compare simple 1-compartment vs. realistic 2-compartment kinetics
        - **Distribution**: How drug distributes between central and peripheral tissues
        - **Metabolism**: Michaelis-Menten kinetics showing saturable metabolism
        - **Saturation**: Degree of metabolic pathway saturation over time
        """)
    
    with tab2:
        st.markdown("### Pharmacodynamic Model Internals")
        st.markdown("Shows how drug concentration translates to physiological effects.")
        
        from utils.visualization import plot_pd_internals
        fig_pd = plot_pd_internals(time_points, params['dose'], params['drug_profile'], heart_model)
        st.pyplot(fig_pd)
        
        st.markdown("""
        **Key Insights:**
        - **Tolerance**: How drug effects diminish over time due to receptor desensitization
        - **Receptor Binding**: Percentage of target receptors occupied by drug
        - **Circadian Effects**: How time of day modulates drug response
        - **Hill Equation**: Dose-response relationship shape and steepness
        """)
    
    with tab3:
        st.markdown("### Hemodynamic Model Internals")
        st.markdown("Shows detailed cardiovascular mechanics and energetics.")
        
        from utils.visualization import plot_hemodynamic_internals
        fig_hemo = plot_hemodynamic_internals(time_points, params['dose'], params['drug_profile'], heart_model)
        st.pyplot(fig_hemo)
        
        st.markdown("""
        **Key Insights:**
        - **Frank-Starling**: How preload affects stroke volume
        - **Windkessel Model**: Systolic and diastolic pressure generation
        - **Oxygen Demand**: Myocardial energy requirements
        - **Efficiency**: Mechanical work vs. energy consumption ratio
        """)
    
    with tab4:
        st.markdown("### Machine Learning Model Internals")
        st.markdown("Shows how the ML ensemble makes predictions and model uncertainty.")
        
        from utils.visualization import plot_ml_internals
        fig_ml = plot_ml_internals(time_points, params['dose'], params['drug_profile'], heart_model)
        st.pyplot(fig_ml)
        
        st.markdown(f"""
        **Current ML Configuration:**
        - **Models**: {', '.join(heart_model.ml_models.keys())}
        - **Model Weights**: {', '.join([f'{k}: {v:.3f}' for k, v in heart_model.model_weights.items()])}
        - **Features**: 7 physiological and temporal features
        - **Training**: {heart_model.ml_models['rf'].n_estimators} trees in Random Forest
        
        **Key Insights:**
        - **Model Agreement**: How well different algorithms agree
        - **Ensemble Weighting**: Which models contribute most to final predictions
        - **Uncertainty**: Prediction variance indicates model confidence
        - **Feature Importance**: Which inputs most influence cardiac output predictions
        """)
    
    with tab5:
        st.markdown("### Autonomic System Model Internals")
        st.markdown("Shows baroreflex responses and respiratory-cardiovascular coupling.")
        
        from utils.visualization import plot_autonomic_internals
        fig_auto = plot_autonomic_internals(time_points, params['dose'], params['drug_profile'], heart_model)
        st.pyplot(fig_auto)
        
        st.markdown("""
        **Key Insights:**
        - **Baroreflex**: Automatic blood pressure regulation responses
        - **HR Adjustments**: How baroreceptors modulate heart rate
        - **Respiratory Coupling**: Heart rate variability due to breathing
        - **Combined Effects**: Integration of multiple autonomic influences
        """)
    
    # Technical Summary
    st.markdown("---")
    st.markdown("### 🔧 Technical Implementation Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Current Simulation Parameters:**
        - **Drug**: {params['drug_profile'].name} ({params['drug_profile'].mechanism})
        - **Dose**: {params['dose']} mg
        - **Patient**: {params['patient_condition']}
        - **Duration**: {params['sim_duration']} hours
        - **Time Points**: {len(time_points)} calculations
        """)
    
    with col2:
        st.markdown("""
        **Model Features:**
        - 2-compartment pharmacokinetics
        - Receptor binding kinetics  
        - Tolerance development
        - Ensemble machine learning
        - Baroreflex regulation
        - Respiratory coupling
        """)

    # Footer
    render_footer()

if __name__ == "__main__":
    main()
