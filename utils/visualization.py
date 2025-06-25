"""
Visualization utilities for the heart simulator
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import streamlit as st
from typing import List, Tuple, Dict

from models.heart_model import HeartModel
from models.drug_database import DrugProfile
from models.pharmacology import pk_concentration, pd_effect
from models.cardiac_mechanics import ventricular_volume, time_varying_elastance, calculate_hemodynamics
def plot_time_course(time_points: np.ndarray, concentrations: List[float], 
                    hr_values: List[float], contractility_values: List[float], 
                    co_values: List[float], heart_model: HeartModel) -> Figure:
    """
    Create time course plots for drug effects
    Create time course plots for drug effects
    
    Args:
        time_points: Time array
        concentrations: Drug concentrations over time
        hr_values: Heart rate values over time
        contractility_values: Contractility values over time
        co_values: Cardiac output values over time
        heart_model: Heart model instance
        
    Returns:
        Matplotlib figure
    """
    # Create subplots with better spacing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Drug concentration
    ax1.plot(time_points, concentrations, linewidth=3, color='#1f77b4')
    ax1.set_title('Drug Concentration vs Time', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Concentration (mg/L)', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # Heart rate
    ax2.plot(time_points, hr_values, linewidth=3, color='#d62728')
    ax2.axhline(y=heart_model.baseline_hr, color='#d62728', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
    ax2.set_title('Heart Rate vs Time', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Heart Rate (bpm)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    
    # Contractility
    ax3.plot(time_points, contractility_values, linewidth=3, color='#2ca02c')
    ax3.axhline(y=heart_model.baseline_contractility, color='#2ca02c', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
    ax3.set_title('Contractility vs Time', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Time (hours)', fontsize=12)
    ax3.set_ylabel('Contractility (mmHg/mL)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    
    # Cardiac output
    ax4.plot(time_points, co_values, linewidth=3, color='#ff7f0e')
    ax4.set_title('Cardiac Output vs Time (ML Prediction)', fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel('Time (hours)', fontsize=12)
    ax4.set_ylabel('Cardiac Output (L/min)', fontsize=12)
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax4.tick_params(axis='both', which='major', labelsize=10)
    
    # Improve spacing between subplots
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)
    
    return fig
def plot_pv_loops(pv_times: List[int], dose: float, drug_profile: DrugProfile, 
                 heart_model: HeartModel, sim_duration: int) -> Figure:
    """
    Create pressure-volume loop comparison plot
    Create pressure-volume loop comparison plot
    
    Args:
        pv_times: List of time points for comparison
        dose: Drug dose
        drug_profile: Drug profile
        heart_model: Heart model instance
        sim_duration: Simulation duration
        
    Returns:
        Matplotlib figure
    """
    fig_pv, ax_pv = plt.subplots(figsize=(14, 8))

    for i, pv_time in enumerate(pv_times):
        if pv_time <= sim_duration:
            # Calculate drug effects at this time
            C = pk_concentration(pv_time, dose, drug_profile)
            hr = pd_effect(C, heart_model.baseline_hr, drug_profile.emax_hr, 
                          drug_profile.ec50_hr, drug_profile.hill_coefficient)
            contractility = pd_effect(C, heart_model.baseline_contractility, 
                                    drug_profile.emax_contractility, 
                                    drug_profile.ec50_contractility, 
                                    drug_profile.hill_coefficient)
            
            # Generate PV loop
            T = 60 / hr  # Cardiac cycle duration in seconds
            t_cycle = np.linspace(0, T, 1000)
            
            # Calculate volume and pressure
            V_t = ventricular_volume(t_cycle, T, heart_model.baseline_preload, heart_model.esv_base)
            E_t = time_varying_elastance(t_cycle, T, contractility, heart_model.e_min)
            P_t = E_t * (V_t - heart_model.v0)
            
            colors = ['#1f77b4', '#d62728']  # Professional blue and red
            labels = [f"T={pv_time}h (HR={hr:.0f}bpm, E={contractility:.2f})", 
                     f"T={pv_time}h (HR={hr:.0f}bpm, E={contractility:.2f})"]
            
            ax_pv.plot(V_t, P_t, color=colors[i], linewidth=3, label=labels[i], alpha=0.8)

    ax_pv.set_title("Pressure-Volume Loops Comparison", fontsize=16, fontweight='bold', pad=20)
    ax_pv.set_xlabel("Volume (mL)", fontsize=14)
    ax_pv.set_ylabel("Pressure (mmHg)", fontsize=14)
    ax_pv.legend(fontsize=12, loc='upper right')
    ax_pv.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_pv.tick_params(axis='both', which='major', labelsize=12)

    # Add some styling
    ax_pv.spines['top'].set_visible(False)
    ax_pv.spines['right'].set_visible(False)
    ax_pv.spines['left'].set_linewidth(1.5)
    ax_pv.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout(pad=2.0)
    
    return fig_pv

def calculate_time_course_data(time_points: np.ndarray, dose: float, drug_profile: DrugProfile, 
                              heart_model: HeartModel) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    Calculate physiological effects over time
    
    Args:
        time_points: Time array
        dose: Drug dose
        drug_profile: Drug profile
        heart_model: Heart model instance
        
    Returns:
        Tuple of lists: (concentrations, hr_values, contractility_values, co_values, map_values)
    """
    concentrations = [pk_concentration(t, dose, drug_profile) for t in time_points]
    
    hr_values = []
    contractility_values = []
    co_values = []
    map_values = []
    
    for i, t in enumerate(time_points):
        C = concentrations[i]
        
        # Calculate drug effects
        hr = pd_effect(C, heart_model.baseline_hr, drug_profile.emax_hr, 
                      drug_profile.ec50_hr, drug_profile.hill_coefficient)
        contractility = pd_effect(C, heart_model.baseline_contractility, 
                                drug_profile.emax_contractility, 
                                drug_profile.ec50_contractility, 
                                drug_profile.hill_coefficient)
        
        # Use ML model for cardiac output prediction
        co = heart_model.predict_cardiac_output(C, hr, contractility, t)
        
        # Calculate hemodynamics
        afterload = pd_effect(C, heart_model.baseline_afterload, 
                            drug_profile.emax_afterload/100, 
                            drug_profile.ec50_afterload, 
                            drug_profile.hill_coefficient)
        
        hemodynamics = calculate_hemodynamics(hr, contractility, afterload, 
                                            heart_model.baseline_preload, 
                                            heart_model.esv_base, heart_model.v0, 
                                            heart_model.e_min)
        
        hr_values.append(hr)
        contractility_values.append(contractility)
        co_values.append(co)
        map_values.append(hemodynamics['mean_arterial_pressure'])
    
    return concentrations, hr_values, contractility_values, co_values, map_values
def plot_model_internals(time_points: np.ndarray, dose: float, drug_profile: DrugProfile,
                        heart_model: HeartModel) -> Figure:
    """
    Create comprehensive visualization of model internals and computational steps
    
    Args:
        time_points: Time array
        dose: Drug dose
        drug_profile: Drug profile
        heart_model: Heart model instance
        
    Returns:
        Matplotlib figure showing model internals
    """
    from models.pharmacology import (pk_two_compartment_model, pd_effect_with_tolerance, 
                                   receptor_binding_kinetics, drug_metabolism_kinetics)
    from models.cardiac_mechanics import (calculate_myocardial_oxygen_demand, baroreflex_response,
                                        respiratory_cardiovascular_coupling)
    
    # Calculate comprehensive model data
    pk_simple = [pk_concentration(t, dose, drug_profile) for t in time_points]
    pk_complex = [pk_two_compartment_model(t, dose, drug_profile) for t in time_points]
    receptor_data = [receptor_binding_kinetics(c) for c in pk_simple]
    metabolism_data = [drug_metabolism_kinetics(c) for c in pk_simple]
    
    # Calculate enhanced pharmacodynamics
    hr_basic = [pd_effect(c, heart_model.baseline_hr, drug_profile.emax_hr, 
                         drug_profile.ec50_hr, drug_profile.hill_coefficient) for c in pk_simple]
    hr_tolerance = [pd_effect_with_tolerance(c, heart_model.baseline_hr, drug_profile.emax_hr,
                                           drug_profile.ec50_hr, drug_profile.hill_coefficient, 
                                           t, 0.05) for c, t in zip(pk_simple, time_points)]
    
    contractility_basic = [pd_effect(c, heart_model.baseline_contractility, drug_profile.emax_contractility,
                                   drug_profile.ec50_contractility, drug_profile.hill_coefficient) for c in pk_simple]
    contractility_tolerance = [pd_effect_with_tolerance(c, heart_model.baseline_contractility, 
                                                      drug_profile.emax_contractility,
                                                      drug_profile.ec50_contractility, 
                                                      drug_profile.hill_coefficient, t, 0.03) 
                             for c, t in zip(pk_simple, time_points)]
    
    # Calculate hemodynamics and autonomic responses
    hemodynamics = []
    o2_demands = []
    baroreflex_responses = []
    
    for c, hr, contr, t in zip(pk_simple, hr_tolerance, contractility_tolerance, time_points):
        afterload = pd_effect(c, heart_model.baseline_afterload, drug_profile.emax_afterload/100,
                            drug_profile.ec50_afterload, drug_profile.hill_coefficient)
        
        hemo = calculate_hemodynamics(hr, contr, afterload, heart_model.baseline_preload,
                                    heart_model.esv_base, heart_model.v0, heart_model.e_min)
        hemodynamics.append(hemo)
        
        o2_demand = calculate_myocardial_oxygen_demand(hr, contr, afterload, hemo['stroke_volume'])
        o2_demands.append(o2_demand)
        
        baroreflex = baroreflex_response(hemo['mean_arterial_pressure'])
        baroreflex_responses.append(baroreflex)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    # Create grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    # Row 1: Pharmacokinetics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_points, pk_simple, 'b-', linewidth=2, label='Simple PK')
    ax1.plot(time_points, [pk['central_concentration'] for pk in pk_complex], 'r-', linewidth=2, label='Central')
    ax1.plot(time_points, [pk['peripheral_concentration'] for pk in pk_complex], 'g--', linewidth=2, label='Peripheral')
    ax1.set_title('Pharmacokinetics', fontweight='bold')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Concentration (mg/L)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_points, [r['occupancy_percentage'] for r in receptor_data], 'purple', linewidth=2)
    ax2.set_title('Receptor Occupancy', fontweight='bold')
    ax2.set_xlabel('Time (h)')
    ax2.set_ylabel('Occupancy (%)')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time_points, [m['metabolic_rate'] for m in metabolism_data], 'orange', linewidth=2)
    ax3.set_title('Metabolic Rate', fontweight='bold')
    ax3.set_xlabel('Time (h)')
    ax3.set_ylabel('Rate (mg/h)')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(time_points, [m['saturation_percentage'] for m in metabolism_data], 'red', linewidth=2)
    ax4.set_title('Metabolic Saturation', fontweight='bold')
    ax4.set_xlabel('Time (h)')
    ax4.set_ylabel('Saturation (%)')
    ax4.grid(True, alpha=0.3)
    
    # Row 2: Pharmacodynamics
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.plot(time_points, hr_basic, 'b-', linewidth=2, label='Basic PD')
    ax5.plot(time_points, hr_tolerance, 'r-', linewidth=2, label='With Tolerance')
    ax5.axhline(y=heart_model.baseline_hr, color='k', linestyle='--', alpha=0.5, label='Baseline')
    ax5.set_title('Heart Rate Effects', fontweight='bold')
    ax5.set_xlabel('Time (h)')
    ax5.set_ylabel('HR (bpm)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.plot(time_points, contractility_basic, 'b-', linewidth=2, label='Basic PD')
    ax6.plot(time_points, contractility_tolerance, 'r-', linewidth=2, label='With Tolerance')
    ax6.axhline(y=heart_model.baseline_contractility, color='k', linestyle='--', alpha=0.5, label='Baseline')
    ax6.set_title('Contractility Effects', fontweight='bold')
    ax6.set_xlabel('Time (h)')
    ax6.set_ylabel('Contractility')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[1, 2])
    tolerance_hr = []
    for hr_b, hr_t in zip(hr_basic, hr_tolerance):
        if abs(hr_b - heart_model.baseline_hr) > 0.1:
            tolerance_val = (hr_b - hr_t) / (hr_b - heart_model.baseline_hr) * 100
        else:
            tolerance_val = 0
        tolerance_hr.append(tolerance_val)
    ax7.plot(time_points, tolerance_hr, 'darkred', linewidth=2)
    ax7.set_title('HR Tolerance Development', fontweight='bold')
    ax7.set_xlabel('Time (h)')
    ax7.set_ylabel('Tolerance (%)')
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[1, 3])
    # Circadian effects
    circadian_effects = [1 + 0.1 * np.sin(2 * np.pi * (t - 6) / 24) for t in time_points]
    ax8.plot(time_points, circadian_effects, 'gold', linewidth=2)
    ax8.set_title('Circadian Modulation', fontweight='bold')
    ax8.set_xlabel('Time (h)')
    ax8.set_ylabel('Circadian Factor')
    ax8.grid(True, alpha=0.3)
    
    # Row 3: Hemodynamics & Energetics
    ax9 = fig.add_subplot(gs[2, 0])
    ax9.plot(time_points, [h['stroke_volume'] for h in hemodynamics], 'blue', linewidth=2)
    ax9.set_title('Stroke Volume', fontweight='bold')
    ax9.set_xlabel('Time (h)')
    ax9.set_ylabel('SV (mL)')
    ax9.grid(True, alpha=0.3)
    
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.plot(time_points, [h['systolic_pressure'] for h in hemodynamics], 'red', linewidth=2, label='Systolic')
    ax10.plot(time_points, [h['diastolic_pressure'] for h in hemodynamics], 'blue', linewidth=2, label='Diastolic')
    ax10.set_title('Blood Pressure', fontweight='bold')
    ax10.set_xlabel('Time (h)')
    ax10.set_ylabel('Pressure (mmHg)')
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)
    
    ax11 = fig.add_subplot(gs[2, 2])
    ax11.plot(time_points, [o['mvo2_total'] for o in o2_demands], 'green', linewidth=2)
    ax11.set_title('Myocardial O₂ Consumption', fontweight='bold')
    ax11.set_xlabel('Time (h)')
    ax11.set_ylabel('MVO₂ (mL/min/100g)')
    ax11.grid(True, alpha=0.3)
    
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.plot(time_points, [o['mechanical_efficiency'] for o in o2_demands], 'orange', linewidth=2)
    ax12.set_title('Mechanical Efficiency', fontweight='bold')
    ax12.set_xlabel('Time (h)')
    ax12.set_ylabel('Efficiency (%)')
    ax12.grid(True, alpha=0.3)
    
    # Row 4: Autonomic & ML
    ax13 = fig.add_subplot(gs[3, 0])
    ax13.plot(time_points, [b['autonomic_response'] for b in baroreflex_responses], 'darkblue', linewidth=2)
    ax13.set_title('Autonomic Response', fontweight='bold')
    ax13.set_xlabel('Time (h)')
    ax13.set_ylabel('Response')
    ax13.grid(True, alpha=0.3)
    
    ax14 = fig.add_subplot(gs[3, 1])
    ax14.plot(time_points, [b['hr_adjustment'] for b in baroreflex_responses], 'purple', linewidth=2)
    ax14.set_title('Baroreflex HR Adjustment', fontweight='bold')
    ax14.set_xlabel('Time (h)')
    ax14.set_ylabel('HR Adj (bpm)')
    ax14.grid(True, alpha=0.3)
    
    # ML model feature importance (simplified visualization)
    ax15 = fig.add_subplot(gs[3, 2])
    features = ['Drug Conc', 'HR', 'Contractility', 'Time', 'Symp Tone', 'Age', 'Circadian']
    importances = [0.25, 0.20, 0.18, 0.12, 0.10, 0.08, 0.07]  # Example importances
    ax15.barh(range(len(features)), importances, color='lightblue')
    ax15.set_title('ML Feature Importance', fontweight='bold')
    ax15.set_xlabel('Importance')
    ax15.set_ylabel('Features')
    ax15.set_yticks(range(len(features)))
    ax15.set_yticklabels(features, fontsize=8)
    ax15.grid(True, alpha=0.3)
    
    # ML ensemble weights
    ax16 = fig.add_subplot(gs[3, 3])
    models = list(heart_model.model_weights.keys())
    weights = list(heart_model.model_weights.values())
    colors = ['lightcoral', 'lightgreen', 'lightblue'][:len(models)]
    
    # Simple bar chart instead of pie chart
    ax16.bar(models, weights, color=colors)
    ax16.set_title('ML Ensemble Weights', fontweight='bold')
    ax16.set_xlabel('Models')
    ax16.set_ylabel('Weight')
    ax16.tick_params(axis='x', rotation=45)
    ax16.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(f'Model Internals: {drug_profile.name} ({dose}mg)', fontsize=16, fontweight='bold', y=0.98)
    
    return fig
