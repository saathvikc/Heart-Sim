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
    
    Args:
        time_points: Time array
        concentrations: Drug concentrations over time
        hr_values: Heart rate values over time
        contractility_values: Contractility values over time
        co_values: Cardiac output values over time (from best ML model)
        heart_model: Heart model instance
        
    Returns:
        Matplotlib figure
    """
    # Calculate ensemble predictions for comparison
    co_ensemble_values = []
    for i, t in enumerate(time_points):
        C = concentrations[i]
        hr = hr_values[i]
        contractility = contractility_values[i]
        co_ensemble = heart_model.predict_cardiac_output(C, hr, contractility, t, use_best_model=False)
        co_ensemble_values.append(co_ensemble)
    
    # Create subplots with better spacing
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Drug concentration
    ax1.plot(time_points, concentrations, linewidth=3, color='#1f77b4')
    ax1.set_title('Drug Concentration vs Time', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Concentration (mg/L)', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.text(0.5, -0.15, 'Shows how drug levels rise after dosing and then decline due to metabolism and elimination.',
             transform=ax1.transAxes, fontsize=9, ha='center', style='italic', alpha=0.7)
    
    # Heart rate
    ax2.plot(time_points, hr_values, linewidth=3, color='#d62728')
    ax2.axhline(y=heart_model.baseline_hr, color='#d62728', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
    ax2.set_title('Heart Rate vs Time', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Heart Rate (bpm)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.text(0.5, -0.15, 'Drug effects on heart rate through receptor binding and autonomic responses.',
             transform=ax2.transAxes, fontsize=9, ha='center', style='italic', alpha=0.7)
    
    # Contractility
    ax3.plot(time_points, contractility_values, linewidth=3, color='#2ca02c')
    ax3.axhline(y=heart_model.baseline_contractility, color='#2ca02c', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
    ax3.set_title('Contractility vs Time', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Time (hours)', fontsize=12)
    ax3.set_ylabel('Contractility (mmHg/mL)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.tick_params(axis='both', which='major', labelsize=10)
    ax3.text(0.5, -0.15, 'How strongly the heart muscle contracts, directly affecting stroke volume and cardiac output.',
             transform=ax3.transAxes, fontsize=9, ha='center', style='italic', alpha=0.7)
    
    # Cardiac output - compare best ML model vs ensemble vs equation-based
    ax4.plot(time_points, co_values, linewidth=3, color='#ff7f0e', label=f'Best Model ({heart_model.get_best_model_name().upper()})')
    ax4.plot(time_points, co_ensemble_values, linewidth=2, color='#9467bd', linestyle='-', alpha=0.8, label='ML Ensemble')
    
    # Calculate cardiac output from equations (not ML) for comparison
    co_eq_values = []
    for i, t in enumerate(time_points):
        # Use the same HR and contractility as above
        hr = hr_values[i]
        contractility = contractility_values[i]
        # Estimate afterload as baseline (or you can use drug effect if available)
        afterload = heart_model.baseline_afterload
        hemodynamics = calculate_hemodynamics(
            hr, contractility, afterload,
            heart_model.baseline_preload,
            heart_model.esv_base, heart_model.v0,
            heart_model.e_min
        )
        co_eq_values.append(hemodynamics['cardiac_output'])
    ax4.plot(time_points, co_eq_values, linewidth=2, color='#1f77b4', linestyle='--', label='Equation-based')
    
    # Add text annotation showing which ML model is being used
    best_model_name = heart_model.get_best_model_name()
    best_model_weight = heart_model.model_weights[best_model_name]
    ax4.text(0.02, 0.98, f'Best Model: {best_model_name.upper()} (weight: {best_model_weight:.3f})', 
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax4.set_title('Cardiac Output: ML Models vs Equations', fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel('Time (hours)', fontsize=12)
    ax4.set_ylabel('Cardiac Output (L/min)', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax4.tick_params(axis='both', which='major', labelsize=10)
    ax4.text(0.5, -0.15, 'Compares machine learning predictions with traditional physiological equations for cardiac output.',
             transform=ax4.transAxes, fontsize=9, ha='center', style='italic', alpha=0.7)
    
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
    co_ensemble_values = []  # Store ensemble predictions for comparison
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
        
        # Use best ML model for cardiac output prediction
        co = heart_model.predict_cardiac_output(C, hr, contractility, t, use_best_model=True)
        co_ensemble = heart_model.predict_cardiac_output(C, hr, contractility, t, use_best_model=False)
        
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
        co_ensemble_values.append(co_ensemble)
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

# =============================================================================
# PHASE 1: MODULAR VISUALIZATION COMPONENTS
# =============================================================================

def plot_pk_internals(time_points: np.ndarray, dose: float, drug_profile: DrugProfile) -> Figure:
    """
    Create detailed pharmacokinetic internals visualization
    
    Args:
        time_points: Time array
        dose: Drug dose
        drug_profile: Drug profile
        
    Returns:
        Matplotlib figure showing PK internals
    """
    from models.pharmacology import pk_two_compartment_model, drug_metabolism_kinetics
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Pharmacokinetic Model Internals', fontsize=16, fontweight='bold')
    
    # Calculate PK data
    simple_conc = [pk_concentration(t, dose, drug_profile) for t in time_points]
    complex_pk = [pk_two_compartment_model(t, dose, drug_profile) for t in time_points]
    central_conc = [pk['central_concentration'] for pk in complex_pk]
    peripheral_conc = [pk['peripheral_concentration'] for pk in complex_pk]
    
    # 1. Compartment comparison
    ax1.plot(time_points, simple_conc, label='1-Compartment', linewidth=3, color='blue')
    ax1.plot(time_points, central_conc, label='Central (2-Comp)', linewidth=2, color='red', linestyle='--')
    ax1.plot(time_points, peripheral_conc, label='Peripheral (2-Comp)', linewidth=2, color='green', linestyle=':')
    ax1.set_title('Compartment Model Comparison')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Concentration (mg/L)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.5, -0.15, 'Compares simple 1-compartment vs. realistic 2-compartment drug distribution models.',
             transform=ax1.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 2. Distribution ratio over time
    distribution_ratios = [pk['distribution_ratio'] for pk in complex_pk]
    ax2.plot(time_points, distribution_ratios, linewidth=3, color='purple')
    ax2.set_title('Peripheral/Central Distribution Ratio')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Distribution Ratio')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, -0.15, 'Shows how drug distributes between bloodstream and peripheral tissues over time.',
             transform=ax2.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 3. Metabolism kinetics
    metabolism_data = [drug_metabolism_kinetics(conc) for conc in simple_conc]
    metabolic_rates = [m['metabolic_rate'] for m in metabolism_data]
    ax3.plot(time_points, metabolic_rates, linewidth=3, color='orange')
    ax3.set_title('Metabolic Rate (Michaelis-Menten)')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Metabolic Rate')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.5, -0.15, 'Demonstrates saturable enzyme kinetics where metabolism rate plateaus at high concentrations.',
             transform=ax3.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 4. Saturation percentage
    saturation_pct = [m['saturation_percentage'] for m in metabolism_data]
    ax4.plot(time_points, saturation_pct, linewidth=3, color='brown')
    ax4.set_title('Metabolic Saturation')
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Saturation (%)')
    ax4.grid(True, alpha=0.3)
    ax4.text(0.5, -0.15, 'Percentage of metabolic enzymes saturated, affecting drug clearance efficiency.',
             transform=ax4.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_pd_internals(time_points: np.ndarray, dose: float, drug_profile: DrugProfile, 
                     heart_model: HeartModel) -> Figure:
    """
    Create detailed pharmacodynamic internals visualization
    
    Args:
        time_points: Time array
        dose: Drug dose
        drug_profile: Drug profile
        heart_model: Heart model instance
        
    Returns:
        Matplotlib figure showing PD internals
    """
    from models.pharmacology import pd_effect_with_tolerance, receptor_binding_kinetics
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Pharmacodynamic Model Internals', fontsize=16, fontweight='bold')
    
    # Calculate concentrations
    concentrations = [pk_concentration(t, dose, drug_profile) for t in time_points]
    
    # 1. Tolerance development for HR
    hr_no_tolerance = [pd_effect(c, heart_model.baseline_hr, drug_profile.emax_hr,
                                drug_profile.ec50_hr, drug_profile.hill_coefficient) 
                      for c in concentrations]
    hr_with_tolerance = [pd_effect_with_tolerance(c, heart_model.baseline_hr, drug_profile.emax_hr,
                                                 drug_profile.ec50_hr, drug_profile.hill_coefficient, 
                                                 t, 0.05) 
                        for c, t in zip(concentrations, time_points)]
    
    ax1.plot(time_points, hr_no_tolerance, label='No Tolerance', linewidth=3, color='red')
    ax1.plot(time_points, hr_with_tolerance, label='With Tolerance', linewidth=3, color='darkred', linestyle='--')
    ax1.axhline(y=heart_model.baseline_hr, color='gray', linestyle=':', alpha=0.7, label='Baseline')
    ax1.set_title('Heart Rate: Tolerance Development')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Heart Rate (bpm)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.5, -0.15, 'Shows how drug effects diminish over time due to receptor desensitization.',
             transform=ax1.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 2. Receptor occupancy
    receptor_data = [receptor_binding_kinetics(c) for c in concentrations]
    occupancy_pct = [r['occupancy_percentage'] for r in receptor_data]
    ax2.plot(time_points, occupancy_pct, linewidth=3, color='purple')
    ax2.set_title('Receptor Occupancy')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Occupancy (%)')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, -0.15, 'Percentage of target receptors bound by drug molecules at each time point.',
             transform=ax2.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 3. Circadian modulation
    circadian_effect = [1 + 0.1 * np.sin(2 * np.pi * (t - 6) / 24) for t in time_points]
    ax3.plot(time_points, circadian_effect, linewidth=3, color='gold')
    ax3.set_title('Circadian Rhythm Modulation')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Circadian Factor')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.5, -0.15, 'Natural daily rhythm effects on drug sensitivity and cardiovascular responses.',
             transform=ax3.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 4. Hill equation components
    hill_components = [(c**drug_profile.hill_coefficient) / 
                      (c**drug_profile.hill_coefficient + drug_profile.ec50_hr**drug_profile.hill_coefficient)
                      for c in concentrations]
    ax4.plot(time_points, hill_components, linewidth=3, color='green')
    ax4.set_title('Hill Equation Response')
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Fractional Response')
    ax4.grid(True, alpha=0.3)
    ax4.text(0.5, -0.15, 'Sigmoidal dose-response relationship showing receptor binding cooperativity.',
             transform=ax4.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_hemodynamic_internals(time_points: np.ndarray, dose: float, drug_profile: DrugProfile,
                              heart_model: HeartModel) -> Figure:
    """
    Create detailed hemodynamic internals visualization
    
    Args:
        time_points: Time array
        dose: Drug dose
        drug_profile: Drug profile
        heart_model: Heart model instance
        
    Returns:
        Matplotlib figure showing hemodynamic internals
    """
    from models.cardiac_mechanics import calculate_myocardial_oxygen_demand, baroreflex_response
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hemodynamic Model Internals', fontsize=16, fontweight='bold')
    
    # Calculate time course data
    concentrations = [pk_concentration(t, dose, drug_profile) for t in time_points]
    hr_values = [pd_effect(c, heart_model.baseline_hr, drug_profile.emax_hr,
                          drug_profile.ec50_hr, drug_profile.hill_coefficient) 
                for c in concentrations]
    contractility_values = [pd_effect(c, heart_model.baseline_contractility,
                                    drug_profile.emax_contractility,
                                    drug_profile.ec50_contractility,
                                    drug_profile.hill_coefficient) 
                           for c in concentrations]
    
    # Calculate hemodynamics for each time point
    hemodynamics_data = []
    for hr, contractility in zip(hr_values, contractility_values):
        afterload = pd_effect(concentrations[0], heart_model.baseline_afterload,
                             drug_profile.emax_afterload/100, drug_profile.ec50_afterload,
                             drug_profile.hill_coefficient)
        hemo = calculate_hemodynamics(hr, contractility, afterload, heart_model.baseline_preload,
                                    heart_model.esv_base, heart_model.v0, heart_model.e_min)
        hemodynamics_data.append(hemo)
    
    # 1. Stroke volume components
    stroke_volumes = [h['stroke_volume'] for h in hemodynamics_data]
    ax1.plot(time_points, stroke_volumes, linewidth=3, color='blue')
    ax1.set_title('Stroke Volume (Frank-Starling)')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Stroke Volume (mL)')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.5, -0.15, 'Volume of blood ejected per heartbeat, governed by Frank-Starling mechanism.',
             transform=ax1.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 2. Pressure components
    systolic_pressures = [h['systolic_pressure'] for h in hemodynamics_data]
    diastolic_pressures = [h['diastolic_pressure'] for h in hemodynamics_data]
    ax2.plot(time_points, systolic_pressures, label='Systolic', linewidth=3, color='red')
    ax2.plot(time_points, diastolic_pressures, label='Diastolic', linewidth=3, color='darkred')
    ax2.set_title('Blood Pressure Components')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Pressure (mmHg)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, -0.15, 'Peak contraction and relaxation pressures calculated from Windkessel model.',
             transform=ax2.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 3. Myocardial oxygen demand
    o2_data = [calculate_myocardial_oxygen_demand(hr, contractility, 1.0, sv)
               for hr, contractility, sv in zip(hr_values, contractility_values, stroke_volumes)]
    mvo2_values = [o2['mvo2_total'] for o2 in o2_data]
    efficiency_values = [o2['mechanical_efficiency'] for o2 in o2_data]
    
    ax3.plot(time_points, mvo2_values, linewidth=3, color='orange')
    ax3.set_title('Myocardial O₂ Consumption')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('MVO₂ (mL O₂/min/100g)')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.5, -0.15, 'Heart muscle oxygen demand based on workload and metabolic requirements.',
             transform=ax3.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 4. Mechanical efficiency
    ax4.plot(time_points, efficiency_values, linewidth=3, color='green')
    ax4.set_title('Mechanical Efficiency')
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Efficiency (%)')
    ax4.grid(True, alpha=0.3)
    ax4.text(0.5, -0.15, 'Ratio of mechanical work output to total energy consumption by the heart.',
             transform=ax4.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_ml_internals(time_points: np.ndarray, dose: float, drug_profile: DrugProfile,
                     heart_model: HeartModel) -> Figure:
    """
    Create detailed machine learning internals visualization
    
    Args:
        time_points: Time array
        dose: Drug dose
        drug_profile: Drug profile
        heart_model: Heart model instance
        
    Returns:
        Matplotlib figure showing ML internals
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Machine Learning Model Internals', fontsize=16, fontweight='bold')
    
    # Calculate predictions from individual models
    concentrations = [pk_concentration(t, dose, drug_profile) for t in time_points]
    hr_values = [pd_effect(c, heart_model.baseline_hr, drug_profile.emax_hr,
                          drug_profile.ec50_hr, drug_profile.hill_coefficient) 
                for c in concentrations]
    contractility_values = [pd_effect(c, heart_model.baseline_contractility,
                                    drug_profile.emax_contractility,
                                    drug_profile.ec50_contractility,
                                    drug_profile.hill_coefficient) 
                           for c in concentrations]
    
    # 1. Individual model predictions
    rf_predictions = []
    gb_predictions = []
    nn_predictions = []
    ensemble_predictions = []
    
    for i, t in enumerate(time_points):
        # Prepare features
        features = np.array([[concentrations[i], hr_values[i], contractility_values[i], t, 
                            heart_model.sympathetic_tone, 65, t % 24]])
        features_scaled = heart_model.scaler.transform(features)
        
        # Individual predictions
        rf_pred = heart_model.ml_models['rf'].predict(features_scaled)[0]
        gb_pred = heart_model.ml_models['gb'].predict(features_scaled)[0]
        nn_pred = heart_model.ml_models['nn'].predict(features_scaled)[0]
        
        rf_predictions.append(rf_pred)
        gb_predictions.append(gb_pred)
        nn_predictions.append(nn_pred)
        
        # Ensemble prediction
        ensemble_pred = (heart_model.model_weights['rf'] * rf_pred +
                        heart_model.model_weights['gb'] * gb_pred +
                        heart_model.model_weights['nn'] * nn_pred)
        ensemble_predictions.append(ensemble_pred)
    
    ax1.plot(time_points, rf_predictions, label='Random Forest', linewidth=2, color='blue')
    ax1.plot(time_points, gb_predictions, label='Gradient Boosting', linewidth=2, color='red')
    ax1.plot(time_points, nn_predictions, label='Neural Network', linewidth=2, color='green')
    ax1.plot(time_points, ensemble_predictions, label='Ensemble', linewidth=3, color='black')
    ax1.set_title('Individual Model Predictions')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Cardiac Output (L/min)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.5, -0.15, 'Compares predictions from different ML algorithms and their weighted ensemble.',
             transform=ax1.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 2. Model weights
    models = list(heart_model.model_weights.keys())
    weights = list(heart_model.model_weights.values())
    colors = ['blue', 'red', 'green']
    
    ax2.bar(models, weights, color=colors)
    ax2.set_title('Ensemble Model Weights')
    ax2.set_ylabel('Weight')
    ax2.set_ylim(0, 1)
    for i, v in enumerate(weights):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    ax2.text(0.5, -0.15, 'Performance-based weights determining each model\'s contribution to ensemble.',
             transform=ax2.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 3. Prediction uncertainty (variance between models)
    uncertainties = []
    for i in range(len(time_points)):
        preds = [rf_predictions[i], gb_predictions[i], nn_predictions[i]]
        uncertainty = np.std(preds)
        uncertainties.append(uncertainty)
    
    ax3.plot(time_points, uncertainties, linewidth=3, color='purple')
    ax3.set_title('Prediction Uncertainty')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Standard Deviation')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.5, -0.15, 'Disagreement between models indicating prediction confidence levels.',
             transform=ax3.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 4. Feature importance (simplified visualization)
    feature_names = ['Drug Conc', 'HR', 'Contractility', 'Time', 'Symp Tone', 'Age', 'Circadian']
    # Use Random Forest feature importance as example
    rf_importance = heart_model.ml_models['rf'].feature_importances_
    
    ax4.barh(feature_names, rf_importance, color='lightblue')
    ax4.set_title('Feature Importance (Random Forest)')
    ax4.set_xlabel('Importance')
    ax4.text(0.5, -0.15, 'Relative importance of input features in predicting cardiac output.',
             transform=ax4.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    plt.tight_layout()
    return fig

def plot_autonomic_internals(time_points: np.ndarray, dose: float, drug_profile: DrugProfile,
                           heart_model: HeartModel) -> Figure:
    """
    Create detailed autonomic system internals visualization
    
    Args:
        time_points: Time array
        dose: Drug dose
        drug_profile: Drug profile
        heart_model: Heart model instance
        
    Returns:
        Matplotlib figure showing autonomic internals
    """
    from models.cardiac_mechanics import baroreflex_response, respiratory_cardiovascular_coupling
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Autonomic System Model Internals', fontsize=16, fontweight='bold')
    
    # Calculate hemodynamic data
    concentrations = [pk_concentration(t, dose, drug_profile) for t in time_points]
    hr_values = [pd_effect(c, heart_model.baseline_hr, drug_profile.emax_hr,
                          drug_profile.ec50_hr, drug_profile.hill_coefficient) 
                for c in concentrations]
    contractility_values = [pd_effect(c, heart_model.baseline_contractility,
                                    drug_profile.emax_contractility,
                                    drug_profile.ec50_contractility,
                                    drug_profile.hill_coefficient) 
                           for c in concentrations]
    
    # Calculate mean arterial pressures
    map_values = []
    for hr, contractility in zip(hr_values, contractility_values):
        afterload = pd_effect(concentrations[0], heart_model.baseline_afterload,
                             drug_profile.emax_afterload/100, drug_profile.ec50_afterload,
                             drug_profile.hill_coefficient)
        hemo = calculate_hemodynamics(hr, contractility, afterload, heart_model.baseline_preload,
                                    heart_model.esv_base, heart_model.v0, heart_model.e_min)
        map_values.append(hemo['mean_arterial_pressure'])
    
    # 1. Baroreflex response
    baroreflex_data = [baroreflex_response(map_val) for map_val in map_values]
    autonomic_responses = [b['autonomic_response'] for b in baroreflex_data]
    ax1.plot(time_points, autonomic_responses, linewidth=3, color='blue')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_title('Baroreflex Autonomic Response')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Autonomic Response')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.5, -0.15, 'Automatic blood pressure regulation through baroreceptor feedback.',
             transform=ax1.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 2. Heart rate adjustments
    hr_adjustments = [b['hr_adjustment'] for b in baroreflex_data]
    ax2.plot(time_points, hr_adjustments, linewidth=3, color='red')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_title('Baroreflex HR Adjustment')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('HR Adjustment (bpm)')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, -0.15, 'Heart rate changes triggered by baroreceptor responses to pressure changes.',
             transform=ax2.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 3. Respiratory coupling
    respiratory_phases = [(t * 0.25) % 1 for t in time_points]  # 15 breaths/min
    respiratory_data = [respiratory_cardiovascular_coupling(phase) for phase in respiratory_phases]
    hr_modulations = [r['hr_modulation'] for r in respiratory_data]
    ax3.plot(time_points, hr_modulations, linewidth=3, color='green')
    ax3.set_title('Respiratory Sinus Arrhythmia')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('HR Modulation (bpm)')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.5, -0.15, 'Natural heart rate variability synchronized with breathing patterns.',
             transform=ax3.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    # 4. Combined autonomic effects
    combined_effects = [hr + adj + mod for hr, adj, mod in 
                       zip(hr_values, hr_adjustments, hr_modulations)]
    ax4.plot(time_points, hr_values, label='Base HR', linewidth=2, color='blue', alpha=0.7)
    ax4.plot(time_points, combined_effects, label='With Autonomic', linewidth=3, color='darkblue')
    ax4.set_title('Combined Autonomic Effects on HR')
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Heart Rate (bpm)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.text(0.5, -0.15, 'Integration of all autonomic influences on heart rate control.',
             transform=ax4.transAxes, fontsize=8, ha='center', style='italic', alpha=0.7)
    
    plt.tight_layout()
    return fig
