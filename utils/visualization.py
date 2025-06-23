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
