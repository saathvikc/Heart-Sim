"""
Cardiovascular Mechanics Models
"""

import numpy as np
from typing import Dict

def time_varying_elastance(t: np.ndarray, T: float, E_max: float, E_min: float) -> np.ndarray:
    """
    Time-varying elastance model (Suga-Sagawa)
    
    Args:
        t: Time array
        T: Cardiac cycle duration
        E_max: Maximum elastance
        E_min: Minimum elastance
        
    Returns:
        Elastance values over time
    """
    t_norm = (t % T) / T
    # More physiologically accurate elastance curve using vectorized operations
    return np.where(
        t_norm < 0.3,  # Systole
        E_min + (E_max - E_min) * np.sin(np.pi * t_norm / 0.3)**2,
        E_min  # Diastole
    )

def ventricular_volume(t: np.ndarray, T: float, EDV: float, ESV: float) -> np.ndarray:
    """
    Ventricular volume throughout cardiac cycle
    
    Args:
        t: Time array
        T: Cardiac cycle duration
        EDV: End-diastolic volume
        ESV: End-systolic volume
        
    Returns:
        Volume values over time
    """
    t_norm = (t % T) / T
    
    # Use numpy.where for vectorized operations
    volume = np.where(
        t_norm < 0.05,  # Isovolumic contraction
        EDV,
        np.where(
            t_norm < 0.35,  # Ejection
            EDV - (EDV - ESV) * (1 - np.cos(np.pi * (t_norm - 0.05) / 0.3 / 2))**2,
            np.where(
                t_norm < 0.4,  # Isovolumic relaxation
                ESV,
                ESV + (EDV - ESV) * (1 - np.cos(np.pi * (t_norm - 0.4) / 0.6 / 2))**2  # Filling
            )
        )
    )
    return volume

def calculate_hemodynamics(hr: float, contractility: float, afterload: float, 
                         preload: float, esv: float, v0: float, e_min: float) -> Dict[str, float]:
    """
    Calculate comprehensive hemodynamics with advanced Frank-Starling and Windkessel mechanics
    
    Args:
        hr: Heart rate
        contractility: Contractility (Emax)
        afterload: Afterload
        preload: Preload (EDV)
        esv: End-systolic volume
        v0: Unstressed volume
        e_min: Minimum elastance
        
    Returns:
        Dictionary with hemodynamic parameters
    """
    T = 60 / hr  # Cardiac cycle duration
    
    # Enhanced Frank-Starling mechanism with length-tension relationship
    optimal_preload = 120  # mL, optimal preload for maximum force
    starling_factor = 1.0 + 0.8 * np.tanh((preload - optimal_preload) / 30)
    
    # Improved stroke volume calculation using elastance model
    # SV = (Ees * (EDV - V0) - Ped) / (Ees + Ea)
    # Where Ees is end-systolic elastance, Ea is arterial elastance
    ees = contractility * starling_factor  # End-systolic elastance
    ea = afterload  # Arterial elastance (simplified)
    
    # End-diastolic pressure (simplified preload-dependent)
    ped = 5 + 0.1 * (preload - 80)**2 / 100  # mmHg, nonlinear EDPVR
    
    # Calculate stroke volume from elastance coupling
    sv = max(10, (ees * (preload - v0) - ped) / (ees + ea))
    
    # Ensure physiological bounds
    sv = min(sv, preload - 20)  # Cannot eject below minimum ESV
    sv = max(sv, 20)  # Minimum stroke volume
    
    # Cardiac output with frequency-dependent effects
    frequency_factor = 1.0 - 0.1 * np.exp(-(hr - 70)/30)  # Optimal around 70 bpm
    co = sv * hr * frequency_factor / 1000  # L/min
    
    # Advanced arterial pressure calculation using Windkessel model
    # Include systolic and diastolic pressure estimation
    arterial_compliance = 1.5  # mL/mmHg
    peripheral_resistance = 1.2  # mmHg·s/mL
    
    # Systolic pressure from ventricular-arterial coupling
    p_sys = (ees * sv) / arterial_compliance + 80  # Base pressure
    
    # Diastolic pressure from exponential decay during diastole
    tau = arterial_compliance * peripheral_resistance  # Time constant
    p_dias = p_sys * np.exp(-0.6 * T / tau) + 60  # Minimum diastolic
    
    # Mean arterial pressure (more accurate formula)
    map_pressure = p_dias + (p_sys - p_dias) / 3
    
    # Enhanced ejection fraction with contractility dependence
    ef = (sv / preload) * 100
    
    # Additional hemodynamic parameters
    stroke_work = map_pressure * sv / 1000  # Joules
    cardiac_index = co / 1.73  # L/min/m² (assuming 1.73 m² BSA)
    
    return {
        'stroke_volume': sv,
        'cardiac_output': co,
        'systolic_pressure': p_sys,
        'diastolic_pressure': p_dias,
        'mean_arterial_pressure': map_pressure,
        'ejection_fraction': ef,
        'stroke_work': stroke_work,
        'cardiac_index': cardiac_index,
        'arterial_compliance': arterial_compliance,
        'peripheral_resistance': peripheral_resistance
    }

def calculate_myocardial_oxygen_demand(hr: float, contractility: float, afterload: float, 
                                     sv: float, wall_thickness: float = 10.0) -> Dict[str, float]:
    """
    Calculate myocardial oxygen consumption using advanced energetics
    
    Args:
        hr: Heart rate (bpm)
        contractility: Contractility index
        afterload: Afterload (mmHg·s/mL)
        sv: Stroke volume (mL)
        wall_thickness: LV wall thickness (mm)
        
    Returns:
        Dictionary with oxygen demand parameters
    """
    # Baseline myocardial oxygen consumption
    mvo2_baseline = 8.0  # mL O2/min/100g
    
    # Heart rate component (linear relationship)
    hr_component = 0.1 * (hr - 70)
    
    # Contractility component (exponential relationship)
    contractility_component = 2.0 * (contractility - 1.0)
    
    # Pressure-volume work component
    stroke_work = afterload * sv * 0.0001333  # Convert to J
    pv_work_component = stroke_work * 0.5
    
    # Wall stress component (Law of Laplace)
    # Stress = P × r / (2 × h)
    chamber_radius = (3 * sv / (4 * np.pi))**(1/3) * 10  # mm
    wall_stress = afterload * chamber_radius / (2 * wall_thickness)
    stress_component = wall_stress * 0.01
    
    # Total MVO2
    total_mvo2 = mvo2_baseline + hr_component + contractility_component + pv_work_component + stress_component
    
    # Mechanical efficiency
    external_work = stroke_work
    total_energy = total_mvo2 * 20  # Convert O2 to energy (J)
    efficiency = (external_work / total_energy) * 100 if total_energy > 0 else 0
    
    return {
        'mvo2_total': max(5.0, total_mvo2),  # mL O2/min/100g
        'mvo2_baseline': mvo2_baseline,
        'hr_component': hr_component,
        'contractility_component': contractility_component,
        'wall_stress': wall_stress,
        'mechanical_efficiency': min(40.0, max(10.0, efficiency)),  # %
        'stroke_work': stroke_work
    }

def baroreflex_response(current_pressure: float, setpoint_pressure: float = 90.0,
                       sensitivity: float = 1.0) -> Dict[str, float]:
    """
    Model baroreflex response to pressure changes
    
    Args:
        current_pressure: Current mean arterial pressure (mmHg)
        setpoint_pressure: Target pressure setpoint (mmHg)
        sensitivity: Baroreflex sensitivity (0-2)
        
    Returns:
        Dictionary with autonomic adjustments
    """
    pressure_error = current_pressure - setpoint_pressure
    
    # Sigmoid baroreflex response
    max_response = 2.0 * sensitivity
    autonomic_response = max_response * np.tanh(pressure_error / 20.0)
    
    # Heart rate adjustment (inverse relationship with pressure)
    hr_adjustment = -autonomic_response * 15  # bpm
    
    # Contractility adjustment (direct relationship)
    contractility_adjustment = autonomic_response * 0.3
    
    # Peripheral resistance adjustment
    resistance_adjustment = autonomic_response * 0.25
    
    # Venous return adjustment
    venous_return_adjustment = autonomic_response * 0.15
    
    return {
        'pressure_error': pressure_error,
        'autonomic_response': autonomic_response,
        'hr_adjustment': hr_adjustment,
        'contractility_adjustment': contractility_adjustment,
        'resistance_adjustment': resistance_adjustment,
        'venous_return_adjustment': venous_return_adjustment
    }

def respiratory_cardiovascular_coupling(respiratory_phase: float, tidal_volume: float = 500.0) -> Dict[str, float]:
    """
    Model respiratory effects on cardiovascular parameters
    
    Args:
        respiratory_phase: Current phase of respiratory cycle (0-1)
        tidal_volume: Tidal volume (mL)
        
    Returns:
        Dictionary with respiratory modulation effects
    """
    # Respiratory sinus arrhythmia
    rsa_amplitude = 8.0 * (tidal_volume / 500.0)  # Scales with tidal volume
    hr_modulation = rsa_amplitude * np.sin(2 * np.pi * respiratory_phase)
    
    # Venous return modulation (increased during inspiration)
    venous_return_modulation = 0.15 * np.sin(2 * np.pi * respiratory_phase - np.pi/2)
    
    # Intrathoracic pressure effects
    pleural_pressure = -5.0 + 8.0 * (tidal_volume / 500.0) * np.sin(2 * np.pi * respiratory_phase)
    
    # Afterload modulation (decreased during inspiration due to reduced LV transmural pressure)
    afterload_modulation = -0.1 * np.sin(2 * np.pi * respiratory_phase)
    
    return {
        'hr_modulation': hr_modulation,
        'venous_return_modulation': venous_return_modulation,
        'pleural_pressure': pleural_pressure,
        'afterload_modulation': afterload_modulation
    }
