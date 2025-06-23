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
    Calculate comprehensive hemodynamics
    
    Args:
        hr: Heart rate
        contractility: Contractility
        afterload: Afterload
        preload: Preload (EDV)
        esv: End-systolic volume
        v0: Unstressed volume
        e_min: Minimum elastance
        
    Returns:
        Dictionary with hemodynamic parameters
    """
    T = 60 / hr  # Cardiac cycle duration
    
    # Stroke volume calculation (simplified Starling mechanism)
    sv = preload - esv
    if contractility > 1.5:
        sv *= (1 + 0.2 * (contractility - 1.5))
    
    # Cardiac output
    co = sv * hr / 1000  # L/min
    
    # Mean arterial pressure (simplified)
    map_pressure = co * afterload * 80  # mmHg
    
    return {
        'stroke_volume': sv,
        'cardiac_output': co,
        'mean_arterial_pressure': map_pressure,
        'ejection_fraction': (sv / preload) * 100
    }
