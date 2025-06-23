"""
Pharmacokinetic and Pharmacodynamic Models
"""

import numpy as np
from .drug_database import DrugProfile

def pk_concentration(t_hours: float, dose_mg: float, drug_profile: DrugProfile) -> float:
    """
    Calculate drug concentration using first-order kinetics
    
    Args:
        t_hours: Time in hours
        dose_mg: Dose in milligrams
        drug_profile: Drug profile containing pharmacokinetic parameters
        
    Returns:
        Drug concentration in mg/L
    """
    k_el = np.log(2) / drug_profile.half_life
    C = (dose_mg / drug_profile.vd) * np.exp(-k_el * t_hours)
    return C

def pd_effect(concentration: float, baseline: float, emax: float, ec50: float, hill_coeff: float) -> float:
    """
    Calculate pharmacodynamic effect using Hill equation
    
    Args:
        concentration: Drug concentration
        baseline: Baseline value of the parameter
        emax: Maximum effect
        ec50: Concentration at 50% effect
        hill_coeff: Hill coefficient
        
    Returns:
        Modified parameter value
    """
    if emax >= 0:  # Agonist
        return baseline + (emax * concentration**hill_coeff) / (concentration**hill_coeff + ec50**hill_coeff)
    else:  # Antagonist
        return baseline * (1 + (emax * concentration**hill_coeff) / (concentration**hill_coeff + ec50**hill_coeff))
