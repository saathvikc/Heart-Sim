"""
Pharmacokinetic and Pharmacodynamic Models
"""

import numpy as np
from typing import Dict
from .drug_database import DrugProfile
from typing import Dict

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

def pk_two_compartment_model(t_hours: float, dose_mg: float, drug_profile: DrugProfile) -> Dict[str, float]:
    """
    Two-compartment pharmacokinetic model for more realistic drug distribution
    
    Args:
        t_hours: Time in hours
        dose_mg: Dose in milligrams
        drug_profile: Drug profile
        
    Returns:
        Dictionary with central and peripheral concentrations
    """
    # Distribution parameters (estimated from single compartment)
    v1 = drug_profile.vd * 0.6  # Central compartment
    v2 = drug_profile.vd * 0.4  # Peripheral compartment
    
    # Rate constants
    k_el = np.log(2) / drug_profile.half_life  # Elimination
    k12 = 0.5  # Central to peripheral
    k21 = 0.3  # Peripheral to central
    
    # Hybrid rate constants
    k10 = k_el
    alpha = 0.5 * ((k10 + k12 + k21) + np.sqrt((k10 + k12 + k21)**2 - 4*k10*k21))
    beta = 0.5 * ((k10 + k12 + k21) - np.sqrt((k10 + k12 + k21)**2 - 4*k10*k21))
    
    # Initial concentration
    c0 = dose_mg / v1
    
    # Coefficients
    A = c0 * (alpha - k21) / (alpha - beta)
    B = c0 * (k21 - beta) / (alpha - beta)
    
    # Central compartment concentration
    c_central = A * np.exp(-alpha * t_hours) + B * np.exp(-beta * t_hours)
    
    # Peripheral compartment (simplified)
    c_peripheral = (c_central * k12 / k21) * (1 - np.exp(-k21 * t_hours))
    
    return {
        'central_concentration': max(0, c_central),
        'peripheral_concentration': max(0, c_peripheral),
        'total_concentration': max(0, c_central + c_peripheral * v2/v1),
        'distribution_ratio': c_peripheral / c_central if c_central > 0 else 0
    }

def pd_effect_with_tolerance(concentration: float, baseline: float, emax: float, 
                           ec50: float, hill_coeff: float, time_hours: float = 0,
                           tolerance_rate: float = 0.1) -> float:
    """
    Enhanced Hill equation with tolerance development over time
    
    Args:
        concentration: Drug concentration
        baseline: Baseline value of the parameter
        emax: Maximum effect
        ec50: Concentration at 50% effect
        hill_coeff: Hill coefficient
        time_hours: Time since first dose (for tolerance)
        tolerance_rate: Rate of tolerance development
        
    Returns:
        Modified parameter value with tolerance
    """
    # Basic Hill equation
    if emax >= 0:  # Agonist
        basic_effect = baseline + (emax * concentration**hill_coeff) / (concentration**hill_coeff + ec50**hill_coeff)
    else:  # Antagonist
        basic_effect = baseline * (1 + (emax * concentration**hill_coeff) / (concentration**hill_coeff + ec50**hill_coeff))
    
    # Tolerance development (exponential decay of effect over time)
    tolerance_factor = np.exp(-tolerance_rate * time_hours)
    
    # Apply tolerance to the drug effect component only
    drug_effect_component = basic_effect - baseline
    effect_with_tolerance = baseline + drug_effect_component * tolerance_factor
    
    return effect_with_tolerance

def receptor_binding_kinetics(concentration: float, receptor_density: float = 1.0,
                            kon: float = 1.0, koff: float = 0.1) -> Dict[str, float]:
    """
    Model receptor binding kinetics and occupancy
    
    Args:
        concentration: Drug concentration
        receptor_density: Total receptor density
        kon: Association rate constant
        koff: Dissociation rate constant
        
    Returns:
        Receptor occupancy parameters
    """
    kd = koff / kon  # Dissociation constant
    
    # Receptor occupancy (Langmuir binding)
    occupancy = (concentration * receptor_density) / (concentration + kd)
    
    # Free receptors
    free_receptors = receptor_density - occupancy
    
    # Binding affinity and cooperativity
    binding_affinity = 1 / kd
    
    return {
        'receptor_occupancy': occupancy,
        'free_receptors': free_receptors,
        'occupancy_percentage': (occupancy / receptor_density) * 100,
        'binding_affinity': binding_affinity,
        'dissociation_constant': kd
    }

def drug_metabolism_kinetics(concentration: float, vmax: float = 10.0, km: float = 5.0) -> Dict[str, float]:
    """
    Model hepatic drug metabolism using Michaelis-Menten kinetics
    
    Args:
        concentration: Drug concentration
        vmax: Maximum metabolic rate
        km: Michaelis constant
        
    Returns:
        Metabolism parameters
    """
    # Michaelis-Menten equation
    metabolic_rate = (vmax * concentration) / (km + concentration)
    
    # Metabolic clearance
    clearance = vmax / (km + concentration)
    
    # Saturation percentage
    saturation = (concentration / (km + concentration)) * 100
    
    return {
        'metabolic_rate': metabolic_rate,
        'clearance': clearance,
        'saturation_percentage': saturation,
        'km': km,
        'vmax': vmax
    }
