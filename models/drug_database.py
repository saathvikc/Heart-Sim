"""
Drug Database Module
Contains drug profiles and pharmacological data
"""

from dataclasses import dataclass
from typing import Dict

@dataclass
class DrugProfile:
    """Drug profile with pharmacokinetic and pharmacodynamic parameters"""
    name: str
    vd: float  # Volume of distribution (L)
    half_life: float  # Half-life (hours)
    ec50_contractility: float  # EC50 for contractility (mg/L)
    emax_contractility: float  # Max effect on contractility
    ec50_hr: float  # EC50 for heart rate (mg/L)
    emax_hr: float  # Max effect on heart rate (bpm)
    ec50_afterload: float  # EC50 for afterload (mg/L)
    emax_afterload: float  # Max effect on afterload (%)
    hill_coefficient: float
    mechanism: str

# Drug database with realistic pharmacological profiles
DRUG_DATABASE: Dict[str, DrugProfile] = {
    "Dobutamine": DrugProfile(
        name="Dobutamine",
        vd=20,
        half_life=2.5,
        ec50_contractility=2.0,
        emax_contractility=1.5,
        ec50_hr=1.8,
        emax_hr=25,
        ec50_afterload=3.0,
        emax_afterload=-15,
        hill_coefficient=1.2,
        mechanism="β1-agonist"
    ),
    "Epinephrine": DrugProfile(
        name="Epinephrine",
        vd=15,
        half_life=0.083,
        ec50_contractility=0.5,
        emax_contractility=2.0,
        ec50_hr=0.3,
        emax_hr=40,
        ec50_afterload=0.8,
        emax_afterload=20,
        hill_coefficient=1.5,
        mechanism="α/β-agonist"
    ),
    "Propranolol": DrugProfile(
        name="Propranolol",
        vd=25,
        half_life=4,
        ec50_contractility=1.0,
        emax_contractility=-0.8,
        ec50_hr=0.8,
        emax_hr=-20,
        ec50_afterload=2.0,
        emax_afterload=10,
        hill_coefficient=1.0,
        mechanism="β-blocker"
    ),
    "Digoxin": DrugProfile(
        name="Digoxin",
        vd=5,
        half_life=36,
        ec50_contractility=1.2,
        emax_contractility=0.6,
        ec50_hr=1.5,
        emax_hr=-5,
        ec50_afterload=4.0,
        emax_afterload=5,
        hill_coefficient=1.8,
        mechanism="Na/K-ATPase inhibitor"
    ),
    "Milrinone": DrugProfile(
        name="Milrinone",
        vd=30,
        half_life=2.3,
        ec50_contractility=1.8,
        emax_contractility=1.2,
        ec50_hr=2.0,
        emax_hr=15,
        ec50_afterload=2.5,
        emax_afterload=-20,
        hill_coefficient=1.1,
        mechanism="PDE3 inhibitor"
    ),
    "Atenolol": DrugProfile(
        name="Atenolol",
        vd=40,
        half_life=6,
        ec50_contractility=2.5,
        emax_contractility=-0.5,
        ec50_hr=2.0,
        emax_hr=-15,
        ec50_afterload=3.0,
        emax_afterload=8,
        hill_coefficient=1.0,
        mechanism="β1-selective blocker"
    )
}
