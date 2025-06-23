import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1.0  # duration of cardiac cycle in seconds
HR = 60  # heart rate (beats per minute)
V0 = 10  # unstressed volume (mL)
EDV = 120  # end-diastolic volume (mL)
ESV = 50  # end-systolic volume (mL)
E_max = 2.0  # max elastance (mmHg/mL)
E_min = 0.06  # min elastance (mmHg/mL)

# Time vector
t = np.linspace(0, T, 1000)

# Elastance function (Suga-Sagawa model)
def elastance(t, T, E_max, E_min):
    t_norm = (t % T) / T
    return E_min + 0.5 * (E_max - E_min) * (1 - np.cos(2 * np.pi * t_norm))

# Volume over time (synthetic model of filling & ejection)
def volume(t, T, EDV, ESV):
    t_norm = (t % T) / T
    vol = np.where(
        t_norm < 0.3,
        EDV - (EDV - ESV) * (t_norm / 0.3),  # ejection phase
        ESV + (EDV - ESV) * ((t_norm - 0.3) / 0.7)  # filling phase
    )
    return vol

# Compute elastance and volume
E_t = elastance(t, T, E_max, E_min)
V_t = volume(t, T, EDV, ESV)

# Compute pressure using time-varying elastance model
P_t = E_t * (V_t - V0)

# Plot pressure-volume loop
plt.figure(figsize=(8, 6))
plt.plot(V_t, P_t, color='blue', linewidth=2)
plt.title("Left Ventricular Pressure-Volume Loop")
plt.xlabel("Volume (mL)")
plt.ylabel("Pressure (mmHg)")
plt.grid(True)
plt.tight_layout()
plt.show()
