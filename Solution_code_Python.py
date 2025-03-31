# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Input parameters from the task sheet
V_OC = 3.7
R0 = 0.005
R1 = 0.005
C1 = 100
Cn = 1 * 3600  # Ah to Coulombs Conversion
J = 0.1
b = 0.05
K = 0.01
Rm = 0.01
Lm = 0.05

# External torque function
def external_torque(t):
    return 0.05 * (1 + np.sin(t))

# Define the system of differential equations without external torque
def behaviour_no_load(t, y):
    # State variables
    IL, omega, V1, z = y
    
    # Differential equations
    dIL_dt = (V_OC - IL * R0 - V1 - Rm * IL - K * omega) / Lm
    domega_dt = (K * IL - b * omega) / J
    dV1_dt = -V1 / (C1 * R1) + IL / C1
    dz_dt = -IL / Cn
    
    # Prevent SoC (state of charge) from becoming negative
    if z <= 0:
        dz_dt = 0
    
    return [dIL_dt, domega_dt, dV1_dt, dz_dt]

# Initial conditions
initial_conditions = [0, 0, 0, 1]  # IL(0), omega(0), V1(0), z(1)

# Time span and evaluation points
time_span = (0, 60)  # 1 minute
time_eval = np.linspace(0, 60, 1200)  # 1200 points for high resolution

# Solve the system of differential equations for no external load torque
output_no_load = solve_ivp(behaviour_no_load, time_span, initial_conditions, t_eval=time_eval, method='RK45', atol=1e-6, rtol=1e-3)

# Extract time points
time_points_no_load = output_no_load.t

# Extract results for no load condition
IL_no_load, omega_no_load, V1_no_load, z_no_load = output_no_load.y
Vt_no_load = V_OC - IL_no_load * R0 - V1_no_load

# Plot results for no external load torque
plt.figure(figsize=(12, 8))
plt.plot(time_points_no_load, omega_no_load)
plt.title('Angular Velocity vs Time (No Load)')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.grid(True)
plt.savefig('Angular_Velocity_vs_Time_No_Load.jpeg')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(time_points_no_load, z_no_load)
plt.title('State of Charge vs Time (No Load)')
plt.xlabel('Time (s)')
plt.ylabel('State of Charge')
plt.grid(True)
plt.savefig('State_of_Charge_vs_Time_No_Load.jpeg')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(time_points_no_load, IL_no_load)
plt.title('Current vs Time (No Load)')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.grid(True)
plt.savefig('Current_vs_Time_No_Load.jpeg')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(time_points_no_load, Vt_no_load)
plt.title('Voltage vs Time (No Load)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.savefig('Voltage_vs_Time_No_Load.jpeg')
plt.show()

# Define the system of differential equations with external torque
def behaviour_with_load(t, y):
    # State variables
    IL, omega, V1, z = y
    
    # External torque at time t
    T_ext = external_torque(t)
    
    # Differential equations
    dIL_dt = (V_OC - IL * R0 - V1 - Rm * IL - K * omega) / Lm
    domega_dt = (K * IL - b * omega - T_ext) / J
    dV1_dt = -V1 / (C1 * R1) + IL / C1
    dz_dt = -IL / Cn
    
    # Prevent SoC (state of charge) from becoming negative
    if z <= 0:
        dz_dt = 0
    
    return [dIL_dt, domega_dt, dV1_dt, dz_dt]

# Solve the system of differential equations for external load torque
output_with_load = solve_ivp(behaviour_with_load, time_span, initial_conditions, t_eval=time_eval, method='RK45', atol=1e-6, rtol=1e-3)

# Extract time points
time_points_with_load = output_with_load.t

# Extract results for load condition
IL_with_load, omega_with_load, V1_with_load, z_with_load = output_with_load.y
Vt_with_load = V_OC - IL_with_load * R0 - V1_with_load

# Plot results for external load torque
plt.figure(figsize=(12, 8))
plt.plot(time_points_with_load, omega_with_load)
plt.title('Angular Velocity vs Time (With Load)')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.grid(True)
plt.savefig('Angular_Velocity_vs_Time_With_Load.jpeg')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(time_points_with_load, z_with_load)
plt.title('State of Charge vs Time (With Load)')
plt.xlabel('Time (s)')
plt.ylabel('State of Charge')
plt.grid(True)
plt.savefig('State_of_Charge_vs_Time_With_Load.jpeg')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(time_points_with_load, IL_with_load)
plt.title('Current vs Time (With Load)')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.grid(True)
plt.savefig('Current_vs_Time_With_Load.jpeg')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(time_points_with_load, Vt_with_load)
plt.title('Voltage vs Time (With Load)')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.savefig('Voltage_vs_Time_With_Load.jpeg')
plt.show()

# Extracting and plotting time steps
time_steps_no_load = np.diff(output_no_load.t)
time_steps_with_load = np.diff(output_with_load.t)

# Plot adaptive time steps over time
plt.figure(figsize=(12, 8))
plt.plot(output_no_load.t[:-1], time_steps_no_load, label='No Load')
plt.plot(output_with_load.t[:-1], time_steps_with_load, label='With Load')
plt.title('Adaptive Time Steps Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Time Step Size (s)')

# Setting the y-axis limits to avoid exaggeration of negligible changes
plt.ylim([time_steps_no_load.min() - 1e-5, time_steps_no_load.max() + 1e-5])  # Adjusting y-axis limits

plt.legend()
plt.grid(True)
plt.savefig('Adaptive_Time_Steps_Adjusted.jpeg')
plt.show()
