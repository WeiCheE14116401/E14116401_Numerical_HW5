import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 定義微分方程系統
def f(t, u):
    u1, u2 = u
    du1_dt = 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)
    du2_dt = -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)
    return np.array([du1_dt, du2_dt])

# exact
def exact_solution(t):
    u1 = 2*np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)
    u2 = -np.exp(-3*t) + 2*np.exp(-39*t) - (1/3)*np.cos(t)
    return u1, u2

# RK4 formula
def rk4_step(t, u, h):
    k1 = h * f(t, u)
    k2 = h * f(t + h/2, u + k1/2)
    k3 = h * f(t + h/2, u + k2/2)
    k4 = h * f(t + h, u + k3)
    return u + (k1 + 2*k2 + 2*k3 + k4)/6

# setting
h = 0.05 
t_values = np.arange(0, 1.01, h)  # t 從 0 到 1，包含 1
u_initial = np.array([4/3, 2/3])  # IC
u1_values = []
u2_values = []
exact_u1_values = []
exact_u2_values = []

# solve
u = u_initial
for t in t_values:
    u1, u2 = u
    u1_values.append(u1)
    u2_values.append(u2)
    exact_u1, exact_u2 = exact_solution(t)
    exact_u1_values.append(exact_u1)
    exact_u2_values.append(exact_u2)
    u = rk4_step(t, u, h)  

# print
df = pd.DataFrame({
    't': t_values,
    'u1_rk4': u1_values,
    'u1_exact': exact_u1_values,
    'u1_error': np.abs(np.array(u1_values) - np.array(exact_u1_values)),
    'u2_rk4': u2_values,
    'u2_exact': exact_u2_values,
    'u2_error': np.abs(np.array(u2_values) - np.array(exact_u2_values))
})
print("數值解與精確解比較：")
print(df.round(6))  

# max error
max_u1_error = df['u1_error'].max()
max_u2_error = df['u2_error'].max()
print(f"\nu1 的最大誤差: {max_u1_error:.6e}")
print(f"u2 的最大誤差: {max_u2_error:.6e}")