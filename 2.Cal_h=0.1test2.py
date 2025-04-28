import numpy as np
import math

# 定義微分方程系統
def f(t, u):
    u1, u2 = u
    du1 = 9*u1 + 24*u2 + 5*math.cos(t) - (1/3)*math.sin(t)
    du2 = -24*u1 - 52*u2 - 9*math.cos(t) + (1/3)*math.sin(t)
    return np.array([du1, du2])

# exact
def u1_exact(t):
    return 2*math.exp(-3*t) - math.exp(-39*t) + (1/3)*math.cos(t)

def u2_exact(t):
    return -math.exp(-3*t) + 2*math.exp(-39*t) - (1/3)*math.cos(t)

# solve
def rk4_system(f, u0, t0, t_end, h):
    # step
    n_steps = int((t_end - t0) / h) + 1
    t_values = np.linspace(t0, t_end, n_steps)
    
    # init
    u_values = np.zeros((n_steps, len(u0)))
    u_values[0] = u0
    
    # RK4 formula
    for i in range(1, n_steps):
        t = t_values[i-1]
        u = u_values[i-1]
        
        k1 = f(t, u)
        k2 = f(t + h/2, u + h*k1/2)
        k3 = f(t + h/2, u + h*k2/2)
        k4 = f(t + h, u + h*k3)
        
        u_values[i] = u + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return t_values, u_values

# para init
t0, t_end = 0.0, 1.0
u0 = np.array([4/3, 2/3])
h = 0.1
t_vals, u_vals = rk4_system(f, u0, t0, t_end, h)

# exact&error
print(f"Runge-Kutta方法 (h={h}) 與精確解的比較:")
print(f"{'t':>6} {'u1_num':>12} {'u1_ex':>12} {'err1':>12} {'u2_num':>12} {'u2_ex':>12} {'err2':>12}")
print('-'*80)
for i in range(len(t_vals)):
    t = t_vals[i]
    u1_n = u_vals[i, 0]
    u2_n = u_vals[i, 1]
    u1_e = u1_exact(t)
    u2_e = u2_exact(t)
    err1 = abs(u1_n - u1_e)
    err2 = abs(u2_n - u2_e)
    print(f"{t:6.2f} {u1_n:12.6f} {u1_e:12.6f} {err1:12.2e} {u2_n:12.6f} {u2_e:12.6f} {err2:12.2e}")
