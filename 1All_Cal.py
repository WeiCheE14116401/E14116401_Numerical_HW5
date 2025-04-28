import numpy as np
import pandas as pd

# 定義微分方程 f(t, y) = 1 + y/t + (y/t)^2
def f(t, y):
    return 1 + y/t + (y/t)**2

# f 的導數 f'(t, y) = f_t + f_y * f(t, y) 
def f_prime(t, y):
    f_t = -y/t**2 - 2*y**2/t**3
    f_y = 1/t + 2*y/t**2
    return f_t + f_y * f(t, y)

# exact sol y(t) = t*tan(ln(t))
def exact_solution(t):
    return t * np.tan(np.log(t))

# (A) euler:  
#  y_i+1 = y_i + h * f(t_i, y_i)
def euler(t0, y0, h, tn):
    t_values = np.arange(t0, tn+ h, h)
    y_values = [y0]
    
    for i in range(len(t_values) - 1):
        t = t_values[i]
        y = y_values[-1]
        y_next = y + h * f(t, y)
        y_values.append(y_next)
    
    return t_values, y_values

# (B) Taylor’s method of order 2:  
#  y_i+1 = y_i + h * f(t_i, y_i) + (h^2 / 2) * f'(t_i, y_i)
def taylor_order2(t0, y0, h, tn):
    t_values = np.arange(t0, tn + h, h)
    y_values = [y0]
    
    for i in range(len(t_values) - 1):
        t = t_values[i]
        y = y_values[-1]
        y_next = y + h * f(t, y) + (h**2 / 2) * f_prime(t, y)
        y_values.append(y_next)
    
    return t_values, y_values

# 主程式
def main():
    # INIT
    t0 = 1.0
    y0 = 0.0
    h = 0.1
    tn = 2.0
    
    # Euler
    t_euler, y_euler = euler(t0, y0, h, tn)
    
    # Taylor
    t_taylor, y_taylor = taylor_order2(t0, y0, h, tn)
    
    # exact
    y_exact = [exact_solution(t) for t in t_euler]
    
    # errors
    error_euler = [abs(y_euler[i] - y_exact[i]) for i in range(len(y_euler))]
    error_taylor = [abs(y_taylor[i] - y_exact[i]) for i in range(len(y_taylor))]
    
    results = {
        't': t_euler,
        'Euler': y_euler,
        'Taylor': y_taylor,
        'Exact': y_exact,
        'Error (Euler)': error_euler,
        'Error (Taylor)': error_taylor
    }
    
    df = pd.DataFrame(results)
    df = df.round(6)  
    print(df)

if __name__ == "__main__":
    main()