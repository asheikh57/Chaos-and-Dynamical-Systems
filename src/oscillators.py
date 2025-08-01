import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def vanderpol(t : float, y : list[float], mu: float= 0.5) -> list[float]:
    x, v = y
    dxdt = v
    dvdt = (1-x**2) * mu * v - x
    return [dxdt, dvdt]

def duffing(t: float, y: list[float], 
            alpha: float = 1, 
            beta: float = 1, 
            gamma: float = 1, 
            delta: float = 1,
            omega: float = 1) -> list[float]:
    x, v = y
    dxdt = v
    dvdt = gamma * np.cos(omega * t) - delta * v - alpha * x - beta * pow(x,3)
    return [dxdt, dvdt]    

def rk4(partial_eq, t, y, dt):
    # need to integrate this before computing sequential terms. This becomes a pattern
    k1 = partial_eq(t,y)  
    k2 = partial_eq(t + dt / 2, [y[i] + k1[i]*(dt/2) for i in range(len(y))])
    k3 = partial_eq(t + dt / 2, [y[i] + k2[i]*(dt/2) for i in range(len(y))])
    k4 = partial_eq(t + dt, [y[i] + k3[i]*dt for i in range(len(y))])
    
    out = [y[i] + (dt/6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(len(y))]
    return out

def solve_equation(diff_eq,
                   t_span : list[float],
                   y_initial : list[float],
                   t_eval : list[float],
                   args : tuple = ()):
    
    if len(t_eval) == 0:
        return []
    if max(t_eval) > t_span[1] or min(t_eval) < t_span[0]:
        raise OverflowError("solve_equation: t_eval exceeds the bounds of t_span")
    
    partial_eq = lambda y, t : diff_eq(y, t, *args)
    
    
    sols = np.zeros((len(t_eval), len(y_initial)))
    sols[0] = np.array(y_initial)
    y_next = y_initial
    
    # i+1 is the index of the solution we're solving for
    # y_initial corresponds to t_eval[0] and sols[0]
    for i in range(len(t_eval) - 1):
        y_curr = y_next
        y_next = rk4(partial_eq, t_eval[i], y_curr, t_eval[i+1] - t_eval[i])
        sols[i+1] = y_next
    #dxdt, dvdt = diff_eq(t, y, *args)
    return sols.transpose()
    

if __name__ == "__main__":
    t = np.linspace(0, 100, 10000)
    y0_iv  = [1.0,0.0]
    mu = 1
    alpha = -1.5
    beta = 0.25
    delta = 0.1
    omega = 2
    gamma = 2.5
    #sols =  solve_ivp(duffing, [0,100], y0_iv, method='RK45',t_eval=t, args=(alpha, beta, gamma, delta, omega,))
    sols = solve_equation(duffing, [0,100], y0_iv, t,args=(alpha, beta, gamma, delta, omega,))
    plt.figure()
    plt.plot(sols[0],sols[1])
    plt.show()
    '''plt.figure()
    plt.plot(sols.y[0],  (lambda x : 0.5*alpha*x*x + 0.25*beta*pow(x,4)) (sols.y[0]))
    plt.figure()
    plt.plot(sols.y[0],  (lambda x : 0.5*x*x) (sols.y[1]))
    plt.show()
'''
'''
    print(type(sols.y[1]))
    plt.figure()
    plt.plot(sols.y[0],sols.y[1])
    plt.figure()
    plt.plot(sols.y[0],  (lambda x : 0.5*alpha*x*x + 0.25*beta*pow(x,4)) (sols.y[0]))
    plt.figure()
    plt.plot(sols.y[0],  (lambda x : 0.5*x*x) (sols.y[1]))
    plt.show()
'''