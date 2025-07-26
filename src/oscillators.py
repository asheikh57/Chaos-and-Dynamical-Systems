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

if __name__ == "__main__":
    t = np.linspace(0,100, 10000)
    y0_iv  = [1.0,0.0]
    mu = 1
    alpha = -1.5
    beta = 0.25
    delta = 0.1
    omega = 2
    gamma = 2.5
    sols =  solve_ivp(duffing, [0,100], y0_iv, method='RK45',t_eval=t, args=(alpha, beta, gamma, delta, omega,))
    plt.figure()
    plt.plot(sols.y[0],sols.y[1])
    plt.show()
    print("hello")
