import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def vanderpol(t : float, y : list[float], mu = 0.5) -> list[float]:
    x, v = y
    dxdt = v
    dvdt = (1-x**2) * mu * v - x
    return [dxdt, dvdt]

if __name__ == "__main__":
    t = np.linspace(0,100, 1000)
    y0_iv  = [5.0,0.0]
    mu = 1
    sols =  solve_ivp(vanderpol, [0,100], y0_iv, method='RK45',t_eval=t, args=(mu,))
    plt.figure()
    plt.plot(sols.y[0],sols.y[1])
    plt.show()
    print("hello")
