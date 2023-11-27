import numpy as np
from findiff import FinDiff
import matplotlib.pyplot as plt
from Euler_evolution_implicit import *

# We will be looking into some of the initial conditions given in the paper 

p_star = 2.116895853824

# b - bald and  s - scalar
pb1 = p_star - np.exp(-12)
pb2 = p_star - np.exp(-15)
ps1 = p_star + np.exp(-12)
ps2 = p_star + np.exp(-15)

tb1, rb1, phib1 = Apparant_horizon_implicit(pb1)
tb2, rb2, phib2 = Apparant_horizon_implicit(pb2)
ts1, rs1, phis1 = Apparant_horizon_implicit(ps1)
ts2, rs2, phis2 = Apparant_horizon_implicit(ps2)

plt.plot(tb1, rb1, label = 'b1', marker = '.')
plt.plot(tb2, rb2,  label = 'b2', marker = '.')
plt.plot(ts1, rs1,  label = 's1', marker = '.')
plt.plot(ts2, rs2, label = 's2', marker = '.')
plt.xlabel("Time")
plt.ylabel("r_h")
plt.legend()
plt.show()

plt.plot(tb1, phib1, label = 'b1', marker = '.')
plt.plot(tb2, phib2, label = 'b2', marker = '.')
plt.plot(ts1, phis1, label = 's1', marker = '.')
plt.plot(ts2, phis2, label = 's2', marker = '.')
plt.xlabel("Time")
plt.ylabel("phi_h")
plt.legend()
plt.show()

dt = tb1[1] - tb1[0]
d_dt = FinDiff(0, dt, 1, acc = 4)

plt.plot(np.log(abs(d_dt(phib1))), tb1)
plt.show()