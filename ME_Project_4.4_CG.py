import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

t = sp.symbols('t')
A, ω, Rs, RL, L, N, V_s = sp.symbols('A ω Rs RL L N V_s')
V_s = A * sp.cos(ω * t)
i_L = sp.Function('i_L')(t)
eq = sp.Eq(V_s - Rs * i_L - N * sp.diff(i_L, t), 0)
i_L_solution = sp.dsolve(eq, i_L, ics={i_L.subs(t, 0): 0})
v_0 = RL * sp.diff(i_L_solution.rhs, t)
v_0_simplified = sp.simplify(v_0)
A_val = 1.0
ω_val = 2 * np.pi
Rs_val = 10
RL_val = 50
L_val = 0.1
N_val = 1.0
time_vals = np.linspace(0, 2, 1000)
v_0_num = sp.lambdify(t, v_0_simplified.subs({A: A_val, ω: ω_val, Rs: Rs_val, RL: RL_val, L: L_val, N: N_val}), "numpy")
v_0_vals = v_0_num(time_vals)
plt.plot(time_vals, v_0_vals, label="v_0(t)")
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Problem 4.4: Voltage v_0(t)')
plt.legend()
plt.grid(True)
plt.show()
print("Problem 4.4: v_0(t) =", v_0_simplified)

