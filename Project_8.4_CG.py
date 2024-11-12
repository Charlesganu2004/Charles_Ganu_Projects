import numpy as np
import scipy.linalg
import scipy.integrate
import matplotlib.pyplot as plt

m = 0.2
L = 0.3
I = (1/3) * m * L**2
g = 9.81

A = np.array([[0, 1, 0, 0],
              [0, 0, -m*g/I, 0],
              [0, 0, 0, 1],
              [0, 0, (m*g*L/I), 0]])

B = np.array([[0],
              [1/I],
              [0],
              [1/(m*L**2)]])
C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
D = np.zeros((2, 1))

eigvals_upright = np.linalg.eigvals(A)
print("Eigenvalues around upright position:", eigvals_upright)

Q = np.diag([10, 1, 10, 1])
R = np.array([[0.01]])

# Solve the continuous-time Algebraic Riccati equation (ARE) for LQR gain matrix K
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print("LQR gain matrix K:", K)

# Construct the observability matrix
obs_matrix = np.vstack([C, C @ A, C @ A @ A, C @ A @ A @ A])
print("Observability matrix rank:", np.linalg.matrix_rank(obs_matrix))

def state_space_model(t, x):
    u = -K @ x
    dxdt = A @ x + B.flatten() * u
    return dxdt

x0 = np.array([0.1, 0, 0.1, 0])
t = np.linspace(0, 10, 1000)

sol = scipy.integrate.solve_ivp(state_space_model, [0, 10], x0, t_eval=t)

plt.figure()
plt.plot(sol.t, sol.y[0, :], label='Theta (rad)')
plt.plot(sol.t, sol.y[2, :], label='Phi (rad)')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('State variables')
plt.title('Rotary Inverted Pendulum with LQR Control')
plt.show()

Q_kf = np.eye(4) * 1e-3
R_kf = np.eye(2) * 1e-2
P = np.eye(4)
x_hat = np.zeros(4)

def kalman_filter(y, u):
    global P, x_hat
    A_kf = A
    B_kf = B
    C_kf = C
    K_kf = P @ C_kf.T @ np.linalg.inv(C_kf @ P @ C_kf.T + R_kf)
    x_hat = A_kf @ x_hat + B_kf.flatten() * u + K_kf @ (y - C_kf @ x_hat)
    P = (np.eye(4) - K_kf @ C_kf) @ P
    return x_hat

def state_space_kalman_model(t, x):
    u = -K @ x_hat
    y = C @ x + np.random.normal(0, 0.1, 2)
    x_hat = kalman_filter(y, u)
    dxdt = A @ x + B.flatten() * u
    return dxdt

sol_kf = scipy.integrate.solve_ivp(state_space_kalman_model, [0, 10], x0, t_eval=t)

plt.figure()
plt.plot(sol_kf.t, sol_kf.y[0, :], label='Theta (rad)')
plt.plot(sol_kf.t, sol_kf.y[2, :], label='Phi (rad)')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('State variables')
plt.title('Rotary Inverted Pendulum with Kalman Filter and LQR Control')
plt.show()

def lqg_control(t, x):
    u = -K @ x_hat
    y = C @ x + np.random.normal(0, 0.1, 2)
    x_hat = kalman_filter(y, u)
    dxdt = A @ x + B.flatten() * u
    return dxdt

sol_lqg = scipy.integrate.solve_ivp(lqg_control, [0, 10], x0, t_eval=t)

plt.figure()
plt.plot(sol_lqg.t, sol_lqg.y[0, :], label='Theta (rad)')
plt.plot(sol_lqg.t, sol_lqg.y[2, :], label='Phi (rad)')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('State variables')
plt.title('Rotary Inverted Pendulum with LQG Control')
plt.show()