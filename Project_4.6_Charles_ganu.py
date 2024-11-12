import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


m = 1  
g = 9.81  
u_max = 15 * m * g


x0 = np.array([0, 5, 0, 0])
xd = np.array([10, 5, 0, 0])


dt = 0.1
A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 0],
              [0, 0, 0, 0]])
B = np.array([[0, 0],
              [0, 0],
              [1/m, 0],
              [0, 1/m]])
A_d = np.eye(4) + A * dt
B_d = B * dt


T = 3  
Tu = 1  
N = int(T / dt)  
N_steps = 50  


x_trajectory = np.zeros((4, N_steps))
u_trajectory = np.zeros((2, N_steps))
x = x0

for k in range(N_steps):
    
    x_var = cp.Variable((4, N+1))
    u_var = cp.Variable((2, N))
   
    
    cost = 0
    for i in range(N):
        cost += cp.quad_form(x_var[:, i] - xd, np.eye(4)) + cp.quad_form(u_var[:, i], np.eye(2))
   
    
    constraints = []
    constraints.append(x_var[:, 0] == x)
    for i in range(N):
        constraints.append(x_var[:, i+1] == A_d @ x_var[:, i] + B_d @ u_var[:, i])
        constraints.append(u_var[:, i] >= 0)
        constraints.append(u_var[:, i] <= u_max)
   
    
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()
   
    
    u = u_var[:, 0].value
    x = A_d @ x + B_d @ u
   
    
    x_trajectory[:, k] = x
    u_trajectory[:, k] = u


plt.figure()
plt.plot(x_trajectory[0, :], x_trajectory[1, :], label='Trajectory')
plt.plot(x0[0], x0[1], 'ro', label='Start')
plt.plot(xd[0], xd[1], 'go', label='Goal')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.title('Position Trajectory')
plt.grid()
plt.show()

plt.figure()
plt.plot(np.arange(N_steps) * Tu, u_trajectory[0, :], label='Thrust F1')
plt.plot(np.arange(N_steps) * Tu, u_trajectory[1, :], label='Thrust F2')
plt.xlabel('Time (s)')
plt.ylabel('Thrust (N)')
plt.legend()
plt.title('Control Inputs')
plt.grid()
plt.show()