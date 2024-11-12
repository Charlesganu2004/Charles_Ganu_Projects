import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import Lasso


def rossler(X, t, a=0.2, b=0.2, c=14):
    x, y, z = X
    x_dot = -y - z
    y_dot = x + a * y
    z_dot = b + z * (x - c)
    return [x_dot, y_dot, z_dot]

def generate_trajectory(a=0.2, b=0.2, c=14, t_max=100, dt=0.01, initial_state=[1, 1, 1]):
    t = np.arange(0, t_max, dt)
    X = odeint(rossler, initial_state, t, args=(a, b, c))
    return t, X


def plot_trajectory(X, title='Rossler System Trajectory'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:, 0], X[:, 1], X[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    plt.show()

def sindy(X, dt, threshold=0.1):
    n_samples, n_features = X.shape
    Theta = np.zeros((n_samples, 10))  
    Theta[:, 0] = 1
    Theta[:, 1:4] = X
    Theta[:, 4:7] = X**2
    Theta[:, 7] = X[:, 0] * X[:, 1]
    Theta[:, 8] = X[:, 0] * X[:, 2]
    Theta[:, 9] = X[:, 1] * X[:, 2]
   
    dX_dt = np.diff(X, axis=0) / dt
   
    model = Lasso(alpha=threshold, fit_intercept=False)
    Xi = np.zeros((10, n_features))
    for i in range(n_features):
        model.fit(Theta[:-1], dX_dt[:, i])
        Xi[:, i] = model.coef_
   
    return Xi


t, X = generate_trajectory()
plot_trajectory(X)


dt = t[1] - t[0]
Xi = sindy(X, dt, threshold=0.1)
print('Identified SINDy model coefficients (without noise):')
print(Xi)


noise_level = 0.01
X_noisy = X + noise_level * np.random.randn(*X.shape)
plot_trajectory(X_noisy, title='Rossler System Trajectory with Noise')

Xi_noisy = sindy(X_noisy, dt, threshold=0.1)
print('Identified SINDy model coefficients (with noise):')
print(Xi_noisy)


thresholds = [0.01, 0.05, 0.1, 0.2]
lengths = [100, 500, 1000, 5000]

for thresh in thresholds:
    Xi_thresh = sindy(X, dt, threshold=thresh)
    print('Threshold: {thresh}, Coefficients:')
    print(Xi_thresh)

for length in lengths:
    t, X_short = generate_trajectory(t_max=length*dt)
    Xi_length = sindy(X_short, dt, threshold=0.1)
    print('Trajectory length: {length}, Coefficients:')
    print(Xi_length)

plt.show()