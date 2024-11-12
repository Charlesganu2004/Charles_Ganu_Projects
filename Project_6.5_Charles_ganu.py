import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def lorenz(x, t, sigma=10, beta=8/3, rho=28):
    x_dot = sigma * (x[1] - x[0])
    y_dot = x[0] * (rho - x[2]) - x[1]
    z_dot = x[0] * x[1] - beta * x[2]
    return [x_dot, y_dot, z_dot]

def generate_data(n_samples, n_timesteps, dt, sigma=10, beta=8/3, rho=28):
    x = np.zeros((n_samples, n_timesteps, 3))
    t = np.linspace(0, (n_timesteps-1)*dt, n_timesteps)
    for i in range(n_samples):
        np.random.seed(i)
        x0 = np.random.uniform(-15, 15, 3)
        x[i] = odeint(lorenz, x0, t, args=(sigma, beta, rho))
    return x

n_samples = 5
n_timesteps = 1000
dt = 0.01
data = generate_data(n_samples, n_timesteps, dt)

X = np.zeros((n_samples * (n_timesteps - 1), 3))
Y = np.zeros((n_samples * (n_timesteps - 1), 3))
for i in range(n_samples):
    X[i*(n_timesteps-1):(i+1)*(n_timesteps-1)] = data[i, :-1]
    Y[i*(n_timesteps-1):(i+1)*(n_timesteps-1)] = data[i, 1:]

def build_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(3,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3))
    return model

model = build_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


history = model.fit(X, Y, epochs=60, batch_size=32, validation_split=0.2, shuffle=True)


model.save('model_28.h5')


def predict_future_states(model, initial_state, timesteps, dt, rho):
    states = np.zeros((timesteps, 3))
    states[0] = initial_state
    for i in range(1, timesteps):
        states[i] = model.predict(states[i-1].reshape(1, -3))
        
        t = np.linspace(0, dt, 2)
        states[i] = odeint(lorenz, states[i-1], t, args=(10, 8/3, rho))[-1]
    return states


timesteps = 1000
initial_state = data[0, 0]

states_17 = predict_future_states(model, initial_state, timesteps, dt, rho=17)
states_35 = predict_future_states(model, initial_state, timesteps, dt, rho=35)


def plot_states(states, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states[:, 0], states[:, 1], states[:, 2])
    ax.set_title(title)
    plt.show()

plot_states(states_17, 'Lorenz system with rho=17')
plot_states(states_35, 'Lorenz system with rho=35')