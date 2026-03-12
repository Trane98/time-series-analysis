import numpy as np
import matplotlib.pyplot as plt

# Simulate ARX(2,2) model
N = 500
u = np.random.randn(N)
e = np.random.randn(N) * 0.1  # noise
y = np.zeros(N)
a1 = 0.5
a2 = -0.3
b1 = 1.2
b2 = 0.8
for t in range(2, N):
    y[t] = a1 * y[t-1] + a2 * y[t-2] + b1 * u[t-1] + b2 * u[t-2] + e[t]

# Construct the regressor matrix Phi and output vector Y.
# We start from t = 3 so that we have valid y(t-1), y(t-2), u(t-1), and u(t-2)
Phi = np.zeros((N-2, 4))
Y = y[2:N]
for t in range(2, N):
    Phi[t-2, :] = [y[t-1], y[t-2], u[t-1], u[t-2]]

# Compute the least squares estimate using the normal equation:
# theta_est = (Phi' * Phi)^{-1} * Phi' * Y
theta_est = np.linalg.solve(Phi.T @ Phi, Phi.T @ Y)

# Compare Model Output Using Estimated Parameters
y_est = np.zeros(N)
for t in range(2, N):
    y_est[t] = theta_est[0] * y_est[t-1] + theta_est[1] * y_est[t-2] + theta_est[2] * u[t-1] + theta_est[3] * u[t-2]

plt.figure()
plt.plot(y, label='Actual y(t)')
plt.plot(y_est, '--', linewidth=1.5, label='Estimated y(t)')
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
