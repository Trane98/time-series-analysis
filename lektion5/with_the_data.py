import numpy as np
import matplotlib.pyplot as plt

# Load real data
data = np.loadtxt('DataRPM.txt', delimiter=',')
N = len(data)
y = data[:, 0]  # Output (RPM)
u = data[:, 1]  # Input (constant in this case)
Z = np.column_stack([np.ones(N), u])  # Regressors: bias and input

# Initialize Recursive Least Squares (RLS)
n = 2  # Number of parameters
r = 1  # Dimension of output
theta_hat = np.array([1.0, 2.0])  # Initial estimate of parameters
P = np.eye(n)  # Initial covariance matrix
R = (0.5 ** 2) * np.eye(r)  # Measurement noise covariance
Q = np.eye(n) * 1e-12  # Process noise covariance
I = np.eye(n)

# Run RLS Algorithm
theta_history = np.zeros((n, N))  # Store estimates
y_hat_history = np.zeros(N)  # Store predicted outputs
for k in range(N):
    Zk = Z[k, :]  # Current input vector
    yk = y[k]  # Current measurement
    y_hat = Zk @ theta_hat  # Predicted output
    y_hat_history[k] = y_hat
    Kk = (P @ Zk) / (R[0, 0] + Zk @ P @ Zk)  # Compute the gain
    theta_hat = theta_hat + Kk * (yk - y_hat)  # Update parameter estimate
    P = (I - np.outer(Kk, Zk)) @ P @ (I - np.outer(Kk, Zk)).T + R[0, 0] * np.outer(Kk, Kk) + Q  # Update covariance matrix
    theta_history[:, k] = theta_hat  # Store history

# Plot the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.plot(y, label='Measured output (RPM)')
plt.plot(y_hat_history, label='Estimated output')
plt.legend()
plt.title('Output Data')

plt.subplot(1, 4, 2)
plt.plot(theta_history[0, :], label='Estimated theta0 (bias)')
plt.title('Parameter 0 Estimation')

plt.subplot(1, 4, 3)
plt.plot(theta_history[1, :], label='Estimated theta1')
plt.title('Parameter 1 Estimation')

plt.subplot(1, 4, 4)
plt.plot(y - y_hat_history, label='Residuals')
plt.title('Residuals')

plt.tight_layout()
plt.show()

print("Final estimate:", theta_hat)
print("Final P matrix:")
print(P)