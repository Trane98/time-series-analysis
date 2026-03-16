import numpy as np
import matplotlib.pyplot as plt

# Generate Synthetic Data
N = 500  # Number of samples
theta_true = np.array([5, 1])  # True parameters
Z = np.column_stack([np.ones(N), np.random.randn(N, 1).flatten()])  # Input data (first column is bias)
w = 0.5 * np.random.randn(N)  # Measurement noise
y = Z @ theta_true + w  # Output data

# Initialize Recursive Least Squares (RLS)
n = 2  # Number of parameters
r = 1  # Dimension of output
theta_hat = np.array([1, 2])  # Initial estimate of parameters
P = np.eye(n)  # Initial covariance matrix
R = (0.5 ** 2) * np.eye(r)  # Measurement noise covariance
Q = np.eye(n) * 1e-12  # Process noise covariance
I = np.eye(n)

# Run RLS Algorithm
theta_history = np.zeros((n, N))  # Store estimates
for k in range(N):
    Zk = Z[k, :]  # Current input vector
    yk = y[k]  # Current measurement
    Kk = (P @ Zk) / (R[0, 0] + Zk @ P @ Zk)  # Compute the gain (R is scalar)
    theta_hat = theta_hat + Kk * (yk - Zk @ theta_hat)  # Update parameter estimate
    P = (I - np.outer(Kk, Zk)) @ P @ (I - np.outer(Kk, Zk)).T + np.outer(Kk, R @ Kk) + Q  # Update covariance matrix
    theta_history[:, k] = theta_hat  # Store history

# Plot the results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(y, label='Measured output')
plt.plot(Z @ theta_true, label='True output')
plt.legend()
plt.title('Output Data')

plt.subplot(1, 3, 2)
plt.plot(theta_history[0, :], label='Estimated theta1')
plt.axhline(y=theta_true[0], color='r', linestyle='--', label='True theta1')
plt.legend()
plt.title('Parameter 1 Estimation')

plt.subplot(1, 3, 3)
plt.plot(theta_history[1, :], label='Estimated theta2')
plt.axhline(y=theta_true[1], color='r', linestyle='--', label='True theta2')
plt.legend()
plt.title('Parameter 2 Estimation')

plt.tight_layout()
plt.show()

print("Final estimate:", theta_hat)
print("True parameters:", theta_true)
