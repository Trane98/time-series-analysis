"""
ARX(2,2) Model Simulation - Naive Implementation

ARX model: y_t = a1*y_{t-1} + a2*y_{t-2} + b1*x_{t-1} + b2*x_{t-2} + epsilon_t

Where:
- AR coefficients: a1=0.5, a2=0.3
- X coefficients (exogenous): b1=1.2, b2=0.8
- epsilon_t ~ N(0, 1) (white noise)
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 500
a1, a2 = 0.5, 0.3  # AR coefficients
b1, b2 = 1.2, 0.8  # X coefficients

# Generate exogenous input X (random white noise)
X = np.random.normal(0, 1, n_samples)

# Initialize y with zeros
y = np.zeros(n_samples)

# Generate noise
epsilon = np.random.normal(0, 1, n_samples)

# Simulate ARX(2,2) model
for t in range(2, n_samples):
    y[t] = a1 * y[t-1] + a2 * y[t-2] + b1 * X[t-1] + b2 * X[t-2] + epsilon[t]

# Display results
print("=" * 60)
print("ARX(2,2) Model Simulation")
print("=" * 60)
print(f"Number of samples: {n_samples}")
print(f"AR coefficients: a1={a1}, a2={a2}")
print(f"X coefficients: b1={b1}, b2={b2}")
print(f"\nFirst 10 samples of y:")
print(y[:10])
print(f"\nStatistics of y:")
print(f"  Mean: {y.mean():.4f}")
print(f"  Std Dev: {y.std():.4f}")
print(f"  Min: {y.min():.4f}")
print(f"  Max: {y.max():.4f}")

# Plot the simulated series
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Plot 1: Full time series
axes[0].plot(y, linewidth=1, alpha=0.8)
axes[0].set_title('ARX(2,2) Simulated Time Series', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('y(t)')
axes[0].grid(True, alpha=0.3)

# Plot 2: Exogenous input
axes[1].plot(X, linewidth=1, alpha=0.8, color='orange')
axes[1].set_title('Exogenous Input X(t)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('X(t)')
axes[1].grid(True, alpha=0.3)

# Plot 3: First 100 samples for detail
axes[2].plot(y[:100], linewidth=1, alpha=0.8, marker='o', markersize=3)
axes[2].set_title('First 100 Samples (Zoom)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('y(t)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('arx_22_simulation.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'arx_22_simulation.png'")
plt.show()
