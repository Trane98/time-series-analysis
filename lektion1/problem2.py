import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

N = 100
w = np.random.randn(N)  # w_k ~ N(0,1)

# x_{-2}, x_{-1}, x_0 = 0  (vi håndterer det ved padding)
x = np.zeros(N)

for k in range(N):
    if k - 2 >= 0:
        x[k] = -0.9 * x[k-2] + w[k]
    else:
        # for k = 0,1: x_{k-2} refers to x_-2, x_-1 which are 0
        x[k] = w[k]

# Moving average: y_k = (x_k + x_{k-1} + x_{k-2} + x_{k-3})/4
y = np.zeros(N)
for k in range(N):
    y[k] = (x[k]
            + (x[k-1] if k-1 >= 0 else 0)
            + (x[k-2] if k-2 >= 0 else 0)
            + (x[k-3] if k-3 >= 0 else 0)) / 4

plt.figure(figsize=(12,5))
plt.plot(x, label="x_k (AR: x_k = -0.9 x_{k-2} + w_k)")
plt.plot(y, linewidth=2, label="y_k (4-punkt moving average)")
plt.title("Problem 2: Sammenligning af x_k og y_k")
plt.xlabel("k")
plt.ylabel("amplitude")
plt.legend()
plt.tight_layout()
plt.show()