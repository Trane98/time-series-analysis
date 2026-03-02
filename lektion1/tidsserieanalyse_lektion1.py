import numpy as np
import matplotlib.pyplot as plt

# Ensure reproducibility (optional)
np.random.seed(0)

# %% Example 1: white noise
x = np.random.randn(1000)

plt.figure()
plt.plot(x)
plt.xlabel("samples")
plt.ylabel("amplitude")
plt.title("white noise")
plt.show()


# %% Example 2: Moving Average
w = np.zeros_like(x)
w[0] = x[0]
w[-1] = x[-1]

for i in range(1, len(x) - 1):
    w[i] = (x[i-1] + x[i] + x[i+1]) / 3

plt.figure()
plt.plot(x)
plt.plot(w, "--", linewidth=2)
plt.xlabel("samples")
plt.ylabel("amplitude")
plt.title("moving average")
plt.show()


# %% Example 3: Auto-regressions
xx = np.zeros_like(x)

for i in range(len(x)):
    if i == 0:
        xx[i] = x[i]
    elif i == 1:
        xx[i] = xx[i-1] + x[i]
    else:
        xx[i] = xx[i-1] - 0.9 * xx[i-2] + x[i]

plt.figure()
plt.plot(xx)
plt.xlabel("samples")
plt.ylabel("amplitude")
plt.title("auto-regressions")
plt.show()


# %% Example 4: Random walk with drift
yy = np.zeros_like(x)

for i in range(len(x)):
    if i == 0:
        yy[i] = x[i] + 0.2
    else:
        yy[i] = yy[i-1] + x[i] + 0.2

plt.figure()
plt.plot(yy)
plt.plot(np.arange(0.2, 0.2*len(x)+0.2, 0.2), linewidth=2)
plt.xlabel("samples")
plt.ylabel("amplitude")
plt.title("random walk with drift")
plt.show()


# %% Example 5: Signal in noise
t = np.linspace(0, 8*np.pi, 1000)

a = 2 * np.cos(2*t)
b = a + x

plt.figure()
plt.plot(b)
plt.plot(a, linewidth=2)
plt.xlabel("samples")
plt.ylabel("amplitude")
plt.title("signal in noise")
plt.show()