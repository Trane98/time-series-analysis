import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

N = 200
t = np.arange(1, N+1)

w1 = np.random.randn(N)
w2 = np.random.randn(N)

s1 = 10*np.exp(-(t-100)/20)  * np.cos(2*np.pi*t/4)
x1 = s1 + w1

s2 = 10*np.exp(-(t-100)/200) * np.cos(2*np.pi*t/4) - 0.1
x2 = s2 + w2

plt.figure(figsize=(12,6))

# Model A
plt.subplot(2,1,1)
plt.plot(x1, label="x_k (Model A)")
plt.plot(s1, linewidth=2, label="s_k (Model A)")
plt.title("Model A")
plt.legend()

# Model B
plt.subplot(2,1,2)
plt.plot(x2, label="x_k (Model B)")
plt.plot(s2, linewidth=2, label="s_k (Model B)")
plt.title("Model B")
plt.legend()

plt.tight_layout()
plt.show()