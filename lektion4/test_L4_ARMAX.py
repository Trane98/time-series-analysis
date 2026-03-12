import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ============================================================
# Generate the data
# ============================================================
np.random.seed(0)

N = 200               # Number of data points
Ts = 1                # Sampling time
t = np.arange(0, N * Ts, Ts)   # Time vector
noise_var = 0.1       # Variance of the noise

# True system (data generating process)
# MATLAB:
# a = [1 -1.5 0.7]
# b = [0 1 0.5]
# c = [1 -1 0.2]
a = np.array([1, -1.5, 0.7])    # AR polynomial
b = np.array([0, 1, 0.5])       # Input polynomial
c = np.array([1, -1, 0.2])      # MA/noise polynomial

# Input signal
u_in = np.random.randn(N)

# White Gaussian noise
e = np.sqrt(noise_var) * np.random.randn(N)

# ============================================================
# Simulate:
# A(q) y(t) = B(q) u(t) + C(q) e(t)
#
# Equivalent:
# y = lfilter(B, A, u) + lfilter(C, A, e)
# ============================================================
y_u = lfilter(b, a, u_in)
y_e = lfilter(c, a, e)
y = y_u + y_e

# ============================================================
# ARMAX estimation helper
# We use SARIMAX with:
#   order=(p,0,q)
#   exog = lagged input terms up to order nb
#
# This is a practical Python replacement for MATLAB armax(DAT,[p nb q nk])
# ============================================================
def build_exog(u, nb=1, nk=1):
    N = len(u)
    X = np.zeros((N, nb))
    for j in range(nb):
        lag = nk + j
        X[lag:, j] = u[:N-lag]
    return X

def fit_armax(y, u, p, nb, q, nk=1):
    exog = build_exog(u, nb=nb, nk=nk)

    model = SARIMAX(
        y,
        exog=exog,
        order=(p, 0, q),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # More iterations + explicit optimizer
    res = model.fit(method="lbfgs", maxiter=500, disp=False)

    # Reject non-converged models
    if not res.mle_retvals.get("converged", False):
        raise RuntimeError(
            f"Model did not converge: p={p}, nb={nb}, q={q}, "
            f"retvals={res.mle_retvals}"
        )

    return res, exog

# ============================================================
# Define maximum orders to consider
# ============================================================
maxP = 3   # Maximum AR order
maxQ = 3   # Maximum MA order
maxU = 3   # Maximum input order

aicValues = np.full((maxP, maxQ, maxU), np.nan)
bicValues = np.full((maxP, maxQ, maxU), np.nan)

# ============================================================
# Loop over possible model orders
# MATLAB: m = armax(DAT,[p u q 1])
# Python approximation: SARIMAX(order=(p,0,q), exog=lagged input of size u)
# ============================================================
aicValues = np.full((maxP, maxQ, maxU), np.nan)
bicValues = np.full((maxP, maxQ, maxU), np.nan)

for p in range(1, maxP + 1):
    for q in range(1, maxQ + 1):
        for nb in range(1, maxU + 1):
            try:
                res, exog = fit_armax(y, u_in, p, nb, q, nk=1)
                aicValues[p-1, q-1, nb-1] = res.aic
                bicValues[p-1, q-1, nb-1] = res.bic
            except Exception:
                pass

# ============================================================
# Plot AIC values
# ============================================================
plt.figure()
plt.plot(aicValues.ravel())
plt.title("AIC values over all tested models")
plt.xlabel("Model index")
plt.ylabel("AIC")
plt.grid(True)
plt.show()

# ============================================================
# Find minimum AIC / BIC
# ============================================================
minAIC = np.nanmin(aicValues)
minBIC = np.nanmin(bicValues)

best_idx = np.unravel_index(np.nanargmin(bicValues), bicValues.shape)
r = best_idx[0] + 1   # p
c_ord = best_idx[1] + 1   # q
v = best_idx[2] + 1   # input order nb

print("Best model by BIC:")
print(f"p = {r}, q = {c_ord}, nb = {v}")
print(f"minAIC = {minAIC:.4f}")
print(f"minBIC = {minBIC:.4f}")

# ============================================================
# Compare the optimal model with data
# MATLAB: m = armax(DAT,[r v c 1]); compare(DAT,m)
# ============================================================
best_model, best_exog = fit_armax(y, u_in, r, v, c_ord, nk=1)
y_hat_best = best_model.fittedvalues

plt.figure()
plt.plot(y, label="Measured output y")
plt.plot(y_hat_best, '--', label=f"Best ARMAX({r},{v},{c_ord}) fit")
plt.title("Comparison: data vs best model")
plt.xlabel("Time")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# Try out the true model structure: ARMAX(2,2,2)
# MATLAB: m = armax(DAT,[2 2 2 1])
# ============================================================
true_model, true_exog = fit_armax(y, u_in, 2, 2, 2, nk=1)
y_hat_true = true_model.fittedvalues

plt.figure()
plt.plot(y, label="Measured output y")
plt.plot(y_hat_true, '--', label="ARMAX(2,2,2) fit")
plt.title("Comparison: data vs ARMAX(2,2,2)")
plt.xlabel("Time")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# ARMA(2,2): MATLAB m1 = armax(DAT,[2 0 2 1])
# In Python: SARIMAX without exogenous input
# ============================================================
arma_model = SARIMAX(
    y,
    order=(2, 0, 2),
    trend='n',
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)
y_hat_arma = arma_model.fittedvalues

# ============================================================
# ARX(2,2): MATLAB m2 = arx(DAT,[2 2 1])
# LS estimation:
# y(t) = a1*y(t-1) + a2*y(t-2) + b1*u(t-1) + b2*u(t-2) + e(t)
# ============================================================
Phi = np.zeros((N - 2, 4))
Y = y[2:]

for k in range(2, N):
    Phi[k - 2, :] = [y[k - 1], y[k - 2], u_in[k - 1], u_in[k - 2]]

theta_arx = np.linalg.lstsq(Phi, Y, rcond=None)[0]

y_hat_arx = np.zeros(N)
for k in range(2, N):
    y_hat_arx[k] = (
        theta_arx[0] * y_hat_arx[k - 1]
        + theta_arx[1] * y_hat_arx[k - 2]
        + theta_arx[2] * u_in[k - 1]
        + theta_arx[3] * u_in[k - 2]
    )

# ============================================================
# Compare models
# ============================================================
plt.figure()
plt.plot(y, label="Measured y", linewidth=1.5)
plt.plot(y_hat_true, '--', label="ARMAX(2,2,2)")
plt.plot(y_hat_arma, ':', label="ARMA(2,2)")
plt.plot(y_hat_arx, '-.', label="ARX(2,2)")
plt.title("Comparison of different models")
plt.xlabel("Time")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# Print summaries
# ============================================================
print("\nBest BIC model summary:")
print(best_model.summary())

print("\nARMAX(2,2,2) summary:")
print(true_model.summary())

print("\nARMA(2,2) summary:")
print(arma_model.summary())

print("\nARX(2,2) LS parameters:")
print(theta_arx)