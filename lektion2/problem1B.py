"""
Problem 1B — Moving Average (MA) proces, sample ACF til lag 20 (N=500),
og sammenligning med white noise fra 1A.

VIGTIG TEORI:
- White noise: x_t = w_t, hvor w_t ~ iid N(0,1)
  => Sand ACF: rho(0)=1 og rho(h)=0 for h>0

- MA(q)-proces (generel form):
    x_t = w_t + θ1 w_{t-1} + ... + θq w_{t-q}
  hvor w_t ~ iid N(0, σ^2)

  => MA(q) har en KENDETEGNS-signatur i ACF:
     ACF "cutoff'er" efter lag q.
     Dvs. teoretisk: rho(h)=0 for h>q

  Intuition:
  x_t og x_{t-h} deler kun de samme w-led hvis h <= q.
  Hvis h > q deler de ingen fælles noise-led -> ingen korrelation.

- Sample ACF varierer omkring den sande ACF pga. sampling-støj.
  For white noise bruger man ofte 95% bounds ±1.96/sqrt(N).
  (For MA er bounds stadig en nyttig tommelregel visuelt.)
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf


def simulate_white_noise(N: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=N)


def simulate_ma_from_weights(N: int, weights, seed: int = 42):
    """
    Simulerer en MA-proces ved at bygge:
        x_t = sum_{k=0..q} weights[k] * w_{t-k}
    hvor w_t ~ iid N(0,1)

    Hvis weights har længde m:
      - q = m-1
      - fx weights=[1, theta] svarer til MA(1): x_t = w_t + theta*w_{t-1}
      - fx weights=[1/3,1/3,1/3] svarer til 3-punkts moving average (MA(2))
    """
    rng = np.random.default_rng(seed)

    weights = np.asarray(weights, dtype=float)
    q = len(weights) - 1

    # Vi skal bruge ekstra w i starten for at kunne lave de første x_t
    w = rng.normal(0.0, 1.0, size=N + q)

    x = np.zeros(N)
    for t in range(N):
        # x_t = w_{t+q}*w0 + w_{t+q-1}*w1 + ... + w_{t}*w_q
        # så vi bruger en "bagud" vægtning
        x[t] = np.dot(weights, w[t : t + len(weights)][::-1])

    return x


def plot_series_and_acf(x, title: str, nlags: int = 20):
    N = len(x)
    r = acf(x, nlags=nlags, fft=True)
    ci = 1.96 / np.sqrt(N)  # klassisk 95% tommelregel-bånd

    lags = np.arange(nlags + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(x, linewidth=1)
    plt.title(f"{title} — time series (N={N})")
    plt.xlabel("t")
    plt.ylabel("x_t")
    plt.tight_layout()
    plt.show(block=False)

    plt.figure(figsize=(9, 4.5))
    plt.axhline(0, linewidth=1)
    plt.vlines(lags, 0, r, linewidth=2)
    plt.scatter(lags, r, s=30)
    plt.axhline(ci, linestyle="--", linewidth=1)
    plt.axhline(-ci, linestyle="--", linewidth=1)

    plt.title(f"{title} — sample ACF (lags 0..{nlags})\nApprox 95% bounds: ±{ci:.3f}")
    plt.xlabel("Lag h")
    plt.ylabel("Sample ACF  r̂(h)")
    plt.xticks(lags)
    plt.tight_layout()
    plt.show(block=False)

    outside = np.sum(np.abs(r[1:]) > ci)
    print(f"{title}: spikes outside bounds (lags 1..{nlags}): {outside}/{nlags}")
    print("-" * 60)


def compare_acf_two_series(x1, label1, x2, label2, nlags: int = 20):
    """
    Ekstra: plot ACF for to serier i samme figur, så forskellen er tydelig.
    """
    r1 = acf(x1, nlags=nlags, fft=True)
    r2 = acf(x2, nlags=nlags, fft=True)
    lags = np.arange(nlags + 1)

    plt.figure(figsize=(9, 4.5))
    plt.axhline(0, linewidth=1)
    plt.plot(lags, r1, marker="o", linewidth=2, label=label1)
    plt.plot(lags, r2, marker="o", linewidth=2, label=label2)
    plt.title("ACF comparison")
    plt.xlabel("Lag h")
    plt.ylabel("Sample ACF  r̂(h)")
    plt.xticks(lags)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)


if __name__ == "__main__":
    N = 500
    nlags = 20
    seed = 1

    # --- VÆLG DIN MOVING AVERAGE HER ---
    # Mest klassiske "moving average smoothing" med vindue på 3:
    # x_t = (w_t + w_{t-1} + w_{t-2}) / 3
    # -> dette er en MA(2) proces (q=2), så ACF bør være ~0 for lag > 2 (teoretisk).
    weights = [1/3, 1/3, 1/3]

    # Hvis din slide 14 i stedet bruger MA(1), kan du fx bruge:
    # weights = [1.0, 0.7]   # x_t = w_t + 0.7*w_{t-1}

    # 1) White noise (til sammenligning)
    wn = simulate_white_noise(N=N, seed=seed)

    # 2) MA-proces
    ma = simulate_ma_from_weights(N=N, weights=weights, seed=seed)

    # 3) Plot + ACF for hver
    plot_series_and_acf(wn, title="White noise (baseline, 1A)", nlags=nlags)
    plot_series_and_acf(ma, title=f"Moving Average / MA(q) with weights={weights}", nlags=nlags)

    # 4) Direkte ACF-sammenligning i samme plot
    compare_acf_two_series(
        wn, "White noise",
        ma, f"MA weights={weights}",
        nlags=nlags
    )

    # Show all plots and keep them open
    plt.show()