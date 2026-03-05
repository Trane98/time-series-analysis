"""
Problem 1A — Gaussian white noise (N=500 og N=50), sample ACF til lag 20.

TEORI (kort):
- White noise: x_t ~ iid N(0, 1)  (uafhængige observationer)
- Sand (teoretisk) ACF:
    rho(0) = 1
    rho(h) = 0 for h > 0
- Sample ACF (estimat) vil ikke blive præcis 0 pga. tilfældig sampling-variation.
- For white noise gælder approx:
    r_hat(h) ≈ N(0, 1/N)  for h>0
  => Standardafvigelse ≈ 1/sqrt(N)
  => 95% "signifikansgrænser" ≈ ± 1.96/sqrt(N)
  Hvis sample ACF spikes ligger indenfor grænserne, er det typisk bare støj.
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

def plot_white_noise_acf(N: int, nlags: int = 20, seed: int = 42):
    # Sæt random seed så du får samme eksempel hver gang (godt til rapport/aflevering).
    rng = np.random.default_rng(seed)

    # 1) Simulér Gaussian white noise:
    #    iid N(0, 1) -> ingen tidslig afhængighed (uafhængige samples).
    x = rng.normal(loc=0.0, scale=1.0, size=N)

    # 2) Beregn sample ACF op til lag 20.
    #    acf(...) returnerer r_hat(0..nlags).
    #    fft=True gør det hurtigere for store N (har ingen betydning for tolkningen).
    r = acf(x, nlags=nlags, fft=True)

    # 3) Teoretiske 95%-grænser for white noise (tommelregel):
    #    r_hat(h) ~ N(0, 1/N) -> 95% ca. ±1.96/sqrt(N)
    ci = 1.96 / np.sqrt(N)

    # Plot sample ACF som “sticks”
    lags = np.arange(nlags + 1)

    plt.figure(figsize=(9, 4.5))
    plt.axhline(0, linewidth=1)

    # Sticks for hver lag
    plt.vlines(lags, 0, r, linewidth=2)
    plt.scatter(lags, r, s=30)

    # 95% konfidensbånd (approx)
    plt.axhline(ci, linestyle="--", linewidth=1)
    plt.axhline(-ci, linestyle="--", linewidth=1)

    # 4) Vis også den “sande” ACF i ord:
    #    For white noise er sand ACF 0 for h>0, så det vi forventer:
    #    - r_hat(0) ~ 1
    #    - r_hat(1..20) omkring 0, med tilfældige udsving,
    #      hvor de fleste ligger indenfor ±1.96/sqrt(N).

    plt.title(f"Sample ACF for Gaussian white noise (N={N}), lags 0..{nlags}\n"
              f"Approx 95% bounds: ±{ci:.3f}")
    plt.xlabel("Lag h")
    plt.ylabel("Sample ACF  r̂(h)")
    plt.xticks(lags)
    plt.tight_layout()

    # En lille tekst-output der gør det nemt at kommentere i rapport:
    print(f"N={N}")
    print(f"95% bounds approx: ±{ci:.3f}")
    # Hvor mange af lags 1..nlags ligger udenfor båndet?
    outside = np.sum(np.abs(r[1:]) > ci)
    print(f"ACF spikes outside bounds (lags 1..{nlags}): {outside} / {nlags}")
    print("-" * 60)


if __name__ == "__main__":
    # Problem 1A kræver N=500 og derefter N=50
    plot_white_noise_acf(N=500, nlags=20, seed=1)
    plot_white_noise_acf(N=50, nlags=20, seed=1)
    plt.show()