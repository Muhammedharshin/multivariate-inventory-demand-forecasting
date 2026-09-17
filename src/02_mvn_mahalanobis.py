"""
02_mvn_mahalanobis.py
CIA3 — Multivariate Inventory Demand Forecasting with Correlated SKU Analysis
MCAI513B-3 | Muhammed Harshin N K | Roll No. 2548611

Purpose:
    Tests the multivariate normality (MVN) assumption on the correlated
    10-department demand basket (Store 1, Walmart Recruiting - Store Sales
    Forecasting dataset) using Mardia's multivariate skewness/kurtosis test,
    then computes Mahalanobis distances to flag multivariate demand-spike
    outlier weeks.

Input:
    basket_top10.csv  — weekly demand matrix, Store 1, 10 correlated depts
                          (produced by 01_eda_basket_selection.py)

Output:
    basket_log_D2.csv    — demand matrix + Mahalanobis D^2 per week (log scale)
    mahalanobis_plot.png — labelled time-series plot with outlier flags
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALPHA = 0.05


def mardia_test(X: np.ndarray):
    """Mardia's multivariate skewness and kurtosis test for MVN.

    Returns (b1p, skew_stat, skew_pval, b2p, kurt_z, kurt_pval, D2)
    where D2 is the vector of squared Mahalanobis distances (= diag(G)).
    """
    n, p = X.shape
    Xbar = X.mean(axis=0)
    Xc = X - Xbar
    S = np.cov(X, rowvar=False, bias=True)          # MLE covariance (n in denom)
    Sinv = np.linalg.inv(S)
    G = Xc @ Sinv @ Xc.T                             # n x n Gram-type matrix
    g_ii = np.diag(G)                                # squared Mahalanobis distances

    b1p = (G ** 3).sum() / (n ** 2)                  # multivariate skewness
    b2p = (g_ii ** 2).sum() / n                       # multivariate kurtosis

    df_skew = p * (p + 1) * (p + 2) / 6
    skew_stat = n * b1p / 6
    skew_pval = 1 - stats.chi2.cdf(skew_stat, df_skew)

    kurt_expected = p * (p + 2)
    kurt_var = 8 * p * (p + 2) / n
    kurt_z = (b2p - kurt_expected) / np.sqrt(kurt_var)
    kurt_pval = 2 * (1 - stats.norm.cdf(abs(kurt_z)))

    return b1p, skew_stat, df_skew, skew_pval, b2p, kurt_z, kurt_pval, g_ii


def main():
    basket = pd.read_csv('basket_top10.csv', index_col=0, parse_dates=True)

    # Edge case: non-positive sales values (returns weeks) cannot be log-transformed.
    # Treated as missing and forward-filled, per the Section A4 pseudo-algorithm.
    n_invalid = (basket <= 0).sum().sum()
    if n_invalid:
        print(f"Note: {n_invalid} non-positive value(s) found — forward-filled as missing.")
        basket[basket <= 0] = np.nan
        basket = basket.ffill()

    # --- Raw levels ---
    X_raw = basket.values
    res_raw = mardia_test(X_raw)
    print("=== Mardia's Test — RAW demand levels ===")
    print(f"Skewness: chi2={res_raw[1]:.2f} (df={res_raw[2]:.0f}), p={res_raw[3]:.4g}")
    print(f"Kurtosis: Z={res_raw[5]:.2f}, p={res_raw[6]:.4g}")

    # --- Log-transformed (standard variance-stabilising transform for sales data) ---
    X_log = np.log(basket.values)
    res_log = mardia_test(X_log)
    print("\n=== Mardia's Test — LOG-transformed demand ===")
    print(f"Skewness: chi2={res_log[1]:.2f} (df={res_log[2]:.0f}), p={res_log[3]:.4g}")
    print(f"Kurtosis: Z={res_log[5]:.2f}, p={res_log[6]:.4g}")

    p_dim = X_log.shape[1]
    chi2_crit = stats.chi2.ppf(1 - ALPHA, df=p_dim)
    D2 = res_log[7]

    basket_out = basket.copy()
    basket_out['D2_log'] = D2
    basket_out.to_csv('basket_log_D2.csv')

    flagged = basket_out.index[D2 > chi2_crit]
    print(f"\nMahalanobis outliers (log scale, alpha={ALPHA}): "
          f"{len(flagged)}/{len(basket_out)} weeks flagged (threshold D2={chi2_crit:.2f})")
    for d, v in zip(flagged, D2[D2 > chi2_crit]):
        print(f"  {d.date()}: D2={v:.2f}")

    # --- Visualization (Section B5 deliverable) ---
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(basket_out.index, D2, color='#1f4e79', linewidth=1.3, label='Mahalanobis $D^2_t$')
    ax.axhline(chi2_crit, color='#c0392b', linestyle='--', linewidth=1.2,
               label=f'$\\chi^2_{{{p_dim},0.95}}$ threshold = {chi2_crit:.1f}')
    out_pts = basket_out.loc[flagged]
    ax.scatter(out_pts.index, out_pts['D2_log'], color='#c0392b', zorder=5, s=40,
               label='Flagged demand-spike weeks')
    ax.set_title('Multivariate Demand-Spike Detection — Mahalanobis Distance\n'
                  '(Store 1, 10-Department Correlated Basket, Log-Transformed Demand)', fontsize=12)
    ax.set_xlabel('Week')
    ax.set_ylabel('Mahalanobis $D^2_t$')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig('mahalanobis_plot.png', dpi=150)
    print("\nSaved: basket_log_D2.csv, mahalanobis_plot.png")


if __name__ == '__main__':
    main()
