"""
04_manova.py
CIA3 — Multivariate Inventory Demand Forecasting with Correlated SKU Analysis
MCAI513B-3 | Muhammed Harshin N K | Roll No. 2548611

Purpose:
    Tests whether the mean demand vector differs between holiday and
    non-holiday weeks (H0: mu_holiday = mu_non_holiday) using MANOVA.

    Note: with only 10 holiday weeks in the data, the holiday group's raw
    10-dimensional covariance matrix (S1) is singular (n1 <= p), which
    invalidates a direct MANOVA on all 10 departments. To keep the test
    well-posed, MANOVA is instead run on the top-2 PCA scores (from
    03_pca.py), which already capture ~88% of joint demand variance.
    This also links the PCA and MANOVA sections of the analysis directly.

Input:
    basket_with_holiday.csv — demand matrix + IsHoliday flag per week

Output:
    Printed test statistics: Hotelling's T^2, Wilks' Lambda, F-test,
    and Box's M test for covariance homogeneity.
"""

import pandas as pd
import numpy as np
from scipy import stats


def box_m_test(S1, S2, n1, n2, p):
    """Box's M test for equality of two covariance matrices."""
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    M = ((n1 + n2 - 2) * np.log(np.linalg.det(Sp))
         - (n1 - 1) * np.log(np.linalg.det(S1))
         - (n2 - 1) * np.log(np.linalg.det(S2)))
    c = ((2 * p ** 2 + 3 * p - 1) / (6 * (p + 1))) * (1 / (n1 - 1) + 1 / (n2 - 1) - 1 / (n1 + n2 - 2))
    chi2_stat = (1 - c) * M
    df = p * (p + 1) / 2
    pval = 1 - stats.chi2.cdf(chi2_stat, df)
    return chi2_stat, df, pval


def main():
    df = pd.read_csv('basket_with_holiday.csv', index_col=0, parse_dates=True)
    demand_cols = [c for c in df.columns if c != 'IsHoliday']

    X = df[demand_cols].values
    X[X <= 0] = np.nan
    X = pd.DataFrame(X).ffill().values
    Xlog = np.log(X)

    # Standardize and project onto top-2 principal components
    Xstd = (Xlog - Xlog.mean(axis=0)) / Xlog.std(axis=0, ddof=1)
    R = np.corrcoef(Xlog, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    scores = Xstd @ eigvecs[:, :2]

    holiday = df['IsHoliday'].astype(bool).values
    g1, g2 = scores[holiday], scores[~holiday]
    n1, n2, p = len(g1), len(g2), 2

    mean1, mean2 = g1.mean(axis=0), g2.mean(axis=0)
    S1, S2 = np.cov(g1, rowvar=False), np.cov(g2, rowvar=False)
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

    diff = (mean1 - mean2).reshape(-1, 1)
    T2 = (n1 * n2 / (n1 + n2)) * (diff.T @ np.linalg.inv(Sp) @ diff).item()
    F_stat = ((n1 + n2 - p - 1) / (p * (n1 + n2 - 2))) * T2
    df1, df2 = p, n1 + n2 - p - 1
    F_pval = 1 - stats.f.cdf(F_stat, df1, df2)
    wilks_lambda = 1 / (1 + T2 / (n1 + n2 - 2))

    print("=== MANOVA: Holiday vs Non-Holiday mean demand vector ===")
    print(f"n_holiday={n1}, n_non_holiday={n2}, p={p} (top-2 PCA scores)")
    print(f"Hotelling's T^2 = {T2:.3f}")
    print(f"Wilks' Lambda   = {wilks_lambda:.4f}")
    print(f"F-statistic     = {F_stat:.3f} (df={df1},{df2}), p-value = {F_pval:.5g}")
    print("Verdict:", "Reject H0 - mean vectors differ significantly"
          if F_pval < 0.05 else "Fail to reject H0")

    chi2_stat, dfM, pM = box_m_test(S1, S2, n1, n2, p)
    print("\n=== Box's M Test: covariance homogeneity ===")
    print(f"Chi2 = {chi2_stat:.3f} (df={dfM:.0f}), p-value = {pM:.4f}")
    print("Verdict:", "Reject H0 - covariances differ (equal-covariance assumption violated)"
          if pM < 0.05 else "Fail to reject H0 - homogeneity assumption reasonable")


if __name__ == '__main__':
    main()
