"""
03_pca.py
CIA3 — Multivariate Inventory Demand Forecasting with Correlated SKU Analysis
MCAI513B-3 | Muhammed Harshin N K | Roll No. 2548611

Purpose:
    Runs PCA (correlation-matrix based, since department sales magnitudes
    differ substantially) on the 10-department correlated demand basket to
    identify dominant common demand factor(s).

Input:
    basket_top10.csv — weekly demand matrix, Store 1, 10 correlated depts

Output:
    pca_scree_plot.png — scree plot (% variance explained + cumulative)
    Printed eigenvalues, variance explained, and PC1/PC2 loadings
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    basket = pd.read_csv('basket_top10.csv', index_col=0, parse_dates=True)
    basket[basket <= 0] = np.nan   # edge case: non-positive sales (returns week)
    basket = basket.ffill()

    X = basket.values
    Xstd = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    R = np.corrcoef(X, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    var_explained = eigvals / eigvals.sum()
    cum_var = np.cumsum(var_explained)

    print("Eigenvalues:", np.round(eigvals, 3))
    print("\nProportion of variance explained:")
    for i, (v, c) in enumerate(zip(var_explained, cum_var)):
        print(f"  PC{i+1}: {v*100:.2f}%  (cumulative {c*100:.2f}%)  eigenvalue={eigvals[i]:.3f}")

    n_kaiser = (eigvals > 1).sum()
    print(f"\nKaiser criterion (eigenvalue>1): retain {n_kaiser} component(s)")

    depts = basket.columns.tolist()
    print("\nLoadings (PC1, PC2):")
    for d, l1, l2 in zip(depts, eigvecs[:, 0], eigvecs[:, 1]):
        print(f"  Dept {d}: PC1={l1:.3f}, PC2={l2:.3f}")

    # Scree plot
    pcs = np.arange(1, len(eigvals) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(pcs, var_explained * 100, color='#2e6da4', alpha=0.85, label='% Variance Explained')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('% Variance Explained', color='#2e6da4')
    ax1.set_xticks(pcs)
    ax1.tick_params(axis='y', labelcolor='#2e6da4')

    ax2 = ax1.twinx()
    ax2.plot(pcs, cum_var * 100, color='#c0392b', marker='o', linewidth=1.5, label='Cumulative %')
    ax2.axhline(90, color='gray', linestyle=':', linewidth=1)
    ax2.set_ylabel('Cumulative % Variance Explained', color='#c0392b')
    ax2.tick_params(axis='y', labelcolor='#c0392b')
    ax2.set_ylim(0, 105)

    plt.title(f'PCA Scree Plot — Store 1, 10-Department Correlated Demand Basket\n'
              f'PC1 explains {var_explained[0]*100:.1f}% of variance (single dominant common demand factor)',
              fontsize=11)
    fig.tight_layout()
    plt.savefig('pca_scree_plot.png', dpi=150)
    print("\nSaved: pca_scree_plot.png")


if __name__ == '__main__':
    main()
