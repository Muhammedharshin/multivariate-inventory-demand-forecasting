"""
05_var_reorder.py
CIA3 — Multivariate Inventory Demand Forecasting with Correlated SKU Analysis
MCAI513B-3 | Muhammed Harshin N K | Roll No. 2548611

Purpose:
    Fits a Vector Autoregression on the 10-department correlated demand
    basket, forecasts next-period demand, and computes correlation-aware
    reorder points using the VAR residual covariance matrix.

Lag selection:
    BIC/HQIC both select lag=1 (AIC prefers lag=8, but with p=10 and only
    n=143 observations that implies 800 parameters -- clear overfitting.
    BIC's stronger parameter penalty is the appropriate criterion here).

Known limitation:
    VAR(1) residuals show remaining autocorrelation (Ljung-Box p<0.05) in
    2-5 of the 10 departments. VAR(2) reduces this but has 200 parameters
    against 143 observations, an even worse overfitting risk. VAR(1) is
    retained as the primary model; this trade-off is documented rather
    than hidden.

Input:
    basket_top10.csv

Output:
    reorder_points.csv       — forecast, safety stock, reorder point per dept
    var_residual_covariance.csv — residual covariance matrix
    Printed residual correlation matrix (evidence for correlated risk)
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox

SERVICE_LEVEL = 0.95


def main():
    basket = pd.read_csv('basket_top10.csv', index_col=0, parse_dates=True)
    basket[basket <= 0] = np.nan
    basket = basket.ffill()

    model = VAR(basket)

    print("=== Lag order selection ===")
    print(model.select_order(maxlags=8).summary())

    fitted = model.fit(1)
    print("\nVAR(1) stable:", fitted.is_stable())

    print("\n=== Residual Ljung-Box test (lag=8) ===")
    resid = fitted.resid
    for col in resid.columns:
        lb = acorr_ljungbox(resid[col], lags=[8], return_df=True)
        p = lb['lb_pvalue'].values[0]
        flag = " <-- residual autocorrelation remains" if p < 0.05 else ""
        print(f"  Dept {col}: p={p:.4f}{flag}")

    forecast = fitted.forecast(basket.values[-fitted.k_ar:], steps=1)[0]
    sigma_resid = fitted.sigma_u

    z = stats.norm.ppf(SERVICE_LEVEL)
    safety_stock = z * np.sqrt(np.diag(sigma_resid.values))
    reorder_point = forecast + safety_stock

    results = pd.DataFrame({
        'Dept': basket.columns,
        'Forecast_next_week': forecast.round(2),
        'Safety_stock_95pct': safety_stock.round(2),
        'Reorder_point': reorder_point.round(2)
    })
    print(f"\n=== Reorder points (service level={SERVICE_LEVEL}) ===")
    print(results.to_string(index=False))
    results.to_csv('reorder_points.csv', index=False)
    sigma_resid.to_csv('var_residual_covariance.csv')

    corr_resid = sigma_resid.values / np.outer(
        np.sqrt(np.diag(sigma_resid.values)), np.sqrt(np.diag(sigma_resid.values)))
    print("\n=== Residual correlation matrix ===")
    print("(off-diagonal values show cross-department demand SHOCK correlation,")
    print(" i.e. correlation remaining even after the VAR's own lag structure)")
    print(pd.DataFrame(corr_resid, index=basket.columns, columns=basket.columns).round(2))
    print(f"\nMean off-diagonal residual correlation: "
          f"{(corr_resid.sum() - len(basket.columns)) / (len(basket.columns)**2 - len(basket.columns)):.3f}")


if __name__ == '__main__':
    main()
