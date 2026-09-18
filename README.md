# Multivariate Inventory Demand Forecasting with Correlated SKU Analysis

**Course:** MCAI513B-3, Multivariate Techniques — MSc Computational Statistics & Applied AI, CHRIST (Deemed to be University)
**Student:** Muhammed Harshin N K | Roll No. 2548611
**Instructor:** Dr. Dibu A S

## Problem

Retail SKU/department demand is often correlated (e.g. a spike in one product category drives lagged demand in related categories). Forecasting each department independently ignores this structure and leads to overstock/stockout cycles. This project jointly models demand across a basket of correlated departments and builds a correlated reorder-point system.

## Dataset

Kaggle — [Walmart Recruiting: Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting)
`train.csv`, `features.csv`, `stores.csv` (not committed here due to file size — download from the Kaggle link above and place in `data/raw/`).

**SKU basket used:** Store 1, 10 departments (55, 14, 29, 7, 82, 5, 6, 32, 23, 46) selected via hierarchical clustering on department-level demand correlation (mean pairwise |ρ| ≈ 0.75), out of a 16-department correlated cluster.

## Techniques applied

| Technique | Purpose | Status |
|---|---|---|
| MVN inference (Mardia's test, Mahalanobis distance) | Test joint-normality assumption; flag multivariate demand-spike outliers | Done |
| PCA | Confirm dominant common demand factor across the basket | Done |
| MANOVA | Test holiday vs. non-holiday mean demand vector | In progress |
| VAR | Joint forecasting across the correlated basket | In progress |
| Reorder-point logic | Combine VAR forecast + residual covariance for correlated safety stock | In progress |

## Key results so far

- **MVN assumption:** Rejected by Mardia's test even after log-transformation (skewness chi2=1070.3, kurtosis Z=14.5, both p<0.0001) — expected for retail data with strong holiday effects; documented as a limitation, log-demand used as a practical approximation for downstream covariance-based calculations.
- **Mahalanobis outlier detection:** 12/143 weeks flagged as multivariate demand-spike outliers (log scale), clustering strongly around Thanksgiving and Christmas weeks.
- **PCA:** PC1 explains 82.0% of total variance with near-uniform loadings (0.30-0.33) across all 10 departments — confirms a single dominant common demand factor drives the basket.

## Repository structure

```
├── data/               processed data (CSVs); raw Kaggle files not committed (see .gitignore)
├── src/                analysis scripts, one per pipeline stage
├── outputs/            generated plots and result files
├── app/                Streamlit reorder-point app (in progress)
├── requirements.txt
└── README.md
```

## How to run
https://multivariate-inventory-demand-forecasting-khmuauucwrhzpc6hva9g.streamlit.app/

```bash
pip install -r requirements.txt
python src/02_mvn_mahalanobis.py
```

## AI-assistance disclosure

Statistical methodology and code were developed with AI-assisted drafting (Claude, Anthropic). All formulas, test statistics, and outputs were verified against course-taught methods (Mardia's test, Mahalanobis distance, PCA eigendecomposition, MANOVA, VAR) as per academic integrity requirements for this assignment.

## Author

Muhammed Harshin N K — MSc Computational Statistics & Applied AI, CHRIST (Deemed to be University)
