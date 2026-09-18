"""
app.py — Correlated SKU Reorder-Point Dashboard
CIA3 — Multivariate Inventory Demand Forecasting with Correlated SKU Analysis
MCAI513B-3 | Muhammed Harshin N K | Roll No. 2548611

Run locally:  streamlit run app.py
Deploy:       push to GitHub, connect repo on share.streamlit.io (Streamlit Community Cloud)
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.api import VAR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

st.set_page_config(page_title="Correlated SKU Reorder-Point Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    # Includes the IsHoliday column so the MANOVA tab works out of the box
    return pd.read_csv("data/basket_with_holiday.csv", index_col=0, parse_dates=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[df <= 0] = np.nan
    return df.ffill().dropna()


# ---------------------------------------------------------------------------
# Analysis functions (mirror src/02-05 scripts)
# ---------------------------------------------------------------------------
def mardia_test(X: np.ndarray):
    n, p = X.shape
    Xbar = X.mean(axis=0)
    Xc = X - Xbar
    S = np.cov(X, rowvar=False, bias=True)
    Sinv = np.linalg.inv(S)
    G = Xc @ Sinv @ Xc.T
    g_ii = np.diag(G)
    b1p = (G ** 3).sum() / (n ** 2)
    b2p = (g_ii ** 2).sum() / n
    df_skew = p * (p + 1) * (p + 2) / 6
    skew_stat = n * b1p / 6
    skew_pval = 1 - stats.chi2.cdf(skew_stat, df_skew)
    kurt_expected = p * (p + 2)
    kurt_z = (b2p - kurt_expected) / np.sqrt(8 * p * (p + 2) / n)
    kurt_pval = 2 * (1 - stats.norm.cdf(abs(kurt_z)))
    return skew_stat, df_skew, skew_pval, kurt_z, kurt_pval, g_ii


def run_pca(X: np.ndarray):
    Xstd = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    R = np.corrcoef(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


@st.cache_data
def fit_var(df: pd.DataFrame, lag: int, service_level: float):
    model = VAR(df.values)
    fitted = model.fit(lag)
    forecast = fitted.forecast(df.values[-fitted.k_ar:], steps=1)[0]
    sigma_resid = fitted.sigma_u
    z = stats.norm.ppf(service_level)
    safety_stock = z * np.sqrt(np.diag(sigma_resid))
    reorder_point = forecast + safety_stock
    corr_resid = sigma_resid / np.outer(np.sqrt(np.diag(sigma_resid)), np.sqrt(np.diag(sigma_resid)))
    return forecast, safety_stock, reorder_point, sigma_resid, corr_resid, fitted.is_stable()


# ---------------------------------------------------------------------------
# Sidebar — data source and settings
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")
uploaded = st.sidebar.file_uploader(
    "Upload weekly demand history (CSV: date index, one column per SKU/department)",
    type="csv"
)
service_level = st.sidebar.slider("Target service level", 0.80, 0.99, 0.95, 0.01)
lag = st.sidebar.selectbox("VAR lag order", [1, 2, 3], index=0,
                            help="BIC/HQIC selected lag=1 for the reference dataset")

if uploaded is not None:
    raw = pd.read_csv(uploaded, index_col=0, parse_dates=True)
    st.sidebar.success(f"Loaded {raw.shape[0]} weeks x {raw.shape[1]} SKUs from upload")
else:
    raw = load_default_data()
    st.sidebar.info("Using default Store 1 / 10-department Walmart basket")

demand_cols = [c for c in raw.columns if c.lower() != "isholiday"]
df = clean_data(raw[demand_cols])
n, p = df.shape

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📦 Multivariate Inventory Demand Forecasting")
st.caption("Correlated SKU Analysis — MVN · PCA · MANOVA · VAR · Reorder Points")

col1, col2, col3 = st.columns(3)
col1.metric("Weeks of data", n)
col2.metric("SKUs / departments", p)
col3.metric("Target service level", f"{service_level:.0%}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "MVN & Outliers", "PCA", "MANOVA", "VAR & Reorder Points"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Demand history")
    st.line_chart(df)

    st.subheader("Correlation matrix")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(p)); ax.set_xticklabels(df.columns, rotation=45)
    ax.set_yticks(range(p)); ax.set_yticklabels(df.columns)
    fig.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Department Demand Correlation Matrix")
    st.pyplot(fig)
    st.caption(f"Mean pairwise correlation: {(corr.values.sum() - p) / (p**2 - p):.3f}")

# ---------------------------------------------------------------------------
# Tab 2 — MVN & Mahalanobis
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Multivariate Normality Test (Mardia's)")
    X_log = np.log(df.values)
    skew_stat, df_skew, skew_p, kurt_z, kurt_p, D2 = mardia_test(X_log)

    c1, c2 = st.columns(2)
    c1.metric("Skewness χ² statistic", f"{skew_stat:.1f}", f"p={skew_p:.4g}")
    c2.metric("Kurtosis Z statistic", f"{kurt_z:.1f}", f"p={kurt_p:.4g}")
    verdict = "MVN rejected" if min(skew_p, kurt_p) < 0.05 else "Consistent with MVN"
    st.info(f"Verdict (α=0.05, log-transformed demand): **{verdict}**")

    st.subheader("Mahalanobis Distance — Demand-Spike Outlier Detection")
    chi2_crit = stats.chi2.ppf(0.95, df=p)
    flagged = df.index[D2 > chi2_crit]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, D2, color="#1f4e79", linewidth=1.2, label="Mahalanobis $D^2_t$")
    ax.axhline(chi2_crit, color="#c0392b", linestyle="--", label=f"χ² threshold ({chi2_crit:.1f})")
    ax.scatter(df.index[D2 > chi2_crit], D2[D2 > chi2_crit], color="#c0392b", zorder=5)
    ax.legend(); ax.set_ylabel("$D^2_t$"); ax.set_xlabel("Week")
    st.pyplot(fig)
    st.write(f"**{len(flagged)} of {n} weeks** flagged as multivariate demand-spike outliers.")
    st.dataframe(pd.DataFrame({"Week": flagged.date, "D²": D2[D2 > chi2_crit].round(2)}))

# ---------------------------------------------------------------------------
# Tab 3 — PCA
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Principal Component Analysis")
    eigvals, eigvecs = run_pca(df.values)  # raw levels, matching 03_pca.py / Section A report
    var_exp = eigvals / eigvals.sum()
    cum_var = np.cumsum(var_exp)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(range(1, p + 1), var_exp * 100, color="#2e6da4", alpha=0.85)
    ax1.set_xlabel("Principal Component"); ax1.set_ylabel("% Variance Explained")
    ax2 = ax1.twinx()
    ax2.plot(range(1, p + 1), cum_var * 100, color="#c0392b", marker="o")
    ax2.set_ylabel("Cumulative %"); ax2.set_ylim(0, 105)
    st.pyplot(fig)

    st.metric("PC1 variance explained", f"{var_exp[0]*100:.1f}%")
    n_kaiser = int((eigvals > 1).sum())
    st.write(f"Kaiser criterion (eigenvalue > 1): retain **{n_kaiser}** component(s)")

    loadings = pd.DataFrame({"Department": df.columns, "PC1 loading": eigvecs[:, 0].round(3),
                              "PC2 loading": eigvecs[:, 1].round(3)})
    st.dataframe(loadings)

# ---------------------------------------------------------------------------
# Tab 4 — MANOVA
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("MANOVA — Holiday vs Non-Holiday Mean Demand")
    st.caption("Run on top-2 PCA scores (raw 10-dim test is ill-posed when holiday weeks < departments)")

    candidate_cols = ["None"] + [c for c in raw.columns if c not in demand_cols] + \
                      [c for c in demand_cols if raw[c].dropna().isin([0, 1]).all()]
    default_idx = candidate_cols.index("IsHoliday") if "IsHoliday" in candidate_cols else 0
    holiday_col = st.selectbox(
        "Holiday indicator column (boolean/0-1)",
        candidate_cols, index=default_idx
    )

    if holiday_col != "None":
        holiday = raw[holiday_col].astype(bool).values
        Xlog = np.log(df.values)  # log-transform, matching 04_manova.py
        eigvals, eigvecs = run_pca(Xlog)
        Xstd = (Xlog - Xlog.mean(axis=0)) / Xlog.std(axis=0, ddof=1)
        scores = Xstd @ eigvecs[:, :2]
        g1, g2 = scores[holiday[:len(scores)]], scores[~holiday[:len(scores)]]
        n1, n2, pp = len(g1), len(g2), 2
        if n1 > pp and n2 > pp:
            mean1, mean2 = g1.mean(axis=0), g2.mean(axis=0)
            S1, S2 = np.cov(g1, rowvar=False), np.cov(g2, rowvar=False)
            Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
            diff = (mean1 - mean2).reshape(-1, 1)
            T2 = (n1 * n2 / (n1 + n2)) * (diff.T @ np.linalg.inv(Sp) @ diff).item()
            F_stat = ((n1 + n2 - pp - 1) / (pp * (n1 + n2 - 2))) * T2
            F_pval = 1 - stats.f.cdf(F_stat, pp, n1 + n2 - pp - 1)
            wilks = 1 / (1 + T2 / (n1 + n2 - 2))
            c1, c2, c3 = st.columns(3)
            c1.metric("Wilks' Λ", f"{wilks:.4f}")
            c2.metric("F-statistic", f"{F_stat:.2f}")
            c3.metric("p-value", f"{F_pval:.2g}")
            st.success("Reject H₀ — mean demand vectors differ" if F_pval < 0.05 else "Fail to reject H₀")
        else:
            st.warning("Not enough weeks in one group to run the test.")
    else:
        st.write("No holiday indicator loaded — this tab requires a boolean/0-1 column "
                 "(e.g. `IsHoliday`) alongside SKU demand columns in the uploaded CSV.")

# ---------------------------------------------------------------------------
# Tab 5 — VAR & Reorder Points
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("VAR Forecast & Correlated Reorder Points")
    try:
        forecast, safety_stock, reorder_point, sigma_resid, corr_resid, is_stable = fit_var(df, lag, service_level)
        st.write(f"VAR({lag}) stable: **{is_stable}**")

        results = pd.DataFrame({
            "SKU / Department": df.columns,
            "Next-week forecast": forecast.round(2),
            f"Safety stock ({service_level:.0%})": safety_stock.round(2),
            "Reorder point": reorder_point.round(2)
        })
        st.dataframe(results, use_container_width=True)

        csv = results.to_csv(index=False).encode()
        st.download_button("Download reorder points as CSV", csv, "reorder_points.csv", "text/csv")

        st.subheader("Residual Correlation (shock co-movement across SKUs)")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr_resid, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(p)); ax.set_xticklabels(df.columns, rotation=45)
        ax.set_yticks(range(p)); ax.set_yticklabels(df.columns)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
        mean_corr = (corr_resid.sum() - p) / (p**2 - p)
        st.info(f"Mean residual correlation across department pairs: **{mean_corr:.3f}** — "
                f"demand shocks remain strongly correlated even after accounting for VAR dynamics, "
                f"which is the statistical justification for correlated (rather than independent) "
                f"reorder-point sizing.")
    except Exception as e:
        st.error(f"VAR fitting failed: {e}. Try a lower lag order or check for missing data.")

st.divider()
st.caption("CIA3 — MCAI513B-3, Multivariate Techniques | Muhammed Harshin N K | Roll No. 2548611 | "
           "Dataset: Kaggle Walmart Recruiting - Store Sales Forecasting")
