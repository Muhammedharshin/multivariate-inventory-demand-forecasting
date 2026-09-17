Section A — Conceptual & Analytical Foundations 
Project: Multivariate Inventory Demand Forecasting with Correlated SKU Analysis Student: Muhammed Harshin N K | Roll 
No. 2548611 Course: MCAI513B-3, Multivariate Techniques, MSc Computational Statistics & Applied AI 
A1. Domain Knowledge & Problem Context 
Domain. Multi-category retail inventory management, focused on the interaction between demand for related product 
departments within a single retail location. 
Why this is a multivariate problem. Classical demand forecasting treats each department or SKU as an independent 
univariate series y_{d,t} and fits a separate model per series. This discards information whenever series are not 
independent — and retail demand rarely is: a promotion, macroeconomic shift, or holiday period does not move one 
department in isolation, it moves a cluster of related departments together, often with lead–lag structure between them 
(e.g., an anchor product driving delayed uptake in an accessory category). 
Formally, weekly demand across a basket of p correlated departments is treated not as p separate scalars but as a single 
random vector: 
X_t = (X_1,t, X_2,t, …, X_p,t)ᵀ ∈ ℝ^p, t = 1, …, n 
with population mean vector μ ∈ ℝ^p and covariance matrix Σ ∈ ℝ^(p×p). The off-diagonal entries of Σ (equivalently, the 
correlation matrix) encode exactly the co-movement structure that a univariate approach discards. The core claim 
motivating this project is that Σ is not (approximately) diagonal for a well-chosen basket of departments — i.e., the 
demand vector genuinely has multivariate structure worth modelling jointly, rather than being p unrelated univariate 
problems bundled together for convenience. 
Empirical grounding. This was not assumed but verified against data. Using the Kaggle Walmart Recruiting — Store Sales 
Forecasting dataset, weekly sales for Store 1 across all departments with complete 143-week coverage (n = 143 weeks, Feb 
2010 – Oct 2012) were extracted. Hierarchical clustering (average linkage, 1 − |ρ| distance) on the department-level 
correlation matrix identified a cluster of 16 departments with mean pairwise |ρ| ≈ 0.73. The 10 most strongly correlated 
members of this cluster (Departments 55, 14, 29, 7, 82, 5, 6, 32, 23, 46) were retained as the analysis basket, giving a 
demand matrix X ∈ ℝ^(143×10). Each of the 10 series was confirmed stationary via the Augmented Dickey–Fuller test (all p 
< 0.05 at levels), satisfying the pre-condition required for Vector Autoregression later in the pipeline. 
Applied parallel. This structure mirrors a pattern common in FMCG/electronics procurement: an anchor product (e.g., a 
handset) and its dependent accessories (case, charger) exhibit correlated, lagged demand. Reorder logic that ignores this 
correlation systematically misprices joint stockout risk — either over-stocking uncorrelated safety margins independently, 
or under-stocking because a shared demand shock hits several SKUs simultaneously. 
A2. Problem Identification & Proposed Solution 
Research question. Within a retail store carrying multiple correlated product departments, can jointly modelling their 
demand — rather than forecasting each department independently — produce more reliable, correlation-aware reorder 
points that reduce simultaneous stockout risk across the basket? 
Scope. Store 1 of the Walmart dataset; a 10-department basket selected on empirically verified correlation (Section A1); 
weekly granularity; forecast horizon of one period ahead, consistent with the weekly reorder cycle typical of the underlying 
business process. 
Proposed multivariate solution — a five-stage pipeline, each stage mapped to a specific course technique: 
1. MVN assumption testing (Unit 2). Test whether the joint demand vector (and later, VAR residuals) is consistent 
with a multivariate normal distribution via Mardia's test. This assumption underlies both the covariance-based 
safety-stock formula and the outlier-detection rule in stage 5. 
2. Dimensionality/pattern check (Unit 5 — PCA). Decompose the sample covariance matrix Σ̂ to confirm the basket is 
driven by a small number of dominant common factors, quantifying how correlated the basket truly is beyond a 
raw correlation matrix. 
3. Regime testing (Unit 3 — MANOVA). Test H₀: μ_holiday = μ_non-holiday to determine whether holiday and non
holiday weeks require separate mean-vector treatment before forecasting. 
4. Joint forecasting (VAR). Fit a Vector Autoregression across the 10-series basket to forecast next-period demand 
while explicitly capturing lead–lag effects between departments — the mechanism a univariate model cannot 
represent. 
5. Correlated reorder-point calculation. Combine the VAR point forecast with the residual covariance matrix Σ̂_ε to 
compute a joint, correlation-adjusted safety stock and reorder point per department, and use Mahalanobis 
distance on incoming weekly vectors to flag anomalous demand spikes in production. 
Expected outputs/decisions: 
• Statistical verdicts on the MVN and mean-vector-homogeneity assumptions (accept/reject, with test statistics) 
• One-period-ahead demand forecast per department, jointly estimated 
• A reorder point per department that accounts for cross-department correlation, not just individual variance 
• A flagged list of historically anomalous demand weeks via Mahalanobis distance, illustrating the outlier-detection 
capability 
• A small deployed application exposing forecasts and reorder points per department 
Why this technique set, and not alternatives. Independent per-department ARIMA/exponential smoothing models were 
considered and rejected as the primary method because they cannot represent cross-series correlation by construction — 
they would satisfy the forecasting requirement but not the multivariate requirement of the assignment. Discriminant 
Analysis and clustering-only approaches were considered but do not address the sequential/forecasting nature of the 
reorder-point problem. VAR was selected because it natively extends univariate autoregression to a vector of stationary 
series, and its residual covariance matrix gives a direct, statistically grounded input into the safety-stock calculation, closing 
the loop between forecasting and inventory decision-making. 
A3. Multivariate Technique(s) Applied 
A3.1 Multivariate Normal (MVN) Inference 
Model. The demand vector at time t is assumed X_t ~ N_p(μ, Σ), with density 
f(x) = (2π)^(−p/2) |Σ|^(−1/2) exp{ −½ (x − μ)ᵀ Σ⁻¹ (x − μ) } 
Parameter estimation. μ is estimated by the sample mean vector X̄, and Σ by the sample covariance matrix 
Σ̂ = (1/(n−1)) Σ_{t=1}^{n} (X_t − X̄)(X_t − X̄)ᵀ 
Assumption verification — Mardia's test. Mardia's multivariate skewness (b₁,p) and kurtosis (b₂,p) statistics are computed 
on the standardized demand vectors. Under H₀ (multivariate normality), n·b₁,p/6 is approximately χ² with p(p+1)(p+2)/6 
degrees of freedom, and the standardized kurtosis statistic is approximately N(0,1). Both are computed and compared 
against their reference distributions; the result determines whether the covariance-based reorder formula and 
Mahalanobis rule below are used directly or applied to a transformed (e.g., log-demand) series. 
Outlier / demand-spike detection — Mahalanobis distance. For each observed week, 
D²_t = (X_t − X̄)ᵀ Σ̂⁻¹ (X_t − X̄) 
is approximately χ²_p distributed under MVN. Weeks with D²_t exceeding the χ²_p critical value (α = 0.05) are flagged as 
multivariate demand-spike outliers — a joint anomaly the univariate baseline would not detect if any single department's 
deviation is unremarkable in isolation. 
A3.2 Principal Component Analysis (PCA) 
Model. PCA performs the eigendecomposition Σ̂ = Γ Λ Γᵀ, where Λ = diag(λ₁ ≥ λ₂ ≥ … ≥ λ_p) and Γ is the matrix of 
corresponding eigenvectors. The k-th principal component score is 
Y_k = γ_kᵀ (X − X̄) 
Interpretation. The proportion of total variance explained by the first m components, Σ_{k=1}^{m} λ_k / Σ_{k=1}^{p} λ_k, 
quantifies how much of the basket's joint variability is captured by a small number of common demand factors. Loadings 
(entries of Γ) identify which departments load most heavily on the dominant component(s), giving a data-driven 
confirmation of the correlated basket identified in A1. 
Assumption. PCA does not require multivariate normality to be valid as a variance-decomposition technique, but 
standardization (correlation-matrix PCA rather than covariance-matrix PCA) is used here since department sales 
magnitudes differ substantially, and this is stated explicitly to justify the choice. 
A3.3 MANOVA 
Model. For group j ∈ {holiday, non-holiday}, observation t within group j: 
X_tj = μ + τ_j + ε_tj, ε_tj ~ N_p(0, Σ) 
Hypothesis. H₀: τ_holiday = τ_non-holiday (equivalently μ_holiday = μ_non-holiday) vs. H₁: at least one mean vector differs. 
Test statistics. Using the between-group (B) and within-group (W) SSCP matrices, 
Wilks' Λ = |W| / |W + B| 
with Pillai's trace and Roy's largest root reported as supporting statistics. Small Λ (equivalently large Pillai's trace) leads to 
rejection of H₀. 
Assumption verification. Homogeneity of the group covariance matrices is tested via Box's M test prior to interpreting the 
MANOVA result; multivariate normality within each group (from A3.1) is also a formal precondition and is reported 
alongside the test outcome rather than assumed. 
A3.4 Vector Autoregression (VAR) 
Model. For the stationary p-dimensional series confirmed in A1, 
X_t = c + A₁X_{t−1} + A₂X_{t−2} + … + A_ℓX_{t−ℓ} + ε_t, ε_t ~ N_p(0, Σ_ε) 
where c is a constant vector, A₁,…,A_ℓ are p×p coefficient matrices, and ℓ is the lag order. 
Lag selection. ℓ is chosen by minimizing AIC/BIC over a candidate range (e.g., ℓ = 1,…,4), balancing model fit against the 
parameter burden of p²ℓ coefficients relative to n = 143 observations. 
Assumption verification. Stationarity of each input series is already confirmed (ADF, Section A1). Post-fit, residual white
noise behaviour is checked via the multivariate Ljung–Box test, and model stability is confirmed by verifying that all 
eigenvalues of the companion matrix lie inside the unit circle. 
Link to inventory decision. The fitted Σ̂_ε (residual covariance) feeds directly into the reorder-point calculation: for a target 
service level z, the correlation-adjusted safety stock vector is 
SS = z · sqrt(diag(L · Σ̂_ε · Lᵀ)) 
where L encodes the lead-time scaling per department, so that departments with correlated residual shocks receive jointly 
— not independently — sized buffers. 
A4. Pseudo-Algorithm 
Input: 
• Data matrix X ∈ ℝ^(n×p) — weekly demand, n = 143 weeks, p = 10 departments 
• Holiday indicator vector h ∈ {0,1}^n 
• Lead time and target service level per department 
• Significance level α = 0.05 
Output: 
• MVN and MANOVA test verdicts (with statistics) 
• PCA variance-explained summary and component loadings 
• Fitted VAR(ℓ) coefficients and forecast X̂_{n+1} 
• Reorder point vector R ∈ ℝ^p 
• List of flagged anomalous weeks (Mahalanobis) 
Steps: 
1. Load and align X, h, and metadata; forward-fill isolated missing weeks (≤1 consecutive gap), else drop the 
department from the basket if missing data exceeds a 5% threshold. 
2. Stationarity check. Run ADF on each column of X. If any series is non-stationary, difference it and re-test before 
proceeding (not required for the current basket — all 10 series passed at levels). 
3. Estimate X̄ and Σ̂ from X. 
4. Edge case — near-singular Σ̂. Compute the condition number of Σ̂. If it exceeds a set threshold (indicating near
collinearity among departments), apply a shrinkage estimator (e.g., Ledoit–Wolf) before inversion, since a poorly 
conditioned Σ̂ makes both Mahalanobis distance and the safety-stock formula numerically unstable. 
5. Test MVN via Mardia's skewness/kurtosis statistics; record accept/reject at α. 
6. Run PCA on the standardized X; record eigenvalues, cumulative variance explained, and loadings for the top 
components. 
7. Run MANOVA of X on h; test Box's M for covariance homogeneity first, then compute Wilks' Λ, Pillai's trace, and 
the associated F-test; record accept/reject of H₀. 
8. Select VAR lag order ℓ by AIC/BIC over a candidate grid; fit VAR(ℓ) on X. 
9. Diagnose VAR fit: multivariate Ljung–Box test on residuals; companion-matrix eigenvalue check for stability. If 
either check fails, reduce ℓ or reconsider the basket. 
10. Forecast X̂_{n+1} from the fitted VAR; extract residual covariance Σ̂_ε. 
11. Compute reorder points R = X̂_{n+1} + SS, where SS is the correlation-adjusted safety stock vector derived from 
Σ̂_ε, lead time, and target service level (Section A3.4). 
12. Score historical weeks with D²_t = (X_t − X̄)ᵀΣ̂⁻¹(X_t − X̄); flag t where D²_t > χ²_{p,1−α}. 
13. Return all test verdicts, PCA summary, VAR forecast, reorder points, and flagged weeks for downstream reporting 
and the deployed application. 
Additional edge cases handled: 
• Missing data: handled at Step 1 (forward-fill vs. drop) rather than silently propagating NaNs into Σ̂. 
• Singular/ill-conditioned Σ̂: handled at Step 4 via shrinkage, avoiding failure of the Mahalanobis and safety-stock 
calculations. 
• Non-stationary input: handled at Step 2 via differencing before VAR fitting, since VAR is undefined for non
stationary inputs without a cointegration-based alternative (VECM), which is out of scope here but noted as a 
limitation.
