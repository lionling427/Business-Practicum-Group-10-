"""
Shared statistics functions (numpy + scipy only, no statsmodels needed).

Used by generate_synthetic_data.py (to compute realized effect sizes for the
answer key) and validate_synthetic_data.py (to check planted effects).
These are also the starting point for the agent's deterministic statistical tools.

Models
  logit()  logistic regression            -> odds ratios      (turnover)
  negbin() negative binomial regression   -> rate ratios      (absenteeism)
  ols()    linear regression               -> betas            (performance, on z-scores)
All three support team-clustered standard errors.
"""
import numpy as np
import pandas as pd
from scipy import stats


class Fit:
    """Holds coefficients, clustered or classical standard errors, and helpers."""

    def __init__(self, names, params, cov, n, extra=None):
        self.names = list(names)
        self.params = pd.Series(params, index=self.names)
        self.se = pd.Series(np.sqrt(np.clip(np.diag(cov), 0, None)), index=self.names)
        z = self.params / self.se
        self.pvalues = pd.Series(2 * stats.norm.sf(np.abs(z)), index=self.names)
        self.n = n
        self.extra = extra or {}

    def ci(self, name, level=0.95):
        q = stats.norm.ppf(0.5 + level / 2)
        b, s = self.params[name], self.se[name]
        return b - q * s, b + q * s

    def ratio(self, name):
        """exp(coefficient) with its 95% CI: odds ratio or rate ratio."""
        lo, hi = self.ci(name)
        return float(np.exp(self.params[name])), float(np.exp(lo)), float(np.exp(hi))


def design(df, numeric, categorical=(), interactions=()):
    """Build an intercept + numeric columns + dummy-coded categoricals matrix."""
    parts = [pd.Series(1.0, index=df.index, name="Intercept")]
    for c in numeric:
        parts.append(df[c].astype(float).rename(c))
    for a, b in interactions:
        parts.append((df[a].astype(float) * df[b].astype(float)).rename(f"{a}:{b}"))
    X = pd.concat(parts, axis=1)
    for c in categorical:
        d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True, dtype=float)
        X = pd.concat([X, d], axis=1)
    return X


def _sandwich(X, scores, bread, cluster):
    if cluster is None:
        return bread
    codes = pd.factorize(np.asarray(cluster))[0]
    G = codes.max() + 1
    S = pd.DataFrame(scores).groupby(codes).sum().to_numpy()
    meat = S.T @ S
    n, k = X.shape
    adj = G / (G - 1) * (n - 1) / (n - k)
    return adj * bread @ meat @ bread


def logit(y, X, cluster=None, max_iter=50, tol=1e-9):
    y = np.asarray(y, float)
    Xv = X.to_numpy(float)
    beta = np.zeros(Xv.shape[1])
    for _ in range(max_iter):
        eta = np.clip(Xv @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        H = (Xv * w[:, None]).T @ Xv
        g = Xv.T @ (y - p)
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    p = 1 / (1 + np.exp(-np.clip(Xv @ beta, -30, 30)))
    bread = np.linalg.inv((Xv * (p * (1 - p))[:, None]).T @ Xv)
    cov = _sandwich(Xv, Xv * (y - p)[:, None], bread, cluster)
    return Fit(X.columns, beta, cov, len(y))


def _poisson(y, Xv, max_iter=50, tol=1e-9):
    beta = np.zeros(Xv.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-6))
    for _ in range(max_iter):
        mu = np.exp(np.clip(Xv @ beta, -30, 30))
        H = (Xv * mu[:, None]).T @ Xv
        step = np.linalg.solve(H, Xv.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def negbin(y, X, cluster=None, alpha=None, max_iter=50, tol=1e-9):
    """NB2 regression. Dispersion alpha is estimated by the Cameron-Trivedi
    auxiliary regression unless given."""
    y = np.asarray(y, float)
    Xv = X.to_numpy(float)
    beta = _poisson(y, Xv)
    if alpha is None:
        mu = np.exp(Xv @ beta)
        alpha = max(float(np.sum((y - mu) ** 2 - y) / np.sum(mu ** 2)), 1e-6)
    for _ in range(max_iter):
        mu = np.exp(np.clip(Xv @ beta, -30, 30))
        w = mu / (1 + alpha * mu)
        H = (Xv * w[:, None]).T @ Xv
        g = Xv.T @ ((y - mu) / (1 + alpha * mu))
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    mu = np.exp(Xv @ beta)
    w = mu / (1 + alpha * mu)
    bread = np.linalg.inv((Xv * w[:, None]).T @ Xv)
    cov = _sandwich(Xv, Xv * ((y - mu) / (1 + alpha * mu))[:, None], bread, cluster)
    return Fit(X.columns, beta, cov, len(y), {"alpha": alpha})


def ols(y, X, cluster=None):
    y = np.asarray(y, float)
    Xv = X.to_numpy(float)
    XtX_inv = np.linalg.inv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ y
    e = y - Xv @ beta
    if cluster is None:
        cov = XtX_inv * (e @ e) / (len(y) - Xv.shape[1])
    else:
        cov = _sandwich(Xv, Xv * e[:, None], XtX_inv, cluster)
    return Fit(X.columns, beta, cov, len(y))


def zscore(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def bh_fdr(pvalues, q=0.05):
    """Benjamini-Hochberg: returns a boolean array of discoveries."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    passed = np.zeros(len(p), bool)
    below = np.where(ranked <= q)[0]
    if len(below):
        passed[order[: below.max() + 1]] = True
    return passed


def demean(df, cols, group):
    """Within-group transformation (fixed effects) for the listed columns."""
    g = df.groupby(group)[cols].transform("mean")
    return df[cols] - g


# ---------------------------------------------------------------------------
# Standard models (Matrix A): one fixed setup per outcome
# ---------------------------------------------------------------------------
ITEMS = [f"Q{i:02d}" for i in range(1, 13)]


def turnover_model(d, cluster=True, extra_numeric=(), interactions=()):
    """Logistic regression: 12 items + tenure, department, role level, pay band."""
    X = design(d, list(ITEMS) + list(extra_numeric) + ["tenure_years"],
               ["department", "role_level", "pay_band"], interactions)
    return logit(d["left_voluntarily_12m"], X, d["team_id"] if cluster else None)


def absence_model(d, cluster=True, extra_numeric=()):
    """Negative binomial: 12 items + Q01-below-3 threshold + tenure, department, shift."""
    d = d.assign(q01_low=(d["Q01"] < 3).astype(float))
    X = design(d, list(ITEMS) + ["q01_low", "tenure_years"] + list(extra_numeric),
               ["department", "shift_type"])
    return negbin(d["unplanned_absence_days_6m"], X, d["team_id"] if cluster else None)


def performance_model(d, cluster=True):
    """Linear model on z-scores (approximates the ordinal model): 12 items +
    tenure, department, prior rating. Coefficients are standardized betas."""
    z = pd.DataFrame({c: zscore(d[c]) for c in list(ITEMS) + ["performance_rating", "prior_performance_rating"]},
                     index=d.index)
    z["tenure_years"] = d["tenure_years"]
    z["department"] = d["department"]
    X = design(z, list(ITEMS) + ["prior_performance_rating", "tenure_years"], ["department"])
    return ols(z["performance_rating"], X, d["team_id"] if cluster else None)


def icc_oneway(values, groups):
    """Share of variance between groups (one-way ANOVA ICC) and the F-test p-value."""
    d = pd.DataFrame({"y": np.asarray(values, float), "g": np.asarray(groups)})
    k = d.groupby("g")["y"].agg(["count", "mean"])
    n, G = len(d), len(k)
    grand = d["y"].mean()
    ssb = float((k["count"] * (k["mean"] - grand) ** 2).sum())
    ssw = float(((d["y"] - d.groupby("g")["y"].transform("mean")) ** 2).sum())
    msb, msw = ssb / (G - 1), ssw / (n - G)
    n0 = (n - (k["count"] ** 2).sum() / n) / (G - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    p = stats.f.sf(msb / msw, G - 1, n - G)
    return float(icc), float(p)
