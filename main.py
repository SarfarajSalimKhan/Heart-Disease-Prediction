import argparse
import io
import os
import urllib.request
import warnings
import zipfile
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import kendalltau
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
SITE_FILES = {
    "Cleveland": "processed.cleveland.data",
    "Hungarian": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA Long Beach": "processed.va.data",
}
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
# Features available at all four sites. slope, ca and thal are mostly missing at
# Hungarian / Switzerland / VA, so they are excluded unless --include-sparse is used.
DENSE_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                  "thalach", "exang", "oldpeak"]
SPARSE_FEATURES = ["slope", "ca", "thal"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]


# --------------------------------------------------------------------------- data
def load_uci(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    have = all(os.path.exists(os.path.join(data_dir, f)) for f in SITE_FILES.values())
    if not have:
        print("Downloading UCI Heart Disease data ...")
        try:
            with urllib.request.urlopen(URL, timeout=60) as r:
                z = zipfile.ZipFile(io.BytesIO(r.read()))
            for f in SITE_FILES.values():
                z.extract(f, data_dir)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"Download failed ({e}).\nDownload {URL} manually and copy the four "
                f"processed.*.data files into '{data_dir}'.")
    frames = []
    for site, fname in SITE_FILES.items():
        df = pd.read_csv(os.path.join(data_dir, fname), header=None, names=COLS,
                         na_values=["?", "-9", -9])
        df["site"] = site
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # A value of 0 for cholesterol or resting BP is a missing-data code, not a measurement.
    for c in ["chol", "trestbps"]:
        df.loc[df[c] == 0, c] = np.nan
    df["target"] = (df["num"] > 0).astype(int)
    return df


def make_synthetic(seed=0):
    """Fake four-site data with site shifts. ONLY for testing that the code runs."""
    rng = np.random.default_rng(seed)
    spec = {"Cleveland": (303, 0.46, 0.0), "Hungarian": (294, 0.36, 0.5),
            "Switzerland": (123, 0.93, 1.0), "VA Long Beach": (200, 0.75, -0.5)}
    frames = []
    for site, (n, prev, shift) in spec.items():
        y = (rng.random(n) < prev).astype(int)
        df = pd.DataFrame({
            "age": rng.normal(52 + 4 * y + shift * 2, 8),
            "sex": (rng.random(n) < 0.6 + 0.2 * y).astype(int),
            "cp": np.where(y == 1, rng.choice([1, 2, 3, 4], n, p=[.1, .1, .2, .6]),
                           rng.choice([1, 2, 3, 4], n, p=[.15, .3, .35, .2])),
            "trestbps": rng.normal(128 + 4 * y + 3 * shift, 16),
            "chol": rng.normal(235 + 5 * y + 15 * shift, 50),
            "fbs": (rng.random(n) < 0.15).astype(int),
            "restecg": rng.choice([0, 1, 2], n, p=[.55, .1, .35]),
            "thalach": rng.normal(158 - 20 * y - 4 * shift, 20),
            "exang": (rng.random(n) < 0.15 + 0.4 * y).astype(int),
            "oldpeak": np.abs(rng.normal(0.4 + 1.2 * y, 0.9)),
        })
        miss = {"Cleveland": 0.0, "Hungarian": 0.05, "Switzerland": 0.4, "VA Long Beach": 0.1}[site]
        df.loc[rng.random(n) < miss, "chol"] = np.nan
        df["target"] = y
        df["site"] = site
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------- models
class OversampledClassifier(ClassifierMixin, BaseEstimator):
    """Wraps an estimator and randomly oversamples the minority class INSIDE fit().
    Because it lives inside the model, resampling only ever touches training data
    (also inside the inner folds used for calibration)."""

    def __init__(self, base=None, random_state=0):
        self.base = base
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        Xo, yo = oversample(pd.DataFrame(X).reset_index(drop=True), np.asarray(y), rng)
        self.model_ = clone(self.base).fit(Xo, yo)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def build_models(features, seed, oversample_train=False):
    cat = [c for c in CATEGORICAL if c in features]
    num = [c for c in features if c not in cat]
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    lr = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000))])
    rf = Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                   ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                                  random_state=seed, n_jobs=-1))])
    gb = Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                   ("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                                      max_depth=2, subsample=0.8,
                                                      random_state=seed))])
    wrap = (lambda m: OversampledClassifier(m, seed)) if oversample_train else (lambda m: m)
    vote = VotingClassifier([("lr", wrap(clone(lr))), ("rf", wrap(clone(rf))), ("gb", wrap(clone(gb)))],
                            voting="soft")
    cal = CalibratedClassifierCV(clone(vote), method="sigmoid", cv=5)
    return {"LogReg": wrap(lr), "RandomForest": wrap(rf), "GradBoost": wrap(gb),
            "Ensemble (uncalibrated)": vote, "Ensemble (calibrated)": cal}


def fit_predict(model, X_tr, y_tr, X_te):
    m = clone(model)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


# ------------------------------------------------------------------------ metrics
def _intercept_shift(y, p):
    """Shift 'a' on the logit scale so that mean(predicted) equals the observed rate."""
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    f = lambda a: expit(lp + a).mean() - y.mean()  # noqa: E731
    try:
        return brentq(f, -10, 10)
    except ValueError:
        return np.nan


def _slope(y, p):
    try:
        lr = LogisticRegression(penalty=None, max_iter=1000)
        lr.fit(logit(p).reshape(-1, 1), y)
        return float(lr.coef_[0][0])
    except Exception:  # noqa: BLE001
        return np.nan


def metrics(y, p, n_boot=500, seed=0):
    """AUC, Brier, calibration slope, calibration shift, accuracy, sens/spec,
    with bootstrap 95% CIs for AUC, Brier and slope when n_boot > 0."""
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out = {"n": len(y), "prevalence": y.mean(), "mean_pred": p.mean()}
    two_classes = len(np.unique(y)) == 2
    out["auc"] = roc_auc_score(y, p) if two_classes else np.nan
    out["brier"] = brier_score_loss(y, p)
    out["cal_slope"] = _slope(y, p) if two_classes else np.nan
    for k in ["auc", "brier", "cal_slope"]:
        out[k + "_lo"] = out[k + "_hi"] = np.nan
    if two_classes and n_boot > 0:
        rng = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        bs = {"auc": [], "brier": [], "cal_slope": []}
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            yb, pb = y[idx], p[idx]
            bs["auc"].append(roc_auc_score(yb, pb))
            bs["brier"].append(brier_score_loss(yb, pb))
            bs["cal_slope"].append(_slope(yb, pb))
        for k, v in bs.items():
            out[k + "_lo"], out[k + "_hi"] = np.nanpercentile(v, [2.5, 97.5])
    out["cal_intercept_shift"] = _intercept_shift(y, p)
    pred = (p >= 0.5).astype(int)
    out["accuracy"] = accuracy_score(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out["sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
    out["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    out["n_neg"] = int((y == 0).sum())
    out["caution"] = "few negatives: AUC unreliable" if (y == 0).sum() < 15 or (y == 1).sum() < 15 else ""
    return out


# -------------------------------------------------------------------- experiments
def run_random_cv(df, features, models, n_splits, n_repeats, seed, n_boot=500):
    X, y = df[features], df["target"].values
    strata = (df["site"] + "_" + df["target"].astype(str)).values
    sites = list(SITE_FILES)
    rows, ci_rows = [], []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + rep)
        oof = {m: np.zeros(len(df)) for m in models}
        for tr, te in skf.split(X, strata):
            for name, mdl in models.items():
                oof[name][te] = fit_predict(mdl, X.iloc[tr], y[tr], X.iloc[te])
        for name in models:
            for site in ["ALL"] + sites:
                mask = np.ones(len(df), bool) if site == "ALL" else (df["site"] == site).values
                rows.append(dict(repeat=rep, model=name, site=site,
                                 **metrics(y[mask], oof[name][mask], n_boot=0)))
                if rep == 0:  # bootstrap CIs from the first repeat's out-of-fold predictions
                    ci_rows.append(dict(model=name, site=site,
                                        **metrics(y[mask], oof[name][mask], n_boot=n_boot, seed=seed)))
        print(f"  random CV repeat {rep + 1}/{n_repeats} done")
    res = pd.DataFrame(rows)
    summary = (res.groupby(["model", "site"])[["auc", "brier", "cal_slope", "cal_intercept_shift",
                                               "accuracy", "sensitivity", "specificity"]].mean().reset_index())
    return res, summary, pd.DataFrame(ci_rows)


def run_loso(df, features, models, n_boot, seed):
    X, y = df[features], df["target"].values
    rows, preds = [], {}
    for held in SITE_FILES:
        te = (df["site"] == held).values
        tr = ~te
        for name, mdl in models.items():
            p = fit_predict(mdl, X[tr], y[tr], X[te])
            preds[(held, name)] = (y[te], p)
            rows.append(dict(model=name, held_out_site=held,
                             **metrics(y[te], p, n_boot=n_boot, seed=seed)))
        print(f"  LOSO fold '{held}' done")
    return pd.DataFrame(rows), preds


def oversample(X, y, rng):
    """Random oversampling of the minority class (duplicates rows)."""
    y = np.asarray(y)
    minority = 1 if y.sum() < len(y) / 2 else 0
    idx_min = np.where(y == minority)[0]
    n_add = int((y != minority).sum() - len(idx_min))
    add = rng.choice(idx_min, n_add, replace=True)
    idx = np.concatenate([np.arange(len(y)), add])
    return X.iloc[idx].reset_index(drop=True), y[idx]


def leakage_demo(df, features, seed, n_splits, n_repeats):
    """Compare resampling BEFORE the split (leaky) with resampling INSIDE training folds."""
    # default RF settings (min_samples_leaf=1), as in most published pipelines
    rf = Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                   ("clf", RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1))])
    X, y = df[features].reset_index(drop=True), df["target"].values
    rows = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep)
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + rep)
        # (a) no resampling at all
        a = [roc_auc_score(y[te], fit_predict(rf, X.iloc[tr], y[tr], X.iloc[te]))
             for tr, te in skf.split(X, y)]
        # (b) WRONG: oversample the whole dataset, then cross-validate
        Xo, yo = oversample(X, y, rng)
        b = [roc_auc_score(yo[te], fit_predict(rf, Xo.iloc[tr], yo[tr], Xo.iloc[te]))
             for tr, te in skf.split(Xo, yo)]
        # (c) RIGHT: split first, oversample only the training fold
        c = []
        for tr, te in skf.split(X, y):
            Xt, yt = oversample(X.iloc[tr].reset_index(drop=True), y[tr], rng)
            c.append(roc_auc_score(y[te], fit_predict(rf, Xt, yt, X.iloc[te])))
        rows += [dict(repeat=rep, pipeline="No resampling", auc=np.mean(a)),
                 dict(repeat=rep, pipeline="Oversample BEFORE split (leaky)", auc=np.mean(b)),
                 dict(repeat=rep, pipeline="Oversample INSIDE training folds (correct)", auc=np.mean(c))]
    res = pd.DataFrame(rows)
    return res.groupby("pipeline")["auc"].agg(["mean", "std"]).reset_index()



def recalibration_experiment(preds, frac=0.2, n_rep=50, seed=0, model="Ensemble (calibrated)"):
    """RQ2 extension: how much labelled data from the NEW site is needed to fix calibration?
    Fit an intercept-only shift, or a logistic (slope + intercept) update, on a small labelled
    part of the held-out site, and evaluate on the rest."""
    rng = np.random.default_rng(seed)
    rows = []
    for site in SITE_FILES:
        y, p = preds[(site, model)]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        for _ in range(n_rep):
            idx = rng.permutation(len(y))
            k = max(int(frac * len(y)), 10)
            fit, ev = idx[:k], idx[k:]
            if len(np.unique(y[fit])) < 2 or len(np.unique(y[ev])) < 2:
                continue
            a = _intercept_shift(y[fit], p[fit])
            p_int = expit(logit(p[ev]) + (a if np.isfinite(a) else 0.0))
            lr = LogisticRegression(penalty=None, max_iter=1000).fit(logit(p[fit]).reshape(-1, 1), y[fit])
            p_log = lr.predict_proba(logit(p[ev]).reshape(-1, 1))[:, 1]
            for label, pp in [("No update", p[ev]), ("Intercept update", p_int), ("Slope + intercept update", p_log)]:
                rows.append(dict(site=site, method=label, brier=brier_score_loss(y[ev], pp),
                                 shift=_intercept_shift(y[ev], pp)))
    res = pd.DataFrame(rows)
    out = res.groupby(["site", "method"]).agg(brier=("brier", "mean"),
                                              abs_shift=("shift", lambda s: s.abs().mean())).reset_index()
    return out

# ------------------------------------------------------------------ explainability
def importance_vector(fitted, X_ref, y_ref, use_shap, kind="tree", seed=0):
    """Global importance = mean |SHAP value| per feature.
    kind='tree'     -> exact TreeSHAP (random forest pipeline)
    kind='agnostic' -> model-agnostic permutation SHAP on predict_proba (used for the ENSEMBLE)
    Falls back to permutation importance if shap is unavailable or fails."""
    if use_shap:
        try:
            import shap
            if kind == "tree":
                imp, clf = fitted.named_steps["imp"], fitted.named_steps["clf"]
                Xt = pd.DataFrame(imp.transform(X_ref), columns=X_ref.columns)
                sv = shap.TreeExplainer(clf).shap_values(Xt)
                if isinstance(sv, list):
                    sv = sv[1]
                elif getattr(sv, "ndim", 2) == 3:
                    sv = sv[:, :, 1]
                return np.abs(sv).mean(axis=0)
            rng = np.random.default_rng(seed)
            filled = X_ref.fillna(X_ref.median()).values.astype(float)
            ref = filled[rng.choice(len(filled), min(100, len(filled)), replace=False)]
            bg = filled[rng.choice(len(filled), min(50, len(filled)), replace=False)]
            cols = list(X_ref.columns)
            f = lambda d: fitted.predict_proba(pd.DataFrame(d, columns=cols))[:, 1]  # noqa: E731
            explainer = shap.Explainer(f, shap.maskers.Independent(bg, max_samples=len(bg)),
                                       algorithm="permutation")
            sv = explainer(ref, max_evals=max(2 * len(cols) + 1, 20 * len(cols)))
            return np.abs(np.asarray(sv.values)).mean(axis=0)
        except Exception as e:  # noqa: BLE001
            print(f"    (SHAP failed: {type(e).__name__}: {e}; using permutation importance)")
    r = permutation_importance(fitted, X_ref, y_ref, n_repeats=5, random_state=seed,
                               scoring="roc_auc")
    return np.clip(r.importances_mean, 0, None)


def top_k_overlap(a, b, k):
    ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return len(ta & tb) / k


def shap_stability(df, features, seed, use_shap, ens_model, k=3, n_boot_models=8):
    """Stability of global feature importance across sites, for BOTH the random forest
    (exact TreeSHAP) and the calibrated ensemble (permutation SHAP)."""
    X, y = df[features], df["target"].values
    rf_plain = build_models(features, seed)["RandomForest"]
    sites = list(SITE_FILES)
    families = {"RandomForest (TreeSHAP)": (rf_plain, "tree", n_boot_models),
                "Ensemble (calibrated)": (ens_model, "agnostic", max(3, n_boot_models // 2))}
    tables, all_pairs = {}, []
    for fam, (mdl, kind, n_b) in families.items():
        print(f"  family: {fam}")

        def vec(idx, seed_=seed):
            fitted = clone(mdl).fit(X.iloc[idx], y[idx])
            return importance_vector(fitted, X, y, use_shap, kind, seed_)

        vecs = {"Pooled (all sites)": vec(np.arange(len(df)))}
        for s_ in sites:  # models trained on a single site
            idx = np.where(df["site"] == s_)[0]
            cls = np.bincount(y[idx], minlength=2)
            if len(idx) >= 30 and cls.min() >= 6:
                try:
                    vecs[f"Single site: {s_}"] = vec(idx)
                except Exception as e:  # noqa: BLE001
                    print(f"    skipped single-site model for {s_}: {e}")
            else:
                print(f"    skipped single-site model for {s_}: too few cases in a class")
        for s_ in sites:  # leave-one-site-out models
            vecs[f"LOSO (without {s_})"] = vec(np.where(df["site"] != s_)[0])
        rng = np.random.default_rng(seed)
        boot = [vec(rng.choice(len(df), len(df)), seed_=seed + i) for i in range(n_b)]

        def pair_stats(items, label):
            rows = []
            for (n1, v1), (n2, v2) in combinations(items, 2):
                rows.append(dict(family=fam, set=label, model_a=n1, model_b=n2,
                                 kendall_tau=kendalltau(v1, v2).statistic,
                                 top_k_overlap=top_k_overlap(v1, v2, k)))
            return rows

        single = [(n, v) for n, v in vecs.items() if n.startswith("Single")]
        loso = [(n, v) for n, v in vecs.items() if n.startswith("LOSO")]
        all_pairs += (pair_stats(single, "Across single-site models")
                      + pair_stats(loso, "Across LOSO models")
                      + pair_stats([(f"bootstrap {i}", v) for i, v in enumerate(boot)],
                                   "Baseline: bootstrap of pooled data"))
        tables[fam] = pd.DataFrame(vecs, index=features)
    return tables, pd.DataFrame(all_pairs)


# ------------------------------------------------------------------------- figures
def plot_auc(summary_random, loso, out):
    models = ["LogReg", "RandomForest", "Ensemble (calibrated)"]
    sites = list(SITE_FILES)
    fig, axes = plt.subplots(1, len(models), figsize=(13, 4), sharey=True)
    for ax, mdl in zip(axes, models):
        r = [summary_random[(summary_random["model"] == mdl) & (summary_random["site"] == s)]["auc"].iloc[0]
             for s in sites]
        l = [loso[(loso["model"] == mdl) & (loso["held_out_site" ] == s)]["auc"].iloc[0]
             for s in sites]
        x = np.arange(len(sites))
        ax.bar(x - 0.2, r, 0.4, label="Random split (pooled CV)")
        ax.bar(x + 0.2, l, 0.4, label="Leave-one-site-out")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" ", "\n") for s in sites], fontsize=8)
        ax.set_title(mdl, fontsize=10)
        ax.set_ylim(0.4, 1.0)
    axes[0].set_ylabel("AUC (per site)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Random split vs leave-one-site-out AUC")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_calibration(preds, out):
    sites = list(SITE_FILES)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, s in zip(axes, sites):
        for name in ["Ensemble (uncalibrated)", "Ensemble (calibrated)"]:
            y, p = preds[(s, name)]
            try:
                fp, mp = calibration_curve(y, p, n_bins=5, strategy="quantile")
                ax.plot(mp, fp, marker="o", label=name)
            except ValueError:
                pass
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"Held out: {s}", fontsize=9)
        ax.set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed fraction positive")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_importance(imp_table, out, title=""):
    norm = imp_table / imp_table.sum(axis=0).replace(0, np.nan)
    norm = norm.loc[imp_table["Pooled (all sites)"].sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(norm.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(norm.shape[1]))
    ax.set_xticklabels(norm.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(norm.shape[0]))
    ax.set_yticklabels(norm.index, fontsize=9)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            v = norm.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.25 else "black")
    fig.colorbar(im, label="Share of total importance")
    ax.set_title(f"Feature importance by training data: {title}")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="fewer repeats/bootstraps")
    ap.add_argument("--synthetic", action="store_true", help="fake data, testing only")
    ap.add_argument("--include-sparse", action="store_true",
                    help="also use slope, ca, thal (mostly missing at 3 of 4 sites)")
    ap.add_argument("--no-shap", action="store_true", help="use permutation importance")
    ap.add_argument("--oversample", action="store_true",
                    help="randomly oversample the minority class INSIDE each training fold")
    args, _ = ap.parse_known_args() # Modified to ignore notebook-specific parameters like -f
    os.makedirs(args.out_dir, exist_ok=True)

    n_repeats = 1 if args.quick else 3
    n_boot = 100 if args.quick else 500
    df = make_synthetic(args.seed) if args.synthetic else load_uci(args.data_dir)
    if args.synthetic:
        print("*** SYNTHETIC DATA: results are meaningless, this only tests the code ***")
    features = DENSE_FEATURES + (SPARSE_FEATURES if args.include_sparse else [])

    use_shap = not args.no_shap
    if use_shap:
        try:
            import shap  # noqa: F401
        except ImportError:
            print("shap not installed -> using permutation importance (pip install shap)")
            use_shap = False

    print("\n== Dataset summary ==")
    summ = df.groupby("site").agg(n=("target", "size"), positives=("target", "sum"),
                                  prevalence=("target", "mean"),
                                  chol_missing=("chol", lambda s: s.isna().mean()))
    print(summ.round(3).to_string())
    summ.to_csv(os.path.join(args.out_dir, "00_dataset_summary.csv"))
    few = [site for site, g in df.groupby("site") if (g["target"] == 0).sum() < 15 or (g["target"] == 1).sum() < 15]
    if few:
        print(f"\nNOTE: {', '.join(few)} has fewer than 15 cases in one class, so its AUC is unreliable.\n")

    models = build_models(features, args.seed, oversample_train=args.oversample)
    print(f"In-fold oversampling: {'ON' if args.oversample else 'OFF (use --oversample to enable)'}")

    print("== Experiment A: random split (pooled stratified CV) ==")
    rand_all, rand_sum, rand_ci = run_random_cv(df, features, models, 5, n_repeats, args.seed, n_boot)
    rand_sum.to_csv(os.path.join(args.out_dir, "01_random_cv_summary.csv"), index=False)
    rand_ci.to_csv(os.path.join(args.out_dir, "01b_random_cv_with_CIs.csv"), index=False)

    print("== Experiment B: leave-one-site-out ==")
    loso, preds = run_loso(df, features, models, n_boot, args.seed)
    loso.to_csv(os.path.join(args.out_dir, "02_loso_results.csv"), index=False)

    # RQ1: AUC drop per model and site
    drop = rand_sum[rand_sum.site != "ALL"].merge(
        loso[["model", "held_out_site", "auc"]].rename(columns={"held_out_site": "site", "auc": "auc_loso"}),
        on=["model", "site"])
    drop["auc_drop"] = drop["auc"] - drop["auc_loso"]
    drop = drop[["model", "site", "auc", "auc_loso", "auc_drop"]].rename(columns={"auc": "auc_random_split"})
    drop.to_csv(os.path.join(args.out_dir, "03_auc_drop_rq1.csv"), index=False)

    print("== Experiment B2: does a small labelled sample from the new site fix calibration? ==")
    recal_tbl = recalibration_experiment(preds, seed=args.seed)
    recal_tbl.to_csv(os.path.join(args.out_dir, "03b_recalibration_rq2.csv"), index=False)

    print("== Experiment C: resampling leakage demo ==")
    leak = leakage_demo(df, features, args.seed, 5, n_repeats)
    leak.to_csv(os.path.join(args.out_dir, "04_leakage_demo.csv"), index=False)

    print("== Experiment D: feature-importance stability across sites ==")
    tables, pairs = shap_stability(df, features, args.seed, use_shap, models["Ensemble (calibrated)"],
                                   n_boot_models=4 if args.quick else 8)
    for i, (fam, tbl) in enumerate(tables.items()):
        tag = "ensemble" if fam.startswith("Ensemble") else "randomforest"
        tbl.to_csv(os.path.join(args.out_dir, f"05_importance_{tag}.csv"))
        plot_importance(tbl, os.path.join(args.out_dir, f"fig3_importance_heatmap_{tag}.png"), fam)
    pairs.to_csv(os.path.join(args.out_dir, "06_stability_pairs.csv"), index=False)
    stab = (pairs.groupby(["family", "set"])[["kendall_tau", "top_k_overlap"]]
            .agg(["mean", "min", "max"]).round(3))
    stab.to_csv(os.path.join(args.out_dir, "07_stability_summary.csv"))

    plot_auc(rand_sum, loso, os.path.join(args.out_dir, "fig1_auc_random_vs_loso.png"))
    plot_calibration(preds, os.path.join(args.out_dir, "fig2_calibration_loso.png"))

    pd.set_option("display.width", 200)
    print("\n================ RESULTS ================")
    print("\nRQ1  AUC: random split vs leave-one-site-out (per site)")
    print(drop.round(3).to_string(index=False))
    print("\nRQ2  Calibration under LOSO (slope near 1 and shift near 0 are good)")
    print(loso[["model", "held_out_site", "n", "prevalence", "mean_pred", "accuracy", "auc", "auc_lo",
                "auc_hi", "brier", "brier_lo", "brier_hi", "cal_slope", "cal_slope_lo",
                "cal_slope_hi", "cal_intercept_shift", "caution"]].round(3).to_string(index=False))
    print("\nRQ2b Recalibrating with 20% labelled data from the new site (lower Brier / smaller |shift| is better)")
    print(recal_tbl.round(3).to_string(index=False))
    print("\nRQ3  Importance-ranking stability (", "SHAP" if use_shap else "permutation", ")")
    print(stab.to_string())
    print("\nMethod check: effect of resampling before the split (Random Forest, pooled data)")
    print(leak.round(3).to_string(index=False))
    print(f"\nAll tables and figures saved in: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
