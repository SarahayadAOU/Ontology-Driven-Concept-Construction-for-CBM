import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix
)


def classification_metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "auc": roc_auc_score(y, p),
        "accuracy": accuracy_score(y, pred),
        "sensitivity_recall": recall_score(y, pred, zero_division=0),
        "specificity": specificity,
        "precision_ppv": precision_score(y, pred, zero_division=0),
        "npv": npv,
        "f1": f1_score(y, pred, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def expected_calibration_error(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins, right=True), 1, n_bins)
    ece = 0.0
    for b in range(1, n_bins + 1):
        m = idx == b
        if not np.any(m):
            continue
        ece += np.mean(m) * abs(np.mean(p[m]) - np.mean(y[m]))
    return float(ece)


def topk_jaccard(rankings, k):
    sets = [set(np.argsort(-np.abs(r))[:k]) for r in rankings]
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            vals.append(len(sets[i] & sets[j]) / len(u) if u else 1.0)
    return float(np.mean(vals)) if vals else np.nan


def rank_sd(rankings):
    ranks = []
    for r in rankings:
        ranks.append(rankdata(-np.abs(r), method="average"))
    return float(np.std(np.vstack(ranks), axis=0, ddof=1).mean())


# Fast DeLong implementation adapted from the standard Sun & Xu algorithm.
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(pred_sorted, label_1_count):
    m = label_1_count
    n = pred_sorted.shape[1] - m
    positive = pred_sorted[:, :m]
    negative = pred_sorted[:, m:]
    k = pred_sorted.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m+n))
    for r in range(k):
        tx[r] = _compute_midrank(positive[r])
        ty[r] = _compute_midrank(negative[r])
        tz[r] = _compute_midrank(pred_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / (m*n) - (m+1)/(2*n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx/m + sy/n


def delong_pvalue(y, p1, p2):
    y = np.asarray(y).astype(int)
    order = np.argsort(-y)
    m = int(y.sum())
    preds = np.vstack([p1, p2])[:, order]
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = cov[0,0] + cov[1,1] - 2*cov[0,1]
    if var <= 0:
        return diff, np.nan, np.nan
    z = diff / np.sqrt(var)
    p = 2 * norm.sf(abs(z))
    return float(diff), float(z), float(p)


def stouffer(z_values):
    z = np.asarray([v for v in z_values if np.isfinite(v)], dtype=float)
    if len(z) == 0:
        return np.nan, np.nan
    zc = z.sum() / np.sqrt(len(z))
    return float(zc), float(2 * norm.sf(abs(zc)))
