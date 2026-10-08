from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class GroupEncoder:
    columns: List[int]
    scaler: StandardScaler
    model: object
    kind: str
    sign: float = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(X[:, self.columns])
        if self.kind == "supervised":
            return self.model.decision_function(Z).reshape(-1, 1)
        if self.kind == "unsupervised":
            return (self.sign * self.model.transform(Z)[:, 0]).reshape(-1, 1)
        raise ValueError(self.kind)


class CBM:
    """Two-stage tabular concept bottleneck used in the manuscript replication."""

    def __init__(self, groups: Dict[str, List[str]], encoder="supervised", C=1.0, seed=0):
        self.groups = groups
        self.encoder = encoder
        self.C = C
        self.seed = seed
        self.encoders_ = []
        self.concept_names_ = list(groups)

    def fit(self, X, y, feature_names):
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        self.encoders_ = []
        Ctrain = []
        for concept, names in self.groups.items():
            cols = [name_to_idx[n] for n in names]
            sc = StandardScaler().fit(X[:, cols])
            Z = sc.transform(X[:, cols])
            if self.encoder == "supervised":
                mdl = LogisticRegression(
                    penalty="l1", solver="liblinear", C=self.C,
                    max_iter=5000, random_state=self.seed
                ).fit(Z, y)
                enc = GroupEncoder(cols, sc, mdl, "supervised")
            elif self.encoder == "unsupervised":
                pca = PCA(n_components=1, random_state=self.seed).fit(Z)
                # Label-free orientation: mean loading positive.
                sign = 1.0 if np.mean(pca.components_[0]) >= 0 else -1.0
                enc = GroupEncoder(cols, sc, pca, "unsupervised", sign)
            else:
                raise ValueError("encoder must be supervised or unsupervised")
            self.encoders_.append(enc)
            Ctrain.append(enc.transform(X))

        Ctrain = np.hstack(Ctrain)
        self.concept_scaler_ = StandardScaler().fit(Ctrain)
        Cs = self.concept_scaler_.transform(Ctrain)
        self.head_ = LogisticRegression(
            penalty="l2", solver="liblinear", C=1.0,
            max_iter=5000, random_state=self.seed
        ).fit(Cs, y)
        return self

    def transform_concepts(self, X):
        C = np.hstack([e.transform(X) for e in self.encoders_])
        return self.concept_scaler_.transform(C)

    def predict_proba(self, X):
        C = self.transform_concepts(X)
        return self.head_.predict_proba(C)[:, 1]

    def concept_effects(self):
        return self.head_.coef_.ravel().copy()
