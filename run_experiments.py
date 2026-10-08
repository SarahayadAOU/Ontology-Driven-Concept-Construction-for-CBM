"""Reproduce the main Gallstone CBM comparison.

Outputs:
  results/cv_summary.csv
  results/classification_metrics.csv
  results/oof_predictions.csv
  results/delong_comparisons.csv
  results/leakage_audit.csv
  results/concept_impurity.csv
  results/stability.csv
  results/reliability.csv
  results/selective_abstention.csv

The defaults reproduce the protocol documented in the comparison workbook:
10 repeats x 10 stratified folds, B=100 bootstrap stability runs.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.cbm import CBM
from src.metrics import (
    classification_metrics, expected_calibration_error,
    topk_jaccard, rank_sd, delong_pvalue, stouffer
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "gallstone_uci.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SCHEME_FILES = {
    "manual": ROOT / "config" / "manual_groups.json",
    "ontology_default": ROOT / "config" / "ontology_default_groups.json",
    "ontology_bridged": ROOT / "config" / "ontology_bridged_groups.json",
}
ENCODERS = ["supervised", "unsupervised"]


def load_data():
    if not DATA.exists():
        raise FileNotFoundError(
            f"{DATA} not found. Run `python download_data.py` first, or place the UCI CSV there."
        )
    df = pd.read_csv(DATA)
    target = "Gallstone Status"
    if target not in df.columns:
        raise ValueError(f"Missing target column {target!r}")
    # Raw UCI coding: 0=present, 1=absent. Manuscript: 1=present.
    y = 1 - df[target].astype(int).to_numpy()
    Xdf = df.drop(columns=[target])
    X = Xdf.astype(float).to_numpy()
    if X.shape != (319, 38):
        raise ValueError(f"Expected X shape (319,38), got {X.shape}")
    if int(y.sum()) != 161:
        raise ValueError(f"Expected 161 gallstone-positive patients after recoding, got {int(y.sum())}")
    return X, y, list(Xdf.columns)


def load_groups():
    return {k: json.loads(v.read_text(encoding="utf-8")) for k, v in SCHEME_FILES.items()}


def fit_reference_l2(Xtr, ytr, Xte, seed):
    sc = StandardScaler().fit(Xtr)
    mdl = LogisticRegression(penalty="l2", solver="liblinear", C=1.0,
                             max_iter=5000, random_state=seed).fit(sc.transform(Xtr), ytr)
    return mdl.predict_proba(sc.transform(Xte))[:,1]


def run_cv(X, y, names, schemes, repeats=10, folds=10, seed=42):
    rows, pred_rows = [], []
    # We explicitly create one StratifiedKFold per repeat so every condition uses identical folds.
    for rep in range(repeats):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + rep)
        split_list = list(cv.split(X, y))
        ref_p = np.zeros(len(y))
        for fold, (tr, te) in enumerate(split_list):
            ref_p[te] = fit_reference_l2(X[tr], y[tr], X[te], seed+rep*100+fold)
        rows.append({"scheme":"raw_l2_reference","encoder":"raw","repeat":rep,
                     "auc":roc_auc_score(y,ref_p),"accuracy":accuracy_score(y,ref_p>=.5)})
        for scheme_name, groups in schemes.items():
            for encoder in ENCODERS:
                p = np.zeros(len(y))
                for fold, (tr, te) in enumerate(split_list):
                    model = CBM(groups, encoder=encoder, seed=seed+rep*100+fold)
                    model.fit(X[tr], y[tr], names)
                    p[te] = model.predict_proba(X[te])
                rows.append({"scheme":scheme_name,"encoder":encoder,"repeat":rep,
                             "auc":roc_auc_score(y,p),"accuracy":accuracy_score(y,p>=.5)})
                for i, pi in enumerate(p):
                    pred_rows.append({"patient_index":i,"repeat":rep,"scheme":scheme_name,
                                      "encoder":encoder,"y_true":int(y[i]),"probability":float(pi)})
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def summarize_cv(cvdf):
    s = cvdf.groupby(["scheme","encoder"])[["auc","accuracy"]].agg(["mean","std"]).reset_index()
    s.columns = ["scheme","encoder","auc_mean","auc_sd","accuracy_mean","accuracy_sd"]
    return s


def ensemble_metrics(pred_df):
    out=[]
    for (scheme,encoder), g in pred_df.groupby(["scheme","encoder"]):
        ens = g.groupby("patient_index").agg(y_true=("y_true","first"), probability=("probability","mean"))
        m=classification_metrics(ens.y_true.to_numpy(), ens.probability.to_numpy())
        m.update({"scheme":scheme,"encoder":encoder})
        out.append(m)
    return pd.DataFrame(out)


def delong_table(pred_df):
    rows=[]
    for encoder in ENCODERS:
        for contender in ["ontology_default","ontology_bridged"]:
            zs=[]; diffs=[]; ps=[]
            for rep in sorted(pred_df.repeat.unique()):
                a=pred_df[(pred_df.scheme=="manual")&(pred_df.encoder==encoder)&(pred_df.repeat==rep)].sort_values("patient_index")
                b=pred_df[(pred_df.scheme==contender)&(pred_df.encoder==encoder)&(pred_df.repeat==rep)].sort_values("patient_index")
                diff,z,p=delong_pvalue(a.y_true.to_numpy(),a.probability.to_numpy(),b.probability.to_numpy())
                diffs.append(diff); zs.append(z); ps.append(p)
            zc,pc=stouffer(zs)
            rows.append({"comparison":f"manual vs {contender}","encoder":encoder,
                         "mean_auc_diff_manual_minus_contender":np.nanmean(diffs),
                         "combined_z":zc,"combined_p":pc})
    return pd.DataFrame(rows)


def fit_full_encoders(groups, encoder, X, y, names, seed):
    # Fit complete CBM only to obtain its encoders and concept scaler. The head is discarded in leakage audit.
    return CBM(groups, encoder=encoder, seed=seed).fit(X,y,names)


def leakage_audit(X,y,names,schemes,repeats=10,folds=10,seed=42):
    rows=[]
    for scheme_name,groups in schemes.items():
        for encoder in ENCODERS:
            # Full-data concept encoders = deliberately leaky representation.
            full = fit_full_encoders(groups,encoder,X,y,names,seed)
            Cfull = full.transform_concepts(X)
            proper_aucs=[]; leaky_aucs=[]
            for rep in range(repeats):
                splits=list(StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed+rep).split(X,y))
                pp=np.zeros(len(y)); lp=np.zeros(len(y))
                for fold,(tr,te) in enumerate(splits):
                    proper=CBM(groups,encoder=encoder,seed=seed+rep*100+fold).fit(X[tr],y[tr],names)
                    pp[te]=proper.predict_proba(X[te])
                    head=LogisticRegression(penalty="l2",solver="liblinear",C=1.0,max_iter=5000,
                                            random_state=seed+rep*100+fold).fit(Cfull[tr],y[tr])
                    lp[te]=head.predict_proba(Cfull[te])[:,1]
                proper_aucs.append(roc_auc_score(y,pp)); leaky_aucs.append(roc_auc_score(y,lp))
            pa=float(np.mean(proper_aucs)); la=float(np.mean(leaky_aucs))
            rows.append({"scheme":scheme_name,"encoder":encoder,"proper_auc":pa,
                         "leaky_full_data_encoder_auc":la,"auc_inflation":la-pa})
    return pd.DataFrame(rows)


def concept_impurity(X,y,names,schemes,seed=42):
    rows=[]
    for scheme_name,groups in schemes.items():
        # Workbook reports one scheme-level value; use supervised concept scores.
        model=CBM(groups,encoder="supervised",seed=seed).fit(X,y,names)
        C=model.transform_concepts(X)
        vals=[]
        for i in range(C.shape[1]):
            for j in range(C.shape[1]):
                if i==j: continue
                # For one predictor with intercept, R2 is squared Pearson correlation.
                r=np.corrcoef(C[:,i],C[:,j])[0,1]
                vals.append(float(r*r))
        rows.append({"scheme":scheme_name,"mean_offdiag_r2":np.mean(vals),"max_offdiag_r2":np.max(vals)})
    return pd.DataFrame(rows)


def concept_stability(X,y,names,schemes,B=100,topk=3,seed=123):
    rng=np.random.default_rng(seed)
    rows=[]
    for scheme_name,groups in schemes.items():
        for encoder in ENCODERS:
            rankings=[]
            for b in range(B):
                idx=rng.integers(0,len(y),len(y))
                # Retry degenerate bootstrap if needed.
                while len(np.unique(y[idx]))<2:
                    idx=rng.integers(0,len(y),len(y))
                m=CBM(groups,encoder=encoder,seed=seed+b).fit(X[idx],y[idx],names)
                rankings.append(m.concept_effects())
            rows.append({"scheme":scheme_name,"encoder":encoder,"rank_sd":rank_sd(rankings),
                         "topk_jaccard":topk_jaccard(rankings,min(topk,len(groups))),"bootstraps":B})
    return pd.DataFrame(rows)


def quantile_higher(x,q):
    try:
        return float(np.quantile(x,q,method="higher"))
    except TypeError:
        return float(np.quantile(x,q,interpolation="higher"))


def reliability(X,y,names,schemes,repeats=10,folds=10,alpha=0.10,seed=42):
    rows=[]
    selective_store={}
    # Reliability is evaluated for all six conditions.
    for scheme_name,groups in schemes.items():
        for encoder in ENCODERS:
            eces=[]; platt_eces=[]; iso_eces=[]; coverages=[]; singletons=[]
            all_test_records=[]
            for rep in range(repeats):
                cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed+rep)
                rep_p=np.full(len(y),np.nan)
                for fold,(outer_tr,te) in enumerate(cv.split(X,y)):
                    fit_idx,cal_idx=train_test_split(outer_tr,test_size=0.2,stratify=y[outer_tr],
                                                     random_state=seed+rep*100+fold)
                    m=CBM(groups,encoder=encoder,seed=seed+rep*100+fold).fit(X[fit_idx],y[fit_idx],names)
                    pcal=m.predict_proba(X[cal_idx]); pte=m.predict_proba(X[te]); rep_p[te]=pte
                    # Platt on raw logit score.
                    eps=1e-6
                    xcal=np.log(np.clip(pcal,eps,1-eps)/(1-np.clip(pcal,eps,1-eps))).reshape(-1,1)
                    xte=np.log(np.clip(pte,eps,1-eps)/(1-np.clip(pte,eps,1-eps))).reshape(-1,1)
                    pl=LogisticRegression(solver="lbfgs",max_iter=2000).fit(xcal,y[cal_idx])
                    ppl=pl.predict_proba(xte)[:,1]
                    iso=IsotonicRegression(out_of_bounds="clip").fit(pcal,y[cal_idx])
                    piso=iso.predict(pte)
                    eces.append(expected_calibration_error(y[te],pte))
                    platt_eces.append(expected_calibration_error(y[te],ppl))
                    iso_eces.append(expected_calibration_error(y[te],piso))

                    # Class-conditional split conformal using raw probabilities.
                    qs={}
                    for c in [0,1]:
                        mask=y[cal_idx]==c
                        pc = pcal if c==1 else 1-pcal
                        scores=1-pc[mask]
                        if len(scores)==0:
                            qs[c]=1.0
                        else:
                            qlev=min(1.0,np.ceil((len(scores)+1)*(1-alpha))/len(scores))
                            qs[c]=quantile_higher(scores,qlev)
                    p0=1-pte; p1=pte
                    inc0=(1-p0)<=qs[0]; inc1=(1-p1)<=qs[1]
                    covered=np.where(y[te]==0,inc0,inc1)
                    coverages.append(np.mean(covered))
                    singletons.append(np.mean((inc0.astype(int)+inc1.astype(int))==1))
                all_test_records.append(rep_p)
            P=np.vstack(all_test_records)
            ensemble=np.nanmean(P,axis=0)
            selective_store[(scheme_name,encoder)]=ensemble
            rows.append({"scheme":scheme_name,"encoder":encoder,
                         "ece_raw":np.mean(eces),"ece_platt":np.mean(platt_eces),"ece_isotonic":np.mean(iso_eces),
                         "conformal_coverage_90":np.mean(coverages),"singleton_fraction":np.mean(singletons)})

    sel=[]
    for (scheme,encoder),p in selective_store.items():
        conf=np.abs(p-.5)
        order=np.argsort(-conf)
        for coverage in [1.0,.9,.8,.7,.6,.5]:
            n=max(1,int(round(len(y)*coverage)))
            idx=order[:n]
            sel.append({"scheme":scheme,"encoder":encoder,"coverage":coverage,
                        "accuracy":accuracy_score(y[idx],p[idx]>=.5)})
    return pd.DataFrame(rows),pd.DataFrame(sel)


def completeness(cv_summary):
    ref=float(cv_summary[(cv_summary.scheme=="raw_l2_reference")].auc_mean.iloc[0])
    rows=[]
    for _,r in cv_summary[cv_summary.encoder=="supervised"].iterrows():
        rows.append({"scheme":r.scheme,"auc":r.auc_mean,"reference_auc":ref,
                     "auc_based_completeness":(r.auc_mean-.5)/(ref-.5),
                     "interpretability_tax":ref-r.auc_mean})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--repeats",type=int,default=10)
    ap.add_argument("--folds",type=int,default=10)
    ap.add_argument("--bootstraps",type=int,default=100)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--skip-reliability",action="store_true")
    args=ap.parse_args()

    X,y,names=load_data(); schemes=load_groups()
    cv,pred=run_cv(X,y,names,schemes,args.repeats,args.folds,args.seed)
    cv.to_csv(RESULTS/"cv_by_repeat.csv",index=False)
    pred.to_csv(RESULTS/"oof_predictions.csv",index=False)
    summary=summarize_cv(cv); summary.to_csv(RESULTS/"cv_summary.csv",index=False)
    ensemble_metrics(pred).to_csv(RESULTS/"classification_metrics.csv",index=False)
    delong_table(pred).to_csv(RESULTS/"delong_comparisons.csv",index=False)
    leak=leakage_audit(X,y,names,schemes,args.repeats,args.folds,args.seed)
    leak.to_csv(RESULTS/"leakage_audit.csv",index=False)
    concept_impurity(X,y,names,schemes,args.seed).to_csv(RESULTS/"concept_impurity.csv",index=False)
    concept_stability(X,y,names,schemes,args.bootstraps,3,args.seed+1000).to_csv(RESULTS/"stability.csv",index=False)
    completeness(summary).to_csv(RESULTS/"predictive_completeness.csv",index=False)
    if not args.skip_reliability:
        rel,sel=reliability(X,y,names,schemes,args.repeats,args.folds,.10,args.seed)
        rel.to_csv(RESULTS/"reliability.csv",index=False)
        sel.to_csv(RESULTS/"selective_abstention.csv",index=False)
    print(summary.to_string(index=False))
    print("\nLeakage audit:\n",leak.to_string(index=False))
    print(f"\nResults written to {RESULTS}")


if __name__ == "__main__":
    main()
