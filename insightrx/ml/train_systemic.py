"""
Patient-level systemic association models (research signals, P1).

For each target we compare, on the validation split:
  image      mean-pooled image embeddings (+ ensemble systemic-head logits) -> PCA -> logistic regression
  metadata   age, sex, diabetes duration, insulin, oral treatment        -> gradient boosting
  fused      both                                                      -> logistic regression
The best variant by validation AUROC is kept; the image-only variant is kept as a fallback when metadata is
missing. A head is enabled in the app only if its validation AUROC >= SYSTEMIC_ENABLE_MIN_AUROC.

  python -m insightrx.ml.train_systemic
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .data import load_tables
from .evaluate import binary_metrics, bootstrap_ci, load_split_outputs, seeds_available


def patient_image_features(idx, z, emb):
    """Mean over a patient's images of [embedding, systemic-head logits]."""
    sys_logits = np.stack([z[t] for t in C.SYSTEMIC_TARGETS], 1)
    X = np.concatenate([emb, sys_logits], 1)
    df = pd.DataFrame(X)
    df['patient_id'] = idx['patient_id'].values
    g = df.groupby('patient_id').mean()
    return g.index.values, g.values


def metadata_matrix(patients, pids):
    p = patients.set_index('patient_id').loc[pids]
    return p[C.METADATA_FEATURES].astype(float).values


def image_model(n_comp):
    return make_pipeline(StandardScaler(), PCA(n_components=n_comp, random_state=C.SEED),
                         LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced'))


def metadata_model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                          class_weight='balanced', random_state=C.SEED)


def fused_model(n_img):
    from sklearn.compose import ColumnTransformer
    ct = ColumnTransformer([
        ('img', make_pipeline(StandardScaler(), PCA(n_components=n_img, random_state=C.SEED)), slice(0, -len(C.METADATA_FEATURES))),
        ('meta', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()),
         slice(-len(C.METADATA_FEATURES), None))])
    return make_pipeline(ct, LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced'))


def youden_threshold(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr - fpr)])


def main():
    seeds = seeds_available()
    _, patients = load_tables()
    data = {}
    for name in ('train', 'val', 'test'):
        idx, z, emb = load_split_outputs(name, seeds)
        pids, Xi = patient_image_features(idx, z, emb)
        data[name] = {'pids': pids, 'img': Xi, 'meta': metadata_matrix(patients, pids)}
    n_comp = int(min(64, len(data['train']['pids']) // 4))
    pat = patients.set_index('patient_id')

    bundle, report = {'seeds': seeds, 'metadata_features': C.METADATA_FEATURES, 'heads': {}}, {}
    for t in C.SYSTEMIC_TARGETS:
        y = {k: pat.loc[d['pids'], t].values.astype(float) for k, d in data.items()}
        ok = {k: ~np.isnan(v) for k, v in y.items()}
        inputs = {'image': lambda d: d['img'], 'metadata': lambda d: d['meta'],
                  'fused': lambda d: np.concatenate([d['img'], d['meta']], 1)}
        builders = {'image': lambda: image_model(n_comp), 'metadata': metadata_model, 'fused': lambda: fused_model(n_comp)}
        fitted, val_auc, test = {}, {}, {}
        for v in ('image', 'metadata', 'fused'):
            m = builders[v]()
            # metadata NaNs: HGB handles them natively; the fused pipeline imputes
            m.fit(inputs[v](data['train'])[ok['train']], y['train'][ok['train']])
            fitted[v] = m
            sv = m.predict_proba(inputs[v](data['val'])[ok['val']])[:, 1]
            val_auc[v] = float(roc_auc_score(y['val'][ok['val']], sv))
            thr = youden_threshold(y['val'][ok['val']], sv)
            st = m.predict_proba(inputs[v](data['test'])[ok['test']])[:, 1]
            yt = y['test'][ok['test']]
            tm = binary_metrics(yt, st, thr)
            tm['auroc_ci95'] = bootstrap_ci([np.array([[a, b]]) for a, b in zip(yt, st)],
                                            lambda a: roc_auc_score(a[:, 0], a[:, 1]))
            tm['val_auroc'] = val_auc[v]
            test[v] = tm
        best = max(val_auc, key=val_auc.get)
        # release gate (SG1/SG2): point estimate AND patient-bootstrap lower bound on validation, so heads backed by
        # a handful of validation positives stay 'Not evaluated'
        sv = fitted[best].predict_proba(inputs[best](data['val'])[ok['val']])[:, 1]
        yv = y['val'][ok['val']]
        val_ci = bootstrap_ci([np.array([[a, b]]) for a, b in zip(yv, sv)], lambda a: roc_auc_score(a[:, 0], a[:, 1]))
        enabled = val_auc[best] >= C.SYSTEMIC_ENABLE_MIN_AUROC and val_ci is not None and val_ci[0] >= C.SYSTEMIC_ENABLE_MIN_LOWER
        bundle['heads'][t] = {'best_variant': best, 'enabled': bool(enabled), 'models': fitted,
                              'thresholds': {v: test[v]['threshold'] for v in test},
                              'val_auroc': val_auc,
                              'test_auroc': {v: test[v]['auroc'] for v in test},
                              'counts': {k: {'pos': int((y[k] == 1).sum()), 'neg': int((y[k] == 0).sum()),
                                             'unknown': int(np.isnan(y[k]).sum())} for k in y}}
        report[t] = {'best_variant': best, 'enabled': bool(enabled), 'variants': test, 'val_auroc_ci95': val_ci,
                     'counts': bundle['heads'][t]['counts']}
        print(f"{t:28s} val AUROC " + ' '.join(f"{v}={val_auc[v]:.3f}" for v in val_auc) +
              f" | test AUROC " + ' '.join(f"{v}={test[v]['auroc']:.3f}" for v in test) +
              f" | best={best} val CI {val_ci} enabled={enabled}", flush=True)

    joblib.dump(bundle, os.path.join(C.CKPT_DIR, 'systemic.joblib'))
    json.dump(report, open(os.path.join(C.METRICS_DIR, 'systemic_metrics.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
