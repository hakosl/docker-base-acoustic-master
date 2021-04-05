from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
import scipy.stats
from sklearn.metrics import roc_auc_score
import numpy as np

def calculate_explicitness(z, factors):
    n_z = z.shape[1]
    n_f = factors.shape[1]
    mean_aucs = np.zeros(n_f)
    all_aucs = []
    all_aucs_factors=[]
    all_aucs_factor_vals=[]
    for factor_idx in range(n_f):
        model = LogisticRegression(C=1e10)
        model.fit(z, factors[:,factor_idx])
        preds = model.predict_proba(z)
        aucs=[]
        for val_idx, val in enumerate(model.classes_):
            y_true = factors[:,factor_idx] == val
            y_pred = preds[:,val_idx]
            auc = roc_auc_score(y_true,y_pred)
            aucs.append(auc)
            all_aucs_factor_vals.append(val)
        mean_aucs[factor_idx] = np.mean(aucs)
        all_aucs.extend(aucs)
        all_aucs_factors.extend([factor_idx] * len(aucs))
    return mean_aucs, all_aucs, all_aucs_factors, all_aucs_factor_vals