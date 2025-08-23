import os
from typing import List, Tuple, Dict, Optional, Union, Any
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import math

# Local TabPFN implementation
from tabpfn import TabPFNClassifier as LocalTabPFNClassifier, TabPFNRegressor as LocalTabPFNRegressor
# Remote client implementation
try:
    from tabpfn_client import TabPFNClassifier as RemoteTabPFNClassifier, \
                                  TabPFNRegressor as RemoteTabPFNRegressor
    _HAS_REMOTE = True
except ImportError:
    RemoteTabPFNClassifier = None
    RemoteTabPFNRegressor = None
    _HAS_REMOTE = False

# ==== PATCH: force TabPFN to ignore the 10k-sample limit ====
# import tabpfn.classifier as _tpcls

# # Keep a reference to the original __init__
# _orig_clf_init = _tpcls.TabPFNClassifier.__init__
# def _patched_clf_init(self, *args, **kwargs):
#     # 1) run the real init
#     _orig_clf_init(self, *args, **kwargs)
#     # 2) then override the limit flag
#     self.ignore_pretraining_limits = True

# # Apply the patch
# _tpc = _tpcls.TabPFNClassifier
# _tpc.__init__ = _patched_clf_init
# ==== end patch ====


def split_train_test(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df, test_size=test_size,
        stratify=df[label_col],
        random_state=random_state
    )
    return train_df, test_df


def subsample_train_set(
    train_df: pd.DataFrame,
    label_col: str,
    subsample_ratio: float,
    random_state: int = 42
) -> pd.DataFrame:
    if not 0 < subsample_ratio <= 1:
        raise ValueError("subsample_ratio must be between 0 and 1.")
    subsampled_df, _ = train_test_split(
        train_df,
        train_size=subsample_ratio,
        stratify=train_df[label_col],
        random_state=random_state
    )
    return subsampled_df


def fit_tabpfn(
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    device: str = "cpu",
    is_classifier: bool = True,
    use_remote: bool = False,
    random_state: Optional[int] = 42
) -> Any:
    """
    Build the TabPFN context by a single fit/inference pass.
    If use_remote=True and client available, uses remote client classes.
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    if use_remote:
        if not _HAS_REMOTE or RemoteTabPFNClassifier is None:
            raise ImportError("tabpfn_client is not installed or unavailable.")
        model = RemoteTabPFNClassifier() if is_classifier else RemoteTabPFNRegressor()
    else:
        model = LocalTabPFNClassifier(device=device) if is_classifier else LocalTabPFNRegressor(device=device)
    model.fit(X_arr, y)
    return model


def predict_tabpfn(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_pred = model.predict(X_arr)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_arr)
    except (AttributeError, NotImplementedError):
        # model may not support predict_proba
        y_proba = None
    return y_pred, y_proba


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    is_classifier: bool = True
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    if is_classifier:
        results["accuracy"] = accuracy_score(y_true, y_pred)
        results["precision"] = precision_score(y_true, y_pred,
                                            average="weighted", zero_division=0)
        results["recall"] = recall_score(y_true, y_pred,
                                         average="weighted", zero_division=0)
        results["f1"] = f1_score(y_true, y_pred,
                                 average="weighted", zero_division=0)
        if y_proba is not None:
            try:
                results["auc"] = roc_auc_score(y_true, y_proba[:, 1])
            except Exception:
                results["auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
    else:
        mse = mean_squared_error(y_true, y_pred)
        results["mse"] = mse
        results["rmse"] = np.sqrt(mse)
        results["r2"] = r2_score(y_true, y_pred)
    return results


def run_tabpfn_experiment_sets(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.2,
    subsample_ratios: List[Optional[float]] = None,       # allow None for flag-based mode
    subsample_sets: Optional[List[str]] = None,      # e.g. ['10K','50K','100K']
    device: str = "cpu",
    is_classifier: bool = True,
    random_state: int = 42,
    use_remote: bool = False
) -> Dict[Union[float,str], Dict[str, float]]:
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # prepare train/test (and validation) depending on mode
    if subsample_sets:
        # confirm flag columns exist
        missing = [s for s in subsample_sets if f"train_{s}" not in df.columns]
        if missing:
            raise ValueError(f"Missing train_* columns: {missing}")
        test_df = df[df['test']]
        val_df  = df[df['validation']]
        train_df = None
    else:
        train_df, test_df = split_train_test(df, label_col, test_size, random_state)

    # define metadata flags to exclude from features
    # flag_cols = [
    #     'test', 'validation',
    #     'train_50','train_100','train_200','train_500',
    #     'train_1K','train_5K','train_10K',
    #     'train_50K','train_100K','train_500K',
    #     'train_1M','train_5M'
    # ]
    # a more robust definition
    flag_cols = [c for c in df.columns if c in {"test","validation"} or c.startswith("train_")]
    
    X_test = test_df.drop(columns=[label_col] + flag_cols)
    y_test = test_df[label_col].values

    results: Dict[float, Dict[str, float]] = {}
    cms: List[Tuple[float, np.ndarray, List[Union[int,str]]]] = []

    # define experiment iterator: (ratio, suffix)
    if subsample_sets:
        exp_iter = [(None, s) for s in subsample_sets]
    else:
        exp_iter = [(r, None) for r in subsample_ratios]
    for ratio, suffix in exp_iter:
        # select training subset
        if suffix is not None:
            sub = df[df[f"train_{suffix}"]]
        else:
            sub = subsample_train_set(train_df, label_col, ratio, random_state)
        X_sup = sub.drop(columns=[label_col] + flag_cols)
        y_sup = sub[label_col].values
        
        model = fit_tabpfn(
            X_sup, y_sup,
            device=device, is_classifier=is_classifier,
            use_remote=use_remote,
        )

        y_pred, y_proba = predict_tabpfn(model, X_test)
        metrics = evaluate_predictions(y_test, y_pred, y_proba, is_classifier)
        key = suffix if suffix is not None else ratio
        results[key] = metrics

        # ——— Print test metrics ———
        n_train = len(sub)
        if subsample_sets:
            # flag‐based mode: use the fixed validation set
            n_val = len(val_df)
        else:
            # legacy mode: remaining of train_df
            n_val = len(train_df) - n_train
        n_test = len(test_df)
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        # label by suffix in flag‐mode, or by numeric ratio otherwise
        label = suffix if suffix is not None else ratio
        print(f"Test metrics (ratio={label}, train={n_train:,}, val={n_val:,}, test={n_test:,}):")
        print(f"  {metrics_str}")

        # collect for confusion‐matrix plotting (use suffix when provided)
        if is_classifier:
            labels = sorted(np.unique(y_test).tolist())
            cm     = confusion_matrix(y_test, y_pred, labels=labels)
            key    = suffix if suffix is not None else ratio
            cms.append((key, cm, labels))

    if is_classifier and cms:
        n = len(cms)
        cols = min(3, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = np.array(axes).reshape(-1)
        for ax, (ratio, cm, labels) in zip(axes, cms):
            im = ax.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
            ax.set(
                xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
                xticklabels=labels, yticklabels=labels,
                xlabel='Pred', ylabel='True', title=f'ratio={ratio}'
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha='center', va='center',
                            color='white' if cm[i, j] > thresh else 'black')
        for ax in axes[len(cms):]:
            ax.axis('off')
        fig.tight_layout(); plt.show()

    if results:
        # plot in natural order: numeric or suffix
        def _to_num(k):
            if isinstance(k, str) and k[-1] in ('K','M'):
                return float(k[:-1]) * (1e3 if k[-1]=='K' else 1e6)
            return float(k)
        order = sorted(results.keys(), key=_to_num)
        metrics_list = list(next(iter(results.values())).keys())
        plt.figure(figsize=(8,6))
        for m in metrics_list:
            vals = [results[k][m] for k in order]
            plt.plot(order, vals, marker='o', label=m)
        plt.xlabel('Subsample Ratio')
        plt.ylabel('Metric Value')
        plt.title('Metrics vs Subsample Ratio')
        plt.legend(); plt.show()

        # ——— Aggregate into DataFrame for reporting ———
        # rows=subsample sizes, cols=metrics
        df = pd.DataFrame(results).T
        # sort keys like '1K','5K','1M'
        df = df.sort_index(key=lambda idx: [
            float(k[:-1]) * (1e3 if k.endswith('K') else 1e6)
            if isinstance(k, str) and k[-1] in ('K','M')
            else float(k)
            for k in idx
        ])
        df = df.T          # metrics as rows
        df = df.round(4)
        print("\n=== Summary table (rows=metrics, cols=subsample sizes) ===")
        print(df.to_string())

    return results
