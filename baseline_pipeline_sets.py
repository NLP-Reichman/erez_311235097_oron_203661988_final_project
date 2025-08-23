import os
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt


def detect_feature_types(
    df: pd.DataFrame,
    label_col: str = "label"
) -> Tuple[List[str], List[str]]:
    """
    Identify numeric vs. categorical feature columns.
    Returns (numeric_cols, categorical_cols).
    """
    # omit all split-indicator flags from the feature set
    flag_cols = {
        "test", "validation",
        "train_50","train_100","train_200","train_500",
        "train_1K","train_5K","train_10K",
        "train_50K","train_100K","train_500K",
        "train_1M","train_5M"
    }
    feature_cols = [
        col for col in df.columns
        if col != label_col and col not in flag_cols
    ]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]
    return numeric_cols, categorical_cols


def detect_target_info(
    y: pd.Series
) -> Dict[str, Union[int, str]]:
    """
    Inspect target series:
      - n_classes
      - task: "binary" | "multiclass"
      - dtype
    """
    n_classes = y.nunique()
    if n_classes == 2:
        task = "binary"
    elif n_classes > 2:
        task = "multiclass"
    else:
        raise ValueError("Target must have at least 2 classes for classification.")
    return {"n_classes": n_classes, "task": task, "dtype": str(y.dtype)}


def split_train_test(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split into train / test.
    Returns (train_df, test_df).
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state
    )
    return train_df, test_df


def subsample_and_split_val(
    train_df: pd.DataFrame,
    label_col: str,
    subsample_ratio: float,
    val_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From train_df:
      1) subsample subsample_ratio (stratified)
      2) split into sub_train / val by val_ratio (stratified)
    Returns (sub_train_df, val_df).
    """
    if not 0 < subsample_ratio <= 1:
        raise ValueError("subsample_ratio must be between 0 and 1.")
    # If subsample_ratio == 1.0 keep the full training set, since scikit-learn forbids train_size=1.0.
    if subsample_ratio == 1.0:
        subsample_df = train_df.copy()
    else:
        subsample_df, _ = train_test_split(
            train_df,
            train_size=subsample_ratio,
            stratify=train_df[label_col],
            random_state=random_state
        )
    # ensure each class has at least two samples for the subsequent validation split
    if subsample_df[label_col].value_counts().min() < 2:
        raise ValueError(
            "Too few samples per class after subsampling; "
            "reduce 'val_ratio' or increase 'subsample_ratio'."
        )
    sub_train_df, val_df = train_test_split(
        subsample_df,
        test_size=val_ratio,
        stratify=subsample_df[label_col],
        random_state=random_state
    )
    return sub_train_df, val_df


def make_catboost_pools(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    categorical_cols: List[str],
    label_col: str
) -> Tuple[Pool, Pool]:
    """
    Build CatBoost Pool objects for train & val.
    """
    train_pool = Pool(
        data=train_df[feature_cols],
        label=train_df[label_col],
        cat_features=categorical_cols
    )
    val_pool = Pool(
        data=val_df[feature_cols],
        label=val_df[label_col],
        cat_features=categorical_cols
    )
    return train_pool, val_pool


def get_catboost_model(
    task: str,
    params: Optional[Dict] = None
) -> Union[CatBoostClassifier, CatBoostRegressor]:
    """
    Instantiate CatBoost model for binary or multiclass tasks
    """
    # Make a protective copy and fill in sensible defaults so that callers can reuse the same dict safely.
    params = params.copy() if params else {}
    params.setdefault("iterations", 100)
    params.setdefault("learning_rate", 0.1)
    params.setdefault("depth", 6)
    # log extra classification metrics so the CatBoost dynamic plot shows them.
    params.setdefault("eval_metric", "Logloss")
    if "custom_metric" not in params:
        params["custom_metric"] = ["AUC", "Precision", "Recall", "F1"]
    params.setdefault("verbose", False)

    # Optional automatic balancing: caller can set balance=True in params to compute weights.
    balance_flag = params.pop("balance", False)
    if balance_flag and "class_weights" not in params and "auto_class_weights" not in params:
        # Build simple balanced weights (total / (n_classes * count)) based on y in train_pool
        # We will compute these on-the-fly later – here we just mark the intent.
        params["auto_class_weights"] = "Balanced"

    if task in {"binary", "multiclass"}:
        return CatBoostClassifier(**params)
    raise ValueError(f"Unsupported task type: {task}")


def train_and_plot_learning_curve(
    model: Union[CatBoostClassifier, CatBoostRegressor],
    train_pool: Pool,
    val_pool: Pool,
    ratio: float,
    train_size: int,
    val_size: int,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Fit model and rely on CatBoost's built-in interactive learning-curve plot.
    A descriptive title is printed **before** the dynamic plot appears so it
    shows up just above the widget in a notebook.
    Returns evals_result dict.
    """
    # Print a header so it's visible above the CatBoost widget
    print(
        f"Learning Curves (ratio={ratio}, train={train_size:,}, val={val_size:,})"
    )

    # CatBoost's built-in interactive plot (requires ipywidgets in Jupyter)
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        plot=True,
    )

    return model.get_evals_result()


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    task: str = "binary"
) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall, f1, and roc-auc (if available).
    """
    results: Dict[str, float] = {}
    results["accuracy"] = accuracy_score(y_true, y_pred)
    if task == "binary":
        results["precision"] = precision_score(y_true, y_pred)
        results["recall"] = recall_score(y_true, y_pred)
        results["f1"] = f1_score(y_true, y_pred)
        if y_proba is not None:
            # assume proba for positive class at column 1
            results["auc"] = roc_auc_score(y_true, y_proba[:, 1])
    elif task == "multiclass":
        results["precision"] = precision_score(y_true, y_pred, average="weighted")
        results["recall"] = recall_score(y_true, y_pred, average="weighted")
        results["f1"] = f1_score(y_true, y_pred, average="weighted")
        if y_proba is not None:
            results["auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
    else:
        raise ValueError(f"Unsupported task type: {task}")
    return results


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[Union[str, int]],
    title: str = ""
) -> None:
    """
    Show confusion matrix via matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    fig.tight_layout()
    plt.show()


def plot_metrics_vs_ratio(
    results: Dict[Union[float, str], Dict[str, float]], metrics: List[str]
) -> None:
    """
    Plot chosen metrics over different subsample ratios.
    """
    # sort keys numerically (handle strings like '5K', '50K', '100K', '500K')
    def _to_num(k):
        if isinstance(k, str) and k[-1] in ('K','M'):
            factor = 1e3 if k[-1] == 'K' else 1e6
            return float(k[:-1]) * factor
        return float(k)
    ratios = sorted(results.keys(), key=_to_num)
    plt.figure(figsize=(8, 6))
    for metric in metrics:
        values = [results[r].get(metric, None) for r in ratios]
        plt.plot(ratios, values, marker='o', label=metric)
    plt.xlabel("Subsample Ratio")
    plt.ylabel("Metric value")
    plt.title("Metrics vs Subsample Ratio")
    plt.legend()
    plt.show()


def run_baseline_experiment_sets(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.2,
    subsample_ratios: Optional[List[float]] = None,
    subsample_sets: Optional[List[str]] = None,
    val_ratio: float = 0.2,
    catboost_params: Optional[Dict] = None,
    random_state: int = 42
) -> Dict[float, Dict[str, float]]:
    """
    Full pipeline for notebook:
      1) detect types
      2) split train/test
      3) for each ratio:
           a) subsample & split val
           b) build pools
           c) init & train
           d) evaluate on test
           e) plot confusion
      4) plot metrics vs ratio
    Returns ratio → metrics.
    """
    catboost_params.setdefault("random_seed", random_state)
    # feature & target info
    numeric_cols, categorical_cols = detect_feature_types(df, label_col)
    feature_cols = numeric_cols + categorical_cols
    target_info = detect_target_info(df[label_col])
    
    # split train/test
    if subsample_sets:
        # flag-based mode: use precomputed columns
        test_df = df[df['test']]
        val_df  = df[df['validation']]
        train_df = None
    else:
        # legacy mode: stratified random split
        train_df, test_df = split_train_test(
            df, label_col, test_size=test_size, random_state=random_state
        )
    
    results: Dict[float, Dict[str, float]] = {}
    confusion_data: List[Tuple[float, np.ndarray, np.ndarray, List[Union[str, int]]]] = []

    # build experiment iterator:
    #  - legacy: (ratio, None) for each float ratio
    #  - flag-based: (None, suffix) for each sampling_experiment
    if subsample_sets:
        exp_iter = [(None, suffix) for suffix in subsample_sets]
    else:
        exp_iter = [(ratio, None) for ratio in subsample_ratios]

    for ratio, suffix in exp_iter:
        if suffix is not None:
            # flag-based: select pre-flagged train subset
            col = f"train_{suffix}"
            sub_train_df = df[df[col]]
        else:
            # legacy sampling + val split
            sub_train_df, val_df = subsample_and_split_val(
                train_df, label_col,
                subsample_ratio=ratio,
                val_ratio=val_ratio,
                random_state=random_state
            )
        train_pool, val_pool = make_catboost_pools(
            sub_train_df, val_df, feature_cols, categorical_cols, label_col
        )
        model = get_catboost_model(target_info['task'], params=catboost_params)
        _ = train_and_plot_learning_curve(
            model, train_pool, val_pool,
            ratio=(suffix or ratio),         # label plots by suffix when in flag mode
            train_size=len(sub_train_df),
            val_size=len(val_df)
        )

        # test evaluation
        X_test = test_df[feature_cols]
        y_test = test_df[label_col].values
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)
        except:
            y_proba = None
        metrics = evaluate_predictions(y_test, y_pred, y_proba, task=target_info['task'])
        key = suffix if suffix is not None else ratio
        results[key] = metrics

        # store confusion info for later joint plotting
        labels = sorted(np.unique(y_test).tolist())
        confusion_data.append((suffix or ratio, y_test, y_pred, labels))

        # Print metrics for transparency
        print(
            f"Test metrics (ratio={ratio}, train={len(sub_train_df)}, val={len(val_df)}, "
            f"test={len(test_df)}):\n  " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

    # Plot all confusion matrices together for easier comparison
    if confusion_data:
        n = len(confusion_data)
        cols = 3
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)  # flatten whether rows==1 or more

        # Plot each confusion matrix
        for idx, (rt, y_t, y_p, lbls) in enumerate(confusion_data):
            ax = axes[idx]
            cm = confusion_matrix(y_t, y_p, labels=lbls)
            ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(f"ratio={rt}")
            ax.set_xlabel("Pred")
            ax.set_ylabel("True")
            ax.set_xticks(np.arange(len(lbls)))
            ax.set_yticks(np.arange(len(lbls)))
            ax.set_xticklabels(lbls)
            ax.set_yticklabels(lbls)
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black"
                    )

        # Hide any unused subplots
        for ax in axes[n:]:
            ax.axis('off')

        fig.tight_layout()
        plt.show()

    # metrics vs ratio (bigger figure)
    if results:
        common_metrics = list(next(iter(results.values())).keys())
        plot_metrics_vs_ratio(results, common_metrics)

        # ——— Aggregate into DataFrame for reporting ———
        # results: { ratio1: {m1:val, m2:val...}, ratio2: {...}, ... }
        df = pd.DataFrame(results).T            # rows=ratios, cols=metrics
        df = df.sort_index(key=lambda idx: [
            float(k[:-1]) * (1e3 if k.endswith('K') else 1e6)
            if isinstance(k, str) and k[-1] in ('K','M')
            else float(k)
            for k in idx
        ])
        df = df.T                              # rows=metrics, cols=ratios
        df = df.round(4)
        print("\n=== Summary table (rows=metrics, cols=subsample sizes) ===")
        print(df.to_string())

    return results