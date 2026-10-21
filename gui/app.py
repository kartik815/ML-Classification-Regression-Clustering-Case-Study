from __future__ import annotations

"""Streamlit GUI for the 23CSE301 Machine Learning Capstone.

The app supports all three tracks:
- Regression: 10 required models, metrics, comparison and single-row predictions.
- Classification: 10 required models, metrics, confusion matrices and predictions.
- Clustering: K-Means and Agglomerative clustering, required metrics and plots.

Place each dataset in <project-root>/data/ or upload it from the relevant page.
"""

import hashlib
import inspect
import io
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    classification_report,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler, label_binarize
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.manifold import TSNE


# -----------------------------------------------------------------------------
# App configuration and paths
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Capstone Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_PATH = Path(__file__).resolve()
ROOT = APP_PATH.parents[1] if APP_PATH.parent.name.lower() in {"gui", "app"} else APP_PATH.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20

DATASET_CANDIDATES = {
    "regression": DATA_DIR / "energydata_complete.csv",
    "classification": DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv",
    "clustering": DATA_DIR / "diabetes_012_health_indicators_BRFSS2021.csv",
}

FEATURE_LABELS = {
    "lights": "Lights energy consumption (Wh)",
    "T1": "Kitchen temperature (°C)",
    "RH_1": "Kitchen humidity (%)",
    "T2": "Living room temperature (°C)",
    "RH_2": "Living room humidity (%)",
    "T3": "Laundry room temperature (°C)",
    "RH_3": "Laundry room humidity (%)",
    "T4": "Office temperature (°C)",
    "RH_4": "Office humidity (%)",
    "T5": "Bathroom temperature (°C)",
    "RH_5": "Bathroom humidity (%)",
    "T6": "North-side outside temperature (°C)",
    "RH_6": "North-side outside humidity (%)",
    "T7": "Ironing room temperature (°C)",
    "RH_7": "Ironing room humidity (%)",
    "T8": "Teenager room temperature (°C)",
    "RH_8": "Teenager room humidity (%)",
    "T9": "Parents room temperature (°C)",
    "RH_9": "Parents room humidity (%)",
    "T_out": "Outside temperature (°C)",
    "RH_out": "Outside humidity (%)",
    "Press_mm_hg": "Atmospheric pressure (mm Hg)",
    "Windspeed": "Wind speed (m/s)",
    "Visibility": "Visibility (km)",
    "Tdewpoint": "Dew point temperature (°C)",
}


# -----------------------------------------------------------------------------
# Common helpers
# -----------------------------------------------------------------------------
def dataset_signature(df: pd.DataFrame) -> str:
    """Lightweight signature used to avoid accidentally showing stale results."""
    cols = "|".join(map(str, df.columns))
    sample = df.head(50).to_csv(index=False).encode("utf-8", errors="ignore")
    meta = f"{df.shape}|{cols}|{hashlib.sha256(sample).hexdigest()}"
    return hashlib.sha256(meta.encode()).hexdigest()


def read_dataset(uploaded_file: Any, local_path: Path) -> tuple[pd.DataFrame | None, str | None]:
    try:
        if uploaded_file is not None:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file), uploaded_file.name
        if local_path.exists():
            return pd.read_csv(local_path), str(local_path)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
    return None, None


def display_upload_help(local_path: Path, track_name: str) -> None:
    st.info(
        f"No local dataset was found for {track_name}. Upload the CSV below, or place it at "
        f"`{local_path.relative_to(ROOT) if local_path.is_relative_to(ROOT) else local_path}` "
        "in your project."
    )


def numeric_column_defaults(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    defaults: dict[str, float] = {}
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        med = values.median()
        defaults[col] = float(med) if pd.notna(med) else 0.0
    return defaults


def safe_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except Exception:
        return default


def encoder_dense() -> OneHotEncoder:
    """Support both older and newer scikit-learn OneHotEncoder APIs."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_tabular_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers: list[tuple[str, Any, list[str]]] = []
    if num_cols:
        transformers.append(
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols)
        )
    if cat_cols:
        transformers.append(
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", encoder_dense())]), cat_cols)
        )
    if not transformers:
        raise ValueError("No usable predictor columns were found.")
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)


def stratified_sample(df: pd.DataFrame, target: str, n_rows: int) -> pd.DataFrame:
    if n_rows <= 0 or n_rows >= len(df):
        return df.copy()
    if df[target].nunique(dropna=True) > 1 and df[target].value_counts().min() >= 2:
        sampled, _ = train_test_split(
            df,
            train_size=n_rows,
            random_state=RANDOM_STATE,
            stratify=df[target],
        )
        return sampled.reset_index(drop=True)
    return df.sample(n=n_rows, random_state=RANDOM_STATE).reset_index(drop=True)


def plot_metric_cards(items: list[tuple[str, str]]) -> None:
    columns = st.columns(len(items))
    for col, (label, value) in zip(columns, items):
        with col:
            st.metric(label, value)


def show_download_csv(df: pd.DataFrame, filename: str, label: str) -> None:
    payload = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, payload, file_name=filename, mime="text/csv")


def figure_to_bytes(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Regression: preprocessing mirrors the project's Appliances notebook workflow.
# -----------------------------------------------------------------------------
def prepare_regression_data(df: pd.DataFrame, target: str) -> dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} was not found.")
    data = df.drop(columns=["date"], errors="ignore").copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target]).copy()
    y = data[target].astype(float)
    X_raw = data.drop(columns=[target]).copy()

    # The assigned appliances dataset is numeric apart from its timestamp.
    non_numeric = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X_raw = X_raw.drop(columns=non_numeric)
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    X_raw = X_raw.dropna(axis=1, how="all")
    if X_raw.shape[1] == 0:
        raise ValueError("No numeric predictor columns remain after cleaning.")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    imputer = SimpleImputer(strategy="mean")
    X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_raw.columns, index=X_train_raw.index)
    X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=X_raw.columns, index=X_test_raw.index)

    if "lights" in X_train.columns:
        X_train["lights"] = np.log1p(X_train["lights"].clip(lower=0))
        X_test["lights"] = np.log1p(X_test["lights"].clip(lower=0))

    clip_bounds: dict[str, tuple[float, float]] = {}
    for col in X_train.columns:
        q1 = float(X_train[col].quantile(0.25))
        q3 = float(X_train[col].quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        clip_bounds[col] = (float(lower), float(upper))
        X_train[col] = X_train[col].clip(lower=lower, upper=upper)
        X_test[col] = X_test[col].clip(lower=lower, upper=upper)

    corr_frame = X_train.copy()
    corr_frame["__target__"] = y_train.to_numpy()
    corr = corr_frame.corr(numeric_only=True)
    to_remove: set[str] = set()
    cols = list(X_train.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            pair_corr = corr.loc[c1, c2]
            if pd.notna(pair_corr) and abs(pair_corr) > 0.90:
                target_corr1 = abs(corr.loc[c1, "__target__"])
                target_corr2 = abs(corr.loc[c2, "__target__"])
                if pd.isna(target_corr1):
                    to_remove.add(c1)
                elif pd.isna(target_corr2):
                    to_remove.add(c2)
                elif target_corr1 < target_corr2:
                    to_remove.add(c1)
                else:
                    to_remove.add(c2)

    final_columns = [c for c in cols if c not in to_remove]
    if not final_columns:
        raise ValueError("Correlation filtering removed all predictor columns.")
    X_train = X_train[final_columns]
    X_test = X_test[final_columns]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "raw_columns": list(X_raw.columns),
        "final_columns": final_columns,
        "removed_columns": sorted(to_remove),
        "imputer": imputer,
        "clip_bounds": clip_bounds,
        "scaler": scaler,
        "defaults": numeric_column_defaults(X_raw, list(X_raw.columns)),
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "X_test_raw": X_test_raw,
    }


def transform_regression_input(values: dict[str, float], prep: dict[str, Any]) -> np.ndarray:
    row = pd.DataFrame([{c: safe_number(values.get(c, prep["defaults"].get(c, 0.0))) for c in prep["raw_columns"]}])
    row = pd.DataFrame(prep["imputer"].transform(row), columns=prep["raw_columns"])
    if "lights" in row.columns:
        row["lights"] = np.log1p(row["lights"].clip(lower=0))
    for col, (lower, upper) in prep["clip_bounds"].items():
        row[col] = row[col].clip(lower=lower, upper=upper)
    row = row[prep["final_columns"]]
    return prep["scaler"].transform(row)


def fit_regression_suite(df: pd.DataFrame, target: str, run_cv: bool) -> dict[str, Any]:
    prep = prepare_regression_data(df, target)
    X_train = prep["X_train_scaled"]
    X_test = prep["X_test_scaled"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]

    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=10000),
        "ElasticNet Regression": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
        "Polynomial Regression (degree 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("poly_scale", StandardScaler()),
            ("regressor", LinearRegression()),
        ]),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
        "Support Vector Regressor (SVR)": SVR(C=10.0, kernel="rbf", epsilon=0.1),
        "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    }

    fitted: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    progress = st.progress(0, text="Training regression models...")
    status = st.empty()
    for i, (name, model) in enumerate(models.items(), start=1):
        status.write(f"Training **{name}** ({i}/{len(models)})")
        model.fit(X_train, y_train)
        pred = np.asarray(model.predict(X_test)).reshape(-1)
        fitted[name] = model
        predictions[name] = pred
        rows.append({
            "Algorithm": name,
            "R²": r2_score(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "MAE": mean_absolute_error(y_test, pred),
        })
        progress.progress(i / len(models), text=f"Trained {name}")
    progress.empty()
    status.empty()

    metrics = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
    if run_cv:
        cv_values: dict[str, float] = {}
        top_models = metrics.head(2)["Algorithm"].tolist()
        cv_progress = st.progress(0, text="Computing 5-fold CV R² for the top two models...")
        for i, name in enumerate(top_models, start=1):
            # n_jobs=1 avoids nested parallelism for tree ensembles.
            scores = cross_val_score(clone(fitted[name]), X_train, y_train, cv=5, scoring="r2", n_jobs=1)
            cv_values[name] = float(scores.mean())
            cv_progress.progress(i / len(top_models), text=f"Cross-validation: {name}")
        cv_progress.empty()
        metrics["5-fold CV R² (top 2)"] = metrics["Algorithm"].map(cv_values)

    return {
        "prep": prep,
        "models": fitted,
        "metrics": metrics,
        "predictions": predictions,
        "target": target,
        "test_y": y_test.to_numpy(),
        "test_x": prep["X_test_raw"],
        "signature": dataset_signature(df),
    }


# -----------------------------------------------------------------------------
# Classification: all ten required algorithms in a common comparison workflow.
# -----------------------------------------------------------------------------
def make_bagging_classifier() -> BaggingClassifier:
    base = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE)
    params = inspect.signature(BaggingClassifier).parameters
    estimator_key = "estimator" if "estimator" in params else "base_estimator"
    kwargs = {
        estimator_key: base,
        "n_estimators": 30,
        "max_samples": 0.7,
        "max_features": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    return BaggingClassifier(**kwargs)


def make_adaboost_classifier() -> AdaBoostClassifier:
    params = inspect.signature(AdaBoostClassifier).parameters
    estimator_key = "estimator" if "estimator" in params else "base_estimator"
    kwargs = {
        estimator_key: DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
        "n_estimators": 500,
        "learning_rate": 0.05,
        "random_state": RANDOM_STATE,
    }
    return AdaBoostClassifier(**kwargs)


def classification_estimators() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Support Vector Classifier (SVC)": SVC(C=1.0, kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=300, max_depth=20, max_features="sqrt", min_samples_split=15,
            min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "AdaBoost Classifier": make_adaboost_classifier(),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE,
        ),
        "Bagging Classifier (Decision Tree)": make_bagging_classifier(),
        "MLP Classifier (Neural Network)": MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.001,
            batch_size=256, learning_rate_init=0.001, max_iter=100,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
            random_state=RANDOM_STATE,
        ),
    }


def calculate_roc_auc(model: Any, y_true: pd.Series, probabilities: np.ndarray) -> float:
    classes = np.asarray(model.classes_)
    try:
        if len(classes) == 2:
            positive_column = 1 if len(classes) > 1 else 0
            return float(roc_auc_score((np.asarray(y_true) == classes[positive_column]).astype(int), probabilities[:, positive_column]))
        return float(roc_auc_score(y_true, probabilities, labels=classes, multi_class="ovr", average="weighted"))
    except Exception:
        return float("nan")


def fit_classification_suite(df: pd.DataFrame, target: str, max_rows: int) -> dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} was not found.")
    data = df.copy().drop_duplicates()
    data[target] = data[target].where(data[target].notna(), np.nan)
    data = data.dropna(subset=[target]).reset_index(drop=True)
    rare = data[target].value_counts()
    removable = rare[rare < 2].index
    if len(removable):
        data = data.loc[~data[target].isin(removable)].reset_index(drop=True)
    if data[target].nunique() < 2:
        raise ValueError("Classification requires at least two target classes with at least two rows each.")
    if len(data) > max_rows:
        data = stratified_sample(data, target, max_rows)

    X = data.drop(columns=[target]).copy()
    X = X.dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("No predictor columns remain after removing the target.")
    y = data[target].copy()
    if y.nunique() < 2:
        raise ValueError("The selected sample contains fewer than two target classes.")

    # Split before fitting imputers, encoders, and scalers to avoid leakage.
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )
    preprocessor = build_tabular_preprocessor(X_train_raw)
    X_train = np.asarray(preprocessor.fit_transform(X_train_raw), dtype=np.float64)
    X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=np.float64)

    fitted: dict[str, Any] = {}
    predicted: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray | None] = {}
    reports: dict[str, dict[str, Any]] = {}
    confusion: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    estimators = classification_estimators()
    progress = st.progress(0, text="Training classification models...")
    status = st.empty()
    for i, (name, model) in enumerate(estimators.items(), start=1):
        status.write(f"Training **{name}** ({i}/{len(estimators)})")
        try:
            model.fit(X_train, y_train)
            pred = np.asarray(model.predict(X_test))
            prob: np.ndarray | None = None
            if hasattr(model, "predict_proba"):
                prob = np.asarray(model.predict_proba(X_test))
            elif hasattr(model, "decision_function"):
                decision = np.asarray(model.decision_function(X_test))
                if decision.ndim == 1:
                    decision = np.column_stack([-decision, decision])
                # These scores are only a fallback; probability-based AUC is preferred.
                prob = decision

            precision = precision_score(y_test, pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, pred, average="weighted", zero_division=0)
            f1w = f1_score(y_test, pred, average="weighted", zero_division=0)
            roc_auc = calculate_roc_auc(model, y_test, prob) if prob is not None and hasattr(model, "classes_") and hasattr(model, "predict_proba") else float("nan")
            labels = list(model.classes_) if hasattr(model, "classes_") else sorted(pd.Series(y_test).unique().tolist())
            full_cm = np.zeros((len(labels), len(labels)), dtype=int)
            label_index = {label: idx for idx, label in enumerate(labels)}
            for actual, pred_label in zip(y_test, pred):
                if actual in label_index and pred_label in label_index:
                    full_cm[label_index[actual], label_index[pred_label]] += 1
            report = classification_report(y_test, pred, output_dict=True, zero_division=0)

            fitted[name] = model
            predicted[name] = pred
            probabilities[name] = prob
            reports[name] = report
            confusion[name] = full_cm
            rows.append({
                "Algorithm": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision (weighted)": precision,
                "Recall (weighted)": recall,
                "F1-score (weighted)": f1w,
                "ROC-AUC (OvR for multiclass)": roc_auc,
            })
        except Exception as exc:
            reports[name] = {"error": str(exc)}
        progress.progress(i / len(estimators), text=f"Processed {name}")
    progress.empty()
    status.empty()

    if not rows:
        reasons = "\n".join(f"{k}: {v.get('error', 'unknown error')}" for k, v in reports.items())
        raise RuntimeError(f"No classification model trained successfully.\n{reasons}")

    metrics = pd.DataFrame(rows).sort_values("F1-score (weighted)", ascending=False).reset_index(drop=True)
    raw_defaults: dict[str, Any] = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            med = pd.to_numeric(X[col], errors="coerce").median()
            raw_defaults[col] = float(med) if pd.notna(med) else 0.0
        else:
            modes = X[col].mode(dropna=True)
            raw_defaults[col] = modes.iloc[0] if not modes.empty else ""

    return {
        "models": fitted,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "predictions": predicted,
        "probabilities": probabilities,
        "reports": reports,
        "confusion": confusion,
        "classes": list(pd.Series(y).dropna().unique()),
        "model_classes": {name: list(model.classes_) for name, model in fitted.items()},
        "feature_columns": list(X.columns),
        "defaults": raw_defaults,
        "test_y": y_test.to_numpy(),
        "test_x_raw": X_test_raw,
        "target": target,
        "sample_rows": len(data),
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "signature": dataset_signature(df),
        "labels": sorted(pd.Series(y).unique().tolist(), key=lambda v: str(v)),
    }


def transform_classifier_input(values: dict[str, Any], bundle: dict[str, Any]) -> np.ndarray:
    row = pd.DataFrame([{c: values.get(c, bundle["defaults"].get(c)) for c in bundle["feature_columns"]}])
    return np.asarray(bundle["preprocessor"].transform(row), dtype=np.float64)


# -----------------------------------------------------------------------------
# Clustering: no ground-truth labels are used to fit either algorithm.
# -----------------------------------------------------------------------------
def prepare_clustering_matrix(df: pd.DataFrame, label_col: str | None, max_rows: int) -> dict[str, Any]:
    data = df.drop_duplicates().reset_index(drop=True).copy()
    labels_true: pd.Series | None = None
    if label_col and label_col in data.columns:
        labels_true = data[label_col].copy()
        features = data.drop(columns=[label_col]).copy()
    else:
        features = data.copy()
    features = features.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    if features.shape[1] < 2:
        raise ValueError("Clustering needs at least two usable predictor columns.")

    if len(features) > max_rows:
        selected_idx = np.sort(np.random.default_rng(RANDOM_STATE).choice(len(features), max_rows, replace=False))
        features = features.iloc[selected_idx].reset_index(drop=True)
        if labels_true is not None:
            labels_true = labels_true.iloc[selected_idx].reset_index(drop=True)
    else:
        features = features.reset_index(drop=True)
        if labels_true is not None:
            labels_true = labels_true.reset_index(drop=True)

    preprocessor = build_tabular_preprocessor(features)
    X_scaled = np.asarray(preprocessor.fit_transform(features), dtype=np.float64)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    if len(X_scaled) < 4:
        raise ValueError("Upload at least four rows for meaningful clustering.")
    return {
        "X": X_scaled,
        "feature_columns": list(features.columns),
        "preprocessor": preprocessor,
        "label_col": label_col,
        "labels_true": labels_true,
        "sample_rows": len(features),
        "feature_count": X_scaled.shape[1],
        "source_signature": dataset_signature(df),
    }


def safe_cluster_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return {
            "Silhouette Score": float("nan"),
            "Davies-Bouldin Index": float("nan"),
            "Calinski-Harabasz Index": float("nan"),
        }
    return {
        "Silhouette Score": float(silhouette_score(X, labels)),
        "Davies-Bouldin Index": float(davies_bouldin_score(X, labels)),
        "Calinski-Harabasz Index": float(calinski_harabasz_score(X, labels)),
    }


def fit_clustering_suite(df: pd.DataFrame, label_col: str | None, n_clusters: int, max_rows: int) -> dict[str, Any]:
    prep = prepare_clustering_matrix(df, label_col, max_rows)
    X = prep["X"]
    k_upper = min(10, len(X) - 1)
    k_values = list(range(2, max(2, k_upper) + 1))
    elbow: list[float] = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X)
        elbow.append(float(km.inertia_))

    final_k = max(2, min(n_clusters, len(X) - 1))
    kmeans = KMeans(n_clusters=final_k, random_state=RANDOM_STATE, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_metrics = safe_cluster_metrics(X, kmeans_labels)
    kmeans_row = {"Algorithm": "K-Means", "Clusters": int(len(np.unique(kmeans_labels))), **kmeans_metrics}
    metric_rows = [kmeans_row]
    labels_by_model: dict[str, np.ndarray] = {"K-Means": kmeans_labels}
    agg_models: dict[str, Any] = {}
    linkage_methods = ["single", "complete", "average", "ward"]
    progress = st.progress(0, text="Fitting agglomerative linkage strategies...")
    status = st.empty()
    for i, method in enumerate(linkage_methods, start=1):
        status.write(f"Fitting **Agglomerative ({method})** ({i}/{len(linkage_methods)})")
        params = inspect.signature(AgglomerativeClustering).parameters
        kwargs: dict[str, Any] = {"n_clusters": final_k, "linkage": method}
        kwargs["metric" if "metric" in params else "affinity"] = "euclidean"
        model = AgglomerativeClustering(**kwargs)
        labels = model.fit_predict(X)
        name = f"Agglomerative - {method.title()} linkage"
        agg_models[name] = model
        labels_by_model[name] = labels
        metric_rows.append({"Algorithm": name, "Clusters": int(len(np.unique(labels))), **safe_cluster_metrics(X, labels)})
        progress.progress(i / len(linkage_methods), text=f"Completed {method} linkage")
    progress.empty()
    status.empty()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_points = pca.fit_transform(X)
    vis_n = min(1200, len(X))
    tsne_idx = np.sort(np.random.default_rng(RANDOM_STATE).choice(len(X), vis_n, replace=False)) if vis_n < len(X) else np.arange(len(X))
    if vis_n >= 3:
        tsne_model = TSNE(
            n_components=2,
            perplexity=min(30.0, max(2.0, (vis_n - 1) / 3.0)),
            init="pca",
            learning_rate="auto",
            random_state=RANDOM_STATE,
        )
        tsne_points = tsne_model.fit_transform(X[tsne_idx])
    else:
        tsne_points = np.zeros((vis_n, 2))

    # Dendrogram linkage is computed on a smaller sample for quadratic complexity.
    dendro_n = min(800, len(X))
    dendro_idx = np.sort(np.random.default_rng(RANDOM_STATE + 1).choice(len(X), dendro_n, replace=False)) if dendro_n < len(X) else np.arange(len(X))
    dendro_X = X[dendro_idx]
    dendrograms: dict[str, np.ndarray] = {}
    for method in linkage_methods:
        dendrograms[method] = linkage(dendro_X, method=method, metric="euclidean")

    label_validation: pd.DataFrame | None = None
    if prep["labels_true"] is not None:
        truth = prep["labels_true"].to_numpy()
        valid = pd.notna(truth)
        truth = truth[valid]
        validation_rows = []
        for name, labels in labels_by_model.items():
            model_labels = labels[valid]
            validation_rows.append({
                "Algorithm": name,
                "Adjusted Rand Index": adjusted_rand_score(truth, model_labels),
                "Normalized Mutual Information": normalized_mutual_info_score(truth, model_labels),
            })
        label_validation = pd.DataFrame(validation_rows)

    return {
        "prep": prep,
        "kmeans_model": kmeans,
        "metrics": pd.DataFrame(metric_rows),
        "k_values": k_values,
        "inertias": elbow,
        "n_clusters": final_k,
        "labels_by_model": labels_by_model,
        "agg_models": agg_models,
        "pca_points": pca_points,
        "pca_variance": float(pca.explained_variance_ratio_.sum()),
        "tsne_points": tsne_points,
        "tsne_idx": tsne_idx,
        "dendrograms": dendrograms,
        "label_validation": label_validation,
        "signature": dataset_signature(df),
    }


# -----------------------------------------------------------------------------
# UI rendering helpers
# -----------------------------------------------------------------------------
def render_regression_results(bundle: dict[str, Any]) -> None:
    st.subheader("Regression model comparison")
    st.caption("All ten models share the same train/test split and training-derived preprocessing.")
    st.dataframe(bundle["metrics"].style.format({"R²": "{:.4f}", "RMSE": "{:.3f}", "MAE": "{:.3f}", "5-fold CV R² (top 2)": "{:.4f}"}), use_container_width=True, hide_index=True)
    show_download_csv(bundle["metrics"], "regression_model_comparison.csv", "Download regression comparison (CSV)")

    best_name = bundle["metrics"].iloc[0]["Algorithm"]
    y_true = bundle["test_y"]
    y_pred = bundle["predictions"][best_name]
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_true, y_pred, alpha=0.45, s=14)
        low = float(min(np.min(y_true), np.min(y_pred)))
        high = float(max(np.max(y_true), np.max(y_pred)))
        ax.plot([low, high], [low, high], linestyle="--")
        ax.set_title(f"Predicted vs Actual — {best_name}")
        ax.set_xlabel("Actual target")
        ax.set_ylabel("Predicted target")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col2:
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_pred, residuals, alpha=0.45, s=14)
        ax.axhline(0, linestyle="--")
        ax.set_title(f"Residuals — {best_name}")
        ax.set_xlabel("Predicted target")
        ax.set_ylabel("Actual − Predicted")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    best_model = bundle["models"][best_name]
    if hasattr(best_model, "feature_importances_"):
        importances = np.asarray(best_model.feature_importances_)
        feat_names = bundle["prep"]["final_columns"]
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=False).head(15)
        st.subheader("Feature importance")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1])
        ax.set_title(f"Top feature importances — {best_name}")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def render_regression_prediction(bundle: dict[str, Any]) -> None:
    st.subheader("Try a prediction")
    model_name = st.selectbox("Regression algorithm", list(bundle["models"].keys()), key="reg_predict_model")
    prep = bundle["prep"]
    st.caption("Defaults are training-data medians. Replace values with your scenario; the app applies the same learned preprocessing.")
    values: dict[str, float] = {}
    with st.form("regression_prediction_form"):
        for start in range(0, len(prep["raw_columns"]), 3):
            row_cols = st.columns(3)
            for ui_col, feature in zip(row_cols, prep["raw_columns"][start:start + 3]):
                default = safe_number(prep["defaults"].get(feature, 0.0))
                with ui_col:
                    values[feature] = st.number_input(
                        FEATURE_LABELS.get(feature, feature), value=default, format="%.4f",
                        key=f"reg_pred_{feature}", help=f"Dataset feature: {feature}",
                    )
        submitted = st.form_submit_button("Predict appliance energy", type="primary", use_container_width=True)
    if submitted:
        try:
            x_user = transform_regression_input(values, prep)
            prediction = float(np.asarray(bundle["models"][model_name].predict(x_user)).reshape(-1)[0])
            st.success(f"Predicted {bundle['target']}: **{prediction:,.2f} Wh**")
            plot_metric_cards([("Prediction", f"{prediction:,.2f} Wh"), ("Model", model_name), ("Predictors used", str(len(prep["final_columns"])))])
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def render_classification_results(bundle: dict[str, Any]) -> None:
    st.subheader("Ten-algorithm classification comparison")
    fmt = {c: "{:.4f}" for c in ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-score (weighted)", "ROC-AUC (OvR for multiclass)"]}
    st.dataframe(bundle["metrics"].style.format(fmt, na_rep="N/A"), use_container_width=True, hide_index=True)
    show_download_csv(bundle["metrics"], "classification_10_algorithm_comparison.csv", "Download classification comparison (CSV)")
    failed_models = {name: result.get("error") for name, result in bundle["reports"].items() if isinstance(result, dict) and result.get("error")}
    if failed_models:
        st.warning("Some classifiers could not be fitted in this environment:")
        for name, reason in failed_models.items():
            st.caption(f"{name}: {reason}")
    top = bundle["metrics"].iloc[0]
    plot_metric_cards([
        ("Top weighted F1", f"{top['F1-score (weighted)']:.4f}"),
        ("Best model", str(top["Algorithm"])),
        ("Rows in modelling sample", f"{bundle['sample_rows']:,}"),
    ])

    available = list(bundle["models"].keys())
    chosen = st.selectbox("Inspect model evaluation", available, index=available.index(str(top["Algorithm"])), key="classification_eval_model")
    if chosen in bundle["confusion"]:
        labels = bundle["model_classes"].get(chosen, bundle["labels"])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=bundle["confusion"][chosen], display_labels=[str(x) for x in labels])
        disp.plot(ax=ax, values_format="d", colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion matrix — {chosen}")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.expander(f"Classification report — {chosen}"):
            report_df = pd.DataFrame(bundle["reports"].get(chosen, {})).T
            st.dataframe(report_df, use_container_width=True)

    with st.expander("Confusion matrices and reports for all models"):
        for name in available:
            if name not in bundle["confusion"]:
                continue
            st.markdown(f"**{name}**")
            labels = bundle["model_classes"].get(name, bundle["labels"])
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay(confusion_matrix=bundle["confusion"][name], display_labels=[str(x) for x in labels]).plot(ax=ax, values_format="d", colorbar=False, cmap="Blues")
            ax.set_title(name)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            report_df = pd.DataFrame(bundle["reports"].get(name, {})).T
            st.dataframe(report_df, use_container_width=True)


def render_classification_prediction(bundle: dict[str, Any]) -> None:
    st.subheader("Try a classification prediction")
    model_name = st.selectbox("Classification algorithm", list(bundle["models"].keys()), key="cls_predict_model")
    values: dict[str, Any] = {}
    with st.form("classification_prediction_form"):
        for start in range(0, len(bundle["feature_columns"]), 3):
            row_cols = st.columns(3)
            for ui_col, feature in zip(row_cols, bundle["feature_columns"][start:start + 3]):
                default = bundle["defaults"].get(feature, 0)
                with ui_col:
                    if pd.api.types.is_numeric_dtype(bundle["test_x_raw"][feature]):
                        values[feature] = st.number_input(str(feature), value=safe_number(default), format="%.4f", key=f"cls_pred_{feature}")
                    else:
                        observed = bundle["test_x_raw"][feature].dropna().astype(str).unique().tolist()
                        choices = observed[:100] if observed else [str(default)]
                        default_str = str(default)
                        idx = choices.index(default_str) if default_str in choices else 0
                        chosen_value = st.selectbox(str(feature), choices, index=idx, key=f"cls_pred_{feature}")
                        values[feature] = chosen_value
        submitted = st.form_submit_button("Predict class", type="primary", use_container_width=True)
    if submitted:
        try:
            x = transform_classifier_input(values, bundle)
            model = bundle["models"][model_name]
            pred = model.predict(x)[0]
            st.success(f"Predicted class: **{pred}** using **{model_name}**")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x)[0]
                prob_df = pd.DataFrame({"Class": [str(c) for c in model.classes_], "Probability": probs})
                st.dataframe(prob_df.style.format({"Probability": "{:.4%}"}), use_container_width=True, hide_index=True)
                st.bar_chart(prob_df.set_index("Class"))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def render_clustering_results(bundle: dict[str, Any]) -> None:
    st.subheader("Clustering comparison")
    metrics = bundle["metrics"].copy()
    st.dataframe(metrics.style.format({"Silhouette Score": "{:.4f}", "Davies-Bouldin Index": "{:.4f}", "Calinski-Harabasz Index": "{:.2f}"}, na_rep="N/A"), use_container_width=True, hide_index=True)
    show_download_csv(metrics, "clustering_metrics.csv", "Download clustering metrics (CSV)")
    plot_metric_cards([
        ("Rows clustered", f"{len(bundle['prep']['X']):,}"),
        ("Features used", f"{bundle['prep']['feature_count']:,}"),
        ("K-Means k", str(bundle["n_clusters"])),
    ])
    st.caption("Metric interpretation: higher Silhouette and Calinski–Harabasz are generally better; lower Davies–Bouldin is generally better. K-Means inertia is used for the elbow curve.")

    st.subheader("K-Means elbow curve")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bundle["k_values"], bundle["inertias"], marker="o")
    ax.set_title("K-Means Elbow Curve")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_xticks(bundle["k_values"])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("PCA cluster visualisations")
    st.caption(f"PCA 2D projection; first two components explain {bundle['pca_variance']:.2%} of the variance in the preprocessed modelling sample.")
    names = list(bundle["labels_by_model"].keys())
    for start in range(0, len(names), 2):
        cols = st.columns(2)
        for col, name in zip(cols, names[start:start + 2]):
            with col:
                fig, ax = plt.subplots(figsize=(6, 4.5))
                scatter = ax.scatter(bundle["pca_points"][:, 0], bundle["pca_points"][:, 1], c=bundle["labels_by_model"][name], s=10, alpha=0.7, cmap="tab10")
                ax.set_title(name)
                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    st.subheader("Agglomerative dendrograms")
    methods = list(bundle["dendrograms"].keys())
    for start in range(0, len(methods), 2):
        cols = st.columns(2)
        for col, method in zip(cols, methods[start:start + 2]):
            with col:
                fig, ax = plt.subplots(figsize=(7, 3.5))
                dendrogram(bundle["dendrograms"][method], ax=ax, no_labels=True, color_threshold=None)
                ax.set_title(f"{method.title()} linkage")
                ax.set_xlabel("Sampled observations")
                ax.set_ylabel("Linkage distance")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    st.subheader("t-SNE visualisation (K-Means labels)")
    vis_labels = bundle["labels_by_model"]["K-Means"][bundle["tsne_idx"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(bundle["tsne_points"][:, 0], bundle["tsne_points"][:, 1], c=vis_labels, s=12, alpha=0.75, cmap="tab10")
    ax.set_title("K-Means clusters — t-SNE 2D")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if bundle["label_validation"] is not None:
        st.subheader("Optional ground-truth validation (not used during fitting)")
        st.caption(f"`{bundle['prep']['label_col']}` was excluded from the input features. These scores are post-hoc comparisons only.")
        st.dataframe(bundle["label_validation"].style.format({"Adjusted Rand Index": "{:.4f}", "Normalized Mutual Information": "{:.4f}"}), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Main page shell
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background: rgba(120, 120, 120, 0.07); padding: 0.8rem 1rem; border-radius: 0.75rem;}
    .stCaption {font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 ML Capstone Studio")
st.caption("23CSE301 · Regression · Classification · Clustering · Random state 42")

with st.sidebar:
    st.title("Capstone Studio")
    page = st.radio(
        "Navigate",
        ["Overview", "Regression", "Classification", "Clustering", "Run Instructions"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Expected local datasets")
    for path in DATASET_CANDIDATES.values():
        mark = "✓" if path.exists() else "Upload needed"
        st.caption(f"{mark} · {path.name}")
    st.caption("Random state: 42")

if page == "Overview":
    st.header("Project overview")
    st.write(
        "A single web interface for the three tracks in the capstone. Upload a dataset or use the expected project `data/` file; "
        "then train, compare and inspect models. Each track uses its own relevant evaluation metrics."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Regression", "10 algorithms")
        st.caption("R², RMSE, MAE; optional 5-fold CV for the top two")
    with c2:
        st.metric("Classification", "10 algorithms")
        st.caption("Accuracy, weighted precision/recall/F1, ROC-AUC, confusion matrices")
    with c3:
        st.metric("Clustering", "2 required algorithms")
        st.caption("Silhouette, Davies–Bouldin, Calinski–Harabasz, elbow, dendrogram, PCA and t-SNE")
    st.divider()
    st.subheader("Models included")
    left, right = st.columns(2)
    with left:
        st.markdown("**Regression**")
        st.write("Linear Regression · Ridge · Lasso · ElasticNet · Polynomial Regression · Decision Tree · Random Forest · Gradient Boosting · SVR · KNN")
        st.markdown("**Classification**")
        st.write("Logistic Regression · KNN · Gaussian Naive Bayes · Decision Tree · SVC · Random Forest · AdaBoost · Gradient Boosting · Bagging · MLP Neural Network")
    with right:
        st.markdown("**Clustering**")
        st.write("K-Means · Agglomerative Hierarchical Clustering (single, complete, average and Ward linkages)")
        st.markdown("**Data handling**")
        st.write("Train/test splitting is reproducible. Classification preprocessing is fitted only on the training split; clustering labels, when selected, are excluded from fitting and used only for post-hoc interpretation.")
    st.info("The uploaded capstone guideline describes the GUI as a bonus: a working web interface that accepts inputs and returns predictions from a trained model (page 6).")

elif page == "Regression":
    st.header("Regression track")
    st.write("Predict appliance energy consumption and compare the ten required regression algorithms.")
    uploaded = st.file_uploader("Upload Appliances Energy Prediction CSV", type=["csv"], key="regression_csv")
    df, source = read_dataset(uploaded, DATASET_CANDIDATES["regression"])
    if df is None:
        display_upload_help(DATASET_CANDIDATES["regression"], "regression")
    else:
        st.caption(f"Source: {source}")
        st.write(f"Dataset shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
        numeric_targets = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or c == "Appliances"]
        default_target = numeric_targets.index("Appliances") if "Appliances" in numeric_targets else max(0, len(numeric_targets) - 1)
        target = st.selectbox("Target column", numeric_targets, index=default_target, key="regression_target")
        run_cv = st.checkbox("Compute 5-fold cross-validated R² for the top two models (takes longer)", value=False, key="regression_cv")
        st.caption("Preprocessing follows the project workflow: drop timestamp, impute using training means, log1p-transform `lights`, clip IQR outliers using training bounds, remove one feature from highly correlated pairs (>0.90), then fit StandardScaler on training data.")
        if st.button("Train and compare all 10 regression models", type="primary", use_container_width=True, key="train_regression"):
            with st.spinner("Preparing data and training the regression suite..."):
                try:
                    bundle = fit_regression_suite(df, target, run_cv)
                    st.session_state["regression_bundle"] = bundle
                    st.session_state["regression_source"] = source
                    st.success("All regression models completed.")
                except Exception as exc:
                    st.error(f"Regression training failed: {exc}")
        bundle = st.session_state.get("regression_bundle")
        if bundle:
            if bundle.get("signature") != dataset_signature(df) or bundle.get("target") != target:
                st.warning("The current dataset/target differs from the dataset used for the stored results. Train again to refresh metrics and predictions.")
            else:
                st.caption(f"Model training used {len(bundle['prep']['y_train']):,} rows; evaluation used {len(bundle['prep']['y_test']):,} held-out rows.")
                render_regression_results(bundle)
                st.divider()
                render_regression_prediction(bundle)
                buffer = io.BytesIO()
                joblib.dump({"track": "regression", "models": bundle["models"], "prep": bundle["prep"], "target": bundle["target"]}, buffer)
                st.download_button("Download trained regression model bundle (.joblib)", buffer.getvalue(), file_name="regression_models.joblib", mime="application/octet-stream")

elif page == "Classification":
    st.header("Classification track")
    st.write("Train and compare all ten algorithms on the diabetes-health-indicators classification dataset (or a compatible labelled CSV).")
    uploaded = st.file_uploader("Upload Diabetes Binary Health Indicators CSV", type=["csv"], key="classification_csv")
    df, source = read_dataset(uploaded, DATASET_CANDIDATES["classification"])
    if df is None:
        display_upload_help(DATASET_CANDIDATES["classification"], "classification")
    else:
        st.caption(f"Source: {source}")
        st.write(f"Dataset shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
        preferred = "Diabetes_binary" if "Diabetes_binary" in df.columns else df.columns[-1]
        target_options = list(df.columns)
        target = st.selectbox("Target / class column", target_options, index=target_options.index(preferred), key="classification_target")
        max_allowed = max(1000, min(50000, len(df)))
        default_rows = min(15000, max_allowed, len(df))
        cls_min_rows = min(1000, max_allowed)
        cls_default_rows = max(cls_min_rows, min(15000, max_allowed))
        cls_step = 1000 if max_allowed >= 2000 else 100 if max_allowed >= 1100 else 1
        max_rows = st.slider("Rows for a reproducible model comparison", min_value=cls_min_rows, max_value=max_allowed, value=cls_default_rows, step=cls_step, key="classification_rows")
        st.caption("To keep all ten estimators practical in an interactive GUI, this sample is stratified before the 80:20 split. Every model uses the same train/test rows. Increase the row limit for a more representative comparison; the notebook remains the source of record for your submitted results.")
        if st.button("Train and compare all 10 classification models", type="primary", use_container_width=True, key="train_classification"):
            with st.spinner("Cleaning, preprocessing and training all ten classification algorithms..."):
                try:
                    bundle = fit_classification_suite(df, target, int(max_rows))
                    st.session_state["classification_bundle"] = bundle
                    st.session_state["classification_source"] = source
                    st.success("Classification model comparison completed.")
                except Exception as exc:
                    st.error(f"Classification training failed: {exc}")
        bundle = st.session_state.get("classification_bundle")
        if bundle:
            if bundle.get("signature") != dataset_signature(df) or bundle.get("target") != target:
                st.warning("The current dataset/target differs from the data used for the stored results. Train again before relying on these metrics.")
            else:
                st.caption(f"Modelling sample: {bundle['sample_rows']:,}; training: {bundle['train_rows']:,}; testing: {bundle['test_rows']:,}.")
                render_classification_results(bundle)
                st.divider()
                render_classification_prediction(bundle)
                buffer = io.BytesIO()
                joblib.dump({"track": "classification", "models": bundle["models"], "preprocessor": bundle["preprocessor"], "feature_columns": bundle["feature_columns"], "target": bundle["target"]}, buffer)
                st.download_button("Download trained classification model bundle (.joblib)", buffer.getvalue(), file_name="classification_models.joblib", mime="application/octet-stream")

elif page == "Clustering":
    st.header("Clustering track")
    st.write("Evaluate K-Means and Agglomerative Hierarchical Clustering with clustering-specific metrics and visualisations.")
    uploaded = st.file_uploader("Upload Diabetes 012 Health Indicators CSV", type=["csv"], key="clustering_csv")
    df, source = read_dataset(uploaded, DATASET_CANDIDATES["clustering"])
    if df is None:
        display_upload_help(DATASET_CANDIDATES["clustering"], "clustering")
    else:
        st.caption(f"Source: {source}")
        st.write(f"Dataset shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
        label_options = ["None — no ground-truth label"] + list(df.columns)
        default_label = "Diabetes_012" if "Diabetes_012" in df.columns else "None — no ground-truth label"
        label_col_choice = st.selectbox("Optional ground-truth column (excluded from clustering; used only for post-hoc validation)", label_options, index=label_options.index(default_label), key="clustering_label_col")
        label_col = None if label_col_choice.startswith("None") else label_col_choice
        cluster_max_rows = min(5000, max(500, len(df)))
        cluster_default_rows = min(2500, cluster_max_rows)
        cluster_step = 500 if cluster_max_rows >= 1000 and cluster_max_rows % 500 == 0 else 1
        max_rows = st.slider("Rows to use for clustering", min_value=500, max_value=cluster_max_rows, value=cluster_default_rows, step=cluster_step, key="clustering_rows")
        n_clusters = st.slider("Number of clusters (k) for the final comparison", min_value=2, max_value=10, value=3, key="clustering_k")
        st.caption("Agglomerative clustering and dendrograms are computationally expensive on large datasets. The GUI uses the selected reproducible sample for comparisons and a smaller subsample for dendrogram drawing.")
        if st.button("Run K-Means and Agglomerative Clustering", type="primary", use_container_width=True, key="train_clustering"):
            with st.spinner("Scaling features, comparing k values, fitting clustering models and preparing visualisations..."):
                try:
                    bundle = fit_clustering_suite(df, label_col, int(n_clusters), int(max_rows))
                    st.session_state["clustering_bundle"] = bundle
                    st.session_state["clustering_source"] = source
                    st.success("Clustering evaluation completed. Ground-truth labels, if selected, were excluded from fitting.")
                except Exception as exc:
                    st.error(f"Clustering failed: {exc}")
        bundle = st.session_state.get("clustering_bundle")
        if bundle:
            current_signature = dataset_signature(df)
            if bundle.get("signature") != current_signature or bundle["prep"].get("label_col") != label_col or bundle.get("n_clusters") != int(n_clusters):
                st.warning("The current dataset/settings differ from the stored clustering run. Run clustering again to refresh results.")
            else:
                st.caption(f"Source: {st.session_state.get('clustering_source', source)} · {len(bundle['prep']['X']):,} rows used.")
                render_clustering_results(bundle)
                buffer = io.BytesIO()
                joblib.dump({"track": "clustering", "kmeans_model": bundle["kmeans_model"], "preprocessor": bundle["prep"]["preprocessor"], "feature_columns": bundle["prep"]["feature_columns"], "n_clusters": bundle["n_clusters"]}, buffer)
                st.download_button("Download K-Means model bundle (.joblib)", buffer.getvalue(), file_name="clustering_kmeans.joblib", mime="application/octet-stream")

elif page == "Run Instructions":
    st.header("Run instructions")
    st.markdown(
        """
        ### 1. Put the app into your repository

        Keep the following structure (the GUI is in `gui/app.py`):

        ```text
        project-root/
        ├── data/
        │   ├── energydata_complete.csv
        │   ├── diabetes_binary_health_indicators_BRFSS2015.csv
        │   └── diabetes_012_health_indicators_BRFSS2021.csv
        ├── gui/
        │   └── app.py
        ├── models/       # optional; downloaded bundles can be stored here
        ├── notebooks/
        ├── requirements.txt
        └── README.md
        ```

        If a dataset is not in `data/`, open its track and upload the CSV in the interface.

        ### 2. Install dependencies

        Run this from the project root:

        ```bash
        python -m pip install -r requirements.txt
        ```

        ### 3. Start Streamlit

        ```bash
        python -m streamlit run gui/app.py
        ```

        ### 4. Important notes

        - Classification removes duplicate rows, stratifies the sample and uses the same train/test split across all ten algorithms. Precision, recall and F1 are weighted; multi-class ROC-AUC uses one-vs-rest (OvR).
        - For responsiveness, classification defaults to a 15,000-row stratified modelling sample. The GUI displays exactly how many rows were used. Use the notebooks for final report numbers and their tuning comparisons.
        - Clustering does not use ground-truth labels while fitting. When selected, the label column is removed before fitting and used only to calculate optional post-hoc Adjusted Rand Index and Normalized Mutual Information.
        - Clustering reports Silhouette Score, Davies–Bouldin Index and Calinski–Harabasz Index. It includes K-Means elbow, a dendrogram for each linkage strategy, PCA 2D plots for each clustering configuration and t-SNE for K-Means.
        - Trained models remain in the current app session. Use the download buttons to save reusable `.joblib` bundles before ending the session.
        - The app is designed as an interactive GUI bonus component; local datasets and model-training metrics are not sent to any external service.
        """
    )
    st.subheader("Evaluation metrics by track")
    st.dataframe(pd.DataFrame([
        {"Track": "Regression", "Required metrics": "R², RMSE, MAE; 5-fold CV R² for the two best models"},
        {"Track": "Classification", "Required metrics": "Accuracy, Precision, Recall, weighted F1-score, confusion matrix and ROC-AUC (OvR for multiclass)"},
        {"Track": "Clustering", "Required metrics": "Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Index; elbow, dendrogram and PCA 2D plots"},
    ]), use_container_width=True, hide_index=True)

st.divider()
st.caption("23CSE301 Machine Learning Capstone · reproducibility seed: 42")
