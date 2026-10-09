from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Appliance Energy Modeling",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "energydata_complete.csv"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_REGISTRY = {
    "Decision Tree": {
        "file": "decision_tree_appliances_randomsplit.pkl",
        "description": "A tree-based regression model that recursively partitions the feature space.",
        "note": "Tuned mainly through tree complexity controls such as max_depth, min_samples_split and min_samples_leaf.",
    },
    "Random Forest": {
        "file": "random_forest_appliances_randomsplit.pkl",
        "description": "An ensemble of randomized decision trees whose predictions are aggregated.",
        "note": "Tuned across n_estimators, max_depth, min_samples_split, min_samples_leaf and max_features.",
    },
    "Gradient Boosting": {
        "file": "gradient_boosting_appliances_randomsplit.pkl",
        "description": "An ensemble that builds trees sequentially, with each tree improving the previous model.",
        "note": "Tuned across learning_rate, n_estimators, depth, leaf/split constraints and subsampling.",
    },
    "SVR": {
        "file": "svr_appliances_randomsplit.pkl",
        "description": "Support Vector Regression for learning a function within an epsilon-insensitive margin.",
        "note": "Features are standardized; kernel, C, epsilon and gamma were tuned.",
    },
    "KNN": {
        "file": "knn_appliances_randomsplit.pkl",
        "description": "Predicts appliance energy from nearby training observations in feature space.",
        "note": "Features are standardized; n_neighbors, weights and p were tuned.",
    },
}

MODEL_ORDER = list(MODEL_REGISTRY.keys())

# =============================================================================
# Friendly labels / units
# The raw predictors below are exactly the columns present before the notebook's
# correlation filtering step. rv1 and rv2 are excluded because the notebook
# intentionally removes them from modeling.
# =============================================================================
FEATURE_INFO = {
    "lights": ("Lights energy", "Wh", "Energy used by light fixtures."),
    "T1": ("Kitchen temperature", "°C", "Temperature measured in the kitchen."),
    "RH_1": ("Kitchen humidity", "%", "Relative humidity in the kitchen."),
    "T2": ("Living room temperature", "°C", "Temperature measured in the living room."),
    "RH_2": ("Living room humidity", "%", "Relative humidity in the living room."),
    "T3": ("Laundry room temperature", "°C", "Temperature measured in the laundry room."),
    "RH_3": ("Laundry room humidity", "%", "Relative humidity in the laundry room."),
    "T4": ("Office temperature", "°C", "Temperature measured in the office."),
    "RH_4": ("Office humidity", "%", "Relative humidity in the office."),
    "T5": ("Bathroom temperature", "°C", "Temperature measured in the bathroom."),
    "RH_5": ("Bathroom humidity", "%", "Relative humidity in the bathroom."),
    "T6": ("Outside temperature (north)", "°C", "Temperature measured outside on the north side."),
    "RH_6": ("Outside humidity (north)", "%", "Relative humidity measured outside on the north side."),
    "T7": ("Ironing room temperature", "°C", "Temperature measured in the ironing room."),
    "RH_7": ("Ironing room humidity", "%", "Relative humidity in the ironing room."),
    "T8": ("Teenager room temperature", "°C", "Temperature measured in the teenager's room."),
    "RH_8": ("Teenager room humidity", "%", "Relative humidity in the teenager's room."),
    "T9": ("Parents room temperature", "°C", "Temperature measured in the parents' room."),
    "RH_9": ("Parents room humidity", "%", "Relative humidity in the parents' room."),
    "T_out": ("Outside temperature", "°C", "Temperature from the weather station."),
    "RH_out": ("Outside humidity", "%", "Humidity from the weather station."),
    "Press_mm_hg": ("Atmospheric pressure", "mm Hg", "Atmospheric pressure."),
    "Windspeed": ("Wind speed", "m/s", "Measured wind speed."),
    "Visibility": ("Visibility", "km", "Visibility measured by the weather station."),
    "Tdewpoint": ("Dew point", "°C", "Dew point temperature."),
}

# =============================================================================
# Data loading
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}\n\n"
            "Expected structure:\n"
            "case study/data/energydata_complete.csv"
        )

    df = pd.read_csv(DATA_PATH)

    if "Appliances" not in df.columns:
        raise ValueError("The dataset must contain the target column 'Appliances'.")

    return df


# =============================================================================
# Preprocessing
# IMPORTANT:
# This mirrors the current five-model notebook:
#   1. drop date
#   2. train_test_split(random_state=42)
#   3. training-set mean imputation
#   4. log1p(lights)
#   5. training-set IQR clipping
#   6. remove redundant predictors using training correlations > 0.90
#   7. StandardScaler fitted only on training data
#
# The GUI needs to reproduce these transformations for one new user row.
# =============================================================================
@st.cache_resource(show_spinner=False)
def fit_preprocessor() -> dict[str, Any]:
    df = load_data().copy()

    # The current notebook drops only date before splitting.
    # rv1 and rv2 are then among the predictors and subsequently removed by
    # the high-correlation feature-selection step in the notebook.
    working = df.drop(columns=["date"], errors="ignore").copy()

    X_raw = working.drop(columns=["Appliances"]).copy()
    y = working["Appliances"].copy()

    # Same random split as your five-model notebook.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # ----- Mean imputation learned from training only -----
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    imputation_values = X_train.mean(numeric_only=True)

    for col in X_train.columns:
        if X_train[col].isna().any() or X_test[col].isna().any():
            X_train[col] = X_train[col].fillna(imputation_values[col])
            X_test[col] = X_test[col].fillna(imputation_values[col])

    # ----- Log transform -----
    X_train["lights"] = np.log1p(X_train["lights"].clip(lower=0))
    X_test["lights"] = np.log1p(X_test["lights"].clip(lower=0))

    # ----- IQR clipping learned from training only -----
    clip_bounds: dict[str, tuple[float, float]] = {}

    for col in X_train.columns:
        q1 = X_train[col].quantile(0.25)
        q3 = X_train[col].quantile(0.75)
        iqr = q3 - q1
        lower = float(q1 - 1.5 * iqr)
        upper = float(q3 + 1.5 * iqr)

        clip_bounds[col] = (lower, upper)

        X_train[col] = X_train[col].clip(lower=lower, upper=upper)
        X_test[col] = X_test[col].clip(lower=lower, upper=upper)

    # ----- Correlation filtering learned from training only -----
    train_corr = X_train.copy()
    train_corr["Appliances"] = y_train.to_numpy()

    corr = train_corr.corr(numeric_only=True)
    threshold = 0.90

    to_remove: set[str] = set()
    cols = list(X_train.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            feature1 = cols[i]
            feature2 = cols[j]
            pair_corr = corr.loc[feature1, feature2]

            if abs(pair_corr) > threshold:
                target_corr1 = abs(corr.loc[feature1, "Appliances"])
                target_corr2 = abs(corr.loc[feature2, "Appliances"])

                if target_corr1 < target_corr2:
                    to_remove.add(feature1)
                else:
                    to_remove.add(feature2)

    final_columns = [c for c in X_train.columns if c not in to_remove]

    X_train = X_train[final_columns]
    X_test = X_test[final_columns]

    # ----- Scaling learned from training only -----
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use training medians as practical defaults in the UI.
    # These defaults are only for convenience; users can replace them.
    defaults = X_train_raw.median(numeric_only=True).to_dict()

    return {
        "raw_columns": list(X_raw.columns),
        "final_columns": final_columns,
        "removed_columns": sorted(to_remove),
        "imputation_values": imputation_values.to_dict(),
        "clip_bounds": clip_bounds,
        "scaler": scaler,
        "defaults": defaults,
        "train_size": len(X_train_raw),
        "test_size": len(X_test_raw),
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
    }


# =============================================================================
# Model helpers
# =============================================================================
def model_path(model_name: str) -> Path:
    return MODEL_DIR / MODEL_REGISTRY[model_name]["file"]


@st.cache_resource
def load_model(model_name: str) -> Any:
    path = model_path(model_name)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{path}\n\n"
            "Run the five-model regression notebook first so the .pkl file is created."
        )

    return joblib.load(path)


def available_models() -> list[str]:
    return [name for name in MODEL_ORDER if model_path(name).exists()]


def transform_user_input(values: dict[str, float], prep: dict[str, Any]) -> np.ndarray:
    # Start with exactly the same raw predictor set that existed before the
    # notebook's correlation filtering.
    row = pd.DataFrame(
        [
            {
                column: values.get(
                    column,
                    prep["defaults"].get(column, prep["imputation_values"].get(column, 0.0)),
                )
                for column in prep["raw_columns"]
            }
        ]
    )

    # Training-derived imputation values.
    for col, value in prep["imputation_values"].items():
        if col in row.columns:
            row[col] = row[col].fillna(value)

    # Match notebook exactly.
    row["lights"] = np.log1p(row["lights"].clip(lower=0))

    # Training-derived clipping.
    for col, (lower, upper) in prep["clip_bounds"].items():
        row[col] = row[col].clip(lower=lower, upper=upper)

    # Same feature selection and order.
    row = row[prep["final_columns"]]

    # Same scaler.
    return prep["scaler"].transform(row)


def predict(model_name: str, x_scaled: np.ndarray) -> float:
    model = load_model(model_name)
    prediction = model.predict(x_scaled)
    return float(np.asarray(prediction).reshape(-1)[0])


# =============================================================================
# Optional metrics loader
#
# The GUI does not invent evaluation results. Once your notebook exports:
# models/model_metrics.csv
# this page will automatically show it.
# =============================================================================
@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame | None:
    candidates = [
        MODEL_DIR / "model_metrics.csv",
        ROOT / "results" / "model_metrics.csv",
    ]

    for path in candidates:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                return None

    return None


# =============================================================================
# UI helpers
# =============================================================================
def show_model_badges() -> None:
    cols = st.columns(5)

    for col, name in zip(cols, MODEL_ORDER):
        with col:
            if model_path(name).exists():
                st.success(f"✓ {name}")
            else:
                st.warning(f"Missing: {name}")


def pretty_feature_label(feature: str) -> str:
    label, unit, _ = FEATURE_INFO.get(
        feature,
        (feature, "", "Dataset predictor."),
    )

    return f"{label} ({unit})" if unit else label


# =============================================================================
# Load data + preprocessing
# =============================================================================
try:
    data = load_data()
    prep = fit_preprocessor()
except Exception as exc:
    st.error(str(exc))
    st.stop()

available = available_models()

# =============================================================================
# Header
# =============================================================================
st.title("⚡ Appliance Energy Modeling")
st.caption(
    "UCI Appliances Energy Prediction • Your 5 Regression Models • Interactive Prediction"
)

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("Navigation")

    page = st.radio(
        "Select page",
        [
            "Overview",
            "Predict",
            "Model Comparison",
            "Model Details",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.subheader("My Models")

    for name in MODEL_ORDER:
        if model_path(name).exists():
            st.caption(f"✅ {name}")
        else:
            st.caption(f"⚠️ {name}")

    st.divider()

    st.caption("Dataset: Appliances Energy Prediction")
    st.caption("Target: Appliances (Wh)")
    st.caption("Train/test split: 80:20")
    st.caption("Random state: 42")

# =============================================================================
# OVERVIEW
# =============================================================================
if page == "Overview":
    st.header("Project Overview")

    st.write(
        "This dashboard is the interactive GUI for my five regression models on the "
        "UCI Appliances Energy Prediction dataset. The target variable is "
        "`Appliances`, representing appliance energy consumption in watt-hours."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Dataset rows", f"{len(data):,}")

    with c2:
        st.metric("Dataset columns", f"{data.shape[1]}")

    with c3:
        st.metric("Final predictors", f"{len(prep['final_columns'])}")

    with c4:
        st.metric("Models available", f"{len(available)}/5")

    st.subheader("My five regression models")
    show_model_badges()

    st.subheader("Current preprocessing pipeline")

    st.markdown(
        """
        **1.** Remove the raw `date` column  
        **2.** 80:20 `train_test_split(random_state=42)`  
        **3.** Mean imputation using training data  
        **4.** `log1p()` transformation on `lights`  
        **5.** IQR-based clipping using training-derived limits  
        **6.** Remove highly correlated predictors above 0.90  
        **7.** Standardize with `StandardScaler` fitted on training data only  
        **8.** Pass the same transformed feature vector to all five models
        """
    )

    st.subheader("Feature information")

    feature_table = pd.DataFrame(
        [
            {
                "Feature": feature,
                "Meaning": FEATURE_INFO.get(feature, (feature, "", ""))[0],
                "Unit": FEATURE_INFO.get(feature, (feature, "", ""))[1],
            }
            for feature in prep["raw_columns"]
        ]
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Final features used by the trained models"):
        st.write(prep["final_columns"])

    with st.expander("Features removed by correlation filtering"):
        st.write(
            prep["removed_columns"]
            if prep["removed_columns"]
            else "No predictors were removed."
        )

# =============================================================================
# PREDICT
# =============================================================================
elif page == "Predict":
    st.header("Predict Appliance Energy Consumption")

    st.write(
        "Enter the environmental and household measurements below and select one "
        "of the five trained regression models."
    )

    if not available:
        st.error(
            "No model .pkl files were found in the models/ directory. "
            "Run your five-model notebook first."
        )
        st.stop()

    selected_model = st.selectbox(
        "Choose a regression model",
        available,
    )

    st.info(MODEL_REGISTRY[selected_model]["description"])
    st.caption(f"Model note: {MODEL_REGISTRY[selected_model]['note']}")

    st.divider()

    st.subheader("Input measurements")

    st.caption(
        "The default values are training-set medians for convenience. "
        "Replace them with the measurements you want to test."
    )

    input_values: dict[str, float] = {}

    # Three UI columns for a compact layout.
    raw_columns = prep["raw_columns"]

    for start in range(0, len(raw_columns), 3):
        ui_cols = st.columns(3)

        for ui_col, feature in zip(ui_cols, raw_columns[start : start + 3]):
            feature_name, unit, description = FEATURE_INFO.get(
                feature,
                (feature, "", "Dataset feature."),
            )

            default = float(
                prep["defaults"].get(
                    feature,
                    prep["imputation_values"].get(feature, 0.0),
                )
            )

            with ui_col:
                input_values[feature] = st.number_input(
                    feature_name,
                    value=default,
                    format="%.4f",
                    key=f"input_{feature}",
                    help=f"{description} Unit: {unit}" if unit else description,
                )

    predict_clicked = st.button(
        "⚡ Predict Appliance Energy",
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:
        try:
            x_user = transform_user_input(input_values, prep)
            prediction = predict(selected_model, x_user)

            st.success(
                f"Predicted appliance energy consumption: **{prediction:,.2f} Wh**"
            )

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("Prediction", f"{prediction:,.2f} Wh")

            with m2:
                st.metric("Model", selected_model)

            with m3:
                st.metric("Features used", str(len(prep["final_columns"])))

            st.subheader("Entered values")

            summary = pd.DataFrame(
                {
                    "Feature": [
                        pretty_feature_label(feature)
                        for feature in input_values.keys()
                    ],
                    "Value": [
                        f"{value:.4f}"
                        for value in input_values.values()
                    ],
                }
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.warning(
                "Make sure this .pkl file was trained with the same preprocessing "
                "and feature order used in your current regression notebook."
            )

# =============================================================================
# MODEL COMPARISON
# =============================================================================
elif page == "Model Comparison":
    st.header("My Five-Model Comparison")

    st.write(
        "Comparison of the five regression models using the results exported "
        "from the regression notebook."
    )

    metrics_df = load_metrics()

    if metrics_df is None:
        st.warning(
            "No `model_metrics.csv` found yet. Export the comparison table "
            "from the notebook to `models/model_metrics.csv`."
        )

        st.code(
            "comparison_df.reset_index(names='Model').to_csv("
            "'../models/model_metrics.csv', index=False)",
            language="python",
        )

    else:
        # ---------------------------------------------------------------------
        # Normalize the CSV structure
        # ---------------------------------------------------------------------
        metrics_df = metrics_df.copy()

        # Remove accidental CSV index columns such as "Unnamed: 0"
        metrics_df = metrics_df.loc[
            :,
            ~metrics_df.columns.astype(str).str.startswith("Unnamed")
        ]

        # Handle files where the model name was saved as the DataFrame index
        # instead of as a "Model" column.
        if "Model" not in metrics_df.columns:
            metrics_df = metrics_df.reset_index()

        # After reset_index(), the model column may be called "index".
        if "Model" not in metrics_df.columns:
            if "index" in metrics_df.columns:
                metrics_df = metrics_df.rename(columns={"index": "Model"})
            else:
                # Fallback: use the first column as model name
                first_col = metrics_df.columns[0]
                metrics_df = metrics_df.rename(columns={first_col: "Model"})

        # Clean column names
        metrics_df.columns = [
            str(col).strip()
            for col in metrics_df.columns
        ]

        # Normalize common metric column names
        rename_map = {}

        for col in metrics_df.columns:
            clean = str(col).strip().lower()

            if clean in {"r²", "r2", "r^2"}:
                rename_map[col] = "R²"
            elif clean == "rmse":
                rename_map[col] = "RMSE"
            elif clean == "mae":
                rename_map[col] = "MAE"
            elif clean == "mse":
                rename_map[col] = "MSE"
            elif clean == "train rmse":
                rename_map[col] = "Train RMSE"

        metrics_df = metrics_df.rename(columns=rename_map)

        # Ensure metric columns are numeric
        for col in ["R²", "RMSE", "MAE", "MSE", "Train RMSE"]:
            if col in metrics_df.columns:
                metrics_df[col] = pd.to_numeric(
                    metrics_df[col],
                    errors="coerce"
                )

        # Remove completely invalid rows
        if "Model" in metrics_df.columns:
            metrics_df["Model"] = metrics_df["Model"].astype(str).str.strip()
            metrics_df = metrics_df[
                metrics_df["Model"].notna()
                & (metrics_df["Model"] != "")
            ]

        metrics_df = metrics_df.reset_index(drop=True)

        # ---------------------------------------------------------------------
        # Results table
        # ---------------------------------------------------------------------
        st.subheader("Results Table")

        display_cols = [
            col
            for col in [
                "Model",
                "R²",
                "RMSE",
                "MAE",
                "MSE",
                "Train RMSE",
            ]
            if col in metrics_df.columns
        ]

        st.dataframe(
            metrics_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        # ---------------------------------------------------------------------
        # Charts
        # ---------------------------------------------------------------------
        try:
            import plotly.express as px

            # R²
            if "R²" in metrics_df.columns:
                st.subheader("R² Comparison")

                chart_df = metrics_df[["Model", "R²"]].dropna()

                fig = px.bar(
                    chart_df,
                    x="Model",
                    y="R²",
                    text_auto=".3f",
                    title="R² Score by Regression Model",
                )

                fig.update_layout(
                    xaxis_title="Regression Model",
                    yaxis_title="R²",
                    xaxis_tickangle=-20,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            # RMSE
            if "RMSE" in metrics_df.columns:
                st.subheader("RMSE Comparison")

                chart_df = metrics_df[["Model", "RMSE"]].dropna()

                fig = px.bar(
                    chart_df,
                    x="Model",
                    y="RMSE",
                    text_auto=".2f",
                    title="RMSE by Regression Model",
                )

                fig.update_layout(
                    xaxis_title="Regression Model",
                    yaxis_title="RMSE",
                    xaxis_tickangle=-20,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            # MAE
            if "MAE" in metrics_df.columns:
                st.subheader("MAE Comparison")

                chart_df = metrics_df[["Model", "MAE"]].dropna()

                fig = px.bar(
                    chart_df,
                    x="Model",
                    y="MAE",
                    text_auto=".2f",
                    title="MAE by Regression Model",
                )

                fig.update_layout(
                    xaxis_title="Regression Model",
                    yaxis_title="MAE",
                    xaxis_tickangle=-20,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        except ImportError:
            st.warning(
                "Plotly is not installed. Install it with:\n\n"
                "`pip install plotly`"
            )

        # ---------------------------------------------------------------------
        # Model ranking information
        # ---------------------------------------------------------------------
        if "R²" in metrics_df.columns:
            valid_r2 = metrics_df.dropna(subset=["R²"])

            if not valid_r2.empty:
                highest_r2 = valid_r2.loc[valid_r2["R²"].idxmax()]

                st.subheader("Current Results")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Highest R²",
                        f"{highest_r2['R²']:.4f}",
                    )

                with c2:
                    st.metric(
                        "Model",
                        str(highest_r2["Model"]),
                    )

                if "RMSE" in highest_r2:
                    with c3:
                        st.metric(
                            "RMSE",
                            f"{highest_r2['RMSE']:.2f}",
                        )

# =============================================================================
# MODEL DETAILS
# =============================================================================
elif page == "Model Details":
    st.header("My Five Regression Algorithms")

    for name in MODEL_ORDER:
        info = MODEL_REGISTRY[name]

        with st.expander(name):
            st.write(info["description"])
            st.write(f"**Assignment focus:** {info['note']}")
            st.write(f"**Artifact:** `{info['file']}`")

            if model_path(name).exists():
                st.success("Trained model artifact found.")
            else:
                st.warning("Trained model artifact not found.")

    st.divider()

    st.subheader("Evaluation metrics")

    metric_df = pd.DataFrame(
        {
            "Metric": ["R²", "RMSE", "MAE"],
            "Meaning": [
                "Measures the proportion of target variance explained by the model.",
                "Square root of mean squared error; larger errors are penalized more heavily.",
                "Mean absolute prediction error in Wh.",
            ],
        }
    )

    st.dataframe(
        metric_df,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The assignment also requires 5-fold cross-validated R² for the two "
        "best-performing regression models. Those values should come from the "
        "notebook's cross-validation results."
    )

# =============================================================================
# ABOUT
# =============================================================================
elif page == "About":
    st.header("About")

    st.markdown(
        """
        ### Appliance Energy Modeling

        This project applies regression algorithms to the **UCI Appliances Energy
        Prediction** dataset.

        **Target:** `Appliances` (Wh)

        **My models:**
        - Decision Tree Regressor
        - Random Forest Regressor
        - Gradient Boosting Regressor
        - Support Vector Regressor (SVR)
        - K-Nearest Neighbors Regressor (KNN)

        The GUI is intended as the interactive component of the machine-learning
        project. It accepts user measurements and returns an appliance-energy
        prediction from a selected trained model.
        """
    )

    st.subheader("Dataset workflow")

    st.write(
        "The prediction interface reproduces the preprocessing used by the "
        "current five-model notebook so that a new input is transformed in the "
        "same way as the training data."
    )

    st.subheader("Repository structure")

    st.code(
        "case study/\n"
        "├── data/\n"
        "│   └── energydata_complete.csv\n"
        "├── models/\n"
        "│   ├── decision_tree_appliances_randomsplit.pkl\n"
        "│   ├── random_forest_appliances_randomsplit.pkl\n"
        "│   ├── gradient_boosting_appliances_randomsplit.pkl\n"
        "│   ├── svr_appliances_randomsplit.pkl\n"
        "│   └── knn_appliances_randomsplit.pkl\n"
        "├── notebooks/\n"
        "│   └── regression_appliances_5_models_randomsplit_tuned.ipynb\n"
        "└── gui/\n"
        "    └── app.py\n",
        language="text",
    )

    st.caption(
        "Model training and hyperparameter tuning remain in the notebook; "
        "this application is the interactive prediction interface."
    )
