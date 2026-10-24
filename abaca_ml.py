"""Random-forest engine for the Abaca Girth Prediction System.

This module contains the model code from abaca_rf.py (same data checks, same
random-forest settings, same out-of-fold validation) reorganised so that both
the training script (train_model.py) and the web app (app.py) can use it.

Nothing here imports Streamlit, so it can be tested on its own.
Units: cm, degrees Celsius, percentages 0-100, leaf count a whole number.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

# ==================== SETTINGS ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "abaca_real_200.xlsx"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "abaca_rf_model.joblib"
RESULTS_DIR = BASE_DIR / "abaca_results"
SEED = 42
N_FOLDS = 5

# For repeated plants or clustered plots, add their ID column to Excel and put
# its name here (e.g. "plant_id"). Records with the same ID stay in one fold.
GROUP_COLUMN = None

EXPECTED_FEATURES = ["height_cm", "leaf_count", "moisture_pct", "soil_pH",
                     "temperature", "humidity_pct", "sun_shade_pct"]
FEATURES = EXPECTED_FEATURES.copy()
TARGET = "girth_cm"
LABELS = ["Height", "Leaf count", "Moisture", "Soil pH", "Temperature",
          "Humidity", "Shade"]
LABEL_OF = dict(zip(FEATURES, LABELS))

# Display settings for the web app. "phys" is the physically possible range.
FEATURE_INFO = {
    "height_cm": dict(label="Plant height", unit="cm", kind="float", step=1.0,
                      decimals=0, phys=(1.0, 1000.0),
                      help="Height of the plant in centimeters."),
    "leaf_count": dict(label="Leaf count", unit="leaves", kind="int", step=1,
                       decimals=0, phys=(0, 100),
                       help="Number of leaves (a whole number)."),
    "moisture_pct": dict(label="Soil moisture", unit="%", kind="float", step=0.1,
                         decimals=1, phys=(0.0, 100.0),
                         help="Soil moisture in percent (0-100)."),
    "soil_pH": dict(label="Soil pH", unit="", kind="float", step=0.01,
                    decimals=2, phys=(0.0, 14.0),
                    help="Soil acidity on the 0-14 pH scale."),
    "temperature": dict(label="Temperature", unit="°C", kind="float", step=0.1,
                        decimals=1, phys=(0.0, 60.0),
                        help="Air temperature in degrees Celsius."),
    "humidity_pct": dict(label="Relative humidity", unit="%", kind="float",
                         step=0.5, decimals=1, phys=(0.0, 100.0),
                         help="Relative humidity in percent (0-100)."),
    "sun_shade_pct": dict(label="Shade level", unit="%", kind="float", step=1.0,
                          decimals=0, phys=(0.0, 100.0),
                          help="Sunshade index in percent. Use the same "
                               "definition of shade as the training dataset."),
}

REQUIRED_KEYS = ("model", "features", "training_ranges", "metrics", "validation",
                 "importance_impurity", "importance_permutation",
                 "residual_quantiles", "medians", "girth_values", "train_values",
                 "familiarity", "fold_metrics", "data_notes", "trained_at",
                 "validation_scheme", "n_records", "sklearn_version", "input_units")


class ModelUnavailable(RuntimeError):
    """Raised when there is neither a saved model nor data to train one."""


# ==================== MODEL ====================
def make_model():
    # Conservative fixed settings; no parameter search on validation results.
    return RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                 max_features=0.8, random_state=SEED, n_jobs=-1)


def validate_features(x):
    """Raise on invalid measurements instead of silently replacing them."""
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Predictors contain missing or infinite values.")
    if (x["height_cm"] <= 0).any():
        raise ValueError("Height must be positive.")
    leaves = x["leaf_count"]
    if ((leaves < 0) | (leaves != np.floor(leaves))).any():
        raise ValueError("Leaf count must be a nonnegative whole number.")
    for col in ["moisture_pct", "humidity_pct", "sun_shade_pct"]:
        if not x[col].between(0, 100).all():
            raise ValueError(f"{col} must be between 0 and 100.")
    if not x["soil_pH"].between(0, 14).all():
        raise ValueError("Soil pH must be between 0 and 14.")
    if (x["temperature"] <= -273.15).any():
        raise ValueError("Temperature must exceed absolute zero in Celsius.")


def load_data(path, group_column=GROUP_COLUMN):
    """Read and check the Excel file. Returns (DataFrame, list of data notes)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Excel file not found: {path}")
    df = pd.read_excel(path)
    required = ["height_cm", "leaf_count", "moisture", "soil_pH",
                "temperature", "humidity", "sun_shade", TARGET]
    if group_column:
        required.append(group_column)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing Excel columns: {missing}")
    for col in required:
        if col != group_column:
            df[col] = pd.to_numeric(df[col], errors="raise")
    if df[required].isna().any().any():
        raise ValueError("Required cells contain missing values. Check Excel first.")
    for raw in ["moisture", "humidity", "sun_shade"]:
        if not df[raw].between(0, 1).all():
            raise ValueError(f"Excel {raw} must use fractions, e.g. 0.96, not 96.")
        df[raw + "_pct"] = df[raw] * 100
    validate_features(df[FEATURES])
    if not np.isfinite(df[TARGET]).all() or (df[TARGET] <= 0).any():
        raise ValueError("Girth must contain finite, positive centimeter values.")
    if len(df) < N_FOLDS * 4 or df[TARGET].nunique() < 2:
        raise ValueError("Too few records or no target variation for evaluation.")

    notes = []
    # Flag duplicate observations for investigation; do not delete real plants.
    duplicates = int(df.duplicated(FEATURES + [TARGET]).sum())
    if duplicates:
        notes.append(f"{duplicates} identical predictor/target rows found. "
                     "Verify where these records came from.")
    corr = df[FEATURES].corr()
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            if abs(corr.loc[a, b]) >= 0.95:
                notes.append(f"{LABEL_OF[a]} and {LABEL_OF[b]} are almost the same "
                             f"signal (correlation {corr.loc[a, b]:.3f}), so they "
                             "can share importance.")
    return df, notes


def _scores(y, pred):
    return {"r2": float(r2_score(y, pred)),
            "rmse_cm": float(np.sqrt(mean_squared_error(y, pred))),
            "mae_cm": float(mean_absolute_error(y, pred))}


def evaluate(df, group_column=GROUP_COLUMN):
    """Out-of-fold validation: every row is predicted by a model that never saw it."""
    x, y = df[FEATURES], df[TARGET]
    if group_column:
        if df[group_column].nunique() < N_FOLDS:
            raise ValueError(f"Need at least {N_FOLDS} distinct groups.")
        splits = GroupKFold(N_FOLDS).split(x, y, groups=df[group_column])
        scheme = f"Grouped {N_FOLDS}-fold (by {group_column})"
    else:
        splits = KFold(N_FOLDS, shuffle=True, random_state=SEED).split(x, y)
        scheme = f"Random {N_FOLDS}-fold (assumes independent plants)"
    oof, baseline = np.full(len(df), np.nan), np.full(len(df), np.nan)
    importances, folds = [], []
    for fold, (tr, va) in enumerate(splits, 1):
        model = make_model().fit(x.iloc[tr], y.iloc[tr])
        pred = model.predict(x.iloc[va])
        oof[va] = pred
        dummy = DummyRegressor(strategy="mean").fit(x.iloc[tr], y.iloc[tr])
        baseline[va] = dummy.predict(x.iloc[va])
        folds.append({"fold": fold, "n_validation": len(va),
                      **_scores(y.iloc[va], pred)})
        # Only held-out rows determine importance. Increase in MAE is in cm.
        imp = permutation_importance(model, x.iloc[va], y.iloc[va],
                                     scoring="neg_mean_absolute_error",
                                     n_repeats=15, random_state=SEED + fold,
                                     n_jobs=1)
        importances.append(imp.importances)
    return {"oof": oof, "baseline": baseline, "folds": folds, "scheme": scheme,
            "importances": np.array(importances)}


def _familiarity(x):
    """Standardised training data plus a 'familiar combination' distance limit.

    A new input is 'familiar' if its nearest training plant is no farther away
    than 95% of training plants are from their own nearest neighbour.
    """
    arr = x.to_numpy(dtype=float)
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    std[std == 0] = 1.0
    z = (arr - mean) / std
    dist = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)
    threshold = max(float(np.quantile(nearest, 0.95)), 1e-6)
    return {"mean": mean, "std": std, "train_z": z, "threshold": threshold}


def train(data_path=DATA_PATH, model_path=MODEL_PATH, save=True,
          group_column=GROUP_COLUMN):
    """Validate, fit the final model on all records, and build the app bundle."""
    df, notes = load_data(data_path, group_column)
    ev = evaluate(df, group_column)
    y = df[TARGET].to_numpy(dtype=float)
    # Final deployment model uses all available observations AFTER validation.
    model = make_model().fit(df[FEATURES], df[TARGET])

    fold_imp = ev["importances"].mean(axis=2)             # folds x features
    perm_sd = fold_imp.std(axis=0, ddof=1)
    impurity = model.feature_importances_ * 100
    residual = y - ev["oof"]

    bundle = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "training_ranges": {f: (float(df[f].min()), float(df[f].max()))
                            for f in FEATURES},
        "sklearn_version": sklearn.__version__,
        "input_units": "cm, Celsius, and percentages 0-100; leaf count integer",
        "note": "Current girth estimate. Correlations do not establish causation.",
        "n_records": int(len(df)),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "validation_scheme": ev["scheme"],
        "metrics": {"rf": _scores(y, ev["oof"]),
                    "baseline": _scores(y, ev["baseline"])},
        "fold_metrics": ev["folds"],
        "validation": {"excel_row": (np.arange(len(df)) + 2).tolist(),
                       "actual": y.tolist(), "predicted": ev["oof"].tolist(),
                       "baseline": ev["baseline"].tolist()},
        "importance_impurity": {f: float(v) for f, v in zip(FEATURES, impurity)},
        "importance_permutation": {
            f: {"mean_mae_increase_cm": float(m), "fold_sd_cm": float(s)}
            for f, m, s in zip(FEATURES, fold_imp.mean(axis=0), perm_sd)},
        "residual_quantiles": {"q10": float(np.quantile(residual, 0.10)),
                               "q90": float(np.quantile(residual, 0.90))},
        "medians": {f: float(df[f].median()) for f in FEATURES},
        "girth_values": y.tolist(),
        "train_values": {f: df[f].astype(float).tolist() for f in FEATURES},
        "familiarity": _familiarity(df[FEATURES]),
        "data_notes": notes,
    }
    if save:
        save_bundle(bundle, model_path)
    return bundle


def save_bundle(bundle, model_path=MODEL_PATH):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)


def load_or_train(model_path=MODEL_PATH, data_path=DATA_PATH):
    """Return (bundle, notice). Loads the saved model, retraining if needed.

    Retraining happens only when the saved model is missing, unreadable, or was
    built with a different scikit-learn version, AND the Excel data is present.
    """
    model_path, data_path = Path(model_path), Path(data_path)
    bundle, notice = None, None
    if model_path.is_file():
        try:
            loaded = joblib.load(model_path)
            if all(k in loaded for k in REQUIRED_KEYS):
                bundle = loaded
        except Exception:  # unreadable or incompatible pickle
            bundle = None
    stale = (bundle is None
             or bundle.get("sklearn_version") != sklearn.__version__)

    if stale and data_path.is_file():
        try:
            fresh = train(data_path, model_path, save=False)
        except Exception as exc:
            if bundle is None:
                raise ModelUnavailable(f"Could not train from {data_path.name}: {exc}")
            notice = ("The saved model was built with a different scikit-learn "
                      "version and could not be rebuilt: " + str(exc))
        else:
            bundle = fresh
            try:
                save_bundle(bundle, model_path)
            except Exception:
                pass  # read-only host: keep the in-memory model
    elif stale and bundle is not None:
        notice = ("This model was built with scikit-learn "
                  f"{bundle.get('sklearn_version')} but the app runs "
                  f"{sklearn.__version__}. Results may differ slightly; "
                  "retrain with train_model.py to remove this warning.")
    elif bundle is None:
        raise ModelUnavailable(
            "No trained model found. Put your Excel file at "
            f"data/{data_path.name} and run `python train_model.py`, "
            "then commit the new model/ folder.")
    return bundle, notice


# ==================== PREDICTION ====================
def predict(bundle, measurements):
    """Predict girth and describe how much to trust the number."""
    if set(measurements) != set(FEATURES):
        raise ValueError(f"Measurements must contain exactly: {FEATURES}")
    sample = pd.DataFrame([measurements], columns=FEATURES).apply(
        pd.to_numeric, errors="raise")
    validate_features(sample)
    model = bundle["model"]
    pred = float(model.predict(sample)[0])
    row = sample.to_numpy(dtype=float)
    tree_sd = float(np.std([t.predict(row)[0] for t in model.estimators_]))

    q = bundle["residual_quantiles"]
    low, high = pred + q["q10"], pred + q["q90"]
    girths = np.asarray(bundle["girth_values"])
    percentile = float((girths < pred).mean() * 100)

    outside = []
    for f in FEATURES:
        lo, hi = bundle["training_ranges"][f]
        v = float(sample[f].iloc[0])
        if v < lo or v > hi:
            outside.append(f)

    fam = bundle["familiarity"]
    z = (row[0] - fam["mean"]) / fam["std"]
    nearest = float(np.sqrt(((fam["train_z"] - z) ** 2).sum(axis=1)).min())
    return {"prediction": pred, "low": low, "high": high, "tree_sd": tree_sd,
            "percentile": percentile, "outside": outside,
            "nearest_distance": nearest, "familiar_limit": fam["threshold"],
            "familiar": nearest <= fam["threshold"], "inputs": dict(measurements)}


def sweep(bundle, measurements, feature, lo, hi, points=60):
    """Predicted girth as one input varies and all others stay fixed."""
    if FEATURE_INFO[feature]["kind"] == "int":
        xs = np.arange(int(np.ceil(lo)), int(np.floor(hi)) + 1, dtype=float)
    else:
        xs = np.linspace(lo, hi, points)
    frame = pd.DataFrame([measurements] * len(xs), columns=FEATURES).astype(float)
    frame[feature] = xs
    return xs, bundle["model"].predict(frame)
