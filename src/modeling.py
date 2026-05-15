from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .config import (
        MODELS_DIR,
        PROCESSED_DATA_DIR,
        PROJECT_ROOT,
        RANDOM_STATE,
        REPORT_IMAGES_DIR,
    )
    from .preprocessing import clean_dataset, load_raw_data, split_dataset
except ImportError:
    from config import (
        MODELS_DIR,
        PROCESSED_DATA_DIR,
        PROJECT_ROOT,
        RANDOM_STATE,
        REPORT_IMAGES_DIR,
    )
    from preprocessing import clean_dataset, load_raw_data, split_dataset
TARGET_COLUMN = "message_length"
BASELINE_FEATURES = ["category", "sentiment_missing", "platform"]
ENGINEERED_FEATURES = [
    "category",
    "sentiment_missing",
    "platform",
    "word_count",
    "unique_word_count",
    "avg_word_length",
    "unique_word_ratio",
    "digit_count",
    "digit_ratio",
    "has_digits",
]


@dataclass(frozen=True)
class ExperimentResult:
    family: str
    variant: str
    feature_set: str
    params: dict
    val_mae: float
    val_rmse: float
    val_r2: float
    test_mae: float
    test_rmse: float
    test_r2: float

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "variant": self.variant,
            "feature_set": self.feature_set,
            "params": json.dumps(self.params, ensure_ascii=False, sort_keys=True),
            "val_mae": self.val_mae,
            "val_rmse": self.val_rmse,
            "val_r2": self.val_r2,
            "test_mae": self.test_mae,
            "test_rmse": self.test_rmse,
            "test_r2": self.test_r2,
        }


def _project_root() -> Path:
    return PROJECT_ROOT


def prepare_dirs() -> tuple[Path, Path, Path]:
    processed_dir = PROCESSED_DATA_DIR
    images_dir = REPORT_IMAGES_DIR
    models_dir = MODELS_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir, images_dir, models_dir


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cleaned = clean_dataset(load_raw_data())
    return split_dataset(cleaned)


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_preprocessor(features: list[str], scale_numeric: bool) -> ColumnTransformer:
    categorical = [feature for feature in features if feature == "platform"]
    numeric = [feature for feature in features if feature != "platform"]

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def numeric_pipeline(estimator: object, scale_numeric: bool) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(ENGINEERED_FEATURES, scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def baseline_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(BASELINE_FEATURES, scale_numeric=True)),
            ("model", LinearRegression()),
        ]
    )


def text_ridge_pipeline(alpha: float, max_features: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=5,
                    max_features=max_features,
                ),
            ),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def text_svd_ridge_pipeline(alpha: float, max_features: int, n_components: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=5,
                    max_features=max_features,
                ),
            ),
            ("svd", TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def fit_and_score(
    estimator: object,
    X_train: pd.DataFrame | pd.Series,
    y_train: pd.Series,
    X_val: pd.DataFrame | pd.Series,
    y_val: pd.Series,
    X_test: pd.DataFrame | pd.Series,
    y_test: pd.Series,
    family: str,
    variant: str,
    feature_set: str,
    params: dict,
) -> ExperimentResult:
    model = clone(estimator)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = regression_metrics(y_val, val_pred)
    test_metrics = regression_metrics(y_test, test_pred)
    return ExperimentResult(
        family=family,
        variant=variant,
        feature_set=feature_set,
        params=params,
        val_mae=val_metrics["mae"],
        val_rmse=val_metrics["rmse"],
        val_r2=val_metrics["r2"],
        test_mae=test_metrics["mae"],
        test_rmse=test_metrics["rmse"],
        test_r2=test_metrics["r2"],
    )


def run_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    y_train = train_df[TARGET_COLUMN]
    y_val = val_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    results: list[ExperimentResult] = []
    fitted_candidates: dict[str, object] = {}

    baseline = baseline_pipeline()
    result = fit_and_score(
        baseline,
        train_df[BASELINE_FEATURES],
        y_train,
        val_df[BASELINE_FEATURES],
        y_val,
        test_df[BASELINE_FEATURES],
        y_test,
        family="baseline_linear",
        variant="baseline_linear",
        feature_set="minimal_tabular",
        params={"features": BASELINE_FEATURES},
    )
    results.append(result)
    fitted_candidates["baseline_linear"] = baseline

    linear = numeric_pipeline(LinearRegression(), scale_numeric=True)
    results.append(
        fit_and_score(
            linear,
            train_df[ENGINEERED_FEATURES],
            y_train,
            val_df[ENGINEERED_FEATURES],
            y_val,
            test_df[ENGINEERED_FEATURES],
            y_test,
            family="linear_regression",
            variant="linear_regression",
            feature_set="engineered_tabular",
            params={"features": ENGINEERED_FEATURES},
        )
    )

    for alpha in [0.1, 1.0, 10.0, 30.0]:
        ridge = numeric_pipeline(Ridge(alpha=alpha), scale_numeric=True)
        results.append(
            fit_and_score(
                ridge,
                train_df[ENGINEERED_FEATURES],
                y_train,
                val_df[ENGINEERED_FEATURES],
                y_val,
                test_df[ENGINEERED_FEATURES],
                y_test,
                family="ridge",
                variant=f"ridge_alpha_{alpha}",
                feature_set="engineered_tabular",
                params={"alpha": alpha},
            )
        )

    for max_depth in [20, None]:
        for min_samples_leaf in [1, 3]:
            forest = numeric_pipeline(
                RandomForestRegressor(
                    n_estimators=180,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                ),
                scale_numeric=False,
            )
            results.append(
                fit_and_score(
                    forest,
                    train_df[ENGINEERED_FEATURES],
                    y_train,
                    val_df[ENGINEERED_FEATURES],
                    y_val,
                    test_df[ENGINEERED_FEATURES],
                    y_test,
                    family="random_forest",
                    variant=f"rf_depth_{max_depth}_leaf_{min_samples_leaf}",
                    feature_set="engineered_tabular",
                    params={
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "n_estimators": 180,
                    },
                )
            )

    for max_depth in [None, 30]:
        for min_samples_leaf in [1, 3]:
            extra_trees = numeric_pipeline(
                ExtraTreesRegressor(
                    n_estimators=220,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                ),
                scale_numeric=False,
            )
            results.append(
                fit_and_score(
                    extra_trees,
                    train_df[ENGINEERED_FEATURES],
                    y_train,
                    val_df[ENGINEERED_FEATURES],
                    y_val,
                    test_df[ENGINEERED_FEATURES],
                    y_test,
                    family="extra_trees",
                    variant=f"et_depth_{max_depth}_leaf_{min_samples_leaf}",
                    feature_set="engineered_tabular",
                    params={
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "n_estimators": 220,
                    },
                )
            )

    for learning_rate in [0.05, 0.1]:
        for max_depth in [3, 5]:
            hgb = numeric_pipeline(
                GradientBoostingRegressor(
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    n_estimators=220,
                    random_state=RANDOM_STATE,
                ),
                scale_numeric=False,
            )
            results.append(
                fit_and_score(
                    hgb,
                    train_df[ENGINEERED_FEATURES],
                    y_train,
                    val_df[ENGINEERED_FEATURES],
                    y_val,
                    test_df[ENGINEERED_FEATURES],
                    y_test,
                    family="gradient_boosting",
                    variant=f"gbr_lr_{learning_rate}_depth_{max_depth}",
                    feature_set="engineered_tabular",
                    params={
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                        "n_estimators": 220,
                    },
                )
            )

    for alpha in [0.5, 1.0, 5.0]:
        for max_features in [3000, 5000]:
            tfidf_ridge = text_ridge_pipeline(alpha=alpha, max_features=max_features)
            results.append(
                fit_and_score(
                    tfidf_ridge,
                    train_df["text"],
                    y_train,
                    val_df["text"],
                    y_val,
                    test_df["text"],
                    y_test,
                    family="tfidf_ridge",
                    variant=f"tfidf_ridge_alpha_{alpha}_mf_{max_features}",
                    feature_set="text_tfidf",
                    params={"alpha": alpha, "max_features": max_features},
                )
            )

    for n_components in [50, 100, 200]:
        for alpha in [0.5, 1.0, 5.0]:
            tfidf_svd = text_svd_ridge_pipeline(
                alpha=alpha,
                max_features=5000,
                n_components=n_components,
            )
            results.append(
                fit_and_score(
                    tfidf_svd,
                    train_df["text"],
                    y_train,
                    val_df["text"],
                    y_val,
                    test_df["text"],
                    y_test,
                    family="tfidf_svd_ridge",
                    variant=f"tfidf_svd_{n_components}_alpha_{alpha}",
                    feature_set="text_tfidf_svd",
                    params={"alpha": alpha, "max_features": 5000, "n_components": n_components},
                )
            )

    results_df = pd.DataFrame([item.to_dict() for item in results])
    results_df = results_df.sort_values(by=["val_mae", "test_mae"]).reset_index(drop=True)

    best_ridge_row = results_df[results_df["family"] == "ridge"].iloc[0]
    best_extra_row = results_df[results_df["family"] == "extra_trees"].iloc[0]
    best_hgb_row = results_df[results_df["family"] == "gradient_boosting"].iloc[0]

    ridge_alpha = json.loads(best_ridge_row["params"])["alpha"]
    extra_params = json.loads(best_extra_row["params"])
    hgb_params = json.loads(best_hgb_row["params"])

    ensemble = Pipeline(
        [
            ("preprocessor", build_preprocessor(ENGINEERED_FEATURES, scale_numeric=False)),
            (
                "model",
                VotingRegressor(
                    estimators=[
                        ("ridge", Ridge(alpha=ridge_alpha)),
                        (
                            "extra",
                            ExtraTreesRegressor(
                                n_estimators=extra_params["n_estimators"],
                                max_depth=extra_params["max_depth"],
                                min_samples_leaf=extra_params["min_samples_leaf"],
                                n_jobs=1,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                        (
                            "hgb",
                            GradientBoostingRegressor(
                                learning_rate=hgb_params["learning_rate"],
                                max_depth=hgb_params["max_depth"],
                                n_estimators=hgb_params["n_estimators"],
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                [
                    fit_and_score(
                        ensemble,
                        train_df[ENGINEERED_FEATURES],
                        y_train,
                        val_df[ENGINEERED_FEATURES],
                        y_val,
                        test_df[ENGINEERED_FEATURES],
                        y_test,
                        family="voting_ensemble",
                        variant="voting_ridge_extra_hgb",
                        feature_set="engineered_tabular",
                        params={
                            "ridge_alpha": ridge_alpha,
                            "extra_params": extra_params,
                            "hgb_params": hgb_params,
                        },
                    ).to_dict()
                ]
            ),
        ],
        ignore_index=True,
    ).sort_values(by=["val_mae", "test_mae"]).reset_index(drop=True)

    best_overall = results_df.iloc[0]
    artifacts = {
        "best_overall": best_overall.to_dict(),
        "best_by_family": (
            results_df.sort_values(by="val_mae").groupby("family", as_index=False).first()
        ),
    }
    return results_df, artifacts


def fit_final_model(best_row: pd.Series, train_df: pd.DataFrame, val_df: pd.DataFrame) -> object:
    combined_train = pd.concat([train_df, val_df], ignore_index=True)
    y_combined = combined_train[TARGET_COLUMN]
    family = best_row["family"]
    params = json.loads(best_row["params"])

    if family == "baseline_linear":
        estimator = baseline_pipeline()
        estimator.fit(combined_train[BASELINE_FEATURES], y_combined)
        return estimator

    if family == "linear_regression":
        estimator = numeric_pipeline(LinearRegression(), scale_numeric=True)
        estimator.fit(combined_train[ENGINEERED_FEATURES], y_combined)
        return estimator

    if family == "ridge":
        estimator = numeric_pipeline(Ridge(alpha=params["alpha"]), scale_numeric=True)
        estimator.fit(combined_train[ENGINEERED_FEATURES], y_combined)
        return estimator

    if family == "random_forest":
        estimator = numeric_pipeline(
            RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            scale_numeric=False,
        )
        estimator.fit(combined_train[ENGINEERED_FEATURES], y_combined)
        return estimator

    if family == "extra_trees":
        estimator = numeric_pipeline(
            ExtraTreesRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            scale_numeric=False,
        )
        estimator.fit(combined_train[ENGINEERED_FEATURES], y_combined)
        return estimator

    if family == "gradient_boosting":
        estimator = numeric_pipeline(
            GradientBoostingRegressor(
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                random_state=RANDOM_STATE,
            ),
            scale_numeric=False,
        )
        estimator.fit(combined_train[ENGINEERED_FEATURES], y_combined)
        return estimator

    if family == "tfidf_ridge":
        estimator = text_ridge_pipeline(alpha=params["alpha"], max_features=params["max_features"])
        estimator.fit(combined_train["text"], y_combined)
        return estimator

    if family == "tfidf_svd_ridge":
        estimator = text_svd_ridge_pipeline(
            alpha=params["alpha"],
            max_features=params["max_features"],
            n_components=params["n_components"],
        )
        estimator.fit(combined_train["text"], y_combined)
        return estimator

    if family == "voting_ensemble":
        estimator = Pipeline(
            [
                ("preprocessor", build_preprocessor(ENGINEERED_FEATURES, scale_numeric=False)),
                (
                    "model",
                    VotingRegressor(
                        estimators=[
                            ("ridge", Ridge(alpha=params["ridge_alpha"])),
                            (
                                "extra",
                                ExtraTreesRegressor(
                                    n_estimators=params["extra_params"]["n_estimators"],
                                    max_depth=params["extra_params"]["max_depth"],
                                    min_samples_leaf=params["extra_params"]["min_samples_leaf"],
                                    n_jobs=1,
                                    random_state=RANDOM_STATE,
                                ),
                            ),
                            (
                                "hgb",
                                GradientBoostingRegressor(
                                    learning_rate=params["hgb_params"]["learning_rate"],
                                    max_depth=params["hgb_params"]["max_depth"],
                                    n_estimators=params["hgb_params"]["n_estimators"],
                                    random_state=RANDOM_STATE,
                                ),
                            ),
                        ]
                    ),
                ),
            ]
        )
        estimator.fit(combined_train[ENGINEERED_FEATURES], y_combined)
        return estimator

    raise ValueError(f"Unsupported family: {family}")


def plot_model_comparison(best_by_family: pd.DataFrame, images_dir: Path) -> Path:
    sns.set_theme(style="whitegrid")
    chart_df = best_by_family.sort_values(by="val_mae").copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=chart_df, x="family", y="val_mae", ax=ax, color="#5B8FF9")
    ax.set_title("Validation MAE by best model in each family")
    ax.set_xlabel("Model family")
    ax.set_ylabel("MAE")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = images_dir / "model_comparison_mae.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_svd_explained_variance(train_df: pd.DataFrame, images_dir: Path) -> Path:
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5, max_features=5000)
    matrix = vectorizer.fit_transform(train_df["text"])
    component_grid = [10, 25, 50, 100, 150, 200]
    curve = []
    for components in component_grid:
        svd = TruncatedSVD(n_components=components, random_state=RANDOM_STATE)
        svd.fit(matrix)
        curve.append(
            {
                "components": components,
                "explained_variance": float(svd.explained_variance_ratio_.sum()),
            }
        )

    curve_df = pd.DataFrame(curve)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=curve_df, x="components", y="explained_variance", marker="o", ax=ax)
    ax.set_title("TruncatedSVD explained variance on TF-IDF features")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Explained variance ratio")
    fig.tight_layout()
    path = images_dir / "svd_explained_variance.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_feature_importance(
    final_model: object,
    processed_dir: Path,
    images_dir: Path,
) -> Path | None:
    if not isinstance(final_model, Pipeline):
        return None

    model = final_model.named_steps.get("model")
    preprocessor = final_model.named_steps.get("preprocessor")
    if model is None or preprocessor is None or not hasattr(model, "feature_importances_"):
        return None

    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    csv_path = processed_dir / "final_model_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    top_features = importance_df.head(12).copy()
    sns.barplot(data=top_features, x="importance", y="feature", ax=ax, color="#5B8FF9")
    ax.set_title("Final model feature importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    image_path = images_dir / "final_model_feature_importance.png"
    fig.savefig(image_path, dpi=200)
    plt.close(fig)
    return csv_path


def save_artifacts(
    results_df: pd.DataFrame,
    artifacts: dict[str, object],
    final_model: object,
    test_metrics: dict[str, float],
    processed_dir: Path,
    models_dir: Path,
) -> None:
    results_df.to_csv(processed_dir / "model_experiments.csv", index=False)
    artifacts_payload = {
        "best_overall": artifacts["best_overall"],
        "best_by_family": artifacts["best_by_family"].to_dict(orient="records"),
        "final_test_metrics": test_metrics,
    }
    (processed_dir / "model_summary.json").write_text(
        json.dumps(artifacts_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    joblib.dump(final_model, models_dir / "final_model.joblib")


def main() -> None:
    processed_dir, images_dir, models_dir = prepare_dirs()
    train_df, val_df, test_df = load_splits()

    results_df, artifacts = run_experiments(train_df=train_df, val_df=val_df, test_df=test_df)
    best_row = pd.Series(artifacts["best_overall"])
    final_model = fit_final_model(best_row=best_row, train_df=train_df, val_df=val_df)

    if best_row["feature_set"] in {"minimal_tabular"}:
        test_pred = final_model.predict(test_df[BASELINE_FEATURES])
    elif best_row["feature_set"] == "engineered_tabular":
        test_pred = final_model.predict(test_df[ENGINEERED_FEATURES])
    else:
        test_pred = final_model.predict(test_df["text"])

    test_metrics = regression_metrics(test_df[TARGET_COLUMN], test_pred)

    plot_model_comparison(artifacts["best_by_family"], images_dir)
    plot_svd_explained_variance(train_df, images_dir)
    save_feature_importance(final_model, processed_dir, images_dir)
    save_artifacts(
        results_df=results_df,
        artifacts=artifacts,
        final_model=final_model,
        test_metrics=test_metrics,
        processed_dir=processed_dir,
        models_dir=models_dir,
    )

    print(processed_dir / "model_experiments.csv")
    print(processed_dir / "model_summary.json")
    print(images_dir / "model_comparison_mae.png")
    print(images_dir / "svd_explained_variance.png")
    print(models_dir / "final_model.joblib")


if __name__ == "__main__":
    main()
