"""Reproducible, fold-fitted BigMart model selection and final inference.

Run ``python3 -m modeling`` from the project folder for the complete experiment.
The notebook calls the same functions. Content-addressed caches avoid rerunning
unchanged searches; changing inputs, code, versions or search settings invalidates
them. The reserved modelling check is used only after a winner has been frozen.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import parallel_config
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, cross_val_predict,
    cross_validate, train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from preprocessing import BigMartPreprocessor, ID_COLUMNS, RAW_FEATURES, TARGET


SEED = 42
MODEL_ORDER = ["LinearRegression", "KNN", "SVR", "RandomForest"]
PROFILES = ["compact", "descriptive_ordered", "descriptive_onehot"]
SEARCH_BUDGETS = {"KNN": 16, "SVR": 20, "RandomForest": 24}
VISIBILITY_OPTIONS = [
    ("impute", "raw"), ("impute", "sqrt"),
    ("impute", "capped"), ("keep_zero", "raw"),
]
TARGET_MODES = ["raw", "sqrt", "log1p"]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def frame_hash(frame):
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def package_versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__}


def experiment_fingerprint(train):
    digest = hashlib.sha256(frame_hash(train).encode())
    for filename in ["modeling.py", "preprocessing.py"]:
        digest.update(Path(__file__).with_name(filename).read_bytes())
    digest.update(json.dumps(package_versions(), sort_keys=True).encode())
    return digest.hexdigest()


@contextmanager
def computation_context():
    # Threads work without local worker sockets in restricted environments.
    # Each estimator and numerical library stays single-threaded inside a fold.
    with threadpool_limits(limits=1), parallel_config(backend="threading"):
        yield


class ModelFeatures(TransformerMixin, BaseEstimator):
    """Select a documented profile using only the fitted input feature schema."""

    def __init__(self, profile="compact", visibility="raw", include_ratio=False):
        self.profile = profile
        self.visibility = visibility
        self.include_ratio = include_ratio

    @staticmethod
    def _block(frame, prefix):
        # The first training-observed level is the reference category.
        return sorted(c for c in frame if c.startswith(prefix))[1:]

    def fit(self, X, y=None):
        if self.profile not in PROFILES + ["price_interactions"]:
            raise ValueError(f"Unknown feature profile: {self.profile}")
        if self.visibility not in {"raw", "sqrt", "capped"}:
            raise ValueError(f"Unknown visibility representation: {self.visibility}")
        self.item_columns_ = self._block(X, "Item_Type_")
        prefix = "Outlet_OUT" if self.profile in {"compact", "price_interactions"} else "Outlet_Type_"
        self.outlet_columns_ = self._block(X, prefix)
        self.interaction_columns_ = self._block(X, "Outlet_Type_")
        self.metadata_levels_ = {
            col: sorted(X[col].unique().tolist())
            for col in ["Outlet_Size_code", "Outlet_Location_code"]
        }
        self.output_columns_ = self._assemble(X).columns.tolist()
        return self

    def _assemble(self, X):
        visibility_col = {"raw": "Item_Visibility", "sqrt": "Item_Visibility_sqrt",
                          "capped": "Item_Visibility_iqr"}[self.visibility]
        out = X[["Item_MRP", "Item_Weight", "Item_Visibility_ZeroOrMissing"]].copy()
        out["Visibility"] = X[visibility_col]
        # With item-type columns, the non-edible indicator is redundant.
        out["Item_Fat_Content_Regular"] = X.get("Item_Fat_Content_Regular", 0)
        for col in self.item_columns_ + self.outlet_columns_:
            out[col] = X[col]
        if self.profile.startswith("descriptive"):
            for col in ["Outlet_Years", "MRP_Band_code", "Item_Weight_Missing", "Outlet_Size_Missing"]:
                out[col] = X[col]
            for col, levels in self.metadata_levels_.items():
                if self.profile == "descriptive_ordered":
                    out[col] = X[col]
                else:
                    for level in levels[1:]:
                        out[f"{col}_level_{level}"] = X[col].eq(level).astype(int)
        if self.profile == "price_interactions":
            for col in self.interaction_columns_:
                out[f"MRP_x_{col}"] = X["Item_MRP"] * X[col]
        if self.include_ratio:
            out["Item_Visibility_MeanRatio"] = X["Item_Visibility_MeanRatio"]
        return out

    def transform(self, X):
        check_is_fitted(self, "output_columns_")
        return self._assemble(X).loc[:, self.output_columns_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_columns_")
        return np.asarray(self.output_columns_, dtype=object)


class FeatureScaler(TransformerMixin, BaseEstimator):
    """Scale numerical magnitudes while leaving binary indicators unchanged."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        self.columns_ = X.columns.tolist()
        binary_prefixes = ("Item_Type_", "Outlet_OUT", "Outlet_Type_")
        binary_names = {"Item_Fat_Content_Regular", "Item_Visibility_ZeroOrMissing",
                        "Item_Weight_Missing", "Outlet_Size_Missing"}
        self.numeric_columns_ = [
            c for c in X if not (c.startswith(binary_prefixes) or c in binary_names or "_level_" in c)
        ]
        if self.enabled:
            self.scaler_ = StandardScaler().fit(X[self.numeric_columns_])
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        out = X.loc[:, self.columns_].copy()
        if self.enabled:
            scaled = self.scaler_.transform(out[self.numeric_columns_])
            for i, col in enumerate(self.numeric_columns_):
                out[col] = scaled[:, i]
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columns_")
        return np.asarray(self.columns_, dtype=object)


def transform_target(y, mode):
    values = np.asarray(y, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Sales targets must be finite and nonnegative.")
    if mode == "raw":
        return values
    if mode == "sqrt":
        return np.sqrt(values)
    if mode == "log1p":
        return np.log1p(values)
    raise ValueError(f"Unknown target mode: {mode}")


def inverse_target(values, mode):
    values = np.asarray(values, dtype=float)
    with np.errstate(over="raise", invalid="raise"):
        if mode == "raw":
            result = values
        elif mode == "sqrt":
            result = np.maximum(values, 0) ** 2
        elif mode == "log1p":
            result = np.expm1(values)
        else:
            raise ValueError(f"Unknown target mode: {mode}")
    if not np.isfinite(result).all():
        raise ValueError("Estimator produced non-finite sales predictions.")
    return np.maximum(result, 0)


class SalesRegressor(RegressorMixin, BaseEstimator):
    """Clone a raw-input pipeline, transform y, and predict original-scale sales."""

    def __init__(self, pipeline, target_mode="raw"):
        self.pipeline = pipeline
        self.target_mode = target_mode

    def fit(self, X, y):
        self.pipeline_ = clone(self.pipeline).fit(X, transform_target(y, self.target_mode))
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X):
        check_is_fitted(self, "pipeline_")
        return inverse_target(self.pipeline_.predict(X), self.target_mode)


def build_model(name):
    if name == "MeanBaseline":
        return SalesRegressor(Pipeline([("model", DummyRegressor(strategy="mean"))]))
    models = {
        "LinearRegression": LinearRegression(),
        "KNN": KNeighborsRegressor(n_neighbors=31, weights="uniform", p=2, n_jobs=1),
        "SVR": SVR(kernel="rbf", C=1000, gamma="scale", epsilon=100),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=100,
            max_features=1.0, random_state=SEED, n_jobs=1,
        ),
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}")
    return SalesRegressor(Pipeline([
        ("preprocess", BigMartPreprocessor()),
        ("features", ModelFeatures()),
        ("scale", FeatureScaler(enabled=name != "RandomForest")),
        ("model", models[name]),
    ]))


def development_split(train):
    dev, check = train_test_split(
        np.arange(len(train)), test_size=.2, random_state=SEED,
        stratify=train["Outlet_Identifier"],
    )
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(dev))
    assignments = pd.DataFrame({"row_index": np.arange(len(train)), "partition": "reserved_check",
                                "cv_fold": pd.array([pd.NA] * len(train), dtype="Int64")})
    assignments.loc[dev, "partition"] = "development"
    for number, (_, valid_idx) in enumerate(folds, 1):
        assignments.loc[dev[valid_idx], "cv_fold"] = number
    return dev, check, folds, assignments


def search_space(name):
    feature_key = "pipeline__features__profile"
    if name == "LinearRegression":
        return {feature_key: PROFILES + ["price_interactions"], "target_mode": TARGET_MODES}
    parameters = {
        "KNN": {
            "n_neighbors": [5, 11, 21, 31, 51, 81],
            "weights": ["uniform", "distance"], "p": [1, 2],
        },
        "SVR": {
            "C": [10, 100, 1000, 10000], "gamma": ["scale", .001, .01, .1],
            "epsilon": [.01, .1, 1, 10, 100],
        },
        "RandomForest": {
            "n_estimators": [200, 400, 600], "max_depth": [4, 6, 8, 10, None],
            "min_samples_leaf": [1, 5, 20, 50, 100], "max_features": [.5, .75, 1.0],
        },
    }[name]
    shared = {f"pipeline__model__{k}": v for k, v in parameters.items()}
    shared.update({feature_key: PROFILES, "target_mode": TARGET_MODES})
    if name == "RandomForest":
        shared["pipeline__features__include_ratio"] = [False, True]
    return [dict(shared, **{
        "pipeline__preprocess__visibility_policy": [policy],
        "pipeline__features__visibility": [representation],
    }) for policy, representation in VISIBILITY_OPTIONS]


def _candidate_record(name, stage, parameters, test_scores, train_scores, fit_times, score_times):
    model = build_model(name).set_params(**parameters)
    settings = model.pipeline.named_steps
    return {
        "model": name, "stage": stage, "parameters": _jsonable(parameters),
        "profile": settings["features"].profile if "features" in settings else "none",
        "visibility": settings["features"].visibility if "features" in settings else "none",
        "visibility_policy": settings["preprocess"].visibility_policy if "preprocess" in settings else "none",
        "include_ratio": settings["features"].include_ratio if "features" in settings else False,
        "target_mode": model.target_mode,
        "cv_rmse": float(np.mean(test_scores)), "cv_std": float(np.std(test_scores, ddof=0)),
        "train_rmse": float(np.mean(train_scores)), "fit_seconds": float(np.mean(fit_times)),
        "predict_seconds": float(np.mean(score_times)),
        "fold_rmse": [float(v) for v in test_scores],
    }


def comparison_table(selection):
    rows = []
    for row in selection["finalists"]:
        rows.append({k: v for k, v in row.items() if k not in {"parameters", "fold_rmse"}})
    return pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)


def run_selection(train, output_dir="model_outputs", n_jobs=4, reuse_results=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    fingerprint = experiment_fingerprint(train)
    cache = output_dir / "selection_results.joblib"
    if reuse_results and cache.exists():
        result = joblib.load(cache)
        if result["fingerprint"] == fingerprint:
            print("Loaded matching development-CV results; no search repeated.", flush=True)
            return result

    def progress(message):
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with (output_dir / "progress.log").open("a") as stream:
            stream.write(line + "\n")

    dev, check, folds, assignments = development_split(train)
    assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    X_dev = train.iloc[dev][RAW_FEATURES]
    y_dev = train.iloc[dev][TARGET]
    baselines, tuned, search_results = [], [], {}
    progress(f"Development={len(dev)}, reserved check={len(check)}; five shared folds.")
    with computation_context():
        for name in ["MeanBaseline"] + MODEL_ORDER:
            progress(f"Baseline: {name}")
            scores = cross_validate(
                build_model(name), X_dev, y_dev, cv=folds, n_jobs=n_jobs,
                scoring="neg_root_mean_squared_error", return_train_score=True,
                error_score="raise",
            )
            row = _candidate_record(name, "baseline", {}, -scores["test_score"],
                                    -scores["train_score"], scores["fit_time"], scores["score_time"])
            baselines.append(row)
            progress(f"{name}: CV RMSE={row['cv_rmse']:.3f} ± {row['cv_std']:.3f}")
        for name in MODEL_ORDER:
            budget = 12 if name == "LinearRegression" else SEARCH_BUDGETS[name]
            progress(f"Search: {name}, {budget} candidates × five folds")
            kwargs = dict(estimator=build_model(name), scoring="neg_root_mean_squared_error",
                          cv=folds, n_jobs=n_jobs, refit=False,
                          return_train_score=True, error_score="raise")
            if name == "LinearRegression":
                search = GridSearchCV(param_grid=search_space(name), **kwargs)
            else:
                search = RandomizedSearchCV(param_distributions=search_space(name),
                                            n_iter=budget, random_state=SEED, **kwargs)
            search.fit(X_dev, y_dev)
            results = pd.DataFrame(search.cv_results_)
            results["params"] = results["params"].map(lambda p: json.dumps(_jsonable(p), sort_keys=True))
            results.to_csv(output_dir / f"search_{name}.csv", index=False)
            search_results[name] = results
            idx = search.best_index_
            raw = search.cv_results_
            row = _candidate_record(
                name, "tuned", search.best_params_,
                [-raw[f"split{i}_test_score"][idx] for i in range(5)],
                [-raw[f"split{i}_train_score"][idx] for i in range(5)],
                [raw["mean_fit_time"][idx]], [raw["mean_score_time"][idx]],
            )
            tuned.append(row)
            progress(f"{name}: best searched CV RMSE={row['cv_rmse']:.3f} ± {row['cv_std']:.3f}")

    def rank(row):
        return (row["cv_rmse"], row["predict_seconds"],
                MODEL_ORDER.index(row["model"]) if row["model"] in MODEL_ORDER else -1)

    finalists = [baselines[0]]
    for name in MODEL_ORDER:
        finalists.append(min([r for r in baselines + tuned if r["model"] == name], key=rank))
    winner = min(finalists[1:], key=rank)
    if winner["cv_rmse"] >= baselines[0]["cv_rmse"]:
        raise RuntimeError("No approved model improved on the development mean baseline.")
    result = {"fingerprint": fingerprint, "seed": SEED, "dev_indices": dev,
              "check_indices": check, "folds": folds, "baselines": baselines,
              "tuned": tuned, "finalists": finalists, "winner": winner,
              "search_results": search_results, "versions": package_versions()}
    joblib.dump(result, cache)
    write_json(output_dir / "frozen_selection.json", {
        "fingerprint": fingerprint, "winner": winner,
        "selection_basis": "Development CV only; reserved-check metrics not yet computed.",
    })
    comparison_table(result).to_csv(output_dir / "model_comparison.csv", index=False)
    write_json(output_dir / "baseline_results.json", baselines)
    progress(f"FROZEN WINNER: {winner['model']} ({winner['stage']}), CV RMSE={winner['cv_rmse']:.3f}")
    return result


def selected_estimator(selection):
    winner = selection["winner"]
    return build_model(winner["model"]).set_params(**winner["parameters"])


def regression_metrics(actual, predicted):
    return {"rmse": float(root_mean_squared_error(actual, predicted)),
            "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted))}


def prediction_frame(raw, predicted, partition, tail_cutoff):
    out = raw[ID_COLUMNS + ["Outlet_Type", "Item_MRP", TARGET]].copy()
    out.insert(0, "row_index", raw.index)
    out["partition"] = partition
    out["prediction"] = predicted
    out["residual"] = out[TARGET] - predicted
    out["absolute_error"] = out["residual"].abs()
    out["MRP_Band"] = pd.cut(out.Item_MRP, [0, 69, 136, 203, np.inf],
                             labels=["Low", "Medium", "High", "Very High"])
    out["Sales_segment"] = np.where(out[TARGET].gt(tail_cutoff), "Above development p90", "At/below development p90")
    return out.sort_index()


def group_errors(predictions, column):
    rows = []
    for level, group in predictions.groupby(column, observed=True):
        rows.append({column: str(level), "rows": len(group),
                     **regression_metrics(group[TARGET], group.prediction),
                     "mean_residual": float(group.residual.mean())})
    return pd.DataFrame(rows)


def evaluate_selected(train, selection, output_dir="model_outputs", n_jobs=4, reuse_results=True):
    output_dir = Path(output_dir)
    if experiment_fingerprint(train) != selection["fingerprint"]:
        raise ValueError("Selection cache does not match the training data/code.")
    frozen = json.loads((output_dir / "frozen_selection.json").read_text())
    if frozen["fingerprint"] != selection["fingerprint"] or frozen["winner"] != _jsonable(selection["winner"]):
        raise ValueError("Frozen selection is missing or inconsistent.")
    cache = output_dir / "evaluation_results.joblib"
    if reuse_results and cache.exists():
        result = joblib.load(cache)
        if result["fingerprint"] == selection["fingerprint"]:
            print("Loaded the frozen winner's saved OOF/check evaluation.", flush=True)
            return result
    dev, check = selection["dev_indices"], selection["check_indices"]
    X_dev, y_dev = train.iloc[dev][RAW_FEATURES], train.iloc[dev][TARGET]
    X_check, y_check = train.iloc[check][RAW_FEATURES], train.iloc[check][TARGET]
    print("Winner already frozen. Generating development OOF predictions...", flush=True)
    with computation_context():
        oof = cross_val_predict(selected_estimator(selection), X_dev, y_dev,
                                cv=selection["folds"], n_jobs=n_jobs, method="predict")
        estimator = selected_estimator(selection).fit(X_dev, y_dev)
        check_predictions = estimator.predict(X_check)
        baseline_predictions = np.full(len(y_check), y_dev.mean())
        metrics = {
            "development_oof": regression_metrics(y_dev, oof),
            "reserved_check": regression_metrics(y_check, check_predictions),
            "reserved_mean_baseline": regression_metrics(y_check, baseline_predictions),
        }
        print(f"Reserved-check RMSE: {metrics['reserved_check']['rmse']:.3f}", flush=True)
        print("Permutation importance: frozen model, reserved check, ten repeats.", flush=True)
        importance = permutation_importance(
            estimator, X_check, y_check, scoring="neg_root_mean_squared_error",
            n_repeats=10, random_state=SEED, n_jobs=n_jobs,
        )
    tail_cutoff = y_dev.quantile(.9)
    oof_frame = prediction_frame(train.iloc[dev], oof, "development_oof", tail_cutoff)
    check_frame = prediction_frame(train.iloc[check], check_predictions, "reserved_check", tail_cutoff)
    importance_frame = pd.DataFrame({"feature": RAW_FEATURES,
        "rmse_increase_mean": importance.importances_mean,
        "rmse_increase_std": importance.importances_std}).sort_values("rmse_increase_mean", ascending=False)
    oof_frame.to_csv(output_dir / "development_oof_predictions.csv", index=False)
    check_frame.to_csv(output_dir / "reserved_check_predictions.csv", index=False)
    importance_frame.to_csv(output_dir / "permutation_importance.csv", index=False)
    for partition, frame in [("development_oof", oof_frame), ("reserved_check", check_frame)]:
        for col in ["Outlet_Identifier", "Outlet_Type", "MRP_Band", "Sales_segment"]:
            group_errors(frame, col).to_csv(output_dir / f"errors_{partition}_{col}.csv", index=False)
    write_json(output_dir / "evaluation_metrics.json", metrics)
    joblib.dump(estimator, output_dir / "development_model.joblib")
    result = {"fingerprint": selection["fingerprint"], "metrics": metrics,
              "oof_predictions": oof_frame, "check_predictions": check_frame,
              "importance": importance_frame, "development_tail_cutoff": float(tail_cutoff)}
    joblib.dump(result, cache)
    return result


def validate_submission(submission, test, sample):
    if list(submission.columns) != ID_COLUMNS + [TARGET]:
        raise ValueError("Submission columns do not match the required order.")
    if len(submission) != len(test) or len(sample) != len(test):
        raise ValueError("Submission/test/template row counts differ.")
    pd.testing.assert_frame_equal(submission[ID_COLUMNS].reset_index(drop=True),
                                  test[ID_COLUMNS].reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(sample[ID_COLUMNS].reset_index(drop=True),
                                  test[ID_COLUMNS].reset_index(drop=True), check_dtype=False)
    values = submission[TARGET].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Submission sales must be finite and nonnegative.")


def build_final(train, test, sample, selection, evaluation, output_dir="model_outputs",
                submission_path="submission.csv"):
    output_dir = Path(output_dir)
    if experiment_fingerprint(train) != selection["fingerprint"] or evaluation["fingerprint"] != selection["fingerprint"]:
        raise ValueError("Selection/evaluation fingerprints do not match the current run.")
    print("Refitting the frozen configuration on all training rows...", flush=True)
    with computation_context():
        estimator = selected_estimator(selection).fit(train[RAW_FEATURES], train[TARGET])
        predictions = estimator.predict(test[RAW_FEATURES])
        model_path = output_dir / "final_model.joblib"
        joblib.dump(estimator, model_path)
        reloaded = joblib.load(model_path)
        np.testing.assert_allclose(reloaded.predict(test[RAW_FEATURES]), predictions, rtol=1e-12, atol=1e-12)
    submission = test[ID_COLUMNS].copy()
    submission[TARGET] = predictions
    validate_submission(submission, test, sample)
    submission.to_csv(submission_path, index=False)
    validate_submission(pd.read_csv(submission_path), test, sample)
    selected_names = estimator.pipeline_.named_steps["features"].get_feature_names_out().tolist()
    model_settings = estimator.pipeline_.named_steps["model"].get_params(deep=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": selection["fingerprint"], "versions": package_versions(),
        "train_frame_sha256": frame_hash(train), "test_frame_sha256": frame_hash(test),
        "raw_file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path("train_v9rqX0R.csv"), Path("test_AbJTz2l.csv"),
        ]},
        "winner": selection["winner"], "estimator_parameters": model_settings,
        "selected_features": selected_names, "selected_feature_count": len(selected_names),
        "metrics": evaluation["metrics"], "development_rows": len(selection["dev_indices"]),
        "reserved_check_rows": len(selection["check_indices"]), "final_fit_rows": len(train),
        "test_rows": len(test), "seed": SEED, "submission": str(submission_path),
        "selection_policy": "Frozen from development CV; no retuning on reserved-check results.",
        "limitations": ["Full-dataset EDA preceded the modelling split.",
                         "Only ten existing outlets; not an assessment of unseen-store generalisation.",
                         "Local RMSE is not a public leaderboard score."],
    }
    write_json(output_dir / "run_metadata.json", metadata)
    print(f"Saved {model_path}; validated {len(submission)} submission rows.", flush=True)
    return metadata


def main():
    train = pd.read_csv("train_v9rqX0R.csv")
    test = pd.read_csv("test_AbJTz2l.csv")
    sample = pd.read_csv("sample_submission_8RXa3c6 (1).csv")
    selection = run_selection(train)
    print(comparison_table(selection).to_string(index=False), flush=True)
    evaluation = evaluate_selected(train, selection)
    build_final(train, test, sample, selection, evaluation)


if __name__ == "__main__":
    # Keep estimator classes importable when loading artifacts in another process.
    import modeling
    modeling.main()
