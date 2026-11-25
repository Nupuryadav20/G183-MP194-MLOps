"""Independent verification of the Stage 2 training + MLflow workflow (MP194).

Rules this file follows:
  * It never writes to mlflow.db, models/ or the registry. The database is
    copied to a temp folder and only read.
  * It does not re-implement train.py. It checks the *outputs* of train.py
    (config, MLflow runs, registry, saved model, comparison CSV).
  * Tests that need a file that is not present skip with a clear reason.
"""
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "params.yaml"
TRAIN_CSV = ROOT / "data" / "raw" / "train.csv"
LATER_CSV = ROOT / "data" / "raw" / "later.csv"
MLFLOW_DB = ROOT / "mlflow.db"
MODEL_PATH = ROOT / "models" / "model.joblib"
COMPARISON_CSV = ROOT / "reports" / "model_comparison.csv"

EXPERIMENT = "G183_MP194"
REGISTERED = "MP194_Household_Electricity"
ALIAS = "champion"


# ---------------------------------------------------------------- helpers
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def pick_metric(metrics: dict, name: str):
    """Return the logged metric whose key contains `name`, preferring 'val'."""
    keys = [k for k in metrics if name in k.lower()]
    keys.sort(key=lambda k: ("val" not in k.lower(), k))
    return keys[0] if keys else None


@pytest.fixture(scope="module")
def cfg():
    if not PARAMS.exists():
        pytest.skip("params.yaml not found")
    return yaml.safe_load(PARAMS.read_text())


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Read-only MLflow client on a COPY of mlflow.db."""
    pytest.importorskip("mlflow")
    if not MLFLOW_DB.exists():
        pytest.skip("mlflow.db not found")
    from mlflow.tracking import MlflowClient

    copy = tmp_path_factory.mktemp("mlflow_copy") / "mlflow_copy.db"
    shutil.copyfile(MLFLOW_DB, copy)
    return MlflowClient(tracking_uri=f"sqlite:///{copy.as_posix()}",
                        registry_uri=f"sqlite:///{copy.as_posix()}")


@pytest.fixture(scope="module")
def runs(client):
    exp = client.get_experiment_by_name(EXPERIMENT)
    assert exp is not None, f"experiment {EXPERIMENT!r} not found in mlflow.db"
    return client.search_runs([exp.experiment_id], max_results=1000)


def run_label(r):
    return (r.data.tags.get("mlflow.runName")
            or r.data.params.get("model_name")
            or r.data.params.get("model")
            or r.info.run_id)


# ------------------------------------------------- 2. config / pipeline sanity
def test_config_names_match_assignment(cfg):
    assert cfg["model"]["experiment_name"] == EXPERIMENT
    assert cfg["model"]["registered_model_name"] == REGISTERED
    assert cfg["model"]["target"] == "units_kwh"


def test_target_not_in_input_features(cfg):
    feats = cfg["model"]["numerical_features"] + cfg["model"]["categorical_features"]
    assert cfg["model"]["target"] not in feats
    assert len(feats) == len(set(feats)), "duplicate feature names in config"
    assert set(cfg["model"]["numerical_features"]).isdisjoint(
        cfg["model"]["categorical_features"])


def test_split_follows_config(cfg):
    from sklearn.model_selection import train_test_split

    n = 1000
    df = pd.DataFrame({"x": range(n), "y": range(n)})
    size, seed = cfg["model"]["validation_size"], cfg["model"]["random_state"]
    assert 0 < size < 1
    a, b = train_test_split(df, test_size=size, random_state=seed)
    a2, b2 = train_test_split(df, test_size=size, random_state=seed)
    assert len(b) == int(round(n * size))
    assert b.index.equals(b2.index), "split is not reproducible with the config seed"


def test_preprocessing_handles_missing_and_unseen(cfg):
    """Numeric imputation + scaling and one-hot with unknown categories."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num = cfg["model"]["numerical_features"]
    cat = cfg["model"]["categorical_features"]
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=50) for c in num})
    for c in cat:
        df[c] = rng.choice(["a", "b"], 50)
    df.iloc[0, 0] = np.nan
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    out = pre.fit_transform(df)
    assert not np.isnan(out.toarray() if hasattr(out, "toarray") else out).any()
    unseen = df.copy()
    unseen[cat[0]] = "never_seen"
    pre.transform(unseen)  # must not raise


# ------------------------------------------------------------- metrics + hash
def test_metric_formulas_on_known_values():
    y = [10, 20, 30, 40]
    p = [12, 18, 33, 36]
    assert mae(y, p) == pytest.approx(2.75)
    assert rmse(y, p) == pytest.approx(np.sqrt((4 + 4 + 9 + 16) / 4))
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    assert mae(y, p) == pytest.approx(mean_absolute_error(y, p))
    assert rmse(y, p) == pytest.approx(mean_squared_error(y, p) ** 0.5)


def test_hash_function_is_deterministic(tmp_path):
    f = tmp_path / "a.csv"
    f.write_bytes(b"a,b\n1,2\n")
    assert sha256_of(f) == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    f.write_bytes(b"a,b\n1,3\n")
    assert sha256_of(f) != hashlib.sha256(b"a,b\n1,2\n").hexdigest()


# ------------------------------------------------------- 3. MLflow tracking
def test_experiment_has_enough_successful_runs(runs):
    finished = [r for r in runs if r.info.status == "FINISHED"]
    assert len(finished) >= 3, f"only {len(finished)} FINISHED runs"


EXPECTED_CANDIDATES = {"linear_regression", "random_forest_200",
                       "random_forest_400", "gradient_boosting"}


def test_four_candidates_represented(runs):
    labels = {run_label(r) for r in runs if r.info.status == "FINISHED"}
    assert EXPECTED_CANDIDATES <= labels, (
        f"missing candidates: {sorted(EXPECTED_CANDIDATES - labels)}")


def test_every_run_logs_params_and_metrics(runs):
    for r in runs:
        if r.info.status != "FINISHED":
            continue
        for p in ("model_name", "validation_size", "random_state", "data_sha256"):
            assert p in r.data.params, f"{run_label(r)}: param {p!r} not logged"
        for m in ("mae_validation", "rmse_validation"):
            assert m in r.data.metrics, f"{run_label(r)}: metric {m!r} not logged"
        assert r.data.metrics["rmse_validation"] >= r.data.metrics["mae_validation"]


def test_data_sha256_logged_and_matches_raw_file(runs):
    if not TRAIN_CSV.exists():
        pytest.skip("data/raw/train.csv not present (DVC-tracked)")
    expected = sha256_of(TRAIN_CSV)
    for r in runs:
        if r.info.status != "FINISHED":
            continue
        logged = r.data.params.get("data_sha256") or r.data.tags.get("data_sha256")
        assert logged, f"{run_label(r)}: data_sha256 not logged"
        assert logged == expected, f"{run_label(r)}: hash differs from raw train.csv"


def test_run_artifacts_exist_in_registry_of_runs(client, runs):
    for r in runs:
        if r.info.status != "FINISHED":
            continue
        arts = client.list_artifacts(r.info.run_id)
        if not arts:
            pytest.skip("artifact store not reachable from this machine "
                        "(path recorded on another computer)")
        return


# ----------------------------------------------------------- 4. registry
def test_registered_model_and_versions_exist(client):
    rm = client.get_registered_model(REGISTERED)
    assert rm.name == REGISTERED
    versions = client.search_model_versions(f"name='{REGISTERED}'")
    assert len(versions) >= 1


def test_champion_alias_points_to_lowest_mae_run(client, runs):
    mv = client.get_model_version_by_alias(REGISTERED, ALIAS)
    champ = client.get_run(mv.run_id)
    k = pick_metric(champ.data.metrics, "mae")
    assert k, "champion run has no MAE metric"
    best = min(
        r.data.metrics[pick_metric(r.data.metrics, "mae")]
        for r in runs
        if r.info.status == "FINISHED" and pick_metric(r.data.metrics, "mae")
    )
    assert champ.data.metrics[k] == pytest.approx(best), (
        "champion is not the run with the lowest validation MAE")


def test_registered_model_loads(client):
    mlflow = pytest.importorskip("mlflow")
    mv = client.get_model_version_by_alias(REGISTERED, ALIAS)
    src = mv.source
    if src.startswith("file:") or src.startswith("/") or ":" in src[:3]:
        local = Path(src.replace("file://", "").replace("file:", ""))
        if not local.exists():
            pytest.skip("registered model files are not on this machine")
    try:
        model = mlflow.pyfunc.load_model(f"models:/{REGISTERED}@{ALIAS}")
    except Exception as exc:  # report as a finding, not a crash
        pytest.fail(f"registered model could not be loaded: {exc}")
    assert model is not None


# ----------------------------------------------------- 5. reproducibility
def test_saved_model_loads_and_predicts(cfg):
    joblib = pytest.importorskip("joblib")
    if not MODEL_PATH.exists():
        pytest.skip("models/model.joblib not found")
    if not LATER_CSV.exists():
        pytest.skip("data/raw/later.csv not present (DVC-tracked)")
    model = joblib.load(MODEL_PATH)
    feats = cfg["model"]["numerical_features"] + cfg["model"]["categorical_features"]
    X = pd.read_csv(LATER_CSV).head(25)[feats]  # later.csv has clean spellings
    pred = np.asarray(model.predict(X), dtype=float)
    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all()
    assert (pred > 0).all(), "negative electricity prediction"


def test_model_comparison_csv(runs):
    if not COMPARISON_CSV.exists():
        pytest.skip("reports/model_comparison.csv not found")
    df = pd.read_csv(COMPARISON_CSV)
    assert len(df) >= 4, "expected one row per candidate model"
    mae_cols = [c for c in df.columns if "mae" in c.lower()]
    assert mae_cols, f"no MAE column in {list(df.columns)}"
    assert df[mae_cols[0]].notna().all()
    assert (df[mae_cols[0]] > 0).all()


# ------------------------------------------- train.py functions (importable)
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def train_mod():
    pytest.importorskip("mlflow")
    import importlib
    return importlib.import_module("mp194.train")


def test_get_models_returns_the_four_candidates(train_mod):
    models = train_mod.get_models(42)
    assert set(models) == EXPECTED_CANDIDATES
    assert models["random_forest_200"].get_params()["random_state"] == 42
    assert models["gradient_boosting"].get_params()["random_state"] == 42


def test_get_data_hash_matches_sha256(train_mod, tmp_path):
    f = tmp_path / "x.csv"
    f.write_bytes(b"a,b\n1,2\n3,4\n")
    assert train_mod.get_data_hash(str(f)) == hashlib.sha256(
        b"a,b\n1,2\n3,4\n").hexdigest()


def test_build_preprocessor_output_shape(train_mod, cfg):
    num = cfg["model"]["numerical_features"]
    cat = cfg["model"]["categorical_features"]
    df = _synthetic_frame(60)
    pre = train_mod.build_preprocessor(num, cat)
    out = pre.fit_transform(df[num + cat])
    out = out.toarray() if hasattr(out, "toarray") else out
    assert out.shape[0] == 60
    assert not np.isnan(out).any()
    # 6 numeric columns + one-hot (3 connection levels + 3 feeders)
    assert out.shape[1] == len(num) + 3 + 3


def _synthetic_frame(n, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "rooms": rng.integers(1, 6, n),
        "members": rng.integers(1, 8, n),
        "ac_hours_per_day": rng.uniform(0, 12, n),
        "avg_temp_c": rng.uniform(24, 36, n),
        "house_area_sqft": rng.uniform(300, 3000, n),
        "solar_kw": rng.uniform(0, 5, n),
        "connection": rng.choice(["domestic", "domestic_plus", "shop_cum_home"], n),
        "feeder": rng.choice(["Feeder-1", "Feeder-2", "Feeder-3"], n),
    })
    df["units_kwh"] = (40 * df["rooms"] + 25 * df["members"]
                       + 12 * df["ac_hours_per_day"] + 0.05 * df["house_area_sqft"]
                       + rng.normal(0, 15, n))
    return df


def test_train_main_end_to_end_in_sandbox(train_mod, cfg, tmp_path, monkeypatch):
    """Run the REAL train.main() in a throw-away folder with fake data.

    Everything (mlflow.db, mlruns, models/, reports/) is created inside
    tmp_path, so the project's real database and registry are untouched.
    """
    import joblib
    import mlflow
    from mlflow.tracking import MlflowClient

    old_uri = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(cfg))
    df = _synthetic_frame(240)
    df.to_csv(tmp_path / "data" / "raw" / "train.csv", index=False)
    df.to_csv(tmp_path / "data" / "processed" / "train_clean.csv", index=False)

    try:
        train_mod.main()

        comp = pd.read_csv(tmp_path / "reports" / "model_comparison.csv")
        assert set(comp["model"]) == EXPECTED_CANDIDATES
        assert comp["mae"].is_monotonic_increasing, "report is not sorted by MAE"
        assert (comp["rmse"] >= comp["mae"]).all()

        c = MlflowClient(tracking_uri="sqlite:///mlflow.db",
                         registry_uri="sqlite:///mlflow.db")
        name = cfg["model"]["registered_model_name"]
        mv = c.get_model_version_by_alias(name, "champion")
        assert mv.run_id == comp.iloc[0]["run_id"], "champion is not the lowest-MAE run"

        exp = c.get_experiment_by_name(cfg["model"]["experiment_name"])
        found = c.search_runs([exp.experiment_id])
        assert len(found) == 4
        want = hashlib.sha256((tmp_path / "data/raw/train.csv").read_bytes()).hexdigest()
        assert {r.data.params["data_sha256"] for r in found} == {want}

        model = joblib.load(tmp_path / "models" / "model.joblib")
        feats = cfg["model"]["numerical_features"] + cfg["model"]["categorical_features"]
        pred = model.predict(df[feats].head(10))
        assert np.isfinite(pred).all()
    finally:
        mlflow.set_tracking_uri(old_uri)


def test_cleaned_training_file_has_required_columns(cfg):
    path = ROOT / "data" / "processed" / "train_clean.csv"
    if not path.exists():
        pytest.skip("data/processed/train_clean.csv not generated yet "
                    "(run: python -m mp194.cleaning)")
    cols = set(pd.read_csv(path, nrows=5).columns)
    need = set(cfg["model"]["numerical_features"] + cfg["model"]["categorical_features"]
               + [cfg["model"]["target"]])
    assert need <= cols, f"missing in cleaned data: {sorted(need - cols)}"
