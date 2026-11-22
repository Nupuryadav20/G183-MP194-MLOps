
import hashlib

import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from mp194.train import (
    load_config,
    get_data_hash,
    build_preprocessor,
    get_models
)


def test_config_loads_correctly():
    config = load_config()

    assert config["model"]["target"] == "units_kwh"
    assert config["model"]["experiment_name"] == "G183_MP194"


def test_feature_columns_are_valid():
    config = load_config()

    numerical = config["model"]["numerical_features"]
    categorical = config["model"]["categorical_features"]

    assert "feeder" in categorical
    assert "connection" in categorical
    assert "feeder%" not in categorical
    assert "units_kwh" not in numerical + categorical


def test_data_hash_is_sha256(tmp_path):
    file = tmp_path / "sample.csv"
    file.write_text("rooms,units_kwh\n3,450\n")

    result = get_data_hash(file)

    expected = hashlib.sha256(
        file.read_bytes()
    ).hexdigest()

    assert result == expected
    assert len(result) == 64


def test_all_four_models_exist():
    models = get_models(random_state=42)

    expected = {
        "linear_regression",
        "random_forest_200",
        "random_forest_400",
        "gradient_boosting"
    }

    assert set(models.keys()) == expected


def test_preprocessor_handles_numeric_and_categorical():
    numerical = ["rooms", "solar_kw"]
    categorical = ["connection", "feeder"]

    X = pd.DataFrame({
        "rooms": [2, 3, 4, 5],
        "solar_kw": [1.2, 2.0, 3.0, 4.0],
        "connection": [
            "domestic",
            "domestic_plus",
            "domestic",
            "shop_cum_home"
        ],
        "feeder": [
            "Feeder-1",
            "Feeder-2",
            "Feeder-1",
            "Feeder-3"
        ]
    })

    preprocessor = build_preprocessor(
        numerical,
        categorical
    )

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > len(numerical)


def test_preprocessor_handles_unknown_category():
    numerical = ["rooms"]
    categorical = ["connection"]

    train = pd.DataFrame({
        "rooms": [2, 3, 4],
        "connection": [
            "domestic",
            "domestic_plus",
            "shop_cum_home"
        ]
    })

    test = pd.DataFrame({
        "rooms": [3],
        "connection": ["new_category"]
    })

    preprocessor = build_preprocessor(
        numerical,
        categorical
    )

    preprocessor.fit(train)
    transformed = preprocessor.transform(test)

    assert transformed.shape[0] == 1
    assert not np.isnan(transformed.toarray()).any() if hasattr(
        transformed, "toarray"
    ) else not np.isnan(transformed).any()


def test_training_pipeline_can_predict():
    X = pd.DataFrame({
        "rooms": [1, 2, 3, 4, 5, 6],
        "solar_kw": [0, 1, 2, 3, 4, 5],
        "connection": [
            "domestic",
            "domestic_plus",
            "domestic",
            "shop_cum_home",
            "domestic_plus",
            "domestic"
        ]
    })

    y = np.array([150, 250, 350, 450, 550, 650])

    preprocessor = build_preprocessor(
        ["rooms", "solar_kw"],
        ["connection"]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)
    assert np.isfinite(predictions).all()
