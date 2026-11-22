
import pandas as pd
import numpy as np
import pytest

from mp194.cleaning import (
    load_config,
    clean_train_data,
    clean_later_data
)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def raw_train(config):
    return pd.read_csv(config["data"]["train_path"])


@pytest.fixture
def raw_later(config):
    return pd.read_csv(config["data"]["later_path"])


@pytest.fixture
def cleaned_train(raw_train, config):
    return clean_train_data(raw_train, config)


@pytest.fixture
def cleaned_later(raw_later, config):
    return clean_later_data(raw_later, config)


def test_train_rows_preserved(raw_train, cleaned_train):
    assert len(cleaned_train) == len(raw_train)


def test_sentinel_values_removed(cleaned_train):
    assert not (cleaned_train["solar_kw"] == -999).any()


def test_feeder3_area_conversion(
    raw_train, cleaned_train
):
    original = raw_train[
        raw_train["feeder"] == "Feeder-3"
    ]

    cleaned = cleaned_train[
        cleaned_train["feeder"] == "Feeder-3"
    ]

    merged = original[
        ["record_id", "house_area_sqft"]
    ].merge(
        cleaned[["record_id", "house_area_sqft"]],
        on="record_id",
        suffixes=("_raw", "_clean")
    )

    expected = (
        merged["house_area_sqft_raw"] / 0.092903
    )

    assert np.allclose(
        merged["house_area_sqft_clean"],
        expected
    )


def test_connection_categories(cleaned_train):
    expected = {
        "domestic",
        "domestic_plus",
        "shop_cum_home"
    }

    assert set(cleaned_train["connection"].unique()) == expected


def test_stuck_sensor_corrected(
    raw_train, cleaned_train
):
    mask = (
        raw_train["date"].between(
            "2026-05-05", "2026-05-21"
        )
        & (raw_train["members"] == 5)
    )

    original = raw_train.loc[
        mask, ["record_id", "members"]
    ]

    cleaned = cleaned_train[
        ["record_id", "members"]
    ]

    merged = original.merge(
        cleaned,
        on="record_id",
        suffixes=("_raw", "_clean")
    )

    changed = (
        merged["members_raw"] != merged["members_clean"]
    ).sum()

    assert changed > 0


def test_no_missing_values(cleaned_train):
    assert cleaned_train.isnull().sum().sum() == 0


def test_target_preserved(cleaned_train):
    assert "units_kwh" in cleaned_train.columns
    assert cleaned_train["units_kwh"].notna().all()


def test_later_categories(cleaned_later):
    expected = {
        "domestic",
        "domestic_plus",
        "shop_cum_home"
    }

    assert set(cleaned_later["connection"].unique()) == expected
