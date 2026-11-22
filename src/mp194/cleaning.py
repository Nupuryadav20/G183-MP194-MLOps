
import pandas as pd
import numpy as np
import yaml
from pathlib import Path


def load_config():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def normalize_connection(series, aliases):
    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )

    result = normalized.map(aliases)

    if result.isna().any():
        unknown = normalized[result.isna()].unique().tolist()
        raise ValueError(
            f"Unknown connection categories: {unknown}"
        )

    return result


def clean_train_data(df, config):
    df = df.copy()

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates()

    # 2. Convert dates
    df["date"] = pd.to_datetime(df["date"])

    area = config["cleaning"]

    # 3. Correct Feeder-3 area values (square metres to sqft)
    area_mask = df["feeder"].eq(area["affected_feeder"])

    df.loc[area_mask, area["area_column"]] = (
        df.loc[area_mask, area["area_column"]]
        / area["area_conversion_factor"]
    )

    # 4. Replace sentinel values
    sentinel_column = area["sentinel_column"]

    df[sentinel_column] = df[sentinel_column].replace(
        area["sentinel_value"],
        np.nan
    )

    # 5. Detect stuck members sensor
    sensor = area["stuck_sensor"]

    stuck_mask = (
        df["date"].between(
            sensor["start_date"],
            sensor["end_date"]
        )
        & df[sensor["column"]].eq(
            sensor["expected_value"]
        )
    )

    # Mark affected readings as missing
    df.loc[stuck_mask, sensor["column"]] = np.nan

    # 6. Impute stuck sensor readings using unaffected records
    valid_members = (
        df.loc[~stuck_mask]
        .groupby("rooms")["members"]
        .median()
    )

    # Room-wise median imputation
    df.loc[stuck_mask, "members"] = (
        df.loc[stuck_mask, "rooms"].map(valid_members)
    )

    # Global fallback, calculated from unaffected records
    global_median = df.loc[~stuck_mask, "members"].median()

    df["members"] = df["members"].fillna(global_median)

    # Members represents a whole-number count
    df["members"] = df["members"].round().astype(int)

    # 7. Impute remaining numerical missing values
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

    # 8. Normalize connection categories
    aliases = area["category_aliases"]

    df["connection"] = normalize_connection(
        df["connection"],
        aliases
    )

    return df


def clean_later_data(df, config):
    df = df.copy()

    # Later data is already clean
    df["date"] = pd.to_datetime(df["date"])

    # Normalize category formatting
    aliases = config["cleaning"]["category_aliases"]

    df["connection"] = normalize_connection(
        df["connection"],
        aliases
    )

    return df


def main():
    config = load_config()

    # Load original datasets
    train = pd.read_csv(config["data"]["train_path"])
    later = pd.read_csv(config["data"]["later_path"])

    # Clean datasets
    train_clean = clean_train_data(train, config)
    later_clean = clean_later_data(later, config)

    # Create output directory
    output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save cleaned datasets
    train_clean.to_csv(
        output_dir / "train_clean.csv",
        index=False
    )

    later_clean.to_csv(
        output_dir / "later_clean.csv",
        index=False
    )

    # Display results
    print("\n========== CLEANING COMPLETED ==========")

    print("Original train shape:", train.shape)
    print("Cleaned train shape:", train_clean.shape)

    print("Original later shape:", later.shape)
    print("Cleaned later shape:", later_clean.shape)

    print("\nRemaining missing values:")
    print(train_clean.isnull().sum())

    print("\nConnection categories:")
    print(train_clean["connection"].value_counts())

    print("\nFeeder-wise area median:")
    print(
        train_clean.groupby("feeder")[
            "house_area_sqft"
        ].median()
    )

    print("\nCleaned members distribution:")
    print(train_clean["members"].value_counts().sort_index())


if __name__ == "__main__":
    main()
