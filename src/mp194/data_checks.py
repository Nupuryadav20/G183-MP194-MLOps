
import pandas as pd
import numpy as np

TRAIN_PATH = "data/raw/train.csv"
LATER_PATH = "data/raw/later.csv"
HOLDOUT_PATH = "data/raw/holdout_features.csv"


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)
    return train, later, holdout


def check_leaking_column(train, holdout):
    print("\n1. LEAKING COLUMN")

    extra = set(train.columns) - set(holdout.columns)
    candidates = extra - {"units_kwh"}

    print("Train-only columns:", candidates)
    print("Target column:", "units_kwh")

    if candidates:
        for col in candidates:
            print("Candidate:", col)
            if pd.api.types.is_numeric_dtype(train[col]):
                print("Target correlation:",
                      train[col].corr(train["units_kwh"]))
    else:
        print("No separate train-only feature found.")


def check_unit_mixup(train):
    print("\n2. UNIT MIXUP")

    stats = train.groupby("feeder")["house_area_sqft"].agg(
        ["min", "mean", "median", "max"]
    )

    print(stats)

    # Feeder-3 records area in square metres.
    factor = 0.0929

    print("Affected group: Feeder-3")
    print("Column: house_area_sqft")
    print("Wrong value = correct value x", factor)


def check_sentinel(train):
    print("\n3. SENTINEL VALUE")

    counts = train["solar_kw"].value_counts()

    print("Solar sentinel count:")
    print(counts.get(-999, 0))

    print("Sentinel value: -999")


def check_duplicates(train):
    print("\n4. DUPLICATE ROWS")

    count = int(train.duplicated().sum())

    print("Extra duplicate rows:", count)


def check_stuck_sensor(train):
    print("\n5. STUCK SENSOR")

    df = train.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "record_id"]).reset_index(drop=True)

    column = "members"

    # Find consecutive runs of identical readings.
    groups = df[column].ne(df[column].shift()).cumsum()

    runs = (
        df.groupby(groups)
        .agg(
            value=(column, "first"),
            count=(column, "size"),
            start_date=("date", "min"),
            end_date=("date", "max")
        )
    )

    runs = runs[runs["count"] >= 5]
    runs = runs.sort_values("count", ascending=False)

    print(runs.to_string())


def check_target_typos(train):
    print("\n6. TARGET TYPOS")

    target = train["units_kwh"]

    # Look for unusually large values relative to the
    # ordinary target distribution.
    threshold = 3 * target.median()

    suspicious = train[target > threshold]

    print("Target minimum:", target.min())
    print("Target maximum:", target.max())
    print("Suspicious values above 3x median:",
          len(suspicious))

    print(suspicious[["record_id", "units_kwh"]].to_string(
        index=False
    ))


def check_spelling_variants(train, later):
    print("\n7. SPELLING VARIANTS")

    column = "connection"

    print("Raw training values:")
    print(train[column].value_counts().to_string())

    print("\nCanonical values from later.csv:")
    print(later[column].value_counts().to_string())

    print("\nNumber of true categories:",
          later[column].nunique())


def inspect_data():
    train, later, holdout = load_data()

    print("DATASET SHAPES")
    print("Train:", train.shape)
    print("Later:", later.shape)
    print("Holdout:", holdout.shape)

    check_leaking_column(train, holdout)
    check_unit_mixup(train)
    check_sentinel(train)
    check_duplicates(train)
    check_stuck_sensor(train)
    check_target_typos(train)
    check_spelling_variants(train, later)


if __name__ == "__main__":
    inspect_data()
