
import hashlib
import yaml
import joblib
import mlflow
import mlflow.sklearn

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error

from mlflow import MlflowClient


def load_config():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def get_data_hash(path):
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


def build_preprocessor(numerical_features, categorical_features):

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor


def get_models(random_state):

    return {
        "linear_regression": LinearRegression(),

        "random_forest_200": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        ),

        "random_forest_400": RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=2,
            random_state=random_state
        )
    }


def main():

    config = load_config()
    model_config = config["model"]

    # 1. Load cleaned training data
    data = pd.read_csv("data/processed/train_clean.csv")

    numerical_features = model_config["numerical_features"]
    categorical_features = model_config["categorical_features"]
    target = model_config["target"]

    feature_columns = numerical_features + categorical_features

    X = data[feature_columns]
    y = data[target]

    # 2. Split training and validation data
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=model_config["validation_size"],
        random_state=model_config["random_state"]
    )

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_val))

    # 3. Original dataset SHA-256
    data_hash = get_data_hash("data/raw/train.csv")

    print("Original train.csv SHA256:", data_hash)

    # 4. Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    experiment_name = model_config["experiment_name"]

    mlflow.set_experiment(experiment_name)

    # 5. Define preprocessing and candidate models
    models = get_models(model_config["random_state"])

    results = []

    best_mae = float("inf")
    best_model = None
    best_run_id = None

    for model_name, estimator in models.items():

        print(f"\nTraining: {model_name}")

        preprocessor = build_preprocessor(
            numerical_features,
            categorical_features
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        with mlflow.start_run(run_name=model_name) as run:

            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("validation_size", model_config["validation_size"])
            mlflow.log_param("random_state", model_config["random_state"])

            # Required by assignment
            mlflow.log_param("data_sha256", data_hash)

            # Train
            pipeline.fit(X_train, y_train)

            # Validate
            predictions = pipeline.predict(X_val)

            mae = mean_absolute_error(y_val, predictions)
            rmse = mean_squared_error(y_val, predictions) ** 0.5

            # Log metrics
            mlflow.log_metric("mae_validation", mae)
            mlflow.log_metric("rmse_validation", rmse)

            # Log model artifact
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle"
            )

            run_id = run.info.run_id

            results.append({
                "model": model_name,
                "mae": mae,
                "rmse": rmse,
                "run_id": run_id
            })

            print("MAE:", round(mae, 4))
            print("RMSE:", round(rmse, 4))
            print("Run ID:", run_id)

            if mae < best_mae:
                best_mae = mae
                best_model = pipeline
                best_run_id = run_id

    # 6. Display model comparison
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mae")

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.to_string(index=False))

    # 7. Register best model
    model_name = model_config["registered_model_name"]

    model_uri = f"runs:/{best_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    # 8. Assign champion alias
    client = MlflowClient()

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=registered_model.version
    )

    # 9. Save local model for future API use
    Path("models").mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, "models/model.joblib")

    # 10. Save comparison report
    Path("reports").mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        "reports/model_comparison.csv",
        index=False
    )

    print("\n========== BEST MODEL ==========")
    print("Model:", results_df.iloc[0]["model"])
    print("Best MAE:", best_mae)
    print("Best run ID:", best_run_id)
    print("Registered model:", model_name)
    print("Registered version:", registered_model.version)
    print("Champion alias: champion")

    print("\nMODEL TRAINING COMPLETED")


if __name__ == "__main__":
    main()
