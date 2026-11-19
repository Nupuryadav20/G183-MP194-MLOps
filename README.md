# MP194 – Household Electricity Use Prediction

## Project Information

- **Course:** SCSE3040 – MLOps
- **Group ID:** G183
- **Problem ID:** MP194
- **Client:** Kochi Power Distribution Office
- **Problem Type:** Regression
- **Target Variable:** `units_kwh`
- **Evaluation Metric:** Mean Absolute Error (MAE)

## Group Members

| Enrolment Number | Name |
|---|---|
| S24CSEU2267 | Aman Choudhari |
| S24CSEU2288 | Parv Pandey |
| S24CSEU2258 | Nupur Yadav |
| S24CSEU2286 | Prateek Kumar |

## Project Description

This project develops an end-to-end MLOps pipeline for predicting monthly household electricity consumption in kWh.

The pipeline includes data validation, data versioning using DVC, model training, experiment tracking using MLflow, model registration, FastAPI model serving, Docker containerization, GitHub Actions CI, Kubernetes deployment configuration, structured logging, data drift monitoring and model retraining.

## Dataset

- `train.csv`: Historical training data.
- `later.csv`: Future data for drift monitoring and retraining.
- `holdout_features.csv`: Unlabelled data for final predictions.

## Technology Stack

- Python
- Pandas
- Scikit-learn
- DVC
- MLflow
- FastAPI
- Docker
- GitHub Actions
- Kubernetes
- Pytest

## Evaluation

The model predicts `units_kwh` and is evaluated using Mean Absolute Error (MAE).

## Repository Structure

The repository is organized into data, source code, tests, reports, API, model artifacts, Kubernetes configurations and CI workflows.

## Setup

Installation and execution instructions will be maintained here as the pipeline is implemented.