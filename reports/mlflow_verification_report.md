# MLflow and Training Verification Report (Stage 2)

**Project:** MP194, Household Electricity Use Prediction (Kochi), Group G183
**Verifier:** Parv Pandey (Testing owner)
**Branch:** `parv/stage2-mlflow-tests`
**Commit verified:** `<paste: git rev-parse --short HEAD>`
**Date:** `<fill>`

> Scope: verification only. `train.py`, `mlflow.db`, `models/` and the registry were not modified.
> `mlflow.db` was copied to a temporary file and read from there.

## 1. Files inspected
`PROBLEM.md`, `STUDENT_HANDBOOK.md`, `params.yaml`, `src/mp194/train.py`, existing tests in `tests/`, `mlflow.db`, `models/model.joblib`, `reports/model_comparison.csv`.

How `train.py` works (from reading the code):
- Data: reads `data/processed/train_clean.csv` (line 92), which `cleaning.py` creates. The raw `data/raw/train.csv` is only used for the SHA256 hash (line 115).
- Features and target: features are the configured numerical and categorical lists, `X = data[feature_columns]` and `y = data[target]`, so `units_kwh` is not an input.
- Split: `train_test_split` with `validation_size` 0.2 and `random_state` 42 from `params.yaml` (line 104).
- Pipeline: `Pipeline([preprocessor, model])`. Numeric columns get median imputation and scaling; categorical columns get most-frequent imputation and one-hot encoding with `handle_unknown="ignore"`.
- Candidates: `linear_regression`, `random_forest_200`, `random_forest_400`, `gradient_boosting` (`get_models`).
- Selection: each model is logged as an MLflow run, the lowest `mae_validation` wins (line 192). The best run is registered as `MP194_Household_Electricity` (line 209), given the `champion` alias (line 217), and saved to `models/model.joblib` (line 226) and `reports/model_comparison.csv`.

## 2. Training pipeline verification
| Check | Result | Evidence (file:line or test) |
|---|---|---|
| Cleaned training data is used | `<PASS/FAIL>` | `train.py:92`; `test_cleaned_training_file_has_required_columns` |
| `units_kwh` not in input features | `<PASS/FAIL>` | `test_target_not_in_input_features`, `test_train_main_end_to_end_in_sandbox` |
| Configured numerical and categorical features are used | `<PASS/FAIL>` | `<train.py:LINE>` |
| Split uses configured size and seed | `<PASS/FAIL>` | `test_split_follows_config`, `<train.py:LINE>` |
| Preprocessing is inside the model pipeline | `<PASS/FAIL>` | `<train.py:LINE>` |
| Numeric and categorical preprocessing works | `<PASS/FAIL>` | `test_preprocessing_handles_missing_and_unseen` |
| All four candidates train | `<PASS/FAIL>` | `test_get_models_returns_the_four_candidates`, `test_train_main_end_to_end_in_sandbox` |
| MAE and RMSE calculated correctly | `<PASS/FAIL>` | `test_metric_formulas_on_known_values` |
| Lowest validation MAE selected | `<PASS/FAIL>` | `test_champion_alias_points_to_lowest_mae_run` |
| Comparison CSV correct | `<PASS/FAIL>` | `test_model_comparison_csv` |

## 3. MLflow tracking verification
Command used (read-only):
```
<paste the search_runs output from the Python snippet>
```
| Check | Result | Evidence |
|---|---|---|
| Experiment `G183_MP194` exists | `<>` | |
| At least 3 FINISHED runs | `<>` (count: `<n>`) | |
| Four candidates represented | `<>` (names: `<list>`) | |
| Parameters logged | `<>` | |
| MAE and RMSE logged | `<>` | |
| `data_sha256` logged | `<>` | |
| Hash equals SHA256 of raw `train.csv` | `<>` | logged `<hash>` vs file `<hash>` |
| Artifacts available | `<>` | |

## 4. Model registry verification
| Check | Result | Evidence |
|---|---|---|
| Registered name `MP194_Household_Electricity` | `<>` | |
| Selected model registered, version exists | `<>` (versions: `<n>`) | |
| `champion` alias configured | `<>` | |
| Alias points to run with lowest MAE | `<>` | alias version `<v>`, run `<id>`, MAE `<x>` |
| Registered model loads | `<>` | |

## 5. Reproducibility
| Check | Result | Evidence |
|---|---|---|
| Training reproducible from the same config | `<>` | re-run in a throwaway folder; MAE `<a>` vs `<b>` |
| Data hash consistent with raw file | `<>` | |
| `models/model.joblib` loads | `<>` | |
| Predictions are finite and positive | `<>` | `test_saved_model_loads_and_predicts` |

## 6. Tests added
File: `tests/test_mlflow_verification.py` (no existing tests duplicated, no writes to the database).

| Test | Covers |
|---|---|
| `test_config_names_match_assignment` | experiment, model and target names |
| `test_target_not_in_input_features` | feature and target separation |
| `test_split_follows_config` | split size and seed |
| `test_preprocessing_handles_missing_and_unseen` | preprocessing |
| `test_metric_formulas_on_known_values` | MAE and RMSE |
| `test_get_models_returns_the_four_candidates` | `get_models` |
| `test_get_data_hash_matches_sha256` | `get_data_hash` from `train.py` |
| `test_build_preprocessor_output_shape` | `build_preprocessor` output |
| `test_train_main_end_to_end_in_sandbox` | runs the real `train.main()` on fake data in a temp folder: 4 runs, sorted comparison CSV, champion = lowest MAE, hash, model.joblib predicts |
| `test_cleaned_training_file_has_required_columns` | cleaned file has features and target |
| `test_hash_function_is_deterministic` | hash generation |
| `test_experiment_has_enough_successful_runs` | at least 3 runs |
| `test_four_candidates_represented` | four candidates |
| `test_every_run_logs_params_and_metrics` | params and metrics |
| `test_data_sha256_logged_and_matches_raw_file` | data hash |
| `test_run_artifacts_exist_in_registry_of_runs` | artifacts |
| `test_registered_model_and_versions_exist` | registry |
| `test_champion_alias_points_to_lowest_mae_run` | alias |
| `test_registered_model_loads` | registry load |
| `test_saved_model_loads_and_predicts` | saved model |
| `test_model_comparison_csv` | comparison CSV |

## 7. Test run output
```
<paste: python -m pytest -v>
```
Summary: `<passed / skipped / failed>`. Explain every skip (for example data files not present).

## 8. Bugs and observations
| # | File:line | Description | Severity |
|---|---|---|---|
| 1 | `train.py:120` | Tracking URI is `sqlite:///mlflow.db` with no artifact location, so model artifacts go to a local `mlruns/` folder. The GitHub root listing I saw had no `mlruns/`, so a teammate who clones may find the registered model and run artifacts missing. **Confirm:** `git ls-files mlruns \| head`, and `ls mlruns`. | Medium (if confirmed) |
| 2 | `train.py:149`, `:209`, `:217` | Every run of `train.py` adds 4 new runs and a new registered model version, and moves `champion`. Nothing prevents accidental re-runs from changing the registry. Not a bug in one run, but it matters for the "do not change the champion" rule. | Low |
| 3 | `train.py:173-175` | `log_model(name="model", ...)` needs MLflow 3 or newer. `requirements.txt` must pin a compatible version. **Confirm:** run `pip install -r requirements.txt` in a fresh environment. An earlier check suggested it fails on line 1; record the exact error here, or delete this row if it works. | Medium (if confirmed) |
| 4 | `train.py:92` | Training depends on `data/processed/train_clean.csv`, which is only produced by running `python -m mp194.cleaning` first. The README Setup section does not say so. Check whether the file is committed. | Low |
| 5 | `cleaning.py:96-102` | The median fill covers every numeric column, including the target `units_kwh`. A missing target would be replaced by a made-up value instead of the row being dropped. Training reads that target. Target typos (10x values) are also not corrected in `cleaning.py`; confirm against the data-checks findings. | Medium |
| 6 | `cleaning.py:62-69` | The stuck-sensor mask is "date in window and members == 5", so genuine households of 5 in that window are also overwritten. Confirm on real data how many rows this touches. | Low |
| 7 | `train.py:104-109` | The same validation split is used to compare models and to report the final score, and there is no separate test set. Acceptable for this project, but the reported MAE is slightly optimistic. | Info |

Add the line numbers you actually see in your copy if they differ from the ones above (taken from the files you shared).

## 9. Conclusion
`<One short paragraph: is the Stage 2 workflow verified? What must the main developer fix?>`
