# MP194 Stage 3 Deployment QA

**Reviewer** Parv Pandey · **Date** 26 November 2026 · **Group** G183

**Overall status: FAIL.** All three GitHub Actions runs failed. Application events were structured JSON, while Uvicorn's default access logs were plain text. Docker and Kubernetes runtime checks remain **NOT TESTED**.

Reviewed local `main` at `d3ada83`, which includes Stage 3 commit `f75d7c1`. The originally supplied GitHub URL returned 404. The checkout points to [G183-MP194-MLOps](https://github.com/Nupuryadav20/G183-MP194-MLOps).

| Area | Status | Result |
| --- | --- | --- |
| Docker configuration | PASS | Python 3.12 slim, expected copy sources, port 8000 and Uvicorn binding verified by inspection. |
| Docker build and runtime | NOT TESTED | Docker daemon unavailable. Build stopped before processing the Dockerfile. |
| Kubernetes configuration | PASS | Deployment, Service, selectors and probes matched the requested settings. |
| Kubernetes runtime | NOT TESTED | No Kubernetes context was configured. |
| GitHub Actions | FAIL | Stage 3 push, pull request and main push all failed. |
| Structured stdout logging | FAIL | Application events were JSON. Default Uvicorn access records were plain text. |

## Docker

The Dockerfile uses `python:3.12-slim`, installs from `requirements.txt`, and copies `app/`, `src/`, `models/model.joblib` and `params.yaml`. All copy sources are tracked and allowed by `.dockerignore`. It exposes port 8000 and starts `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

```text
$ docker version
failed to connect to the docker API ... docker.sock: connect: no such file or directory

$ docker desktop start
Failed to start Docker Desktop

$ docker build -t mp194-api:1.0 .
ERROR: failed to connect to the docker API ... docker.sock: connect: no such file or directory
```

Image dependency installation and copy steps were unverified. The build never reached the point where `docker run --rm -p 8000:8000 mp194-api:1.0` or the container endpoint checks could run.

## Kubernetes

Deployment `mp194-api` requests two replicas of `mp194-api:1.0` on container port 8000. Readiness and liveness both call `/health` on port 8000. Readiness starts at 5 seconds and liveness at 15 seconds. Service `mp194-svc` forwards TCP port 80 to 8000 and selects the matching pod label `app=mp194-api`. The `imagePullPolicy: Never` setting requires the image to be present on each eligible node.

```text
$ kubectl config current-context
error: current-context is not set

$ kubectl config get-contexts
CURRENT   NAME   CLUSTER   AUTHINFO   NAMESPACE

$ kubectl --request-timeout=7s cluster-info
Unable to connect to the server: dial tcp [::1]:8080: connect: operation not permitted
```

No deployment was attempted. Replica readiness, probes and Service routing remain unverified at runtime.

## GitHub Actions

The workflow is configured for `push` and `pull_request`. It uses `actions/setup-python@v5` with Python 3.12, installs `requirements.txt` and runs `pytest -v` with `PYTHONPATH=src`.

| Run | Result |
| --- | --- |
| [CI #1](https://github.com/Nupuryadav20/G183-MP194-MLOps/actions/runs/45677573329), Stage 3 push at `f75d7c1` | FAIL |
| [CI #2](https://github.com/Nupuryadav20/G183-MP194-MLOps/actions/runs/45678053646), pull request #1 | FAIL |
| [CI #3](https://github.com/Nupuryadav20/G183-MP194-MLOps/actions/runs/45679519842/job/139436013990), main push at `d3ada83` | FAIL. The public annotation identifies `Run pytest -v` and exit code 2. |

Run #3 reached pytest, indicating the prior setup and install steps completed. Full hosted logs required sign-in, so the runner's underlying pytest error could not be confirmed directly. A local Python 3.12 import check with the repository root excluded showed that `PYTHONPATH=src` resolves `mp194` but leaves the root-level `app` package unavailable. CI-style collection using available Python 3.14 tooling reproduced that import error. It also hit a separate local error because the available environment lacked MLflow.

```text
$ <Python 3.12 venv>/bin/python -m pip --disable-pip-version-check install --retries 0 --timeout 5 -r requirements.txt
ERROR: No matching distribution found for pandas==2.2.3

$ PYTHONSAFEPATH=1 PYTHONPATH=src python3.12 -c '<find_spec check>'
app: None
mp194: ModuleSpec(name='mp194', ...)

$ PYTHONPATH=src <Python 3.14> -P -c '<pytest -v --collect-only -p no:cacheprovider>'
collected 29 items / 2 errors
ERROR tests/test_api.py
E   ModuleNotFoundError: No module named 'app'
ERROR tests/test_training.py
E   ModuleNotFoundError: No module named 'mlflow'
```

## API and logging

A host-side FastAPI `TestClient` smoke test loaded the committed model. Stdout capture confirmed JSON prediction and exception events. A sample record emitted through Uvicorn's default access logger confirmed that its format is plain text. Container logging remains unverified.

```text
$ PYTHONDONTWRITEBYTECODE=1 python3 -u -   # inline TestClient smoke test
/health 200
/docs 200
POST /predict 200
POST /predict invalid 422
POST /predict forced exception 500

{"level": "INFO", "logger": "mp194.api", "message": "Prediction done", "prediction_kwh": 486.14}
{"level": "ERROR", "logger": "mp194.api", "message": "Unhandled error", "error": "RuntimeError"}
INFO:     127.0.0.1:12345 - "GET /health HTTP/1.1" 200 OK
```

## Findings and repository state

1. `PYTHONPATH=src` makes `mp194` importable but omits the root-level `app` package in CI-style collection. The local reproduction failed at `from app.main import app`.
2. Uvicorn's default access records make stdout a mix of JSON and plain text.

Inspection commands included `git status --short --branch`, `git remote -v`, `git log -4 --oneline --decorate`, `git ls-files app src models params.yaml` and `nl -ba` on the six infrastructure files. The project working tree was clean after verification. No project files were modified, committed or pushed.
