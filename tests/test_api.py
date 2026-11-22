
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def valid_payload():
    return {
        "rooms": 3,
        "members": 4,
        "ac_hours_per_day": 8,
        "avg_temp_c": 30,
        "house_area_sqft": 1200,
        "solar_kw": 2,
        "connection": "domestic",
        "feeder": "Feeder-1"
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_valid_prediction():
    response = client.post(
        "/predict",
        json=valid_payload()
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["unit"] == "kWh"
    assert isinstance(response.json()["prediction"], float)


def test_ac_hours_above_limit():
    payload = valid_payload()
    payload["ac_hours_per_day"] = 26

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_house_area_below_limit():
    payload = valid_payload()
    payload["house_area_sqft"] = 100

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_missing_field():
    payload = valid_payload()
    del payload["rooms"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_extra_field_rejected():
    payload = valid_payload()
    payload["extra_field"] = 123

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
