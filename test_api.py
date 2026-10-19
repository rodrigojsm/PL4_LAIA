from app import app


def test_predict():
    client = app.test_client()

    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
    )

    assert response.status_code == 200

    data = response.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], int)


def test_prediction_batch():
    client = app.test_client()

    response = client.post(
        "/prediction_batch",
        json={
            "features": [
                [5.1, 3.5, 1.4, 0.2],
                [6.7, 3.0, 5.2, 2.3],
            ]
        },
    )

    assert response.status_code == 200

    data = response.get_json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 2