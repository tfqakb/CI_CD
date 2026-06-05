from fastapi.testclient import TestClient
from predict_model import app

client = TestClient(app)

def test_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to the Churn Prediction App!"

def test_predict_churn():
    response = client.get("/predict")

    assert response.status_code == 200
    data = response.json()
    assert "conf_matrix" in data
    assert "acc_score" in data
    
    assert isinstance(data["conf_matrix"], list)
    assert isinstance(data["acc_score"], float)
    