import pytest
import sys
import os
from fastapi.testclient import TestClient

# Add parent directory to path to import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)

    
def test_get_root():
    """
    Test that the GET on the root returns a 200 status code and the correct response body.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Welcome to the Census Income Prediction API"}
    
def test_post_predict_low_income():
    """
    Test that the POST on the predict endpoint returns a 200 status code.
    Tests the ML model inference for low income prediction (<=50K).
    """
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 30,
        "native-country": "United-States"
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    
    response_data = response.json()
    # Test that we get a prediction result (model should be loaded)
    assert "prediction" in response_data, f"Expected prediction, got: {response_data}"
    assert response_data["prediction"] in ["<=50K", ">50K"]

def test_post_predict_high_income():
    """
    Test that the POST on the predict endpoint returns a 200 status code.
    Tests the ML model inference for high income prediction (>50K).
    """
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    
    response_data = response.json()
    # Test that we get a prediction result (model should be loaded)
    assert "prediction" in response_data, f"Expected prediction, got: {response_data}"
    assert response_data["prediction"] in ["<=50K", ">50K"]