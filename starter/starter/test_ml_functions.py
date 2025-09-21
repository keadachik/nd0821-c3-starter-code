import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.model import train_model, save_model,compute_model_metrics
from ml.data import process_data
from ml.model import inference

def test_train_model_returns_expected_type():
    """
    Test that the train_model function returns a model of the expected type.
    """

    X = np.random.rand(100,5)
    y = np.random.randint(0,2,100)

    model = train_model(X,y)

    assert isinstance(model, RandomForestClassifier), f"Expected a RandomForestClassifier, got {type(model)}"
    assert hasattr(model, 'predict'), "Model must have a predict method"
    assert hasattr(model, 'fit'), "Model must have a fit method"


def test_infrence_returns_expected_shape():
    X_train = np.random.rand(50, 5)
    y_train = np.random.randint(0, 2, 50)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    X_test = np.random.rand(10, 5)
    preds = inference(model, X_test)


    assert isinstance(preds, np.ndarray), f"Expected a numpy array, got {type(preds)}"
    assert preds.shape == (10,), f"Expected a shape of (10,), got {preds.shape}"

def test_compute_model_metrics_returns_tuple():
    """
    Test that the compute_model_metrics function returns a tuple of three floats.
    """
   
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])

    metrics = compute_model_metrics(y_true, y_pred)

    assert isinstance(metrics, tuple), f"Expected a tuple, got {type(metrics)}"
    assert len(metrics) == 3, f"Expected a tuple of length 3, got length {len(metrics)}"
    
    precision, recall, fbeta = metrics
    assert isinstance(precision, (float, np.float64)), f"Expected precision to be a float, got {type(precision)}"
    assert isinstance(recall, (float, np.float64)), f"Expected recall to be a float, got {type(recall)}"
    assert isinstance(fbeta, (float, np.float64)), f"Expected fbeta to be a float, got {type(fbeta)}"