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


def test_process_data_training_mode():
    """
    Test that process_data function works correctly in training mode.
    """
    # Create sample data
    data = pd.DataFrame({
        'workclass': ['Private', 'Government', 'Private'],
        'age': [25, 30, 35],
        'salary': ['<=50K', '>50K', '<=50K']
    })
    
    categorical_features = ['workclass']
    X, y, encoder, lb = process_data(
        data, 
        categorical_features=categorical_features, 
        label='salary', 
        training=True
    )
    
    assert isinstance(X, np.ndarray), f"Expected numpy array, got {type(X)}"
    assert isinstance(y, np.ndarray), f"Expected numpy array, got {type(y)}"
    assert X.shape[0] == 3, f"Expected 3 rows, got {X.shape[0]}"
    assert y.shape[0] == 3, f"Expected 3 labels, got {y.shape[0]}"
    assert encoder is not None, "Encoder should not be None in training mode"
    assert lb is not None, "Label binarizer should not be None in training mode"


def test_process_data_inference_mode():
    """
    Test that process_data function works correctly in inference mode.
    """
    # Create sample data
    data = pd.DataFrame({
        'workclass': ['Private', 'Government', 'Private'],
        'age': [25, 30, 35],
        'salary': ['<=50K', '>50K', '<=50K']
    })
    
    categorical_features = ['workclass']
    
    # First train to get encoders
    _, _, encoder, lb = process_data(
        data, 
        categorical_features=categorical_features, 
        label='salary', 
        training=True
    )
    
    # Then test inference mode
    X, y, _, _ = process_data(
        data, 
        categorical_features=categorical_features, 
        label='salary', 
        training=False, 
        encoder=encoder, 
        lb=lb
    )
    
    assert isinstance(X, np.ndarray), f"Expected numpy array, got {type(X)}"
    assert isinstance(y, np.ndarray), f"Expected numpy array, got {type(y)}"
    assert X.shape[0] == 3, f"Expected 3 rows, got {X.shape[0]}"


def test_compute_slice_metrics():
    """
    Test that compute_slice_metrics function returns expected DataFrame.
    """
    from ml.model import compute_slice_metrics
    
    # Create sample data
    df = pd.DataFrame({
        'education': ['HS-grad', 'Bachelors', 'HS-grad', 'Masters'],
        'age': [25, 30, 35, 40]
    })
    
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    
    results = compute_slice_metrics(df, 'education', y_true, y_pred)
    
    assert isinstance(results, pd.DataFrame), f"Expected DataFrame, got {type(results)}"
    assert 'precision' in results.columns, "Missing precision column"
    assert 'recall' in results.columns, "Missing recall column"
    assert 'fbeta' in results.columns, "Missing fbeta column"
    assert 'accuracy' in results.columns, "Missing accuracy column"
    assert 'count' in results.columns, "Missing count column"
    assert len(results) == 3, f"Expected 3 unique education values, got {len(results)}"


def test_save_model():
    """
    Test that save_model function works without errors.
    """
    import tempfile
    import os
    
    # Create a simple model
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)
    model.fit(X, y)
    
    # Test saving to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        save_model(model, tmp.name)
        assert os.path.exists(tmp.name), "Model file was not created"
        
        # Clean up
        os.unlink(tmp.name)