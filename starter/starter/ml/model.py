from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def save_model(model, model_path):
    """
    Saves the trained machine learning model to a file.
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(model_path):
    """
    Loads a machine learning model from a file.
    
    Inputs
    ------
    model_path : str
        Path to the saved model file.
    Returns
    -------
    model
        Loaded machine learning model.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def compute_slice_metrics(df, feature, y_true, y_pred):
    """
    Computes the performance of the model on a slice of the data.

    Inputs:
    -------
    df: pd.DataFrame
        Dataframe containing the features and label.
    feature: str
        Name of the feature to compute the performance on.
    y_true: np.array
        Known labels, binarized.
    y_pred: np.array
        Predicted labels, binarized.

    Returns:
    -------
    results: pd.DataFrame
        Dataframe containing the performance of the model on the slice.
    """
    results = []

    unique_values = df[feature].unique()

    for value in unique_values:
        print(f"Processing: {value}")
        mask = df[feature] == value

        subset_y_true = y_true[mask]
        subset_y_pred = y_pred[mask]
        
        precision, recall, fbeta = compute_model_metrics(subset_y_true, subset_y_pred)
        accuracy = (subset_y_true == subset_y_pred).mean()
        row_data = {
            "value": value, 
            "precision": precision, 
            "recall": recall, 
            "fbeta": fbeta,
            "accuracy": accuracy,
            "count": len(subset_y_true)
        }

        results.append(row_data)

    return pd.DataFrame(results)

def save_slice_output(slice_df, feature_name, filename="slice_output.txt"):
    """
    Saves the slice output to a file.
    """
    with open(filename, "w") as f:
        f.write(f"Slice performance for feature: {feature_name}\n")
        f.write("=" * 50 + "\n\n")

        for index, row in slice_df.iterrows():
            f.write(f"{feature_name}: {row['value']}\n")
            f.write(f"Precision: {row['precision']:.4f}\n")
            f.write(f"Recall: {row['recall']:.4f}\n")
            f.write(f"Fbeta: {row['fbeta']:.4f}\n")
            f.write(f"Accuracy: {row['accuracy']:.4f}\n")
            f.write(f"Count: {row['count']}\n")
            f.write("\n")
            f.write("-" * 50 + "\n\n")