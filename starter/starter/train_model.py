# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.model import train_model, save_model,inference,compute_slice_metrics,save_slice_output
from ml.model import compute_model_metrics

# Add the necessary imports for the starter code.
from ml.data import process_data
# Add code to load in the data.
import os
if os.path.exists('../data/census.csv'):
    data = pd.read_csv('../data/census.csv')
elif os.path.exists('data/census.csv'):
    data = pd.read_csv('data/census.csv')
else:
    raise FileNotFoundError("Could not find census.csv in ../data/ or data/")
# Optional enhancement, use K-fold cross validation instead of
# a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)
# Train and save a model.
model = train_model(X_train, y_train)

# Determine model save path
if os.path.exists('../model'):
    model_path = '../model/model.pkl'
    encoder_path = '../model/encoder.pkl'
    lb_path = '../model/lb.pkl'
elif os.path.exists('model'):
    model_path = 'model/model.pkl'
    encoder_path = 'model/encoder.pkl'
    lb_path = 'model/lb.pkl'
else:
    # Create model directory
    os.makedirs('model', exist_ok=True)
    model_path = 'model/model.pkl'
    encoder_path = 'model/encoder.pkl'
    lb_path = 'model/lb.pkl'

save_model(model, model_path)

# Save encoders for API use
import pickle
with open(encoder_path, 'wb') as f:
    pickle.dump(encoder, f)
with open(lb_path, 'wb') as f:
    pickle.dump(lb, f)

# compute the slice metrics
print("making predictions on the test data")
y_pred = inference(model, X_test)

print("computing the slice metrics")
slice_results = compute_slice_metrics(
    test,
    "education",
    y_test,
    y_pred
)

print("Slice results:")
print(slice_results)

print("saving the slice results")
# Determine slice output path
if os.path.exists('..'):
    slice_output_path = "../slice_output.txt"
else:
    slice_output_path = "slice_output.txt"
save_slice_output(slice_results, "education", slice_output_path)

print("Training and slice evaluation complete!")

precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
accuracy = (y_test == y_pred).mean()
print("Overall metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Fbeta: {fbeta:.4f}")
print(f"Accuracy: {accuracy:.4f}")
