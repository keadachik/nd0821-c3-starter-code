# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.model import train_model, save_model,inference,compute_slice_metrics,save_slice_output

# Add the necessary imports for the starter code.
from ml.data import process_data
# Add code to load in the data.
data = pd.read_csv('../data/census.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
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
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
model = train_model(X_train, y_train)
save_model(model, '../model/model.pkl')

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
save_slice_output(slice_results, "education","../slice_output.txt")

print("Training and slice evaluation complete!")