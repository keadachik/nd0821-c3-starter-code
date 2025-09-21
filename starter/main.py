from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
from starter.ml.model import load_model, inference
from starter.ml.data import process_data

class PersonData(BaseModel):
    age: int = Field(example=30)
    workclass: str = Field(example="Private")
    fnlwgt: int = Field(example=77516,description="Final weight")
    education: str = Field(example="Bachelors",description="Education level")
    
    education_num: int = Field(
        alias="education-num",
        example=13,
        description="Number of years of education"
    )

    marital_status: str = Field(
        alias="marital-status",
        example="Never-married",
        description="Marital status"
    )

    occupation: str = Field(example="Prof-specialty",description="Occupation")
    relationship: str = Field(example="Not-in-family",description="Relationship")
    race: str = Field(example="White",description="Race")
    sex: str = Field(example="Male",description="Sex")
    capital_gain: int = Field(
        alias="capital-gain",
        example=2174,
        description="Capital gain"
    )

    capital_loss: int = Field(
        alias="capital-loss",
        example=0,
        description="Capital loss"
    )

    hours_per_week: int = Field(
        alias="hours-per-week",
        example=40,
        description="Hours per week"
    )

    native_country: str = Field(
        alias="native-country",
        example="United-States",
        description="Native country"
    )

    class Config:
        # allow hyphens in the field names
        allow_population_by_field_name = True

        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }




app = FastAPI(title="Census Income Prediction API")

# Load model and encoders at startup
try:
    print("Loading model from model/model.pkl...")
    model = load_model("model/model.pkl")
    print("Loading encoder from model/encoder.pkl...")
    with open("model/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    print("Loading label binarizer from model/lb.pkl...")
    with open("model/lb.pkl", "rb") as f:
        lb = pickle.load(f)
    print("All models loaded successfully!")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    model = None
    encoder = None
    lb = None
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    encoder = None
    lb = None

@app.get("/")
def welcome():
    return {"greeting": "Welcome to the Census Income Prediction API"}

@app.post("/predict")
def predict(input_data: PersonData):
    if model is None or encoder is None or lb is None:
        return {"error": "Model not loaded. Please train the model first."}
    
    # Convert input data to DataFrame
    input_dict = input_data.dict(by_alias=True)
    input_df = pd.DataFrame([input_dict])
    
    # Define categorical features
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
    
    # Process the input data
    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Make prediction
    prediction = inference(model, X)
    
    # Convert prediction back to label
    prediction_label = lb.inverse_transform(prediction)[0]
    
    return {"prediction": prediction_label}
