from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os 
import pandas as pd
from tensorflow.keras.models import load_model

# Initialize FastAPI app
app = FastAPI()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Path to the deep neural network model
dnn_model_path = os.path.join(BASE_DIR, "models", "dnn_model.keras")
# Paths to models
rf_model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

dt_model_path = os.path.join(BASE_DIR, "models", "decision_tree_model_final.pkl")


# Load models
try:
    with open(rf_model_path, "rb") as rf_file:
        random_forest_model = pickle.load(rf_file)

    with open(dt_model_path, "rb") as dt_file:
        decision_tree_model = pickle.load(dt_file)
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    
    dnn_model = load_model(dnn_model_path)
    # print("Random Forest Model loaded:", random_forest_model)
    # print("Decision Tree Model loaded:", decision_tree_model)
except Exception as e:
    raise RuntimeError(f"Error loading models: {str(e)}")
# Define input structure
class PredictionInput(BaseModel):
    feature_vector: list

# Routes
feature_columns = [
    "pos", "flw", "flg", "bl", "pic", "lin", "cl", "cz", "ni", 
    "erl", "erc", "lt", "hc", "pr", "fo", "cs", "pi"
]

# @app.post("/predict/random-forest")
# async def predict_random_forest(data: PredictionInput):
#     try:
#         # Log the received input
#         print("Received input data:", data.feature_vector)
        
#         # Convert the feature vector (dict) into a DataFrame with one row
#         input_data = pd.DataFrame([data.feature_vector])
        
#         # Ensure column order matches the feature columns used during training
#         input_data = input_data[feature_columns]
        
#         # Convert all values to numeric (in case they are strings)
#         input_data = input_data.apply(pd.to_numeric)
        
#         # Scale the input data using the saved scaler
#         scaled_data = scaler.transform(input_data)
        
#         # Predict using the loaded model
#         prediction = random_forest_model.predict(scaled_data)
        
#         return {"prediction": int(prediction[0])}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#fake
# {
#     "feature_vector": ["44", "48", "325", "33", "1", "0", "12", "0", "0", "0", "0", "0", "0", "0", "0", "0.1111110002", "0.0949850008"]

# }
#real
# {
#     "feature_vector": ["30", "1800", "2400", "17", "1", "0", "35", "0.055555556", "0.0560000017", "11.529999733", "0.6600000262", "0.5559999943", "0.1669999957", "0", "0", "0.004865", "569.24871826"]
# }


@app.post("/predict/dnn")
async def predict_dnn(data: PredictionInput):
    try:
        feature_vector = np.array(data.feature_vector)

        input_data = feature_vector.reshape(1, -1)

        prediction = np.round(dnn_model.predict(input_data).tolist()[0][0], 0)
        return {"prediction": prediction}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/random-forest")
async def predict_random_forest(data: PredictionInput):
    try:
        # Log the received input
        print("Received input data:", data.feature_vector)
        
        # Convert the feature vector (list) into a DataFrame
        input_data = pd.DataFrame([data.feature_vector], columns=feature_columns)
        
        # Convert all values to numeric
        input_data = input_data.apply(pd.to_numeric)
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Predict using the loaded model
        prediction = random_forest_model.predict(scaled_data)
        
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict/decision-tree")
async def predict_decision_tree(data: PredictionInput):
    try:
        
        print("Received input data:", data.feature_vector)
        
        # Convert the feature vector (list) into a DataFrame
        input_data = pd.DataFrame([data.feature_vector], columns=feature_columns)
        
        # Convert all values to numeric
        input_data = input_data.apply(pd.to_numeric)
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Predict using the loaded model
        prediction = decision_tree_model.predict(scaled_data)
        
        
        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the ML API! 1) Use /predict/random-forest 2) /predict/decision-tree 3) Use /predict/dnn to make predictions."}
