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

# Path to the deep neural network models
dnn_model_path = os.path.join(BASE_DIR, "models", "dnn_model.keras")
dnn_model_post_path = os.path.join(BASE_DIR, "models", "dnn_post.keras")

# Paths to RF models
rf_model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
rf_model_post_path = os.path.join(BASE_DIR, "models", "rfc_model_post.pkl")

# Paths to DT models
dt_model_path = os.path.join(BASE_DIR, "models", "decision_tree_model_final.pkl")
dt_model_post_path = os.path.join(BASE_DIR, "models", "dtc_model_post.pkl")

# Load models
try:
    with open(rf_model_path, "rb") as rf_file:
        random_forest_model = pickle.load(rf_file)

    with open(dt_model_path, "rb") as dt_file:
        decision_tree_model = pickle.load(dt_file)

    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    # Load models for post prediction
    with open(rf_model_post_path, "rb") as rf_post_file:
        random_forest_post_model = pickle.load(rf_post_file)

    with open(dt_model_post_path, "rb") as dt_post_file:
        decision_tree_post_model = pickle.load(dt_post_file)
    
    dnn_model = load_model(dnn_model_path)
    dnn_post_model = load_model(dnn_model_post_path)
    
except Exception as e:
    raise RuntimeError(f"Error loading models: {str(e)}")


# Define input structure
class PredictionInput(BaseModel):
    feature_vector: list


# Feature columns used in prediction
feature_columns = [
    "pos", "flw", "flg", "bl", "pic", "lin", "cl", "cz", "ni", 
    "erl", "erc", "lt", "hc", "pr", "fo", "cs", "pi"
]


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
        input_data = pd.DataFrame([data.feature_vector], columns=feature_columns)
        input_data = input_data.apply(pd.to_numeric)
        scaled_data = scaler.transform(input_data)
        prediction = random_forest_model.predict(scaled_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/decision-tree")
async def predict_decision_tree(data: PredictionInput):
    try:
        input_data = pd.DataFrame([data.feature_vector], columns=feature_columns)
        input_data = input_data.apply(pd.to_numeric)
        scaled_data = scaler.transform(input_data)
        prediction = decision_tree_model.predict(scaled_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/dnn-post")
async def predict_dnn_post(data: PredictionInput):
    try:
        feature_vector = np.array(data.feature_vector)
        input_data = feature_vector.reshape(1, -1)
        prediction = np.round(dnn_post_model.predict(input_data).tolist()[0][0], 0)
        return {"prediction": prediction}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/random-forest-post")
async def predict_random_forest_post(data: PredictionInput):
    try:
        input_data = pd.DataFrame([data.feature_vector], columns=feature_columns)
        input_data = input_data.apply(pd.to_numeric)
        scaled_data = scaler.transform(input_data)
        prediction = random_forest_post_model.predict(scaled_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/decision-tree-post")
async def predict_decision_tree_post(data: PredictionInput):
    try:
        input_data = pd.DataFrame([data.feature_vector], columns=feature_columns)
        input_data = input_data.apply(pd.to_numeric)
        scaled_data = scaler.transform(input_data)
        prediction = decision_tree_post_model.predict(scaled_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": (
            "Welcome to the ML API! "
            "1) Use /predict/random-forest and /predict/random-forest-post "
            "2) Use /predict/decision-tree and /predict/decision-tree-post "
            "3) Use /predict/dnn and /predict/dnn-post to make predictions."
        )
    }
