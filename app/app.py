from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os 
from tensorflow.keras.models import load_model

# Disable GPU and avoid TensorFlow pre-allocating memory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize FastAPI app
app = FastAPI()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths to models
model_paths = {
    "dnn": os.path.join(BASE_DIR, "models", "dnn_model.keras"),
    "rf": os.path.join(BASE_DIR, "models", "random_forest_model.pkl"),
    "dt": os.path.join(BASE_DIR, "models", "decision_tree_model_final.pkl"),
    "scaler": os.path.join(BASE_DIR, "models", "scaler.pkl"),
}

# Feature columns used in prediction
feature_columns = [
    "pos", "flw", "flg", "bl", "pic", "lin", "cl", "cz", "ni", 
    "erl", "erc", "lt", "hc", "pr", "fo", "cs", "pi"
]

# Utility function to load models (lazy loading)
def load_model_lazy(model_type: str):
    """Lazy load models when needed."""
    try:
        if model_type == 'dnn':
            return load_model(model_paths['dnn'])
        elif model_type in ['rf', 'dt']:
            with open(model_paths[model_type], 'rb') as model_file:
                return pickle.load(model_file)
        elif model_type == 'scaler':
            with open(model_paths['scaler'], 'rb') as scaler_file:
                return pickle.load(scaler_file)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
    except Exception as e:
        raise RuntimeError(f"Error loading {model_type} model: {str(e)}")


# Define input structure
class PredictionInput(BaseModel):
    feature_vector: list


@app.post("/predict/dnn")
async def predict_dnn(data: PredictionInput):
    """Predict using DNN model."""
    try:
        # Lazy load DNN model
        dnn_model = load_model_lazy('dnn')

        feature_vector = np.array(data.feature_vector, dtype=float).reshape(1, -1)
        prediction = np.round(dnn_model.predict(feature_vector).tolist()[0][0], 0)
        return {"prediction": int(prediction)}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/random-forest")
async def predict_random_forest(data: PredictionInput):
    """Predict using Random Forest model."""
    try:
        # Lazy load Random Forest model and Scaler
        random_forest_model = load_model_lazy('rf')
        scaler = load_model_lazy('scaler')

        input_data = np.array(data.feature_vector, dtype=float).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        prediction = random_forest_model.predict(scaled_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/decision-tree")
async def predict_decision_tree(data: PredictionInput):
    """Predict using Decision Tree model."""
    try:
        # Lazy load Decision Tree model and Scaler
        decision_tree_model = load_model_lazy('dt')
        scaler = load_model_lazy('scaler')

        input_data = np.array(data.feature_vector, dtype=float).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        prediction = decision_tree_model.predict(scaled_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": (
            "Welcome to the ML API! "
            "1) Use /predict/random-forest "
            "2) Use /predict/decision-tree "
            "3) Use /predict/dnn"
        )
    }
