'''
Example of launch of api with 2 params:
python src/api.py "glove-wiki-gigaword-50" "./runs/train1/best_catboost_model.cbm"
Or with only path to model:
python src/api.py "./runs/train1/best_catboost_model.cbm"
'''

import os
import sys
import logging
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRegressor
import gensim.downloader as api
from inference import run

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load word vectors at module level
try:
    logger.info("Loading word vectors...")
    word_vectors = api.load("glove-wiki-gigaword-50")
    logger.info("Word vectors loaded successfully")
except Exception as e:
    logger.error(f"Failed to load word vectors: {str(e)}")
    raise

model = None

class InputData(BaseModel):
    summary: str
    text: str
    HelpfulnessNumerator: int = 1
    HelpfulnessDenominator: int = 1

@app.post("/predict")
def predict(data: InputData):
    logger.info(f"Received prediction request for text: {data.text[:50]}...")
    prediction = run(
        model=model,
        summary=data.summary,
        text=data.text,
        HelpfulnessNumerator=data.HelpfulnessNumerator,
        HelpfulnessDenominator=data.HelpfulnessDenominator,
        output=False,
        word_vectors=word_vectors,
    )
    logger.info(f"Prediction completed: {prediction}")
    return {"prediction": prediction}

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up API server...")

if __name__ == "__main__":
    argv = sys.argv
    match(len(argv)):
        case 1:
            model_path = './runs/train1/best_catboost_model.cbm'
        case 2:  # consider only model path is given
            model_path = argv[1]
        case 3:
            model_path = argv[2]

    if not os.path.exists(model_path):
        logger.error(f"Model not found at path: {model_path}")
        raise FileNotFoundError(f"Model not found that way: {model_path}")
    
    try:
        logger.info("Loading model...")
        model = CatBoostRegressor()
        model.load_model(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    import uvicorn
    logger.info("Starting uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
