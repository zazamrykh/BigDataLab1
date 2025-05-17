'''
Example of launch of model with 2 params:
python src/inference.py ./runs/train1/best_catboost_model.cbm  "Very good!" "I really like that masterpiece!"
 
Example of request sent using curl:
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"summary\": \"Great product!\", \"text\": \"This product works perfectly and I love it.\", \"HelpfulnessNumerator\": 5, \"HelpfulnessDenominator\": 7}"
'''

import os
import sys
import logging
import numpy as np
from catboost import CatBoostRegressor
from utils import get_text_embedding, cosine_sim
import gensim.downloader as api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load word vectors at module level
try:
    logger.info("Loading word vectors...")
    word_vectors = api.load("glove-wiki-gigaword-50")
    logger.info("Word vectors loaded successfully")
except Exception as e:
    logger.error(f"Failed to load word vectors: {str(e)}")
    raise

def run(model=None, summary="", text="", HelpfulnessNumerator=1, HelpfulnessDenominator=1, output=False, word_vectors=None, model_path=None):
    try:
        if model is None:
            if not os.path.exists(model_path):
                logger.error(f"Model not found at path: {model_path}")
                raise FileNotFoundError(f"Model not found: {model_path}")
            logger.info("Loading model...")
            model = CatBoostRegressor()
            model.load_model(model_path)
            logger.info("Model loaded successfully")
        
        if word_vectors is None:
            if output:
                logger.info('Loading embeddings because it is not provided in input...')
            word_vectors = api.load("glove-wiki-gigaword-50")
        
        good_emb = word_vectors["good"]
        bad_emb = word_vectors["bad"]
        
        logger.info("Calculating cosine similarities...")
        cos_sim_good_text = cosine_sim(get_text_embedding(text, word_vectors), good_emb)
        cos_sim_bad_text = cosine_sim(get_text_embedding(text, word_vectors), bad_emb)
        cos_sim_good_summary = cosine_sim(get_text_embedding(summary, word_vectors), good_emb)
        cos_sim_bad_summary = cosine_sim(get_text_embedding(summary, word_vectors), bad_emb)
        
        input_data = np.array([cos_sim_good_text, cos_sim_bad_text, cos_sim_good_summary, cos_sim_bad_summary,
                                HelpfulnessNumerator, HelpfulnessDenominator]).reshape(1, -1)
        
        if input_data.shape[1] != len(model.feature_names_):
            logger.error(f"Feature count mismatch. Expected: {len(model.feature_names_)}, Got: {input_data.shape[1]}")
            raise ValueError(f"Wrong features count {len(model.feature_names_)}, получено {input_data.shape[1]}")
        
        if output:
            logger.info('Running inference...')
        prediction = model.predict(input_data)
        
        if output:
            logger.info(f"Prediction result: {prediction[0]}")
        return prediction[0]
    except Exception as e:
        logger.error(f"Error in run(): {str(e)}")
        raise
 

if __name__ == '__main__':
    try:
        logger.info("Starting inference script")
        argv = sys.argv
        model_path = argv[1]
        input_data = argv[2:]
        summary = input_data[0]
        text = input_data[1]
        
        if len(input_data) == 4:
            HelpfulnessNumerator = input_data[2]
            HelpfulnessDenominator = input_data[3]
        else:
            HelpfulnessNumerator, HelpfulnessDenominator = 1, 1
        
        if(len(input_data) == 5):
            output = bool(input_data[5])
        else:
            output = False
            
        logger.info(f"Running inference for text: {text[:50]}...")
        result = run(model_path, summary, text, HelpfulnessNumerator, HelpfulnessDenominator, output=True)
        logger.info(f"Inference completed with result: {result}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise