'''
Example of launch of api with 2 params:
python src/api.py "glove-wiki-gigaword-50" "./runs/train1/best_catboost_model.cbm"
Or with only path to model:
python src/api.py "./runs/train1/best_catboost_model.cbm"
'''

import os
import sys
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from inference import InferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReviewAPI:
    """API for review prediction service"""
    
    def __init__(self, model_path=None):
        """Initialize API with model path"""
        self.app = FastAPI()
        self.engine = None
        
        if model_path:
            self.load_model(model_path)
        
        self._setup_routes()
    
    def load_model(self, model_path):
        """Load prediction model"""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model not found at path: {model_path}")
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            logger.info("Initializing inference engine...")
            self.engine = InferenceEngine(model_path)
            logger.info("Inference engine ready")
        except Exception as e:
            logger.error(f"Failed to initialize API: {str(e)}")
            raise
    
    def _setup_routes(self):
        """Configure API routes"""
        
        class InputData(BaseModel):
            summary: str
            text: str
            HelpfulnessNumerator: int = 1
            HelpfulnessDenominator: int = 1

        @self.app.post("/predict")
        async def predict(data: InputData):
            logger.info(f"Received prediction request for text: {data.text[:50]}...")
            try:
                prediction = self.engine.predict(
                    summary=data.summary,
                    text=data.text,
                    HelpfulnessNumerator=data.HelpfulnessNumerator,
                    HelpfulnessDenominator=data.HelpfulnessDenominator,
                    verbose=True
                )
                logger.info(f"Prediction completed: {prediction}")
                return {"prediction": prediction}
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                raise

        @self.app.on_event("startup")
        async def startup_event():
            logger.info("API server started")

    def run(self, host="0.0.0.0", port=8000):
        """Run the API server"""
        import uvicorn
        logger.info("Starting uvicorn server...")
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    try:
        logger.info("Starting API service")
        argv = sys.argv
        model_path = argv[1] if len(argv) >= 2 else './runs/train1/best_catboost_model.cbm'
        
        api = ReviewAPI(model_path)
        api.run()
    except Exception as e:
        logger.error(f"API service failed: {str(e)}")
        raise
