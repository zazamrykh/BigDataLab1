"""
python
"""

import logging
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys
import pandas as pd

import utils  # for params access
from utils import load_config
from data_processing import get_dataset, add_features, split_df
from utils import create_dirs, get_output_path, save_params

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train(featured_path=None):
    try:
        logger.info("Starting training process")
        create_dirs()
        output_path = get_output_path()
        
        if featured_path is None:
            logger.info("Loading and processing dataset from scratch")
            df = get_dataset(True, False, filename=output_path + 'tgt_distrib.png')
            add_features(df)
        else:
            logger.info(f"Loading pre-processed dataset from: {featured_path}")
            df = pd.read_csv(featured_path)
            
        logger.info("Splitting dataset into train/test")
        X_train, X_test, y_train, y_test = split_df(df)
        
        logger.info("Training CatBoost model")
        loss_after_train = train_catboost(X_train, X_test, y_train, y_test, hypoptim=False, save_dir=output_path)
        
        logger.info("Saving training parameters")
        save_params(utils.params, output_path + 'params.txt', min_loss=loss_after_train)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error in train(): {str(e)}")
        raise
    
    
def train_catboost(X_train, X_test, y_train, y_test, save_dir='./runs/train1/', hypoptim=True):
    try:
        logger.info("Loading training configuration")
        config = load_config()
        train_config = config['train']
        
        if hypoptim:
            logger.info("Performing hyperparameter optimization")
            param_dict = {
                'depth': [4, 8],
                'learning_rate': [0.03, 0.3],
                'l2_leaf_reg': [1, 5]
            }
        else:
            logger.info("Using fixed hyperparameters")
            param_dict = {
                'depth': [int(train_config['depth'])],
                'learning_rate': [float(train_config['learning_rate'])],
                'l2_leaf_reg': [int(train_config['l2_leaf_reg'])]
            }
            
        model = CatBoostRegressor(loss_function='RMSE', cat_features=[])

        logger.info("Starting grid search")
        grid_search = GridSearchCV(estimator=model, param_grid=param_dict, scoring=None, cv=3, verbose=100, n_jobs=-1)
        grid_search.fit(X_train, y_train, verbose=100)

        logger.info(f"Best params found: {grid_search.best_params_}")
        y_pred = grid_search.best_estimator_.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        logger.info(f'Mean Squared Error (MSE): {mse}')
        logger.info(f'Mean Absolute Error (MAE): {mae}')

        os.makedirs(save_dir, exist_ok=True)

        best_model = grid_search.best_estimator_
        model_path = os.path.join(save_dir, 'model.cbm')
        best_model.save_model(model_path)
        best_model.save_model(os.path.join('models', 'model.cbm'))  # for dvc

        logger.info(f"Model saved to {model_path} and models/model.cbm")
        return mse
    except Exception as e:
        logger.error(f"Error in train_catboost(): {str(e)}")
        raise
    
if __name__ == '__main__':
    try:
        logger.info("Starting training script")
        argv = sys.argv
        if len(argv) == 2:
            featured_path = argv[1]
        else:
            featured_path = None
        train(featured_path=featured_path)
        logger.info("Training script completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise