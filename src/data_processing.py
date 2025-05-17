import numpy as np
import os
import shutil
import pandas as pd
import kagglehub
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import utils
from utils import cosine_sim, get_text_embedding

# Load word vectors at module level
try:
    logger.info("Loading word vectors...")
    word_vectors = api.load("glove-wiki-gigaword-50")
    logger.info("Word vectors loaded successfully")
except Exception as e:
    logger.error(f"Failed to load word vectors: {str(e)}")
    raise

def get_dataset(dataset_path=None, output=False, visualize=False, filename='distribution.png'):
    try:
        if not os.path.exists("./data"):
            logger.info("Creating data directory")
            os.makedirs("./data")
            
        path = kagglehub.dataset_download("arhamrumi/amazon-product-reviews") if dataset_path is None else dataset_path
        logger.info(f"Loading dataset from: {path}")
        df = pd.read_csv(os.path.join(path, 'Reviews.csv'))

        shutil.copy(path + '/Reviews.csv', "./data")
        
        df = df.sample(frac=1, random_state=utils.params.random_seed).reset_index(drop=True)[:utils.params.all_data_size]
        
        if output:
            logger.info("Common information:")
            logger.info(df.info())

        missing_values = df.isnull().sum()
        if output:
            logger.info("\nNulls:")
            logger.info(missing_values[missing_values > 0])

        if output:
            logger.info("\nData examples:")
            logger.info(df.head())

        df["ProfileName"] = df["ProfileName"].fillna("No text")
        df["Summary"] = df["Summary"].fillna("No text")

        if output:
            logger.info("\nNulls after filling:")
            null_sum_after_fill = df.isnull().sum()
            logger.info(null_sum_after_fill)

        assert null_sum_after_fill.sum() == 0

        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, palette="coolwarm", legend=False)
        plt.title("Score distribution")
        plt.xlabel("Score")
        plt.ylabel("N reviews")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        if output:
            logger.info(f"Saving distribution plot to: {filename}")
            plt.savefig(filename)
            
        if visualize:
            plt.show()
        
        return df
    except Exception as e:
        logger.error(f"Error in get_dataset: {str(e)}")
        raise


def add_features(df):
    try:
        logger.info("Adding features to dataset")
        
        good_emb = word_vectors["good"]
        bad_emb = word_vectors["bad"]

        logger.info("Calculating cosine similarities...")
        df["cos_sim_good_text"] = df["Text"].apply(lambda x: cosine_sim(get_text_embedding(x, word_vectors), good_emb))
        df["cos_sim_bad_text"] = df["Text"].apply(lambda x: cosine_sim(get_text_embedding(x, word_vectors), bad_emb))
        df["cos_sim_good_summary"] = df["Summary"].apply(lambda x: cosine_sim(get_text_embedding(x, word_vectors), good_emb))
        df["cos_sim_bad_summary"] = df["Summary"].apply(lambda x: cosine_sim(get_text_embedding(x, word_vectors), bad_emb))

        logger.info("Feature examples:")
        logger.info(df[["Text", "Summary", "cos_sim_good_text", "cos_sim_bad_text", "cos_sim_good_summary", "cos_sim_bad_summary"]].head())

        df.drop(columns=["Id", "UserId", "ProfileName", "ProductId", "Text", "Summary", 'Time'], inplace=True)
        logger.info("Features added successfully")
    except Exception as e:
        logger.error(f"Error in add_features: {str(e)}")
        raise
    
    
def split_df(df):
    try:
        logger.info("Splitting dataset")
        X = df.drop(columns=["Score"])
        y = df["Score"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=utils.params.random_seed, shuffle=True)

        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error in split_df: {str(e)}")
        raise


if __name__ == '__main__':
    try:
        logger.info("Starting data processing")
        df = get_dataset(filename='data/tgt_distrib.png')
        add_features(df)
        df.to_csv('data/Reviews_featurized.csv', index=False)
        logger.info("Data processing completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
