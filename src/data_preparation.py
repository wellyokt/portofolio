import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from config.config import Config
from utils.logger import setup_logger

logger = setup_logger('data_preparation')

def load_and_prepare_data():
    """Load and prepare data for modeling"""
    try:
        # Load data
        logger.info("Loading data from CSV...")
        df = pd.read_csv(Config.DATA_PATH)
        
        # Split features and target
        X = df[Config.FEATURE_COLUMNS]
        y = df[Config.TARGET_COLUMN]
        
        # Save feature names
        feature_names = X.columns.tolist()
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE
        )
        continuous_vars = ['Daily_Time_Spent_on_Site', 'Age', 'Area_Income', 'Daily_Internet_Usage', 'Hour', 'DayOfWeek']
        # Scale features
        scaler = MinMaxScaler()
        X_train[continuous_vars]= scaler.fit_transform(continuous_vars)
        X_test[continuous_vars] = scaler.transform(X_test[continuous_vars])
        
        # Save scaler
        with open(Config.SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        
        logger.info("Data preparation completed successfully")
        return X_train, X_test, y_train, y_test, feature_names
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise