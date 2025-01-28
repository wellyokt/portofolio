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
    # try:
        # Load data
    logger.info("Loading data from CSV...")
    df = pd.read_csv(Config.DATA_PATH)
    
    # Split features and target
    X = df[Config.FEATURE_COLUMNS]
    y = df[Config.TARGET_COLUMN]
    # X['Daily_Time_Spent_on_Site'] =X['Daily_Time_Spent_on_Site'].astype(float)
    # X['Daily_Time_Spent_on_Site'].fillna(0, inplace=True)  # Replace NaN with 0

    
    # Save feature names
    feature_names = X.columns.tolist()
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE
    )
    print(X_train)
    continuous_vars = ['Daily_Time_Spent_on_Site', 'Age', 'Area_Income', 'Daily_Internet_Usage', 'Hour', 'DayOfWeek']
    # Scale features
    scaler = MinMaxScaler()
    X_train[continuous_vars]= scaler.fit_transform(X_train[continuous_vars])
    # ValueError: could not convert string to float: 'Daily_Time_Spent_on_Site'
    X_test[continuous_vars] = scaler.transform(X_test[continuous_vars])
    
    # Save scaler
    with open(Config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info("Data preparation completed successfully")
    return X_train, X_test, y_train, y_test, feature_names
        
    # except Exception as e:
    #     logger.error(f"Error in data preparation: {str(e)}")
    #     raise