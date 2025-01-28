import os
from pathlib import Path

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    LOGS_DIR = BASE_DIR / "logs"
    STATIC_DIR = BASE_DIR / "static"
    CV_DIR = BASE_DIR/'cv'
    
    # Data paths
    DATA_PATH = ARTIFACTS_DIR / "advertising.csv"
    MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
    SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
    METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
    FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.json"

    #Profile
    FOTO_PATH = CV_DIR / "IMG_0236.png"
    project1_image = CV_DIR / "google-adsense-moves-to-pay-per-impression.png"


    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    NUM_FEATURES = 8
    CV_FOLDS = 5
    TARGET_COLUMN = "Clicked_on_Ad"
    
    # Feature columns for model
    FEATURE_COLUMNS = [
        'Daily_Time_Spent_on_Site', 'Age', 'Area_Income',
       'Daily_Internet_Usage', 'Male', 'Hour', 'DayOfWeek'
    ]
    
    
     # Data validation rules
    DATA_VALIDATION = {
        'Daily_Time_Spent_on_Site': {'min': 0.0, 'max': 100.0},
        "Age": {'min': 15, 'max': 80},
        "Area_Income": {'min': 10000.0, 'max': 100000.0},
        'Daily_Internet_Usage': {'min': 100.0, 'max': 300.0},
        'Male': {'min': 0, 'max': 1},
        'Hour': {'min': 0, 'max': 24},
        'NOX': {'min': 0.3, 'max': 0.9},
        'DayOfWeek': {'min': 0, 'max': 400}
    }
    
    # Logging configuration
    LOG_FILE = LOGS_DIR / "app.log"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_LEVEL = "INFO"
    
    # FastAPI settings
    API_TITLE = "Ad Click Prediction API"
    API_DESCRIPTION = "API for predicting ad click"
    API_VERSION = "1.0.0"
    HOST = "0.0.0.0"
    PORT = 8000
    
    
    # Cache settings
    CACHE_TTL = 3600  # 1 hour
    
    # Feature descriptions for documentation
    FEATURE_DESCRIPTIONS ={
    'Daily_Time_Spent_on_Site': 'The average time (in minutes) a user spends on the site daily.',
    'Age': 'The user\'s age.',
    'Area_Income': 'The average income of the user\'s geographical area.',
    'Daily_Internet_Usage': 'The average number of hours a user spends on the internet daily.',
    'Male': 'A binary value indicating whether the user is male (1) or not (0).',
    'Hour': 'The hour of the day when the user visits the site.',
    'DayOfWeek': 'The day of the week when the user visits the site (usually encoded as a number, e.g., 0=Monday, 6=Sunday).'
}
    
    

    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [cls.ARTIFACTS_DIR, cls.LOGS_DIR, cls.STATIC_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_feature_range(cls, feature):
        """Get the valid range for a feature."""
        return cls.DATA_VALIDATION.get(feature, {'min': float('-inf'), 'max': float('inf')})
    
    @classmethod
    def is_valid_feature_value(cls, feature, value):
        """Check if a feature value is within valid range."""
        ranges = cls.get_feature_range(feature)
        return ranges['min'] <= value <= ranges['max']

# Create directories on import
Config.create_directories()