from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from config.config import Config
from utils.logger import setup_logger

logger = setup_logger('model')

def create_pipeline():
    """Create model pipeline for Logistic Regression"""
    return Pipeline([
        ('classifier', LogisticRegression(
            random_state=Config.RANDOM_STATE,
            max_iter=100,
            solver='lbfgs'  # or 'liblinear', depending on the problem
        ))
    ])

def train_model(pipeline, X_train, y_train, feature_names):
    """Train Logistic Regression model with grid search CV"""
    try:
        logger.info("Starting model training with GridSearchCV...")

        # Update parameter grid for Logistic Regression
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10],  # Regularization strength
            'classifier__solver': ['liblinear', 'lbfgs'],
            'classifier__max_iter': [100, 200],
            'classifier__penalty': ['l2'],  # L2 penalty for Logistic Regression
            'classifier__class_weight': [None, 'balanced']  # Handling class imbalance
        }
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=Config.CV_FOLDS,
            n_jobs=-1,
            scoring='accuracy',  # Since it's classification, use accuracy scoring
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        # Get feature importance (Logistic Regression coefficients)
        best_model = grid_search.best_estimator_
        logistic_model = best_model.named_steps['classifier']
        
        # Get coefficients (feature importances for logistic regression)
        feature_importance = dict(zip(feature_names, logistic_model.coef_[0]))
        logger.info(f"Feature importance: {feature_importance}")
        
        # Save model
        import pickle
        with open(Config.MODEL_PATH, 'wb') as f:
            pickle.dump(best_model, f)
        
        return best_model, feature_importance
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise e
