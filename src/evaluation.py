from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
from config.config import Config
from utils.logger import setup_logger
import numpy as np

logger = setup_logger('evaluation')

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """Evaluate model performance for Logistic Regression classifier"""
    try:
        # Make predictions
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        
        # Calculate classification metrics
        metrics = {
            'train_accuracy': float(accuracy_score(y_train, pred_train)),
            'test_accuracy': float(accuracy_score(y_test, pred_test)),
            'train_precision': float(precision_score(y_train, pred_train, average='binary', pos_label=1)),
            'test_precision': float(precision_score(y_test, pred_test, average='binary', pos_label=1)),
            'train_recall': float(recall_score(y_train, pred_train, average='binary', pos_label=1)),
            'test_recall': float(recall_score(y_test, pred_test, average='binary', pos_label=1)),
            'train_f1': float(f1_score(y_train, pred_train, average='binary', pos_label=1)),
            'test_f1': float(f1_score(y_test, pred_test, average='binary', pos_label=1))
        }
        
        # Optionally calculate AUC for binary classification
        if len(np.unique(y_train)) == 2:  # Ensure it's binary classification
            train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
            test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            metrics['train_auc'] = float(train_auc)
            metrics['test_auc'] = float(test_auc)
        
        # Get coefficients as feature importance from Logistic Regression model
        logistic_model = model.named_steps['classifier']
        importance_values = [float(x) for x in logistic_model.coef_[0]]  # For binary classification, coef_ is a 2D array
        feature_importance = dict(zip(feature_names, importance_values))
        
        # Save metrics
        with open(Config.METRICS_PATH, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save feature importance
        with open(Config.FEATURE_IMPORTANCE_PATH, 'w') as f:
            json.dump(feature_importance, f, indent=4)
        
        logger.info("Model evaluation completed and saved")
        return metrics, feature_importance
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise e
