"""
Model training module with LightGBM and hyperparameter tuning
"""
import logging
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import optuna
from optuna.samplers import TPESampler
import joblib
import yaml
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightGBMTrainer:
    """Class to handle LightGBM model training and optimization"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_params = self.config['model_params']['lightgbm']
        self.tuning_params = self.config['model_params']['hyperparam_tuning']
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
    def objective(self, trial, X_train, y_train):
        """Optuna objective function for hyperparameter tuning"""
        
        # Suggest hyperparameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': self.config['data']['random_state'],
            'verbose': -1
        }
        
        # Create model
        model = lgb.LGBMClassifier(**params)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=self.tuning_params['cv_folds'], shuffle=True, random_state=42),
            scoring=self.tuning_params['scoring'],
            n_jobs=-1
        )
        
        return cv_scores.mean()
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: Optional[int] = None):
        """Tune hyperparameters using Optuna"""

        logger.info("Starting hyperparameter tuning...")

        n_trials = n_trials or self.tuning_params['n_trials']

        # Create study with persistent storage
        storage_name = "sqlite:///models/optuna_study.db"
        study_name = "ft3_lightgbm_optimization"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction='maximize',
            sampler=TPESampler(seed=42),
            load_if_exists=True  # Resume if study exists
        )

        logger.info(f"Study has {len(study.trials)} completed trials")

        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_params['objective'] = 'binary'
        self.best_params['metric'] = 'auc'
        self.best_params['boosting_type'] = 'gbdt'
        self.best_params['random_state'] = self.config['data']['random_state']
        self.best_params['verbose'] = -1
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {study.best_value:.4f}")
        
        return self.best_params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              tune_hyperparameters: bool = True):
        """Train LightGBM model"""
        
        logger.info("Starting model training...")
        
        # Split validation set if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=self.config['data']['validation_size'],
                random_state=self.config['data']['random_state'],
                stratify=y_train
            )
        
        # Tune hyperparameters if requested
        if tune_hyperparameters:
            params = self.tune_hyperparameters(X_train, y_train)
        else:
            params = self.model_params
        
        # Train final model
        self.model = lgb.LGBMClassifier(**params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(100)
            ]
        )
        
        # Store feature names for production use
        self.feature_names = X_train.columns.tolist()
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate on validation set
        val_pred = self.model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred)
        
        logger.info(f"Validation AUC: {val_auc:.4f}")
        logger.info(f"Top 10 features:\n{self.feature_importance.head(10)}")
        
        return self.model
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance"""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logger.info(f"Test metrics: {metrics}")
        
        return metrics
    
    def save_model(self, path: str = "models/fasttrack_model.pkl"):
        """Save trained model"""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'feature_names': getattr(self, 'feature_names', None),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
        
        # Also save feature importance as CSV
        if self.feature_importance is not None:
            importance_path = path.replace('.pkl', '_feature_importance.csv')
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")
    
    def load_model(self, path: str = "models/fasttrack_model.pkl"):
        """Load trained model"""
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.best_params = model_data['best_params']
        self.feature_importance = model_data['feature_importance']
        self.feature_names = model_data.get('feature_names', None)
        
        logger.info(f"Model loaded from {path}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict_proba(X)[:, 1]


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_test: pd.DataFrame, y_test: pd.Series,
                config_path: str = "config.yaml",
                tune_hyperparameters: bool = True) -> Tuple[LightGBMTrainer, Dict]:
    """
    Convenience function to train and evaluate model
    
    Returns:
        Tuple of (trainer object, test metrics)
    """
    
    trainer = LightGBMTrainer(config_path)
    
    # Train model
    trainer.train(X_train, y_train, tune_hyperparameters=tune_hyperparameters)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save model
    trainer.save_model()
    
    return trainer, metrics


if __name__ == "__main__":
    # Test model training
    import sys
    sys.path.append('..')
    
    from src.data_loader import load_data_for_training
    from src.feature_engineering import engineer_features
    
    # Load and prepare data
    df, config = load_data_for_training()
    X, y = engineer_features(df, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=y
    )
    
    # Train model
    trainer, metrics = train_model(
        X_train, y_train,
        X_test, y_test,
        tune_hyperparameters=True
    )
    
    print(f"Model training complete!")
    print(f"Test metrics: {metrics}")