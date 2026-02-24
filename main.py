"""
Main script to run the complete pipeline
"""
import os
import sys
import logging
import argparse
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import Variables_cat_train
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import sklearn after path setup
from sklearn.model_selection import train_test_split

# Import our modules
from src.data_loader import SnowflakeDataLoader, load_data_for_training
from src.feature_engineering import FeatureEngineer, engineer_features
from src.model_training import LightGBMTrainer, train_model
from src.model_auditor import run_full_audit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories"""
    directories = ['models', 'results', 'logs', 'reports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("Created necessary directories")


def run_training_pipeline(config_path: str = "config.yaml", 
                         tune_hyperparameters: bool = True):
    """Run the complete training pipeline"""
    
    logger.info("Starting training pipeline...")
    create_directories()
    
    try:
        # Step 1: Load data
        logger.info("Step 1: Loading data from Snowflake...")
        df, config = load_data_for_training(config_path)
        logger.info(f"Loaded {len(df)} rows of data")
        
        # Step 2: Feature engineering
        logger.info("Step 2: Engineering features...")
        X, y = engineer_features(df, config_path, fit=True, calculate_iv=True)
        logger.info(f"Created {X.shape[1]} features")
        
        # Step 2.5: NO FILTRAR FEATURES - Usar todas las features
        logger.info("Step 2.5: Using ALL features (no IV filtering)")
        logger.info(f"Using complete feature set: {X.shape[1]} features")
        
        # Step 3: Split data
        logger.info("Step 3: Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state'],
            stratify=y
        )
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Step 4: Train model
        logger.info("Step 4: Training model...")
        trainer, metrics = train_model(
            X_train, y_train,
            X_test, y_test,
            config_path,
            tune_hyperparameters=tune_hyperparameters
        )
        logger.info(f"Model trained with AUC: {metrics['auc']:.4f}")
        
        # Step 5: Audit model
        logger.info("Step 5: Auditing model...")
        audit_report = run_full_audit(
            trainer.model,
            X_train, y_train,
            X_test, y_test,
            config_path
        )
        
        # Step 6: Save results
        logger.info("Step 6: Saving results...")
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': df.shape,
            'feature_count': X.shape[1],
            'train_size': len(X_train),
            'test_size': len(X_test),
            'model_metrics': metrics,
            'audit_summary': audit_report['audit_summary']
        }
        
        # Save results
        results_df = pd.DataFrame([results])
        results_df.to_csv(f"results/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
        
        logger.info("Training pipeline completed successfully!")
        
        return trainer, metrics, audit_report
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


def run_prediction_pipeline(ids: list, config_path: str = "config.yaml"):
    """Run prediction on new data"""
    
    logger.info("Starting prediction pipeline...")
    
    try:
        # Load model
        logger.info("Loading trained model...")
        trainer = LightGBMTrainer(config_path)
        trainer.load_model()
        
        # Load feature engineer
        logger.info("Loading feature transformers...")
        engineer = FeatureEngineer(config_path)
        engineer.load_transformers()
        
        # Load data
        logger.info(f"Loading data for {len(ids)} IDs...")
        loader = SnowflakeDataLoader(config_path)
        loader.connect()
        
        df = loader.load_prediction_data(
            table_name="ENVIOS_COMPIN",
            id_column="N_LICENCIA",
            id_values=ids
        )
        
        # Engineer features
        logger.info("Engineering features...")
        X = engineer.transform(df)
        
        # Load selected features and filter
        logger.info("Loading selected features...")
        selected_features_path = 'models/selected_features.txt'
        if os.path.exists(selected_features_path):
            with open(selected_features_path, 'r') as f:
                selected_features = [line.strip() for line in f.readlines()]
            
            # Filter X to only include selected features
            X = X[selected_features]
            logger.info(f"Filtered to {len(selected_features)} selected features")
        else:
            logger.warning("Selected features file not found. Using all features.")
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = trainer.predict(X)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'N_LICENCIA': ids,
            'prediction_score': predictions,
            'prediction_class': (predictions > 0.5).astype(int),
            'prediction_timestamp': datetime.now()
        })
        
        # Save predictions to Snowflake
        logger.info("Saving predictions...")
        loader.save_predictions(results_df)
        
        loader.disconnect()
        
        logger.info(f"Predictions completed for {len(results_df)} records")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {str(e)}")
        raise


def main():
    """Main function with argument parsing"""
    
    parser = argparse.ArgumentParser(description='COMPIN LightGBM Model Pipeline')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], 
                       default='train', help='Pipeline mode')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no-tuning', action='store_true',
                       help='Skip hyperparameter tuning')
    parser.add_argument('--ids', type=str, nargs='+',
                       help='IDs for prediction mode')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_training_pipeline(
            config_path=args.config,
            tune_hyperparameters=not args.no_tuning
        )
    
    elif args.mode == 'predict':
        if not args.ids:
            raise ValueError("IDs required for prediction mode")
        
        results = run_prediction_pipeline(
            ids=args.ids,
            config_path=args.config
        )
        
        print(f"\nPrediction Results:")
        print(results)


if __name__ == "__main__":
    main()