"""
Feature engineering module for preprocessing and transforming features
"""
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import yaml
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class to handle feature engineering and preprocessing"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize feature engineering with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Import variable groups from Variables_cat_train.py
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from Variables_cat_train import llaves, fechas, categoricas, numericas, binarias, texto_no_estructurado, target
        
        # Identify leak variables
        all_vars = llaves + fechas + categoricas + numericas + binarias + texto_no_estructurado
        leak = [v for v in all_vars if 'LEAK' in v]
        
        # Store feature groups (excluding leak variables)
        self.features = {
            'keys': llaves,
            'dates': fechas,
            'categorical': categoricas,
            'numeric': numericas,
            'binary': binarias,
            'text': texto_no_estructurado
        }
        
        self.target = self.config['data']['target']
        self.leak_vars = leak
        self.all_targets = target
        
        # Initialize transformers
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.text_vectorizers = {}
        self.column_transformer = None
        
    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from date columns"""
        df = df.copy()
        
        date_cols = self.features.get('dates', [])
        
        for col in date_cols:
            if col in df.columns:
                # Convert to datetime
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Check if the column has any non-null values
                has_valid_dates = df[col].notna().any()
                
                if has_valid_dates:
                    # Extract date components normally
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                    df[f'{col}_quarter'] = df[col].dt.quarter
                    
                    # Days since a reference date (e.g., 2022-01-01)
                    reference_date = pd.Timestamp('2022-01-01')
                    df[f'{col}_days_since_ref'] = (df[col] - reference_date).dt.days
                else:
                    # If all values are NaT, use proxy values
                    logger.warning(f"Date column {col} has all NaT values. Creating derived features with proxy values.")
                    
                    if 'FECHA_RECEPCION' in df.columns and pd.to_datetime(df['FECHA_RECEPCION'], errors='coerce').notna().any():
                        # Use FECHA_RECEPCION as proxy
                        proxy_date = pd.to_datetime(df['FECHA_RECEPCION'], errors='coerce')
                        df[f'{col}_year'] = proxy_date.dt.year
                        df[f'{col}_month'] = proxy_date.dt.month
                        df[f'{col}_day'] = proxy_date.dt.day
                        df[f'{col}_dayofweek'] = proxy_date.dt.dayofweek
                        df[f'{col}_quarter'] = proxy_date.dt.quarter
                        
                        reference_date = pd.Timestamp('2022-01-01')
                        df[f'{col}_days_since_ref'] = (proxy_date - reference_date).dt.days
                    else:
                        # Use default values
                        df[f'{col}_year'] = 2024
                        df[f'{col}_month'] = 1
                        df[f'{col}_day'] = 1
                        df[f'{col}_dayofweek'] = 0
                        df[f'{col}_quarter'] = 1
                        df[f'{col}_days_since_ref'] = 730  # approx days from 2022-01-01 to 2024-01-01
                
                # Drop original date column
                df = df.drop(columns=[col])
            else:
                # IMPORTANT: If date column is missing, still create the derived features
                # This ensures consistency between training and prediction
                # Use FECHA_RECEPCION as proxy if available, otherwise use current date
                logger.warning(f"Date column {col} not found. Creating derived features with proxy values.")
                
                if 'FECHA_RECEPCION' in df.columns:
                    # Use FECHA_RECEPCION as proxy
                    proxy_date = pd.to_datetime(df['FECHA_RECEPCION'], errors='coerce')
                    logger.info(f"Using FECHA_RECEPCION as proxy for {col}")
                else:
                    # Use a default date (median date from training or current date)
                    proxy_date = pd.Timestamp('2024-01-01')
                    logger.info(f"Using default date 2024-01-01 as proxy for {col}")
                
                # Create the same derived features using proxy date
                if isinstance(proxy_date, pd.Series):
                    # If proxy_date is a Series (from FECHA_RECEPCION)
                    df[f'{col}_year'] = proxy_date.dt.year
                    df[f'{col}_month'] = proxy_date.dt.month
                    df[f'{col}_day'] = proxy_date.dt.day
                    df[f'{col}_dayofweek'] = proxy_date.dt.dayofweek
                    df[f'{col}_quarter'] = proxy_date.dt.quarter
                    
                    reference_date = pd.Timestamp('2022-01-01')
                    df[f'{col}_days_since_ref'] = (proxy_date - reference_date).dt.days
                else:
                    # If proxy_date is a single Timestamp
                    df[f'{col}_year'] = proxy_date.year
                    df[f'{col}_month'] = proxy_date.month
                    df[f'{col}_day'] = proxy_date.day
                    df[f'{col}_dayofweek'] = proxy_date.dayofweek
                    df[f'{col}_quarter'] = proxy_date.quarter
                    
                    reference_date = pd.Timestamp('2022-01-01')
                    df[f'{col}_days_since_ref'] = (proxy_date - reference_date).days
                
                # Fill NaN values that might arise from coerce
                for suffix in ['_year', '_month', '_day', '_dayofweek', '_quarter', '_days_since_ref']:
                    feature_name = f'{col}{suffix}'
                    if feature_name in df.columns:
                        df[feature_name] = df[feature_name].fillna(
                            df[feature_name].median() if not df[feature_name].isna().all() else 0
                        )
        
        return df
    
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from text columns using TF-IDF"""
        df = df.copy()
        
        text_cols = self.features.get('text', [])
        
        for col in text_cols:
            if col in df.columns:
                # Fill missing values
                df[col] = df[col].fillna('')
                
                # Initialize TF-IDF vectorizer if not exists
                if col not in self.text_vectorizers:
                    self.text_vectorizers[col] = TfidfVectorizer(
                        max_features=50,  # Reducido de 100 a 50
                        ngram_range=(1, 2),
                        min_df=5,
                        max_df=0.8
                    )
                
                # Fit or transform
                if hasattr(self.text_vectorizers[col], 'vocabulary_'):
                    # Transform only
                    text_features = self.text_vectorizers[col].transform(df[col])
                else:
                    # Fit and transform
                    text_features = self.text_vectorizers[col].fit_transform(df[col])
                
                # Convert to DataFrame
                feature_names = [f'{col}_tfidf_{i}' for i in range(text_features.shape[1])]
                text_df = pd.DataFrame(
                    text_features.toarray(),
                    columns=feature_names,
                    index=df.index
                )
                
                # Concatenate with original DataFrame
                df = pd.concat([df.drop(columns=[col]), text_df], axis=1)
        
        return df
    
    def handle_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()
        
        cat_cols = self.features.get('categorical', [])
        
        for col in cat_cols:
            if col in df.columns:
                # Convert to string and fill missing values
                df[col] = df[col].astype(str).replace('nan', 'MISSING').replace('<NA>', 'MISSING')
                
                if fit:
                    # Fit and transform
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    
                    # Handle unknown categories
                    unique_values = df[col].unique()
                    self.label_encoders[col].fit(unique_values)
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                else:
                    # Transform only, handle unseen categories
                    if col in self.label_encoders:
                        # Get known categories
                        known_categories = set(self.label_encoders[col].classes_)
                        
                        # Find a default category that exists in the encoder
                        # Use 'MISSING' if it exists, otherwise use the first known category
                        if 'MISSING' in known_categories:
                            default_category = 'MISSING'
                        else:
                            default_category = list(known_categories)[0]
                        
                        # Replace unknown categories with default category
                        df[col] = df[col].apply(
                            lambda x: x if x in known_categories else default_category
                        )
                        
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                    else:
                        logger.warning(f"Label encoder not found for column {col}")
                        df[f'{col}_encoded'] = 0
                
                # Drop original column
                df = df.drop(columns=[col])
        
        return df
    
    def handle_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle numeric features - imputation and scaling"""
        df = df.copy()
        
        numeric_cols = self.features.get('numeric', [])
        binary_cols = self.features.get('binary', [])
        
        # Combine numeric and binary columns
        all_numeric = numeric_cols + binary_cols
        
        for col in all_numeric:
            if col in df.columns:
                # Fill missing values with median for numeric, 0 for binary
                if col in binary_cols:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        df = df.copy()
        
        # Define key interactions based on domain knowledge
        interactions = [
            ('COT_EDAD', 'CIE_GRUPO_encoded'),
            ('DIASSOLICITADO', 'CANT_LIC_U12M_CALC'),
            ('MEDICO_EXTRANJERO', 'CIE_GRUPO_PSIQUIATRICAS_12'),
            ('EMPLEADOR_TASA_RECHAZO_ISAPRE_U12M', 'MEDICO_TASA_RATIFICACION_COMPIN_U12M')
        ]
        
        for col1, col2 in interactions:
            if col1 in df.columns and col2 in df.columns:
                # Multiplication interaction
                df[f'{col1}_X_{col2}'] = df[col1] * df[col2]
                
                # Ratio interaction (avoiding division by zero)
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit transformers and transform training data"""
        logger.info("Starting feature engineering fit_transform")
        
        # First, remove any leak variables that might be present
        leak_cols_present = [col for col in df.columns if col in self.leak_vars]
        if leak_cols_present:
            logger.warning(f"Removing leak columns from data: {leak_cols_present}")
            df = df.drop(columns=leak_cols_present)
        
        # Remove any target variables except the one we need
        other_targets = [col for col in self.all_targets if col != self.target and col in df.columns]
        if other_targets:
            logger.info(f"Removing other target columns: {other_targets}")
            df = df.drop(columns=other_targets)
        
        # Separate features and target
        target_col = self.target
        if target_col in df.columns:
            y = df[target_col].copy()
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df.copy()
        
        # Drop ID columns
        id_cols = self.features.get('keys', [])
        X = X.drop(columns=[col for col in id_cols if col in X.columns])
        
        # Apply transformations
        X = self.create_date_features(X)
        X = self.create_text_features(X)
        X = self.handle_categorical_features(X, fit=True)
        X = self.handle_numeric_features(X)
        X = self.create_interaction_features(X)
        
        # Scale numeric features
        numeric_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
        if numeric_features:
            X[numeric_features] = self.scaler.fit_transform(X[numeric_features])
        
        # Store feature names for later use
        self._feature_names = list(X.columns)
        
        logger.info(f"Feature engineering complete. Shape: {X.shape}")
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformers"""
        logger.info("Starting feature engineering transform")
        
        # First, remove any leak variables that might be present
        leak_cols_present = [col for col in df.columns if col in self.leak_vars]
        if leak_cols_present:
            logger.warning(f"Removing leak columns from data: {leak_cols_present}")
            df = df.drop(columns=leak_cols_present)
        
        # Remove ALL target variables (including the one we use for training)
        target_cols_present = [col for col in df.columns if col in self.all_targets]
        if target_cols_present:
            logger.info(f"Removing target columns from prediction data: {target_cols_present}")
            df = df.drop(columns=target_cols_present)
        
        X = df.copy()
        
        # Drop ID columns
        id_cols = self.features.get('keys', [])
        X = X.drop(columns=[col for col in id_cols if col in X.columns])
        
        # Apply transformations
        X = self.create_date_features(X)
        X = self.create_text_features(X)
        X = self.handle_categorical_features(X, fit=False)
        X = self.handle_numeric_features(X)
        X = self.create_interaction_features(X)
        
        # Scale numeric features
        numeric_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
        if numeric_features:
            try:
                X[numeric_features] = self.scaler.transform(X[numeric_features])
            except ValueError as e:
                if 'feature names' in str(e).lower():
                    # Si hay un problema con los nombres de features, intentar alinear
                    logger.warning("Feature name mismatch in scaler, attempting to align...")
                    
                    # Obtener las features esperadas por el scaler
                    if hasattr(self.scaler, 'feature_names_in_'):
                        expected_features = list(self.scaler.feature_names_in_)
                        
                        # Crear DataFrame con las features esperadas
                        X_aligned = pd.DataFrame(index=X.index)
                        for feat in expected_features:
                            if feat in numeric_features and feat in X.columns:
                                X_aligned[feat] = X[feat]
                            else:
                                X_aligned[feat] = 0  # Valor por defecto
                        
                        # Aplicar el scaler
                        X_scaled = self.scaler.transform(X_aligned)
                        
                        # Actualizar solo las columnas numéricas que existen
                        for i, feat in enumerate(expected_features):
                            if feat in X.columns:
                                X[feat] = X_scaled[:, i]
                    else:
                        raise e
                else:
                    raise e
        
        logger.info(f"Feature engineering complete. Shape: {X.shape}")
        
        return X
    
    def save_transformers(self, path: str = "models/feature_fasttrack.pkl"):
        """Save fitted transformers"""
        transformers = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'text_vectorizers': self.text_vectorizers,
            'feature_names': self.get_feature_names()
        }
        
        joblib.dump(transformers, path)
        logger.info(f"Saved transformers to {path}")
    
    def load_transformers(self, path: str = "models/feature_fasttrack.pkl"):
        """Load fitted transformers"""
        transformers = joblib.load(path)
        
        self.label_encoders = transformers['label_encoders']
        self.scaler = transformers['scaler']
        self.text_vectorizers = transformers['text_vectorizers']
        
        logger.info(f"Loaded transformers from {path}")
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names after transformation"""
        # Return the feature names from the scaler if available
        if hasattr(self, 'scaler') and hasattr(self.scaler, 'feature_names_in_'):
            return list(self.scaler.feature_names_in_)
        # Otherwise return the stored feature names if available
        elif hasattr(self, '_feature_names'):
            return self._feature_names
        return []


def engineer_features(df: pd.DataFrame, config_path: str = "config.yaml", 
                     fit: bool = True, calculate_iv: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Convenience function for feature engineering
    
    Args:
        df: Input DataFrame
        config_path: Path to configuration file
        fit: Whether to fit transformers (True for training, False for prediction)
        calculate_iv: Whether to calculate Information Value for features
    
    Returns:
        Tuple of (transformed features, target)
    """
    engineer = FeatureEngineer(config_path)
    
    if fit:
        X, y = engineer.fit_transform(df)
        engineer.save_transformers()
        
        # Calculate Information Value if requested
        if calculate_iv and y is not None:
            from src.information_value import calculate_and_save_iv
            logger.info("Calculating Information Value for features...")
            iv_df, iv_calculator = calculate_and_save_iv(X, y)
            
        return X, y
    else:
        engineer.load_transformers()
        X = engineer.transform(df)
        return X, None


if __name__ == "__main__":
    # Test feature engineering
    import sys
    sys.path.append('..')
    
    from src.data_loader import load_data_for_training
    
    # Load sample data
    df, config = load_data_for_training()
    
    # Engineer features
    X, y = engineer_features(df, fit=True)
    
    print(f"Original shape: {df.shape}")
    print(f"Engineered features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts()}")