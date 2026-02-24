"""
Model auditing module for comprehensive ML model evaluation.

This module provides functionality for:
- Data quality assessment
- Model performance evaluation
- Fairness analysis across protected attributes
- Feature importance analysis using SHAP
- Data drift detection using statistical tests
"""
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy.stats import ks_2samp
# Removed non-existent data_science_auditor dependency
# Implementing auditing functionality with standard libraries
import yaml
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelAuditor:
    """Class to handle comprehensive model auditing"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize auditor with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.auditing_config = self.config['auditing']
        self.thresholds = self.auditing_config['thresholds']
        self.protected_attributes = self.auditing_config.get('protected_attributes', [])
        
        # Initialize audit results storage
        self.audit_results = {}
        
    def audit_data_quality(self, df: pd.DataFrame) -> Dict:
        """Audit data quality"""
        logger.info("Running data quality audit...")
        
        quality_results = {
            'missing_values': {},
            'data_types': {},
            'unique_counts': {},
            'statistics': {}
        }
        
        # Check missing values
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        quality_results['missing_values'] = {
            col: {
                'count': int(missing_counts[col]),
                'percentage': float(missing_pct[col])
            }
            for col in df.columns if missing_counts[col] > 0
        }
        
        # Check data types
        quality_results['data_types'] = df.dtypes.astype(str).to_dict()
        
        # Check unique value counts
        quality_results['unique_counts'] = {
            col: df[col].nunique()
            for col in df.columns
        }
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            quality_results['statistics'] = df[numeric_cols].describe().to_dict()
        
        # Flag potential issues
        issues = []
        for col, missing_info in quality_results['missing_values'].items():
            if missing_info['percentage'] > 20:
                issues.append(f"High missing rate in {col}: {missing_info['percentage']:.1f}%")
        
        quality_results['issues'] = issues
        
        self.audit_results['data_quality'] = quality_results
        return quality_results
    
    def audit_model_performance(self, y_true: pd.Series, y_pred: np.ndarray, 
                               y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """Audit model performance metrics"""
        logger.info("Running model performance audit...")
        
        performance_results = {}
        
        # Binary classification metrics
        if y_pred_proba is not None:
            performance_results['auc'] = float(roc_auc_score(y_true, y_pred_proba))
        
        performance_results['precision'] = float(precision_score(y_true, y_pred))
        performance_results['recall'] = float(recall_score(y_true, y_pred))
        performance_results['f1'] = float(f1_score(y_true, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        performance_results['confusion_matrix'] = cm.tolist()
        performance_results['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True
        )
        
        # Check against thresholds
        threshold_violations = []
        for metric, threshold_key in [
            ('auc', 'auc_min'),
            ('precision', 'precision_min'),
            ('recall', 'recall_min'),
            ('f1', 'f1_min')
        ]:
            if metric in performance_results:
                if performance_results[metric] < self.thresholds.get(threshold_key, 0):
                    threshold_violations.append(
                        f"{metric}: {performance_results[metric]:.3f} < {self.thresholds[threshold_key]}"
                    )
        
        performance_results['threshold_violations'] = threshold_violations
        
        self.audit_results['model_performance'] = performance_results
        return performance_results
    
    def audit_fairness(self, X: pd.DataFrame, y_true: pd.Series, 
                      y_pred: np.ndarray, model) -> Dict:
        """Audit model fairness across protected attributes"""
        logger.info("Running fairness audit...")
        
        fairness_results = {}
        
        for protected_attr in self.protected_attributes:
            if protected_attr in X.columns:
                attr_results = {}
                
                # Get unique groups
                groups = X[protected_attr].unique()
                
                for group in groups:
                    mask = X[protected_attr] == group
                    
                    if mask.sum() > 0:
                        group_metrics = {
                            'count': int(mask.sum()),
                            'positive_rate': float(y_pred[mask].mean()),
                            'true_positive_rate': float(y_true[mask].mean()),
                            'precision': float(precision_score(y_true[mask], y_pred[mask])) if mask.sum() > 0 else 0,
                            'recall': float(recall_score(y_true[mask], y_pred[mask])) if mask.sum() > 0 else 0
                        }
                        attr_results[str(group)] = group_metrics
                
                # Calculate disparity metrics
                positive_rates = [v['positive_rate'] for v in attr_results.values()]
                if len(positive_rates) > 1:
                    disparity = max(positive_rates) / (min(positive_rates) + 1e-10)
                    attr_results['disparity_ratio'] = float(disparity)
                    
                    # Flag if disparity is too high
                    if disparity > 1.2:  # 80% rule
                        attr_results['fairness_violation'] = True
                
                fairness_results[protected_attr] = attr_results
        
        self.audit_results['fairness'] = fairness_results
        return fairness_results
    
    def audit_feature_importance(self, X: pd.DataFrame, model, sample_size: int = 1000) -> Dict:
        """Audit feature importance using SHAP"""
        logger.info("Running feature importance audit...")
        
        importance_results = {}
        
        # Get model feature importance
        if hasattr(model, 'feature_importances_'):
            importance_results['model_importance'] = {
                'features': X.columns.tolist(),
                'importances': model.feature_importances_.tolist()
            }
        
        # SHAP analysis (on sample for speed)
        try:
            sample_idx = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
            X_sample = X.iloc[sample_idx]
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # For binary classification, take positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Average absolute SHAP values
            shap_importance = np.abs(shap_values).mean(axis=0)
            
            importance_results['shap_importance'] = {
                'features': X.columns.tolist(),
                'importances': shap_importance.tolist()
            }
            
            # Top features
            top_n = 20
            top_features_idx = np.argsort(shap_importance)[-top_n:][::-1]
            importance_results['top_features'] = [
                {
                    'feature': X.columns[idx],
                    'importance': float(shap_importance[idx])
                }
                for idx in top_features_idx
            ]
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {str(e)}")
            importance_results['shap_error'] = str(e)
        
        self.audit_results['feature_importance'] = importance_results
        return importance_results
    
    def audit_data_drift(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Dict:
        """Audit data drift between train and test sets"""
        logger.info("Running data drift audit...")
        
        drift_results = {}
        
        # Kolmogorov-Smirnov test for numeric features
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in X_test.columns:
                # Perform KS test
                statistic, p_value = ks_2samp(X_train[col], X_test[col])
                
                drift_results[col] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'drift_detected': p_value < self.auditing_config['drift_detection']['threshold']
                }
        
        # Summary
        drift_detected_features = [
            col for col, result in drift_results.items()
            if result.get('drift_detected', False)
        ]
        
        drift_results['summary'] = {
            'total_features_checked': len(drift_results),
            'features_with_drift': len(drift_detected_features),
            'drift_percentage': (len(drift_detected_features) / len(drift_results) * 100) if drift_results else 0,
            'drifted_features': drift_detected_features
        }
        
        self.audit_results['data_drift'] = drift_results
        return drift_results
    
    def generate_audit_report(self, output_path: str = "reports/audit_report.json"):
        """Generate comprehensive audit report"""
        logger.info("Generating audit report...")
        
        report = {
            'audit_timestamp': datetime.now().isoformat(),
            'model_name': self.config['model']['name'],
            'model_version': self.config['model']['version'],
            'audit_results': self._convert_to_serializable(self.audit_results),
            'audit_summary': self._generate_summary()
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serializer)
        
        logger.info(f"Audit report saved to {output_path}")
        
        return report
    
    def _json_serializer(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)
    
    def _convert_to_serializable(self, obj):
        """Recursively convert numpy types in nested structures"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _generate_summary(self) -> Dict:
        """Generate audit summary"""
        summary = {
            'passed_checks': [],
            'failed_checks': [],
            'warnings': []
        }
        
        # Check data quality
        if 'data_quality' in self.audit_results:
            if self.audit_results['data_quality'].get('issues', []):
                summary['warnings'].extend(self.audit_results['data_quality']['issues'])
            else:
                summary['passed_checks'].append('Data quality check')
        
        # Check model performance
        if 'model_performance' in self.audit_results:
            violations = self.audit_results['model_performance'].get('threshold_violations', [])
            if violations:
                summary['failed_checks'].append('Model performance thresholds')
                summary['warnings'].extend(violations)
            else:
                summary['passed_checks'].append('Model performance check')
        
        # Check fairness
        if 'fairness' in self.audit_results:
            fairness_violations = any(
                attr_results.get('fairness_violation', False)
                for attr_results in self.audit_results['fairness'].values()
            )
            if fairness_violations:
                summary['failed_checks'].append('Fairness check')
            else:
                summary['passed_checks'].append('Fairness check')
        
        # Check data drift
        if 'data_drift' in self.audit_results:
            drift_summary = self.audit_results['data_drift'].get('summary', {})
            if drift_summary.get('drift_percentage', 0) > 20:
                summary['warnings'].append(
                    f"Data drift detected in {drift_summary['drift_percentage']:.1f}% of features"
                )
        
        return summary
    
    def plot_audit_results(self, output_dir: str = "reports/"):
        """Generate visualization plots for audit results"""
        logger.info("Generating audit visualizations...")
        
        # Performance metrics plot
        if 'model_performance' in self.audit_results:
            perf = self.audit_results['model_performance']
            
            plt.figure(figsize=(10, 6))
            metrics = ['precision', 'recall', 'f1', 'auc']
            values = [perf.get(m, 0) for m in metrics]
            
            plt.bar(metrics, values)
            plt.ylabel('Score')
            plt.title('Model Performance Metrics')
            plt.ylim(0, 1)
            
            # Add threshold lines
            for i, metric in enumerate(metrics):
                threshold_key = f"{metric}_min"
                if threshold_key in self.thresholds:
                    plt.axhline(y=self.thresholds[threshold_key], color='r', 
                              linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}performance_metrics.png")
            plt.close()
        
        # Confusion matrix plot
        if 'model_performance' in self.audit_results:
            cm = np.array(self.audit_results['model_performance']['confusion_matrix'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(f"{output_dir}confusion_matrix.png")
            plt.close()
        
        logger.info("Visualizations saved")


def run_full_audit(model, X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series,
                  config_path: str = "config.yaml") -> Dict:
    """
    Run comprehensive model audit
    
    Returns:
        Audit report dictionary
    """
    
    auditor = ModelAuditor(config_path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Run all audits
    auditor.audit_data_quality(X_test)
    auditor.audit_model_performance(y_test, y_pred, y_pred_proba)
    auditor.audit_fairness(X_test, y_test, y_pred, model)
    auditor.audit_feature_importance(X_train, model)
    auditor.audit_data_drift(X_train, X_test)
    
    # Generate report and plots
    report = auditor.generate_audit_report()
    auditor.plot_audit_results()
    
    return report


if __name__ == "__main__":
    # Test auditing
    import sys
    sys.path.append('..')
    
    from src.data_loader import load_data_for_training
    from src.feature_engineering import engineer_features
    from src.model_training import LightGBMTrainer
    from sklearn.model_selection import train_test_split
    
    # Load data and model
    df, config = load_data_for_training()
    X, y = engineer_features(df, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        stratify=y
    )
    
    # Load trained model
    trainer = LightGBMTrainer()
    trainer.load_model()
    
    # Run audit
    report = run_full_audit(
        trainer.model,
        X_train, y_train,
        X_test, y_test
    )
    
    print("Audit complete!")
    print(f"Summary: {report['audit_summary']}")