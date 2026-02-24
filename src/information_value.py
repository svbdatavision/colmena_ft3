"""
Module for calculating Weight of Evidence (WoE) and Information Value (IV)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InformationValueCalculator:
    """Class to calculate WoE and IV for features"""
    
    def __init__(self):
        self.woe_dict = {}
        self.iv_dict = {}
        
    def calculate_woe_iv(self, X: pd.DataFrame, y: pd.Series, 
                        bins: int = 10, 
                        min_bin_size: float = 0.05) -> pd.DataFrame:
        """
        Calculate WoE and IV for all features
        
        Args:
            X: Feature dataframe
            y: Target variable (binary)
            bins: Number of bins for numeric variables
            min_bin_size: Minimum proportion of samples in each bin
            
        Returns:
            DataFrame with IV values for each feature
        """
        
        iv_results = []
        
        # Ensure y is binary
        if len(y.unique()) != 2:
            raise ValueError("Target variable must be binary")
        
        # Calculate total goods and bads
        n_total = len(y)
        n_goods = sum(y == 1)
        n_bads = sum(y == 0)
        
        if n_goods == 0 or n_bads == 0:
            raise ValueError("Target variable must have both classes")
        
        # Process each feature
        for col in X.columns:
            try:
                if X[col].dtype in ['object', 'category']:
                    # Categorical variable
                    woe_df, iv = self._calculate_categorical_woe_iv(
                        X[col], y, n_goods, n_bads
                    )
                else:
                    # Numeric variable
                    woe_df, iv = self._calculate_numeric_woe_iv(
                        X[col], y, n_goods, n_bads, bins, min_bin_size
                    )
                
                # Store results
                self.woe_dict[col] = woe_df
                self.iv_dict[col] = iv
                
                iv_results.append({
                    'feature': col,
                    'iv': iv,
                    'predictive_power': self._get_predictive_power(iv),
                    'type': 'categorical' if X[col].dtype in ['object', 'category'] else 'numeric'
                })
                
            except Exception as e:
                logger.warning(f"Error calculating IV for {col}: {str(e)}")
                iv_results.append({
                    'feature': col,
                    'iv': 0,
                    'predictive_power': 'Error',
                    'type': 'error'
                })
        
        # Create results dataframe
        iv_df = pd.DataFrame(iv_results).sort_values('iv', ascending=False)
        
        logger.info(f"Calculated IV for {len(iv_df)} features")
        logger.info(f"Top 10 features by IV:\n{iv_df.head(10)}")
        
        return iv_df
    
    def _calculate_categorical_woe_iv(self, feature: pd.Series, target: pd.Series,
                                     n_goods: int, n_bads: int) -> Tuple[pd.DataFrame, float]:
        """Calculate WoE and IV for categorical variable"""
        
        # Create crosstab
        crosstab = pd.crosstab(feature, target)
        
        # Handle missing columns
        if 0 not in crosstab.columns:
            crosstab[0] = 0
        if 1 not in crosstab.columns:
            crosstab[1] = 0
        
        # Calculate distributions
        crosstab['total'] = crosstab.sum(axis=1)
        crosstab['bad_rate'] = crosstab[0] / crosstab['total']
        
        # Calculate good and bad distributions
        crosstab['good_dist'] = crosstab[1] / n_goods
        crosstab['bad_dist'] = crosstab[0] / n_bads
        
        # Handle zero distributions
        crosstab['good_dist'] = crosstab['good_dist'].replace(0, 0.0001)
        crosstab['bad_dist'] = crosstab['bad_dist'].replace(0, 0.0001)
        
        # Calculate WoE
        crosstab['woe'] = np.log(crosstab['good_dist'] / crosstab['bad_dist'])
        
        # Calculate IV contribution
        crosstab['iv_contribution'] = (crosstab['good_dist'] - crosstab['bad_dist']) * crosstab['woe']
        
        # Total IV
        iv = crosstab['iv_contribution'].sum()
        
        return crosstab, iv
    
    def _calculate_numeric_woe_iv(self, feature: pd.Series, target: pd.Series,
                                  n_goods: int, n_bads: int, 
                                  bins: int, min_bin_size: float) -> Tuple[pd.DataFrame, float]:
        """Calculate WoE and IV for numeric variable"""
        
        # Handle missing values
        mask = ~feature.isna()
        feature_clean = feature[mask]
        target_clean = target[mask]
        
        if len(feature_clean) == 0:
            return pd.DataFrame(), 0
        
        # Create bins
        try:
            # Use quantile-based binning
            _, bin_edges = pd.qcut(feature_clean, q=bins, retbins=True, duplicates='drop')
            feature_binned = pd.cut(feature_clean, bins=bin_edges, include_lowest=True)
        except:
            # Fall back to equal-width binning
            feature_binned = pd.cut(feature_clean, bins=min(bins, len(feature_clean.unique())), include_lowest=True)
        
        # Create temporary dataframe
        temp_df = pd.DataFrame({
            'feature': feature_binned,
            'target': target_clean
        })
        
        # Calculate WoE and IV using categorical method
        return self._calculate_categorical_woe_iv(temp_df['feature'], temp_df['target'], n_goods, n_bads)
    
    def _get_predictive_power(self, iv: float) -> str:
        """Interpret IV value"""
        if iv < 0.02:
            return "Not useful"
        elif iv < 0.1:
            return "Weak"
        elif iv < 0.3:
            return "Medium"
        elif iv < 0.5:
            return "Strong"
        else:
            return "Very strong"
    
    def plot_top_features(self, iv_df: pd.DataFrame, top_n: int = 20, save_path: str = None):
        """Plot top features by IV"""
        
        plt.figure(figsize=(10, 8))
        
        # Select top features
        top_features = iv_df.head(top_n)
        
        # Create color map based on predictive power
        color_map = {
            'Not useful': 'red',
            'Weak': 'orange',
            'Medium': 'yellow',
            'Strong': 'lightgreen',
            'Very strong': 'darkgreen',
            'Error': 'gray'
        }
        colors = [color_map.get(x, 'gray') for x in top_features['predictive_power']]
        
        # Create horizontal bar plot
        plt.barh(range(len(top_features)), top_features['iv'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Information Value (IV)')
        plt.title(f'Top {top_n} Features by Information Value')
        plt.gca().invert_yaxis()
        
        # Add reference lines
        plt.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='Weak')
        plt.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5, label='Medium')
        plt.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Strong')
        
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved IV plot to {save_path}")
        
        plt.close()
    
    def plot_woe_pattern(self, feature_name: str, save_path: str = None):
        """Plot WoE pattern for a specific feature"""
        
        if feature_name not in self.woe_dict:
            logger.warning(f"No WoE data found for feature: {feature_name}")
            return
        
        woe_df = self.woe_dict[feature_name]
        
        plt.figure(figsize=(10, 6))
        
        # Plot WoE values
        x_labels = [str(idx) for idx in woe_df.index]
        plt.bar(range(len(woe_df)), woe_df['woe'])
        plt.xticks(range(len(woe_df)), x_labels, rotation=45)
        plt.xlabel(feature_name)
        plt.ylabel('Weight of Evidence (WoE)')
        plt.title(f'WoE Pattern for {feature_name}')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add bad rate on secondary axis
        ax2 = plt.gca().twinx()
        ax2.plot(range(len(woe_df)), woe_df['bad_rate'], 'ro-', label='Bad Rate')
        ax2.set_ylabel('Bad Rate', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved WoE plot to {save_path}")
        
        plt.close()
    
    def get_feature_summary(self, feature_name: str) -> Dict:
        """Get summary statistics for a feature"""
        
        if feature_name not in self.woe_dict:
            return None
        
        woe_df = self.woe_dict[feature_name]
        iv = self.iv_dict[feature_name]
        
        return {
            'feature': feature_name,
            'iv': iv,
            'predictive_power': self._get_predictive_power(iv),
            'n_categories': len(woe_df),
            'woe_range': woe_df['woe'].max() - woe_df['woe'].min(),
            'max_bad_rate': woe_df['bad_rate'].max(),
            'min_bad_rate': woe_df['bad_rate'].min()
        }


def calculate_and_save_iv(X: pd.DataFrame, y: pd.Series, 
                         save_path: str = "models/feature_iv_analysis.csv",
                         plot_path: str = "reports/feature_iv_plot.png") -> pd.DataFrame:
    """
    Convenience function to calculate and save IV analysis
    
    Args:
        X: Feature dataframe
        y: Target variable
        save_path: Path to save IV results
        plot_path: Path to save IV plot
        
    Returns:
        DataFrame with IV results
    """
    
    calculator = InformationValueCalculator()
    
    # Calculate IV for all features
    iv_df = calculator.calculate_woe_iv(X, y)
    
    # Save results
    iv_df.to_csv(save_path, index=False)
    logger.info(f"Saved IV analysis to {save_path}")
    
    # Create visualization
    calculator.plot_top_features(iv_df, top_n=30, save_path=plot_path)
    
    # Print summary
    print("\nInformation Value Summary:")
    print("=" * 60)
    print(f"Total features analyzed: {len(iv_df)}")
    print(f"Features with Strong predictive power (IV > 0.3): {len(iv_df[iv_df['iv'] > 0.3])}")
    print(f"Features with Medium predictive power (0.1 < IV < 0.3): {len(iv_df[(iv_df['iv'] > 0.1) & (iv_df['iv'] <= 0.3)])}")
    print(f"Features with Weak predictive power (0.02 < IV < 0.1): {len(iv_df[(iv_df['iv'] > 0.02) & (iv_df['iv'] <= 0.1)])}")
    print(f"Features with No predictive power (IV < 0.02): {len(iv_df[iv_df['iv'] <= 0.02])}")
    
    return iv_df, calculator