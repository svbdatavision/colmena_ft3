"""
Data loader module v2 - Corregido para manejar correctamente el flujo COMPIN.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from dotenv import load_dotenv
import yaml
from src.spark_sql_compat import SparkSQLConnection, get_spark_session

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SnowflakeDataLoader:
    """Backwards-compatible loader class using Databricks SQL."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Spark-based connection-compatible loader."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        snowflake_cfg = self.config.get("data", {}).get("snowflake", {})
        database = snowflake_cfg.get("database", "")
        schema = snowflake_cfg.get("schema", "")

        self.connection_params = {
            # Keep names for backward compatibility with legacy callsites.
            "database": database,
            "schema": schema,
        }
        
        self.conn = None
        self.cursor = None
        self.spark = None
        
    def connect(self):
        """Bind loader to the active Spark session."""
        try:
            self.spark = get_spark_session()
            self.conn = SparkSQLConnection(self.spark)
            self.cursor = self.conn.cursor()
            logger.info("Successfully connected to Spark SQL")
        except Exception as e:
            logger.error(f"Failed to connect to Spark SQL: {str(e)}")
            raise
    
    def disconnect(self):
        """Detach loader from current Spark compatibility connection."""
        if self.cursor:
            try:
                self.cursor.close()
            except Exception:
                pass
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
        self.spark = None
        logger.info("Disconnected from Spark SQL")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a query and return results as DataFrame"""
        if self.spark is None:
            raise RuntimeError("No hay conexión activa. Ejecute connect() primero.")
        try:
            df = self.spark.sql(query).toPandas()
            logger.info(f"Query executed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    def load_training_data(self, 
                          table_name: str = "MODELO_LM_202507_TRAIN",
                          date_from: Optional[str] = None,
                          date_to: Optional[str] = None,
                          limit: Optional[int] = None,
                          analysis_mode: str = "compin") -> pd.DataFrame:
        """
        Load training data from Databricks SQL
        
        Args:
            table_name: Name of the table to load from
            date_from: Start date for filtering (YYYY-MM-DD)
            date_to: End date for filtering (YYYY-MM-DD)
            limit: Number of rows to limit (for testing)
            analysis_mode: 
                - "compin": (Deprecated) This mode is kept for backward compatibility
                - "all_modified": (Deprecated) This mode is kept for backward compatibility
                - "all": Todas las licencias
        
        Returns:
            DataFrame with training data
        """
        # Import variable groups from Variables_cat_train.py
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from Variables_cat_train import llaves, fechas, categoricas, numericas, binarias, texto_no_estructurado, target
        
        # Extract leak variables from the different groups
        leak = []
        # From fechas
        leak.extend([v for v in fechas if 'LEAK' in v])
        # From categoricas
        leak.extend([v for v in categoricas if 'LEAK' in v])
        # From numericas
        leak.extend([v for v in numericas if 'LEAK' in v])
        # From binarias
        leak.extend([v for v in binarias if 'LEAK' in v])
        # From texto_no_estructurado
        leak.extend([v for v in texto_no_estructurado if 'LEAK' in v])
        
        # Remove leak variables from their original groups for feature selection
        fechas_clean = [v for v in fechas if 'LEAK' not in v]
        categoricas_clean = [v for v in categoricas if 'LEAK' not in v]
        numericas_clean = [v for v in numericas if 'LEAK' not in v]
        binarias_clean = [v for v in binarias if 'LEAK' not in v]
        texto_no_estructurado_clean = [v for v in texto_no_estructurado if 'LEAK' not in v]
        
        # Define variables to EXCLUDE
        variables_to_exclude = leak  # Exclude all leak variables
        
        # Define variables to KEEP (excluding leak and only keeping the selected target)
        target_col = self.config['data']['target']  # Should be TARGET_RATIFICA
        
        # Get all feature columns (excluding leak variables)
        feature_cols = []
        
        # Add all variable groups except leak and target
        all_variables = llaves + fechas_clean + categoricas_clean + numericas_clean + binarias_clean + texto_no_estructurado_clean
        
        # Filter out any leak variables that might be in the lists
        feature_cols = [col for col in all_variables if col not in variables_to_exclude]
        
        # Add target column
        all_cols = feature_cols + [target_col]
        
        # Remove duplicates while preserving order
        seen = set()
        all_cols = [x for x in all_cols if not (x in seen or seen.add(x))]
        
        # Build query
        query = f"""
        SELECT {', '.join(all_cols)}
        FROM {table_name}
        WHERE 1=1
        """
        
        if date_from:
            query += f" AND FECHA_RECEPCION >= '{date_from}'"
        
        if date_to:
            query += f" AND FECHA_RECEPCION <= '{date_to}'"
        
        # Apply target filter
        query += f" AND {target_col} IS NOT NULL"
        logger.info(f"Filtering: Only licenses with {target_col} not null")
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info(f"Loading data with query for {len(all_cols)} columns...")
        logger.info(f"Analysis mode: {analysis_mode}")
        logger.info(f"Excluding {len(leak)} leak variables")
        logger.info(f"Using target: {target_col}")
        
        df = self.execute_query(query)
        
        # Basic data validation
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Remove TARGET_MODIFICA column if present as it's not needed for training
        if 'TARGET_MODIFICA' in df.columns:
            df = df.drop(columns=['TARGET_MODIFICA'])
        
        logger.info(f"Target distribution: {df[target_col].value_counts().to_dict()}")
        
        # Double-check no leak variables are present
        leak_cols_present = [col for col in df.columns if col in leak]
        if leak_cols_present:
            logger.warning(f"WARNING: Leak columns found in data: {leak_cols_present}")
            df = df.drop(columns=leak_cols_present)
        
        return df
    
    def get_data_statistics(self, table_name: str = "MODELO_LM_202507_TRAIN",
                           date_from: Optional[str] = None,
                           date_to: Optional[str] = None) -> Dict:
        """Get statistics about the data flow"""
        
        query = f"""
        SELECT 
            COUNT(*) as total_licenses,
            SUM(CASE WHEN TARGET_MODIFICA = 1 THEN 1 ELSE 0 END) as modified_licenses,
            SUM(CASE WHEN TARGET_MODIFICA = 1 AND TARGET_RATIFICA IS NOT NULL THEN 1 ELSE 0 END) as compin_licenses,
            SUM(CASE WHEN TARGET_MODIFICA = 1 AND TARGET_RATIFICA = 1 THEN 1 ELSE 0 END) as compin_ratified,
            SUM(CASE WHEN TARGET_MODIFICA = 1 AND TARGET_RATIFICA = 0 THEN 1 ELSE 0 END) as compin_denied
        FROM {table_name}
        WHERE 1=1
        """
        
        if date_from:
            query += f" AND FECHA_RECEPCION >= '{date_from}'"
        
        if date_to:
            query += f" AND FECHA_RECEPCION <= '{date_to}'"
        
        result = self.execute_query(query)
        
        stats = result.iloc[0].to_dict()
        
        # Calculate percentages
        if stats['total_licenses'] > 0:
            stats['pct_modified'] = stats['modified_licenses'] / stats['total_licenses'] * 100
        
        if stats['modified_licenses'] > 0:
            stats['pct_to_compin'] = stats['compin_licenses'] / stats['modified_licenses'] * 100
        
        if stats['compin_licenses'] > 0:
            stats['pct_ratified'] = stats['compin_ratified'] / stats['compin_licenses'] * 100
            stats['pct_denied'] = stats['compin_denied'] / stats['compin_licenses'] * 100
        
        return stats


def load_data_for_training(config_path: str = "config.yaml", 
                          date_from: Optional[str] = "2022-01-01",
                          date_to: Optional[str] = "2025-05-31",
                          limit: Optional[int] = None,
                          analysis_mode: str = "compin") -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to load training data
    
    Args:
        config_path: Path to configuration file
        date_from: Start date for filtering (YYYY-MM-DD)
        date_to: End date for filtering (YYYY-MM-DD)
        limit: Number of rows to limit (for testing)
        analysis_mode: "compin", "all_modified", or "all"
    
    Returns:
        Tuple of (DataFrame, config dict)
    """
    loader = SnowflakeDataLoader(config_path)
    
    try:
        # Establish connection
        loader.connect()
        
        # First, get statistics
        logger.info("Getting data flow statistics...")
        stats = loader.get_data_statistics(
            date_from=date_from,
            date_to=date_to
        )
        
        logger.info("Data Flow Statistics:")
        logger.info(f"- Total licenses: {stats['total_licenses']:,}")
        logger.info(f"- Modified by ISAPRE: {stats['modified_licenses']:,} ({stats.get('pct_modified', 0):.1f}%)")
        logger.info(f"- Sent to COMPIN: {stats['compin_licenses']:,} ({stats.get('pct_to_compin', 0):.1f}% of modified)")
        logger.info(f"- COMPIN Ratified: {stats['compin_ratified']:,} ({stats.get('pct_ratified', 0):.1f}%)")
        logger.info(f"- COMPIN Denied: {stats['compin_denied']:,} ({stats.get('pct_denied', 0):.1f}%)")
        
        # Load data from Databricks SQL
        df = loader.load_training_data(
            table_name="MODELO_LM_202507_TRAIN",
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            analysis_mode=analysis_mode
        )
        
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        logger.info(f"Successfully loaded training data: {df.shape}")
        
        return df, config
        
    except Exception as e:
        logger.error(f"Failed to load training data: {str(e)}")
        raise
    finally:
        if loader.conn:
            loader.disconnect()


if __name__ == "__main__":
    # Test the data loader
    import os
    
    try:
        # Test loading with different modes
        print("\n1. Testing COMPIN mode (default):")
        df_compin, config = load_data_for_training(limit=1000, analysis_mode="compin")
        print(f"   Loaded {len(df_compin)} COMPIN cases")
        
        print("\n2. Testing ALL MODIFIED mode:")
        df_modified, _ = load_data_for_training(limit=1000, analysis_mode="all_modified")
        print(f"   Loaded {len(df_modified)} modified licenses")
        
        print(f"\n   Percentage that went to COMPIN: {len(df_compin)/len(df_modified)*100:.1f}%")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()