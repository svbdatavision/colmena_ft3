"""
Data loader module for connecting to Databricks SQL and loading data.

`SnowflakeDataLoader` is kept as a backwards-compatible alias.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import databricks.sql as databricks_sql
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabricksDataLoader:
    """Class to handle Databricks SQL connections and data loading"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Databricks SQL connection with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        server_hostname = os.environ.get("DATABRICKS_SERVER_HOSTNAME", "").strip()
        http_path = os.environ.get("DATABRICKS_HTTP_PATH", "").strip()
        access_token = (
            os.environ.get("DATABRICKS_ACCESS_TOKEN")
            or os.environ.get("DATABRICKS_TOKEN")
            or ""
        ).strip()
        catalog = os.environ.get("DATABRICKS_CATALOG", "").strip()
        schema = os.environ.get("DATABRICKS_SCHEMA", "").strip()

        self.connection_params = {
            "server_hostname": server_hostname,
            "http_path": http_path,
            "access_token": access_token,
            "catalog": catalog,
            # Keep database alias for legacy callsites.
            "database": catalog,
            "schema": schema,
        }

        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish connection to Databricks SQL."""
        required = {
            "DATABRICKS_SERVER_HOSTNAME": self.connection_params["server_hostname"],
            "DATABRICKS_HTTP_PATH": self.connection_params["http_path"],
            "DATABRICKS_ACCESS_TOKEN/DATABRICKS_TOKEN": self.connection_params["access_token"],
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(f"Por favor configure las variables de entorno: {', '.join(missing)}")

        conn_kwargs: Dict[str, Any] = {
            "server_hostname": self.connection_params["server_hostname"],
            "http_path": self.connection_params["http_path"],
            "access_token": self.connection_params["access_token"],
            "_user_agent_entry": "FT3",
        }
        if self.connection_params["catalog"]:
            conn_kwargs["catalog"] = self.connection_params["catalog"]
        if self.connection_params["schema"]:
            conn_kwargs["schema"] = self.connection_params["schema"]

        try:
            self.conn = databricks_sql.connect(**conn_kwargs)
            self.cursor = self.conn.cursor()
            logger.info("Successfully connected to Databricks SQL")
        except Exception as e:
            logger.error(f"Failed to connect to Databricks SQL: {str(e)}")
            raise

    def disconnect(self):
        """Close Databricks SQL connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Disconnected from Databricks SQL")

    def execute_query(self, query: str, chunk_size: int = 500000) -> pd.DataFrame:
        """Execute a query and return results as DataFrame.

        Args:
            query: SQL query to execute
            chunk_size: Number of rows to fetch at a time (default 500k)

        Returns:
            DataFrame with query results
        """
        if self.conn is None:
            raise RuntimeError("No hay conexión activa. Ejecute connect() primero.")

        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)

            columns = [desc[0] for desc in (cursor.description or [])]
            if not columns:
                logger.info("Query executed successfully, returned no resultset")
                return pd.DataFrame()

            chunks: List[pd.DataFrame] = []
            total_rows = 0

            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                chunk_df = pd.DataFrame(rows, columns=columns)
                chunks.append(chunk_df)
                total_rows += len(chunk_df)
                logger.info(f"Loaded chunk: {len(chunk_df)} rows (total so far: {total_rows})")

            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Query executed successfully, returned {len(df)} rows")
                return df

            logger.info("Query returned no rows")
            return pd.DataFrame(columns=columns)

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
    
    def load_training_data(self, 
                          table_name: str = "MODELO_LM_202507_TRAIN",
                          date_from: Optional[str] = None,
                          date_to: Optional[str] = None,
                          limit: Optional[int] = None,
                          filter_compin: bool = True,
                          chunk_size: int = 200000,
                          include_leak_ft: bool = False) -> pd.DataFrame:
        """
        Load training data from Snowflake
        
        Args:
            table_name: Name of the table to load from
            date_from: Start date for filtering (YYYY-MM-DD)
            date_to: End date for filtering (YYYY-MM-DD)
            limit: Number of rows to limit (for testing)
            filter_compin: (Deprecated) This parameter is kept for backward compatibility but no longer filters by TARGET_MODIFICA
            chunk_size: Number of rows to fetch at a time (default 200k)
        
        Returns:
            DataFrame with training data
        """
        # Import variable groups from Variables_cat_train.py
        import sys
        import os
        import importlib
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Force reload to get latest changes
        if 'Variables_cat_train' in sys.modules:
            import Variables_cat_train
            importlib.reload(Variables_cat_train)
        
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
        fechas = [v for v in fechas if 'LEAK' not in v]
        categoricas = [v for v in categoricas if 'LEAK' not in v]
        numericas = [v for v in numericas if 'LEAK' not in v]
        binarias = [v for v in binarias if 'LEAK' not in v]
        texto_no_estructurado = [v for v in texto_no_estructurado if 'LEAK' not in v]
        
        # Define variables to EXCLUDE
        variables_to_exclude = leak  # Exclude all leak variables
        
        # Define variables to KEEP (excluding leak and only keeping the selected target)
        target_col = self.config['data']['target']  # Should be TARGET_RATIFICA
        
        # Get all feature columns (excluding leak variables)
        feature_cols = []
        
        # Add all variable groups except leak and target
        all_variables = llaves + fechas + categoricas + numericas + binarias + texto_no_estructurado
        
        # Filter out any leak variables that might be in the lists
        feature_cols = [col for col in all_variables if col not in variables_to_exclude]
        
        # Add target columns we need
        if filter_compin:
            # Need target column
            all_cols = feature_cols + [target_col]
        else:
            all_cols = feature_cols + [target_col]
        
        # Include LEAK_FT if requested
        if include_leak_ft and 'LEAK_FT' not in all_cols:
            all_cols.append('LEAK_FT')
        
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
        
        # Exclude postnatal cases (PARTO and PUERPERIO) and null/empty CIE_GRUPO
        query += " AND CIE_GRUPO NOT IN ('PARTO', 'PUERPERIO')"
        query += " AND CIE_GRUPO IS NOT NULL"
        query += " AND TRIM(CIE_GRUPO) != ''"
        logger.info("Filtering: Excluding PARTO and PUERPERIO cases and null/empty CIE_GRUPO")
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info(f"Loading data with query for {len(all_cols)} columns...")
        logger.info(f"Excluding {len(leak)} leak variables")
        logger.info(f"Using target: {target_col}")
        logger.info(f"Using chunk size: {chunk_size:,} rows")
        
        df = self.execute_query(query, chunk_size=chunk_size)
        
        # Basic data validation
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Remove TARGET_MODIFICA column if present as it's not needed for training
        if 'TARGET_MODIFICA' in df.columns:
            df = df.drop(columns=['TARGET_MODIFICA'])
        
        logger.info(f"Target distribution: {df[target_col].value_counts().to_dict()}")
        
        # Double-check no leak variables are present (except LEAK_FT if explicitly included)
        leak_cols_present = [col for col in df.columns if col in leak]
        if leak_cols_present:
            logger.warning(f"WARNING: Leak columns found in data: {leak_cols_present}")
            # Only drop leak columns if not explicitly requested
            if not include_leak_ft or 'LEAK_FT' not in leak_cols_present:
                df = df.drop(columns=leak_cols_present)
            else:
                # Keep LEAK_FT if requested, drop others
                leak_cols_to_drop = [col for col in leak_cols_present if col != 'LEAK_FT']
                if leak_cols_to_drop:
                    df = df.drop(columns=leak_cols_to_drop)
        
        return df
    
    def load_prediction_data(self, 
                           table_name: str,
                           id_column: str,
                           id_values: List[str]) -> pd.DataFrame:
        """Load data for prediction based on IDs"""
        # Import variable groups from Variables_cat_train.py
        import sys
        import os
        import importlib
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Force reload to get latest changes
        if 'Variables_cat_train' in sys.modules:
            import Variables_cat_train
            importlib.reload(Variables_cat_train)
        
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
        fechas = [v for v in fechas if 'LEAK' not in v]
        categoricas = [v for v in categoricas if 'LEAK' not in v]
        numericas = [v for v in numericas if 'LEAK' not in v]
        binarias = [v for v in binarias if 'LEAK' not in v]
        texto_no_estructurado = [v for v in texto_no_estructurado if 'LEAK' not in v]
        
        # Define variables to EXCLUDE (leak and all targets)
        variables_to_exclude = leak + target
        
        # Get all feature columns (excluding leak and target variables)
        all_variables = llaves + fechas + categoricas + numericas + binarias + texto_no_estructurado
        
        # Filter out any leak or target variables
        feature_cols = [col for col in all_variables if col not in variables_to_exclude]
        
        # Remove duplicates while preserving order
        seen = set()
        feature_cols = [x for x in feature_cols if not (x in seen or seen.add(x))]
        
        # Build query
        id_list = "', '".join(id_values)
        query = f"""
        SELECT {', '.join(feature_cols)}
        FROM {table_name}
        WHERE {id_column} IN ('{id_list}')
        """
        
        logger.info(f"Loading prediction data for {len(id_values)} IDs")
        logger.info(f"Loading {len(feature_cols)} features (excluding {len(leak)} leak vars and {len(target)} target vars)")
        
        df = self.execute_query(query)
        
        # Double-check no leak or target variables are present
        excluded_cols_present = [col for col in df.columns if col in variables_to_exclude]
        if excluded_cols_present:
            logger.warning(f"WARNING: Excluded columns found in prediction data: {excluded_cols_present}")
            df = df.drop(columns=excluded_cols_present)
        
        return df
    
    def save_predictions(self, 
                        predictions_df: pd.DataFrame,
                        table_name: str = "MODEL_PREDICTIONS"):
        """Save predictions back to Databricks SQL."""
        if self.conn is None:
            raise RuntimeError("No hay conexión activa. Ejecute connect() primero.")
        if predictions_df.empty:
            logger.info("No predictions to save")
            return

        try:
            columns = list(predictions_df.columns)
            placeholders = ", ".join(["?"] * len(columns))
            insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

            records: List[Tuple[Any, ...]] = []
            for row in predictions_df.itertuples(index=False, name=None):
                normalized = []
                for value in row:
                    if pd.isna(value):
                        normalized.append(None)
                    elif isinstance(value, pd.Timestamp):
                        normalized.append(value.to_pydatetime())
                    elif hasattr(value, "item"):
                        normalized.append(value.item())
                    else:
                        normalized.append(value)
                records.append(tuple(normalized))

            cursor = self.conn.cursor()
            try:
                batch_size = 1000
                for idx in range(0, len(records), batch_size):
                    batch = records[idx:idx + batch_size]
                    cursor.executemany(insert_sql, batch)
                try:
                    self.conn.commit()
                except Exception:
                    # Some DB-API drivers autocommit; keep compatibility.
                    pass
            finally:
                cursor.close()

            logger.info(f"Saved {len(predictions_df)} predictions to {table_name}")
        except Exception as e:
            logger.error(f"Failed to save predictions: {str(e)}")
            raise


# Backwards compatibility: keep historical class name used across the codebase.
SnowflakeDataLoader = DatabricksDataLoader


def load_data_for_training(config_path: str = "config.yaml",
                          date_from: Optional[str] = "2022-01-01",
                          date_to: Optional[str] = "2025-09-01",
                          limit: Optional[int] = None,
                          filter_compin: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to load training data
    
    Args:
        config_path: Path to configuration file
        date_from: Start date for filtering (YYYY-MM-DD)
        date_to: End date for filtering (YYYY-MM-DD)
        limit: Number of rows to limit (for testing)
        filter_compin: If True, only load licenses that went to COMPIN (TARGET_MODIFICA = 1)
    
    Returns:
        Tuple of (DataFrame, config dict)
    """
    loader = SnowflakeDataLoader(config_path)
    
    try:
        # Establish connection
        loader.connect()
        
        # Load data from Databricks SQL
        df = loader.load_training_data(
            table_name="MODELO_LM_202507_TRAIN",  # Correct table name
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            filter_compin=filter_compin
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
    
    # Check if .env exists
    if not os.path.exists('../.env'):
        print("Please create a .env file with your Databricks credentials")
        print("Copy .env.example to .env and fill in your credentials")
        exit(1)
    
    try:
        # Test loading a small sample
        df, config = load_data_for_training(limit=1000)
        print(f"Successfully loaded {len(df)} rows")
        print(f"Columns: {len(df.columns)}")
        print(f"Target column: {config['data']['target']}")
        print(f"Target distribution:\n{df[config['data']['target']].value_counts()}")
        
        # Check for leak variables
        leak_cols_found = [col for col in df.columns if 'LEAK' in col]
        if leak_cols_found:
            print(f"WARNING: Found leak columns: {leak_cols_found}")
        else:
            print("✓ No leak columns found in data")
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
