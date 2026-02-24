# COMPIN LightGBM Predictive Model

A Python implementation of the COMPIN submission prediction model using LightGBM with comprehensive model auditing capabilities.

## Overview

This project predicts whether COMPIN (medical commission) submissions will be denied (`Deniega`) or approved (`Ratifica`) using machine learning. It includes:

- Snowflake data integration
- Comprehensive feature engineering
- LightGBM model with hyperparameter tuning
- Model auditing with custom implementation using standard libraries
- Production-ready pipeline

## Project Structure

```
python_model/
├── config.yaml              # Main configuration file
├── requirements.txt         # Python dependencies
├── main.py                 # Main pipeline script
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Snowflake data loading
│   ├── feature_engineering.py  # Feature transformation
│   ├── model_training.py   # LightGBM training
│   └── model_auditor.py    # Model auditing
├── models/                 # Saved models
├── results/                # Training results
├── reports/                # Audit reports
└── logs/                   # Application logs
```

## Installation

1. Clone the repository and navigate to the python_model directory:
```bash
cd /Users/alfil/Mi\ unidad/LALO/python_model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Cursor Cloud / Agent Setup

To preinstall the complete FT3 runtime (including Databricks connector and ML stack) on a cloud agent image, run:

```bash
bash scripts/setup_cloud_env.sh
```

Recommended startup command for cloud agents:

```bash
/bin/bash /workspace/scripts/setup_cloud_env.sh
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Snowflake credentials
```

## Configuration

The `config.yaml` file contains all project settings:

- **Model parameters**: LightGBM hyperparameters and tuning settings
- **Data configuration**: Feature groups, target variable, train/test split
- **Auditing settings**: Thresholds, protected attributes, drift detection
- **Snowflake connection**: Database, schema, warehouse settings

## Usage

### Training a Model

Run the complete training pipeline:

```bash
# With hyperparameter tuning (default)
python main.py --mode train

# Without hyperparameter tuning (faster)
python main.py --mode train --no-tuning

# With custom config
python main.py --mode train --config custom_config.yaml
```

### Making Predictions

Predict on new data using license IDs:

```bash
python main.py --mode predict --ids LIC001 LIC002 LIC003
```

### Analyzing Feature Information Value

Run a detailed Information Value analysis:

```bash
# Analyze all data
python analyze_iv.py

# Analyze a sample
python analyze_iv.py --limit 10000
```

This will generate:
- Information Value for all features
- Predictive power categorization
- Weight of Evidence patterns
- Visualizations of top features

### Running Individual Components

You can also run components separately:

```python
# Load data
from src.data_loader import load_data_for_training
df, config = load_data_for_training()

# Engineer features (with Information Value calculation)
from src.feature_engineering import engineer_features
X, y = engineer_features(df, fit=True, calculate_iv=True)

# Train model
from src.model_training import train_model
trainer, metrics = train_model(X_train, y_train, X_test, y_test)

# Audit model
from src.model_auditor import run_full_audit
report = run_full_audit(model, X_train, y_train, X_test, y_test)
```

## Features

### Data Features

The model uses 200+ features organized into categories:

- **Categorical** (21): Gender, diagnosis codes, medical specialties, locations
- **Numeric** (157): Days calculations, counts, rates, historical metrics
- **Binary** (24): Flags for alerts, fraud indicators, specific conditions
- **Text** (2): Diagnosis and clinical history (processed with TF-IDF)
- **Dates** (8): Converted to numeric features (year, month, day, etc.)

### Model Auditing

The model auditing module provides:

1. **Data Quality Checks**
   - Missing value analysis
   - Data type validation
   - Statistical summaries

2. **Model Performance**
   - AUC, precision, recall, F1 score
   - Confusion matrix
   - Threshold violation detection

3. **Fairness Analysis**
   - Performance across protected attributes (gender, age)
   - Disparity ratio calculations
   - 80% rule compliance

4. **Feature Importance**
   - LightGBM feature importance
   - SHAP value analysis
   - Top feature identification

5. **Data Drift Detection**
   - Kolmogorov-Smirnov test
   - Feature-level drift detection
   - Drift summary statistics

## Model Performance

Expected performance metrics (based on configuration thresholds):
- AUC: > 0.70
- Precision: > 0.60
- Recall: > 0.60
- F1 Score: > 0.60

## Output Files

- **models/peritaje_model.pkl**: Trained model and metadata
- **models/peritaje_model_feature_importance.csv**: Feature importance from LightGBM
- **models/feature_peritaje.pkl**: Fitted feature transformers
- **models/feature_iv_analysis.csv**: Information Value analysis for all features
- **results/training_results_*.csv**: Training run summaries
- **reports/audit_report.json**: Comprehensive audit results
- **reports/feature_iv_plot.png**: Top features by Information Value
- **reports/woe_pattern_*.png**: Weight of Evidence patterns for top features
- **reports/*.png**: Performance visualizations

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/

# Type checking
mypy src/
```

## Troubleshooting

1. **Snowflake Connection Issues**
   - Verify credentials in .env file
   - Check network connectivity
   - Ensure proper role/warehouse permissions

2. **Memory Issues**
   - Reduce data sample size in config.yaml
   - Use fewer features for text vectorization
   - Enable chunking for large datasets

3. **Model Performance**
   - Check data quality audit results
   - Review feature importance
   - Adjust hyperparameter search space

## Future Enhancements

- Real-time prediction API
- Model versioning and experiment tracking
- Advanced text analysis with transformers
- Automated retraining pipeline
- Dashboard for monitoring model performance

## License

This project is proprietary to Colmena Isapre.