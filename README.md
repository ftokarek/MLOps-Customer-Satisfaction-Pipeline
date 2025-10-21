# MLOps Customer Satisfaction Pipeline

End-to-end machine learning pipeline for predicting customer satisfaction scores. Built with ZenML for orchestration, MLflow for experiment tracking and model deployment, and Streamlit for interactive predictions.

## Features

- Automated data ingestion and preprocessing
- Model training with MLflow experiment tracking
- Continuous deployment pipeline with accuracy-based triggering
- Real-time predictions via REST API
- Interactive web interface for model inference
- Complete MLOps workflow with versioning and reproducibility

## Tech Stack

- **Orchestration**: ZenML
- **Experiment Tracking**: MLflow
- **Model Deployment**: MLflow Model Deployer
- **Web Interface**: Streamlit
- **ML Framework**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Data Processing**: Pandas, NumPy

## Prerequisites

- Python 3.12+
- Virtual environment (recommended)
- macOS, Linux, or Windows

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MLOps-Customer-Satisfaction-Pipeline.git
cd MLOps-Customer-Satisfaction-Pipeline
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize ZenML:
```bash
zenml init
```

## Usage

### 1. Start ZenML Server

```bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  # macOS only
zenml login --local
```

Access ZenML Dashboard at: `http://127.0.0.1:8237`

### 2. Set Up MLflow Stack

```bash
zenml experiment-tracker register mlflow_tracker_customer --flavor=mlflow
zenml model-deployer register mlflow_customer --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -e mlflow_tracker_customer -d mlflow_customer
zenml stack set mlflow_stack
```

### 3. Run Training Pipeline

```bash
python run_pipeline.py
```

### 4. Deploy Model

```bash
python run_deployment.py --min-accuracy 0.0
```

### 5. Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

Access the application at: `http://localhost:8501`

## Project Structure

```
MLOps-Customer-Satisfaction-Pipeline/
├── data/                          # Dataset storage
├── pipelines/                     # ZenML pipeline definitions
│   ├── training_pipeline.py       # Model training pipeline
│   ├── deployment_pipeline.py     # Deployment and inference pipelines
│   └── utils.py                   # Helper functions
├── steps/                         # Pipeline steps
│   ├── ingest_data.py            # Data ingestion
│   ├── clean_data.py             # Data preprocessing
│   ├── model_train.py            # Model training
│   ├── evaluation.py             # Model evaluation
│   └── config.py                 # Configuration classes
├── src/                          # Core logic
│   ├── data_cleaning.py          # Data cleaning strategies
│   ├── model_dev.py              # Model implementations
│   └── evaluation.py             # Evaluation metrics
├── materializer/                 # Custom ZenML materializers
├── streamlit_app.py              # Web interface
├── run_pipeline.py               # Training pipeline runner
├── run_deployment.py             # Deployment pipeline runner
└── requirements.txt              # Project dependencies
```

## Pipeline Workflows

### Training Pipeline
1. Data ingestion from CSV
2. Data cleaning and preprocessing
3. Model training with MLflow tracking
4. Model evaluation (MSE, RMSE, R2)

### Deployment Pipeline
1. Execute training pipeline
2. Evaluate model performance
3. Deploy model if accuracy threshold is met
4. Start MLflow prediction server

### Inference Pipeline
1. Load test data
2. Retrieve deployed model service
3. Generate predictions

## Model Performance

Current deployed model: LinearRegression
- MSE: 1.864
- RMSE: 1.365
- R2: 0.018

## Configuration

Adjust deployment settings in `run_deployment.py`:
- `--min-accuracy`: Minimum accuracy threshold for deployment (default: 0.92)
- `--config`: Deployment mode (deploy, predict, deploy_and_predict)

## API Endpoints

MLflow prediction server runs at: `http://127.0.0.1:8000/invocations`

## Monitoring

View experiments and runs:
```bash
mlflow ui
```

Access MLflow UI at: `http://localhost:5000`

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

Contributions are welcome. Please ensure code follows existing patterns and includes appropriate tests.

