# MLOps Customer Satisfaction Pipeline

End-to-end machine learning pipeline for predicting customer satisfaction scores. Built with MLflow for experiment tracking and model deployment, ZenML for orchestration and Streamlit for interactive predictions.

## Features

- Automated data ingestion and preprocessing
- Model training with MLflow experiment tracking
- Continuous deployment pipeline with accuracy-based triggering
- Real-time predictions via REST API
- Interactive web interface for model inference
- Complete MLOps workflow with versioning and reproducibility

## Tech Stack

ZenML, MLflow, Streamlit, scikit-learn, Pandas, NumPy

## Prerequisites

- Python 3.12+
- Virtual environment (recommended)
- macOS, Linux or Windows

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
