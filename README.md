# MLOps Assignment 2 - Cats vs Dogs Classification

This project implements an end-to-end MLOps pipeline for a binary image classification model (Cats vs Dogs).

## Project Structure
```
c:/dev/mlops
├── .github/workflows/    # CI/CD pipelines
├── app/                  # FastAPI inference service
│   ├── main.py
├── src/                  # Source code for training
│   ├── train.py
│   ├── model.py
## Tech Stack
- **Language**: Python 3.11
- **Web Framework**: FastAPI
- **ML Framework**: TensorFlow / Keras (MobileNetV2)
- **Experiment Tracking**: MLflow
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus
- **Version Control**: Git + Git LFS

## Architecture

### Application Flow
```mermaid
graph TD
    User[User] -->|POST Image| API[FastAPI /predict]
    API --> Preprocess[Preprocess Image (MobileNetV2)]
    Preprocess --> Model[TensorFlow Model]
    Model -->|Prediction| API
    API -->|JSON Response| User
    API -->|Metrics| Prometheus[Prometheus /metrics]
```

### CI/CD Pipeline
```mermaid
graph LR
    Dev[Developer] -->|Push Code| GitHub[GitHub Repository]
    GitHub -->|Trigger| CI[CI Pipeline (.github/workflows/pipeline.yml)]
    subgraph CI Functions
        Checkout[Checkout Code + LFS] --> Setup[Setup Python]
        Setup --> Install[Install Dependencies]
        Install --> Test[Run Tests]
        Test --> Build[Build Docker Image]
        Build --> Push[Push to Docker Hub]
    end
    Push -->|Deploy| Server[Production Server]
```

## Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Git
- Dataset: `PetImages` (Cats & Dogs) placed in `Dataset/` folder.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd mlops
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Dataset:**
   Ensure your dataset is structure as follows:
   ```
   c:/dev/mlops/Dataset/PetImages/
       ├── Cat/
       └── Dog/
   ```

## Model Development

1. **Train the model:**
   ```bash
   python -m src.train
   ```
   This will train the CNN model and save it as `model.h5`. MLflow logs will be saved in `mlruns/`.

2. **Run Tests:**
   ```bash
   # Run unit tests
   python -m pytest
   ```

## Model Serving

1. **Start the API locally:**
   ```bash
   uvicorn app.main:app --reload
   ```
   - Swagger UI: `http://localhost:8000/docs`
   - Metrics: `http://localhost:8000/metrics`

2. **Docker Deployment:**
   ```bash
   # Build and Run using Docker Compose
   docker-compose up --build -d
   ```

3. **Verify Deployment:**
   ```bash
   # Run smoke tests
   python smoke_test.py
   
   # Simulate traffic for monitoring
   python simulate_traffic.py
   ```

## CI/CD Configuration

- **Workflow**: `.github/workflows/pipeline.yml`
- **CI**: 
  - Verification of code quality with `pytest`
  - Git LFS handling for model artifacts
  - Docker image build
- **Artifacts**: Model (`model.h5`) is uploaded as a workflow artifact.
- **CD**: 
  - Deployment to Docker Hub/GHCR on push to `main`.
  - Deployment trigger via Docker Compose.
- **Secrets Required**:
  - `DOCKER_USERNAME`
  - `DOCKER_PASSWORD`

## Monitoring

- **Application Logs**: Standard output logs for every request.
- **Prometheus Metrics**: `http://localhost:8000/metrics` exposes:
  - Request Count
  - Latency
  - Custom business metrics (via `prometheus-fastapi-instrumentator`)
  
  **To view metrics:**
  1. Open a browser and navigate to `http://localhost:8000/metrics`.
  2. You will see raw metrics. For visualization, you would typically configure a Prometheus server to scrape this endpoint and use Grafana for dashboards.
  3. Simple verification: Refresh the page after making predictions to see counters increase.
