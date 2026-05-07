# Churn Prediction API

This is a FastAPI-based service designed to serve machine learning predictions for customer churn. It is optimized for deployment on **Google Cloud Run**.

## Features

- **FastAPI**: High performance, easy to use, and automatic Swagger documentation.
- **Dockerized**: Ready for containerized deployment.
- **Health Check**: Endpoint to verify service and model status.
- **Pydantic Validation**: Automatic request body validation.

## Local Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your model file in `model/churn_model.pkl`.

4. Run the API:
   ```bash
   uvicorn main:app --reload --port 8080
   ```

5. Open your browser at `http://localhost:8080/docs` to see the interactive API documentation.

## Docker Usage

### Build the image
```bash
docker build -t churn-api .
```

### Run the container
```bash
docker run -p 8080:8080 churn-api
```

## Cloud Run Deployment

1. Build and push to Google Container Registry (GCR) or Artifact Registry:
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/churn-api
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy churn-api --image gcr.io/YOUR_PROJECT_ID/churn-api --platform managed
   ```

## Endpoints

- `GET /`: Welcome message.
- `GET /health`: Check if the API and model are healthy.
- `POST /predict`: Submit customer data to get a churn prediction.
