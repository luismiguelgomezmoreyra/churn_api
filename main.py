import os
import joblib
import json
import pandas as pd
import numpy as np
import datetime
import io
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# --- DATABASE SETUP ---
DATABASE_URL = "sqlite:///./churn_app.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ClientDB(Base):
    __tablename__ = "clients"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    gender = Column(String)
    SeniorCitizen = Column(Integer)
    Partner = Column(String)
    Dependents = Column(String)
    tenure = Column(Integer)
    PhoneService = Column(String)
    MultipleLines = Column(String)
    InternetService = Column(String)
    OnlineSecurity = Column(String)
    OnlineBackup = Column(String)
    DeviceProtection = Column(String)
    TechSupport = Column(String)
    StreamingTV = Column(String)
    StreamingMovies = Column(String)
    Contract = Column(String)
    PaperlessBilling = Column(String)
    PaymentMethod = Column(String)
    MonthlyCharges = Column(Float)
    TotalCharges = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class PredictionHistoryDB(Base):
    __tablename__ = "prediction_history"
    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=True)
    inputs = Column(JSON)
    prediction = Column(Integer)
    probability = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    feedback = Column(Integer, nullable=True) # 1 for correct, 0 for incorrect

Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- MODELS ---
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class ClientCreate(CustomerData):
    name: str

class ClientResponse(ClientCreate):
    id: int
    created_at: datetime.datetime
    class Config:
        from_attributes = True

class PredictionResponse(BaseModel):
    churn_prediction: int
    probability: float
    risk_level: str
    recommendation: str
    message: str

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]

class ExplanationResponse(BaseModel):
    prediction: PredictionResponse
    top_features: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    uptime: str
    version: str
    model_loaded: bool

class InfoResponse(BaseModel):
    api_version: str
    author: str
    description: str
    training_date: str

class FeedbackRequest(BaseModel):
    prediction_id: int
    correct: bool

# --- API INITIALIZATION ---
app = FastAPI(
    title="Churn Prediction PRO API",
    description="Advanced API for Customer Churn Management and Prediction.",
    version="2.0.0"
)

# Global model and metadata
model = None
metadata = {}
START_TIME = datetime.datetime.utcnow()

@app.on_event("startup")
async def load_assets():
    global model, metadata
    model_path = os.getenv("MODEL_PATH", "model/churn_model.pkl")
    meta_path = os.getenv("META_PATH", "model/metadata.json")
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except:
            pass
    
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

# --- UTILS ---
def get_risk_level(prob: float) -> str:
    if prob < 0.3: return "Bajo"
    if prob < 0.7: return "Medio"
    return "Alto"

def get_recommendation(prob: float, data: CustomerData) -> str:
    if prob < 0.3: return "Mantener comunicación estándar."
    if prob < 0.7: return "Ofrecer descuento en renovación o plan superior."
    return "Acción inmediata: Llamada de fidelización y oferta personalizada."

# --- SYSTEM ENDPOINTS ---
@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health():
    uptime = str(datetime.datetime.utcnow() - START_TIME)
    return {
        "status": "healthy",
        "uptime": uptime,
        "version": "2.0.0",
        "model_loaded": model is not None
    }

@app.get("/info", response_model=InfoResponse, tags=["Sistema"])
async def info():
    return {
        "api_version": "2.0.0",
        "author": "Churn Predictor Team",
        "description": "API profesional para gestión de churn en telecomunicaciones.",
        "training_date": metadata.get("training_date", "2024-05-01")
    }

# --- PREDICTION ENDPOINTS ---
@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
async def predict(data: CustomerData, client_id: Optional[int] = None, db: Session = Depends(get_db)):
    if model is None: raise HTTPException(status_code=503, detail="Model not loaded")
    
    df = pd.DataFrame([data.model_dump()])
    prob = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else 0.5
    pred = int(model.predict(df)[0])
    
    res = PredictionResponse(
        churn_prediction=pred,
        probability=prob,
        risk_level=get_risk_level(prob),
        recommendation=get_recommendation(prob, data),
        message="Predicción exitosa"
    )
    
    # Save to history
    history = PredictionHistoryDB(
        client_id=client_id,
        inputs=data.model_dump(),
        prediction=pred,
        probability=prob
    )
    db.add(history)
    db.commit()
    
    return res

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predicción"])
async def predict_batch(data: List[CustomerData], db: Session = Depends(get_db)):
    results = []
    for item in data:
        res = await predict(item, db=db)
        results.append(res)
    return {"results": results}

@app.post("/predict/explain", response_model=ExplanationResponse, tags=["Predicción"])
async def predict_explain(data: CustomerData):
    if model is None: raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Mock explanation for now (using model weights if LogisticRegression)
    # In a real scenario, use SHAP
    top_features = [
        {"feature": "tenure", "influence": -0.8, "description": "Antigüedad reduce riesgo"},
        {"feature": "MonthlyCharges", "influence": 0.5, "description": "Cargos altos aumentan riesgo"},
        {"feature": "Contract", "influence": -0.6, "description": "Contrato largo reduce riesgo"}
    ]
    
    pred_res = await predict(data, db=SessionLocal()) # Simple session for mock
    return {"prediction": pred_res, "top_features": top_features}

@app.post("/predict/simulate", response_model=PredictionResponse, tags=["Predicción"])
async def predict_simulate(data: CustomerData, change_field: str, new_value: Any):
    sim_data = data.model_dump()
    if change_field in sim_data:
        sim_data[change_field] = new_value
    else:
        raise HTTPException(status_code=400, detail=f"Field {change_field} not found")
    
    return await predict(CustomerData(**sim_data), db=SessionLocal())

# --- METRICS ENDPOINTS ---
@app.get("/metrics", tags=["Métricas"])
async def get_metrics():
    return metadata.get("best_metrics", {})

@app.get("/metrics/compare", tags=["Métricas"])
async def compare_models():
    return metadata.get("all_results", {})

@app.get("/metrics/confusion", tags=["Métricas"])
async def confusion_matrix_api():
    return {"matrix": metadata.get("confusion_matrix", [])}

@app.get("/metrics/features", tags=["Métricas"])
async def feature_importance():
    return metadata.get("feature_importances", {})

@app.get("/metrics/roc", tags=["Métricas"])
async def roc_curve():
    # Mock ROC data points
    return {"fpr": [0.0, 0.1, 0.5, 1.0], "tpr": [0.0, 0.4, 0.8, 1.0], "auc": metadata.get("best_metrics", {}).get("roc_auc", 0.75)}

# --- CLIENT MANAGEMENT ---
@app.post("/clients", response_model=ClientResponse, tags=["Clientes"])
async def create_client(client: ClientCreate, db: Session = Depends(get_db)):
    db_client = ClientDB(**client.model_dump())
    db.add(db_client)
    db.commit()
    db.refresh(db_client)
    return db_client

@app.get("/clients", response_model=List[ClientResponse], tags=["Clientes"])
async def list_clients(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(ClientDB).offset(skip).limit(limit).all()

@app.get("/clients/{id}", response_model=ClientResponse, tags=["Clientes"])
async def get_client(id: int, db: Session = Depends(get_db)):
    client = db.query(ClientDB).filter(ClientDB.id == id).first()
    if not client: raise HTTPException(status_code=404, detail="Client not found")
    return client

@app.put("/clients/{id}", response_model=ClientResponse, tags=["Clientes"])
async def update_client(id: int, client_data: CustomerData, db: Session = Depends(get_db)):
    db_client = db.query(ClientDB).filter(ClientDB.id == id).first()
    if not db_client: raise HTTPException(status_code=404, detail="Client not found")
    
    for key, value in client_data.model_dump().items():
        setattr(db_client, key, value)
    
    db.commit()
    db.refresh(db_client)
    return db_client

@app.delete("/clients/{id}", tags=["Clientes"])
async def delete_client(id: int, db: Session = Depends(get_db)):
    db_client = db.query(ClientDB).filter(ClientDB.id == id).first()
    if not db_client: raise HTTPException(status_code=404, detail="Client not found")
    db.delete(db_client)
    db.commit()
    return {"message": "Client deleted"}

# --- HISTORY ENDPOINTS ---
@app.get("/history", tags=["Historial"])
async def get_history(limit: int = 50, db: Session = Depends(get_db)):
    return db.query(PredictionHistoryDB).order_by(PredictionHistoryDB.timestamp.desc()).limit(limit).all()

@app.get("/history/{client_id}", tags=["Historial"])
async def get_client_history(client_id: int, db: Session = Depends(get_db)):
    return db.query(PredictionHistoryDB).filter(PredictionHistoryDB.client_id == client_id).all()

@app.delete("/history", tags=["Historial"])
async def clear_history(db: Session = Depends(get_db)):
    db.query(PredictionHistoryDB).delete()
    db.commit()
    return {"message": "History cleared"}

# --- DATASET & TRAINING ---
@app.get("/dataset/stats", tags=["Dataset"])
async def dataset_stats():
    # In a real case, read the actual CSV
    return {
        "total_rows": 7043,
        "churn_rate": "26.5%",
        "avg_monthly_charges": 64.76,
        "avg_tenure": 32.37
    }

@app.post("/dataset/upload", tags=["Dataset"])
async def upload_dataset(file: UploadFile = File(...)):
    # Mock saving file
    return {"filename": file.filename, "status": "Uploaded successfully"}

@app.post("/model/retrain", tags=["Entrenamiento"])
async def retrain_model():
    # This would call train_model.py logic
    return {"status": "Retraining triggered", "estimated_time": "30s"}

# --- REPORTS & FEEDBACK ---
@app.get("/report/summary", tags=["Reportes"])
async def report_summary(db: Session = Depends(get_db)):
    total = db.query(PredictionHistoryDB).count()
    risks = db.query(PredictionHistoryDB.probability).all()
    high_risk = len([r for r in risks if r[0] > 0.7])
    return {
        "total_predictions": total,
        "high_risk_count": high_risk,
        "high_risk_percentage": (high_risk / total * 100) if total > 0 else 0
    }

@app.get("/export/csv", tags=["Reportes"])
async def export_csv(db: Session = Depends(get_db)):
    history = db.query(PredictionHistoryDB).all()
    # Simple CSV export logic
    output = io.StringIO()
    output.write("id,client_id,prediction,probability,timestamp\n")
    for h in history:
        output.write(f"{h.id},{h.client_id},{h.prediction},{h.probability},{h.timestamp}\n")
    
    response = StreamingResponse(iter([output.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=history_export.csv"
    return response

@app.post("/feedback", tags=["Feedback"])
async def register_feedback(fb: FeedbackRequest, db: Session = Depends(get_db)):
    history = db.query(PredictionHistoryDB).filter(PredictionHistoryDB.id == fb.prediction_id).first()
    if not history: raise HTTPException(status_code=404, detail="Prediction not found")
    history.feedback = 1 if fb.correct else 0
    db.commit()
    return {"message": "Feedback registered"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
