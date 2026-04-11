import io
import time
import logging
import numpy as np
import mlflow.pyfunc
from datetime import datetime
from PIL import Image
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request 
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from tensorflow.keras.applications.resnet50 import preprocess_input
import jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import settings

# --- 1. LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("plant_disease_api.log"), logging.StreamHandler()]
)
logger = logging.getLogger("PlantDiseaseAPI")

# --- 2. DATABASE PERSISTENCE (SQL SERVER) ---
Base = declarative_base()
try:
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database engine initialized.")
except Exception as e:
    logger.critical(f"Database connection failed: {e}")

class PredictionLog(Base):
    """Table to store inference history and performance metrics."""
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    prediction = Column(String(100))
    confidence = Column(Float)
    latency = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# Ensure schema exists
Base.metadata.create_all(bind=engine)

# --- 3. MODEL MANAGEMENT (MLflow & Warm-up) ---

def load_production_model():
    """Retrieves the registered ResNet50 model from MLflow Registry."""
    try:
        logger.info(f"📡 Fetching model from Registry: {settings.MODEL_URI}")
        model = mlflow.pyfunc.load_model(settings.MODEL_URI)
        logger.info("✅ Production model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"❌ Critical Error: Could not load model: {str(e)}")
        raise RuntimeError(f"MLflow model loading failed: {e}")

def model_warmup(model):
    """
    Executes a dummy inference to initialize GPU/CPU kernels.
    Mitigates latency spikes during the first real request.
    """
    try:
        logger.info("🔥 Starting Model Warm-up (Inference Cold-Start Mitigation)...")
        # Generate random noise matching ResNet50 input shape
        dummy_input = np.random.uniform(0, 255, (1, *settings.IMG_SIZE, 3)).astype(np.float32)
        dummy_input = preprocess_input(dummy_input)
        
        # Trigger first prediction
        model.predict(dummy_input)
        logger.info("⚡ Warm-up complete. System is highly responsive.")
    except Exception as e:
        logger.warning(f"⚠️ Warm-up failed, but server will continue: {e}")

# --- 4. LIFESPAN MANAGEMENT ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    global production_model
    # Load and prep model before accepting requests
    production_model = load_production_model()
    model_warmup(production_model)
    
    yield
    logger.info("🛑 Shutting down Plant Disease API...")

app = FastAPI(
    title="Plant Disease Intelligence Platform", 
    version="1.0.0",
    lifespan=lifespan
)

# --- 5. MIDDLEWARE & SECURITY ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS, 
    allow_credentials=True,                
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def track_latency(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    response.headers["X-Inference-Latency"] = f"{time.time() - start_time:.4f}s"
    return response

def get_smart_identifier(request: Request):
    """Identifies users for rate limiting via JWT 'sub' or IP address."""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, options={"verify_signature": False})
            return f"user:{payload.get('sub')}"
        except:
            pass
    return f"ip:{get_remote_address(request)}"

limiter = Limiter(key_func=get_smart_identifier)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 6. INFERENCE ENDPOINTS ---

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_disease(
    request: Request, 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """
    Predicts plant disease from an uploaded image.
    Uses ResNet50 preprocessing and logs results to SQL Server.
    """
    start_time = time.time()
    
    # [A] Content Validation
    if not file.content_type.startswith("image/"):
        logger.warning(f"Rejected non-image upload: {file.content_type}")
        raise HTTPException(status_code=400, detail="Only image files are supported.")

    try:
        # [B] ResNet50 Specialized Preprocessing
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(settings.IMG_SIZE)
        
        # Transform to NumPy array
        img_array = np.array(image)
        # Expand dimensions to create Batch (1, H, W, C)
        img_array = np.expand_dims(img_array, axis=0)
        # Apply ResNet50 specific scaling/normalization
        img_array = preprocess_input(img_array.astype(np.float32))

        # [C] Model Inference
        # pyfunc predicts return results as NumPy or DataFrame
        predictions = production_model.predict(img_array)
        
        # Post-processing results
        idx = np.argmax(predictions)
        confidence = float(np.max(predictions))
        result_label = settings.CLASS_NAMES[idx]

        latency = time.time() - start_time
        
        # [D] Data Persistence
        new_log = PredictionLog(
            filename=file.filename,
            prediction=result_label,
            confidence=confidence,
            latency=latency
        )
        db.add(new_log)
        db.commit()
        db.refresh(new_log)

        logger.info(f"✨ Analysis Complete: {result_label} ({confidence:.2%}) | ID: {new_log.id}")

        return {
            "id": new_log.id,
            "filename": file.filename,
            "prediction": result_label,
            "confidence": round(confidence, 4),
            "latency": f"{latency:.4f} sec",
            "timestamp": new_log.created_at
        }

    except Exception as e:
        if 'db' in locals(): db.rollback()
        logger.error(f"Prediction Pipeline Failure: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal error occurred during image processing.")

@app.get("/health")
def health_check():
    """Monitors service and model registry status."""
    return {
        "status": "ready", 
        "model_version": settings.MODEL_URI, 
        "server_time": datetime.now()
    }

@app.get("/")
def root():
    return {"message": "Plant Disease Intelligence API is online. 🌿"}

if __name__ == "__main__":
    import uvicorn
    # Entry point for local development
    uvicorn.run(app, host="0.0.0.0", port=8000)
