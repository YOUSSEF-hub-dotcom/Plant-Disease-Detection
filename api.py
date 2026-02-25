import io
import time
import logging
import numpy as np
import mlflow.pyfunc
from datetime import datetime
from PIL import Image

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

from contextlib import asynccontextmanager


# --- 1. CONFIGURATION & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("production_app.log"), logging.StreamHandler()]
)
logger = logging.getLogger("PlantAPI")

# --- 2. DATABASE SETUP (SQL SERVER) ---
Base = declarative_base()
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    prediction = Column(String(100))
    confidence = Column(Float)
    latency = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# إنشاء الجداول
Base.metadata.create_all(bind=engine)

# --- 3. MODEL LOADER (MLflow) ---
def load_production_model():
    try:
        logger.info(f"📡 Pulling Model from Registry: {settings.MODEL_URI}")
        model = mlflow.pyfunc.load_model(settings.MODEL_URI)
        logger.info("✅ Production Model Loaded Successfully.")
        return model
    except Exception as e:
        logger.error(f"❌ Critical Error Loading Model: {str(e)}")
        raise RuntimeError(f"Could not load model from MLflow: {e}")

# --- 3.5 WARM-UP FUNCTION ---
def model_warmup(model):
    """إرسال صورة وهمية للموديل لتجهيز الذاكرة والـ Cache و Kernels الـ GPU"""
    try:
        logger.info("🔥 Starting GPU/CPU Warm-up (Cold Start Mitigation)...")
        # إنشاء صورة عشوائية بنفس أبعاد ResNet50
        dummy_input = np.random.uniform(0, 255, (1, *settings.IMG_SIZE, 3)).astype(np.float32)
        # استخدام المعالجة الخاصة بـ ResNet50 لتنشيط الـ GPU Kernels
        dummy_input = preprocess_input(dummy_input)
        # تنفيذ توقع وهمي - هنا الـ 19 ثانية هتحصل والسيرفر بيقوم
        model.predict(dummy_input)
        logger.info("⚡ Warm-up Complete. System is now highly responsive!")
    except Exception as e:
        logger.error(f"⚠️ Warm-up failed, but server will continue: {e}")

# --- 4. FASTAPI APP INITIALIZATION ---

# [مهم جداً] تعريف الـ Lifespan لإدارة التشغيل
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. تحميل الموديل عند بداية التشغيل
    global production_model
    production_model = load_production_model()
    
    # 2. تشغيل الـ Warm-up فوراً
    model_warmup(production_model)
    
    yield
    # كود يتنفذ عند إغلاق السيرفر
    logger.info("🛑 Shutting down PlantAPI...")

app = FastAPI(
    title=settings.PROJECT_NAME, 
    lifespan=lifespan  # دي اللي بتفعل كل الكلام اللي فوق
)
# Middle & CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS, 
    allow_credentials=True,                
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response



# Dependency للـ DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- 5. RATE LIMITING ---

# Rate limiting key function that checks for user ID in JWT or falls back to IP address
def get_smart_identifier(request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get("sub")
            if user_id:
                return f"user:{user_id}"
        except:
            pass
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=get_smart_identifier)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- 6. ENDPOINTS ---

@app.get("/health")
def health():
    return {"status": "healthy", "model": settings.MODEL_URI, "time": datetime.now()}

@app.post("/predict")
@limiter.limit("10/minute")
# لازم نضيف request هنا عشان الـ limiter يشتغل صح
async def predict_disease(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    start_time = time.time()
    
    # [A] Validation: تأكد من نوع الملف
    if not file.content_type.startswith("image/"):
        logger.warning(f"🚫 Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        from tensorflow.keras.applications.resnet50 import preprocess_input

# [B] Image Preprocessing (The Correct Way for ResNet50)
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(settings.IMG_SIZE)

# 1. تحويل الصورة لـ Numpy Array بدون قسمة يدوية
        img_array = np.array(image)

# 2. إضافة بُعد الـ Batch (Batch Dimension)
        img_array = np.expand_dims(img_array, axis=0)

# 3. استخدام المعالجة الخاصة بـ ResNet50 (دي اللي هتحل المشكلة)
        img_array = preprocess_input(img_array)

        # [C] Inference
        # استخدام الموديل اللي حملناه من MLflow
        # ملحوظة: بما إننا شغالين pyfunc، الـ predict ممكن ترجع DataFrame أو Numpy
        predictions = production_model.predict(img_array)
        
        # لو الموديل بيرجع probabilities
        idx = np.argmax(predictions)
        confidence = float(np.max(predictions))
        result_label = settings.CLASS_NAMES[idx]

        # [D] Latency & Monitoring
        latency = time.time() - start_time
        
        # [E] Persistence (Logging to SQL Server)
        new_log = PredictionLog(
            filename=file.filename,
            prediction=result_label,
            confidence=confidence,
            latency=latency
        )
        db.add(new_log)
        db.commit()
        db.refresh(new_log)

        logger.info(f"✨ Prediction: {result_label} | Confidence: {confidence:.2f} | Latency: {latency:.4f}s")

        return {
            "id": new_log.id,
            "filename": new_log.filename,
            "prediction": result_label,
            "confidence": round(confidence, 4),
            "latency": f"{latency:.4f} sec",
            "detected_at": new_log.created_at
        }

    except Exception as e:
        db.rollback()
        logger.error(f"💥 Pipeline Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing image.")

@app.get("/")
def health():
    return {"status": "API is running 🚀"}


if __name__ == "__main__":
    import uvicorn
    # تشغيل السيرفر
    uvicorn.run(app, host="0.0.0.0", port=8000)