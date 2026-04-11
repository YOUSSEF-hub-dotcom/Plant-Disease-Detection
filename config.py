import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv(dotenv_path=".ENV")

class Settings:
    """
    Centralized configuration management for the Plant Disease Platform.
    Environment variables are pulled from .env for security and flexibility.
    """
    
    # --- 1. CORE CONNECTIVITY ---
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    MODEL_URI: str = os.getenv("MODEL_URI")
    # Convert comma-separated string to list for CORS configuration
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # --- 2. PROJECT SPECIFICATIONS ---
    PROJECT_NAME: str = "Plant Disease Intelligence Platform"
    # ResNet50 standard input resolution
    IMG_SIZE: tuple = (224, 224) 
    
    # --- 3. CLASS REGISTRY (38 TARGET CLASSES) ---
    # Maps model index outputs to human-readable plant/disease labels
    CLASS_NAMES: list = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
        "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
        "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
        "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
        "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
    ]

    def __init__(self):
        """Self-validation to ensure critical variables are present at startup."""
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL is missing in .env file")
        if not self.MODEL_URI:
            raise ValueError("MODEL_URI is missing in .env file")

# Global settings instance
settings = Settings()
