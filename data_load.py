import os
import tensorflow as tf
from logger_config import setup_logging # تأكد من اسم الملف اللي فيه كود اللوجينج بتاعك
import logging

# --- 1. INITIALIZE LOGGING ---
setup_logging()
logger = logging.getLogger("DataLoad")

# --- 2. ENVIRONMENT CONFIGURATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

logger.info(f"--- System Test ---")
logger.info(f"TensorFlow Version: {tf.__version__}")

# --- 3. GPU DETECTION & MEMORY GROWTH ---
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logger.info(f"✅ SUCCESS: GPU Detected and Enforced -> {gpus[0]}")
        except RuntimeError as e:
            logger.error(f"⚠️ GPU Runtime Error: {e}")
    else:
        logger.warning("❌ ERROR: No GPU detected. Check WSL CUDA drivers.")
    
    logger.info(f"Detected Devices: {tf.config.list_physical_devices()}")

# --- 4. DATASET VERIFICATION ---
def verify_dataset(base_path):
    logger.info("--- Dataset Verification ---")
    if os.path.exists(base_path):
        all_classes = os.listdir(base_path)
        all_classes.sort()
        logger.info(f"✅ Success! Path exists.")
        logger.info(f"✅ Total Classes Found: {len(all_classes)}")
        logger.info(f"Sample Classes (First 10): {all_classes[:10]}")
        return all_classes
    else:
        logger.critical(f"❌ Critical Error: Path not found at: {base_path}")
        # Debugging parent path
        parent_path = os.path.dirname(base_path)
        if os.path.exists(parent_path):
            logger.debug(f"Available directories in parent path: {os.listdir(parent_path)}")
        return None

# تنفيذ الفحص
BASE_PATH = "/home/youssef/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3/plantvillage dataset/color"

configure_gpu()
classes = verify_dataset(BASE_PATH)