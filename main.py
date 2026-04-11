import argparse
import logging
import os
from logger_config import setup_logging
from data_load import configure_gpu
from data_spliting import run_splitting
from image_process import prepare_datasets
from data_augmentation import apply_augmentation
from model_pipeline import run_full_mlops_lifecycle 

# 1. Initialize Logging Environment
setup_logging()
logger = logging.getLogger("MainPipeline")

def main():
    """
    Main Entry Point for the Plant Disease MLOps Pipeline.
    Manages Environment Setup, Data Splitting, Preprocessing, and Model Lifecycle.
    """
    try:
        # --- PHASE 1: HYPERPARAMETER PARSING ---
        parser = argparse.ArgumentParser(description="Plant Disease Detection Training Pipeline")
        
        # Hyperparameters for Transfer Learning Stages
        parser.add_argument("--lr_stage1", type=float, default=0.0001, help="Learning rate for Phase 1 (Top Layers Only)")
        parser.add_argument("--lr_stage2", type=float, default=0.00005, help="Learning rate for Phase 2 (Fine-tuning)")
        parser.add_argument("--epochs_stage1", type=int, default=10, help="Epochs for initial training")
        parser.add_argument("--epochs_stage2", type=int, default=40, help="Epochs for full model fine-tuning")
        
        args = parser.parse_args()

        logger.info("🎬 Initializing the Full Plant Disease MLOps Pipeline...")

        # --- PHASE 2: ENVIRONMENT & RAW DATA VERIFICATION ---
        # Ensure GPU is recognized and memory growth is managed
        configure_gpu()
        
        # Path for the base PlantVillage dataset (Color version)
        RAW_DATA_PATH = "/home/youssef/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3/plantvillage dataset/color"
        
        if not os.path.exists(RAW_DATA_PATH):
             logger.error(f"Dataset path error: {RAW_DATA_PATH} does not exist.")
             raise FileNotFoundError(f"Source dataset not found at {RAW_DATA_PATH}")

        # --- PHASE 3: DATA ARCHITECTURE & SPLITTING ---
        # Execute automated splitting into Train, Val, and Test folders
        logger.info("📂 Executing Data Splitting into stratified directories...")
        run_splitting() 
        
        # Define the structure for processed splits
        SPLIT_BASE = '/home/youssef/plant_disease_split'
        TRAIN_DIR = os.path.join(SPLIT_BASE, 'train')
        VAL_DIR = os.path.join(SPLIT_BASE, 'val')
        TEST_DIR = os.path.join(SPLIT_BASE, 'test')

        # --- PHASE 4: PREPROCESSING & DATA AUGMENTATION ---
        logger.info("⚙️ Commencing Image Preprocessing (ResNet50 Specialization)...")
        
        # Ingest datasets and extract class names (38 classes)
        train_ds, val_ds, test_ds, class_names = prepare_datasets(
            TRAIN_DIR, VAL_DIR, TEST_DIR, 
            img_size=(224, 224), 
            batch_size=16
        )
        
        # Apply Data Augmentation only to the Training set to prevent overfitting
        logger.info("🎨 Applying Real-time Data Augmentation to Training Pipeline...")
        train_ds = apply_augmentation(train_ds)

        # --- PHASE 5: MLOPS LIFECYCLE (MLFLOW & TRAINING) ---
        logger.info("🚀 Launching Integrated MLflow Lifecycle Governance...")
        
        # Execute training with automated tracking, metric logging, and model registration
        run_id = run_full_mlops_lifecycle(
            train_ds=train_ds, 
            val_ds=val_ds, 
            test_ds=test_ds,
            lr_stage1=args.lr_stage1,
            lr_stage2=args.lr_stage2,
            epochs_stage1=args.epochs_stage1,
            epochs_stage2=args.epochs_stage2
        )

        logger.info(f"✅ Pipeline Successfully Completed! Final Run ID: {run_id}")
        logger.info("🔗 Action Required: Run 'mlflow ui' in your terminal to review results and quality gates.")

    except Exception as e:
        # High-severity logging for pipeline crashes
        logger.critical(f"💥 Pipeline Execution Failed: {str(e)}", exc_info=True)
        # Signal failure for any connected automation tools
        exit(1)

if __name__ == "__main__":
    main()
