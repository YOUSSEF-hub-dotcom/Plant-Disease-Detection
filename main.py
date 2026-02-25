import argparse
import logging
import os
from logger_config import setup_logging
from data_load import configure_gpu, verify_dataset
from data_spliting import run_splitting
from image_process import prepare_datasets
from data_augmentation import apply_augmentation
# تأكد من اسم الدالة واسم الملف
from model_pipeline import run_full_mlops_lifecycle 

setup_logging()
logger = logging.getLogger("MainPipeline")

def main():
    try:
        # --- استلام البارامترات ---
        parser = argparse.ArgumentParser()
        parser.add_argument("--lr_stage1", type=float, default=0.0001)
        parser.add_argument("--lr_stage2", type=float, default=0.00005)
        parser.add_argument("--epochs_stage1", type=int, default=10)
        parser.add_argument("--epochs_stage2", type=int, default=40)
        args = parser.parse_args()

        logger.info("🎬 Starting the Full Plant Disease MLOps Pipeline...")

        # --- المرحلة 1: تهيئة البيئة وفحص الداتا ---
        configure_gpu()
        RAW_DATA_PATH = "/home/youssef/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3/plantvillage dataset/color"
        
        if not os.path.exists(RAW_DATA_PATH):
             raise FileNotFoundError(f"Source dataset not found at {RAW_DATA_PATH}")

        # --- المرحلة 2: تقسيم الداتا ---
        logger.info("📂 Splitting data...")
        run_splitting() 
        
        SPLIT_BASE = '/home/youssef/plant_disease_split'
        TRAIN_DIR = os.path.join(SPLIT_BASE, 'train')
        VAL_DIR = os.path.join(SPLIT_BASE, 'val')
        TEST_DIR = os.path.join(SPLIT_BASE, 'test')

        # --- المرحلة 3: المعالجة ---
        logger.info("⚙️ Preprocessing images...")
        # تأكد إن prepare_datasets بترجع الـ 4 قيم دول بالترتيب
        train_ds, val_ds, test_ds, class_names = prepare_datasets(
            TRAIN_DIR, VAL_DIR, TEST_DIR, 
            img_size=(224, 224), 
            batch_size=16
        )
        
        train_ds = apply_augmentation(train_ds)

        # --- المرحلة 4: MLOps Lifecycle ---
        logger.info("🚀 Launching MLflow Lifecycle...")
        
        # نمرر القيم للدالة في model_pipeline
        run_id = run_full_mlops_lifecycle(
            train_ds=train_ds, 
            val_ds=val_ds, 
            test_ds=test_ds,
            lr_stage1=args.lr_stage1,
            lr_stage2=args.lr_stage2,
            epochs_stage1=args.epochs_stage1,
            epochs_stage2=args.epochs_stage2
        )

        logger.info(f"✅ Pipeline Finished! Run ID: {run_id}")
        logger.info("🔗 Run 'mlflow ui' to see results.")

    except Exception as e:
        logger.critical(f"💥 Pipeline failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()