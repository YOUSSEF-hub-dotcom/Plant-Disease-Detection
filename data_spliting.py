import os
import shutil
import logging
from sklearn.model_selection import train_test_split
from logger_config import setup_logging # تأكد من اسم ملف اللوجر بتاعك

# --- 1. INITIALIZE LOGGING ---
setup_logging()
logger = logging.getLogger("DataSplitting")

# --- 2. CONFIGURATION ---
base_path = "/home/youssef/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/3/plantvillage dataset/color"
base_dir = '/home/youssef/plant_disease_split' 
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

def create_split_folders():
    logger.info("Starting data splitting process...")
    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def copy_images(img_list, source, destination, class_name):
    dest_class_path = os.path.join(destination, class_name)
    if not os.path.exists(dest_class_path):
        os.makedirs(dest_class_path)
    for img_name in img_list:
        shutil.copy(os.path.join(source, img_name), os.path.join(dest_class_path, img_name))

def run_splitting():
    if not os.path.exists(base_path):
        logger.error(f"Source path missing: {base_path}")
        return

    create_split_folders()
    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    logger.info(f"Found {len(classes)} classes to split.")

    for class_name in classes:
        class_path = os.path.join(base_path, class_name)
        images = os.listdir(class_path)
        
        # Split: 80% Train, 20% Temp
        train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=42)
        # Split Temp: 50% Val, 50% Test (10% each of total)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
        
        copy_images(train_imgs, class_path, train_dir, class_name)
        copy_images(val_imgs, class_path, val_dir, class_name)
        copy_images(test_imgs, class_path, test_dir, class_name)
        
        logger.info(f"Successfully split class: {class_name} | Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

    logger.info("✅ SUCCESS: Data splitting complete. (80/10/10 ratio)")

if __name__ == "__main__":
    run_splitting()