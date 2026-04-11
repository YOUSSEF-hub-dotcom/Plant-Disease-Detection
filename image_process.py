import tensorflow as tf
import logging
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("DataPipeline")

def prepare_datasets(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    """
    Loads images, applies ResNet50 specialized preprocessing, and optimizes 
    the data pipeline for high-performance training.
    """
    logger.info("🚀 Initiating data loading from split directories...")

    loader_params = {
        "image_size": img_size,
        "batch_size": batch_size,
        "label_mode": 'categorical' # Required for multi-class classification (38 classes)
    }

    try:
        # 1. Ingest Raw Data from Directories
        # This step automatically infers labels based on folder structure
        raw_train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, **loader_params)
        raw_val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, **loader_params)
        raw_test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, **loader_params)
        
        # 2. Extract and Cache Class Names
        # Crucial to capture these before any mapping or transformation
        class_names = raw_train_ds.class_names
        logger.info(f"✅ Successfully identified {len(class_names)} classes.")

    except Exception as e:
        logger.error(f"❌ Dataset Loading Failure: {e}")
        raise

    # 3. Apply ResNet50 Specialized Preprocessing
    # ResNet50 requires Zero-centering and RGB-to-BGR conversion (handled by preprocess_input)
    AUTOTUNE = tf.data.AUTOTUNE
    
    def apply_resnet_preprocessing(images, labels):
        """Map function to apply specific ResNet50 scaling."""
        return preprocess_input(images), labels

    # Apply the transformation across the entire pipeline using parallel calls
    train_ds = raw_train_ds.map(apply_resnet_preprocessing, num_parallel_calls=AUTOTUNE)
    val_ds = raw_val_ds.map(apply_resnet_preprocessing, num_parallel_calls=AUTOTUNE)
    test_ds = raw_test_ds.map(apply_resnet_preprocessing, num_parallel_calls=AUTOTUNE)

    # 4. Memory & Performance Optimization (Prefetching Strategy)
    # Strategy: Maintain a small buffer to prevent RAM exhaustion during high-res training
    
    # Shuffle only the training set for better generalization
    train_ds = train_ds.shuffle(buffer_size=100) 
    
    # Prefetching: Prepares the next batch while the current batch is training on GPU
    # Buffer_size=2 balances speed and RAM usage (Cold-Start mitigation)
    train_ds = train_ds.prefetch(buffer_size=2)
    val_ds = val_ds.prefetch(buffer_size=2)
    test_ds = test_ds.prefetch(buffer_size=2)

    logger.info("✅ Data Pipeline Optimization: Shuffling and Prefetching Active.")
    
    return train_ds, val_ds, test_ds, class_names
