import tensorflow as tf
from tensorflow.keras import layers
import logging

# Get the logger (configured in main.py)
logger = logging.getLogger("DataAugmentation")

def apply_augmentation(train_ds):
    """
    Defines the augmentation layers and applies them to the training dataset.
    Augmentation is only applied to training data to improve model robustness.
    """
    try:
        logger.info("🛠️ Initializing heavy data augmentation layers...")

        # 1. Define the augmentation logic
        # 
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.3),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
        ], name="heavy_augmentation_layer")

        # 2. Apply augmentation to the training dataset
        # Note: training=True is essential for layers like Dropout or RandomFlip 
        # to ensure they only work during training.
        logger.info("Applying augmentation to the training pipeline...")
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # 3. Re-apply prefetch for performance optimization
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        logger.info("✅ Data Augmentation successfully added to the training pipeline!")
        return train_ds

    except Exception as e:
        logger.error(f"❌ Failed to apply data augmentation: {str(e)}")
        raise