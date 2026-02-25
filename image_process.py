import tensorflow as tf
import logging
from tensorflow.keras.applications.resnet50 import preprocess_input

logger = logging.getLogger("ImageProcess")

def prepare_datasets(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    """
    تحميل الداتا، عمل الـ Normalization الخاص بـ ResNet، وتجهيز الـ Pipeline.
    """
    logger.info("🚀 Loading datasets from split directories...")

    loader_params = {
        "image_size": img_size,
        "batch_size": batch_size,
        "label_mode": 'categorical'
    }

    try:
        # 1. تحميل الداتا الخام (الأسماء موجودة هنا)
        raw_train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, **loader_params)
        raw_val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, **loader_params)
        raw_test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, **loader_params)
        
        # 2. استخراج الأسماء وحفظها "الآن" قبل التحويل
        class_names = raw_train_ds.class_names
        logger.info(f"✅ Successfully loaded classes: {class_names}")

    except Exception as e:
        logger.error(f"❌ Error loading datasets: {e}")
        raise

    # 3. Applying ResNet50 Preprocessing
    AUTOTUNE = tf.data.AUTOTUNE
    
    # تحويل الداتا وتطبيق الـ Preprocessing
    train_ds = raw_train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = raw_val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    test_ds = raw_test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

    # 4. Performance Optimization
    # الـ shuffle هنا للـ Train بس
    # 1. الـ Shuffle: خليه 100 لو الرام تسمح، لو خايف خليها 50 (أفضل من 20 عشان التنوع)
    train_ds = train_ds.shuffle(buffer_size=100) 
    
    # 2. الـ Prefetch: أهم تعديل، هنخليه يحضر 2 batches بس في الذاكرة
    # ده بيمنع الـ Terminal من حجز مساحة صور عملاقة في الرام
    train_ds = train_ds.prefetch(buffer_size=2)
    val_ds = val_ds.prefetch(buffer_size=2)
    test_ds = test_ds.prefetch(buffer_size=2)

    logger.info("✅ Preprocessing & Prefetching Complete!")
    
    # نرجع الـ class_names اللي خزناها في الخطوة رقم 2
    return train_ds, val_ds, test_ds, class_names