import os
import logging
import json
import mlflow
import mlflow.keras
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from mlflow.models.signature import infer_signature
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# استدعاء الملفات الخارجية
from logger_config import setup_logging
from data_load import configure_gpu

setup_logging()
logger = logging.getLogger("FullPipeline")

# --- 1. MLflow Custom Wrapper ---
class PlantDiseaseWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = tf.keras.models.load_model(context.artifacts["keras_model"])
        logger.info(" Production Model Loaded into Wrapper.")

    def predict(self, context, model_input):
        return self.model.predict(model_input)

# --- 2. Training Logic (Architecture & Stages) ---
def train_plant_model(train_ds, val_ds, params):
    base_model = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(38, activation='softmax')
    ])

    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

    # Stage 1: Head Training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr_stage1"]),
        loss='categorical_crossentropy', metrics=metrics
    )

# Stop training if validation loss stops improving
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Dynamically reduce learning rate when the model hits a plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )

    logger.info(" Stage 1: Training the Head...")
    model.fit(train_ds, validation_data=val_ds, epochs=params["epochs_stage1"], callbacks=[early_stopping, reduce_lr])

    # Stage 2: Fine-Tuning
    logger.info(" Stage 2: Unfreezing last 50 layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr_stage2"]),
        loss='categorical_crossentropy', metrics=metrics
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=params["epochs_stage2"], 
                        callbacks=[early_stopping, reduce_lr])
    
    return model, history

# --- 3. Full MLOps Lifecycle Orchestrator ---
def run_full_mlops_lifecycle(train_ds, val_ds, test_ds,lr_stage1, lr_stage2, epochs_stage1, epochs_stage2):
    params = {
        "lr_stage1": lr_stage1,
        "lr_stage2":lr_stage2,
        "epochs_stage1": epochs_stage1,
        "epochs_stage2": epochs_stage2,
        "batch_size": 32,
        "quality_gate": 0.80,  # 80% accuracy threshold for production
        "model_name": "PlantModel_Prod"
    }

    configure_gpu()
    mlflow.set_experiment("Plant_Disease_Intelligence")

    with mlflow.start_run(run_name="Professional_Production_Run") as run:
        run_id = run.info.run_id
        mlflow.log_params(params)
        logger.info(f"Started MLflow Run: {run_id}")

        # [A] تحضير البيانات
        """
        run_splitting()
        train_ds, val_ds, test_ds, classes = prepare_datasets(
            '/home/youssef/plant_disease_split/train',
            '/home/youssef/plant_disease_split/val',
            '/home/youssef/plant_disease_split/test'
        )
        train_ds = apply_augmentation(train_ds)
        """

        # [B] التدريب
        model, history = train_plant_model(train_ds, val_ds, params)

        """
        # [C] التقييم والرسوم البيانية
        test_loss, test_acc = model.evaluate(test_ds)
        mlflow.log_metric("test_accuracy", test_acc)
        """
        

        # 1. استلام النتائج كلها في متغير واحد (قائمة)
        results = model.evaluate(test_ds)

# 2. تفكيك النتائج وتسميتها
        test_loss = results[0]
        test_acc = results[1]
        test_precision = results[2]
        test_recall = results[3]

# 3. تسجيل كل المقاييس في MLflow
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)

# 4. (إضافي ومهم جداً) حساب الـ F1-Score وتسجيله
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
        mlflow.log_metric("test_f1_score", f1_score)

# طباعة النتائج في الـ Terminal عشان تتابعها
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {f1_score:.4f}")

        # رسم الـ Precision والـ Recall للحفظ في الـ MLflow
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['precision'], label='train_precision')
        plt.plot(history.history['val_precision'], label='val_precision')
        plt.title('Precision Evolution')
        plt.savefig("precision_report.png")
        mlflow.log_artifact("precision_report.png")


        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='train_acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.title('Accuracy Evolution')
        plt.savefig("accuracy_report.png")
        mlflow.log_artifact("accuracy_report.png")

        # [D] التغليف (Packaging)
        sample_img = next(iter(test_ds))[0][:1].numpy()
        signature = infer_signature(sample_img, model.predict(sample_img))
        model_temp_path = "final_plant_model.keras"
        model.save(model_temp_path)
        
        mlflow.pyfunc.log_model(
            artifact_path="plant_disease_model",
            python_model=PlantDiseaseWrapper(),
            artifacts={"keras_model": model_temp_path},
            signature=signature
        )

        # --- 4. الـ Registry Workflow اللي طلبته بالترتيب ---
        
        client = mlflow.tracking.MlflowClient()
        model_uri = f"runs:/{run_id}/plant_disease_model"
        model_name = params["model_name"]

        # 1. Registration
        logger.info(f" Step 1: Registering model '{model_name}'...")
        model_details = mlflow.register_model(model_uri, model_name)
        version = model_details.version

        # 2. Transition to Staging
        logger.info(f" Step 2: Transitioning version {version} to STAGING...")
        client.transition_model_version_stage(
            name=model_name, version=version, stage="Staging"
        )

        # 3. Quality Gate
        logger.info(f" Step 3: Checking Quality Gate (Target: {params['quality_gate']*100}%)...")
        
        if test_acc >= params["quality_gate"] and f1_score >= 0.80:
            # 4. Transition to Production
            logger.info(f" Quality Gate Passed! (Accuracy: {test_acc:.4f})")
            logger.info(f" Step 4: Promoting version {version} to PRODUCTION...")
            
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True
            )
            logger.info(f" Model version {version} is now LIVE in Production.")
        else:
            logger.warning(f" Quality Gate Failed (Accuracy: {test_acc:.4f}).")
            logger.warning(f" Model version {version} will remain in STAGING for review.")
        return run_id

if __name__ == "__main__":
    run_full_mlops_lifecycle()