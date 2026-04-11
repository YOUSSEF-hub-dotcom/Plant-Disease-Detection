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

# Internal Module Imports
from logger_config import setup_logging
from data_load import configure_gpu

# Initialize Logging Environment
setup_logging()
logger = logging.getLogger("MLOpsPipeline")

# --- 1. MLFLOW CUSTOM PYFUNC WRAPPER ---
class PlantDiseaseWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow wrapper to ensure seamless production deployment.
    Handles artifact loading and standardized inference logic.
    """
    def load_context(self, context):
        # Load the Keras model from the provided artifacts path
        self.model = tf.keras.models.load_model(context.artifacts["keras_model"])
        logger.info("✅ Production Model loaded into custom MLflow wrapper.")

    def predict(self, context, model_input):
        # Standardized prediction interface for downstream applications (API/Dashboard)
        return self.model.predict(model_input)

# --- 2. MULTI-STAGE TRAINING LOGIC ---
def train_plant_model(train_ds, val_ds, params):
    """
    Executes a two-stage training strategy:
    Stage 1: Frozen backbone (ResNet50) + Custom Head training.
    Stage 2: Selective unfreezing (Fine-tuning) of the last 50 layers.
    """
    # Initialize ResNet50 Backbone with ImageNet weights
    base_model = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = False # Initial freezing for Phase 1

    # Custom Classification Head for 38 plant disease classes
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4), # High dropout to prevent overfitting on specific crops
        layers.Dense(38, activation='softmax')
    ])

    # Performance Metrics for Agricultural Diagnosis
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

    # STAGE 1: Head Initialization
    logger.info("🚀 Stage 1: Initializing custom head training (Top Layers Only)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr_stage1"]),
        loss='categorical_crossentropy', metrics=metrics
    )

    # Callbacks for Training Stability
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6, verbose=1
    )

    model.fit(
        train_ds, validation_data=val_ds, 
        epochs=params["epochs_stage1"], callbacks=[early_stopping, reduce_lr]
    )

    # STAGE 2: Deep Fine-Tuning
    logger.info("🔓 Stage 2: Unfreezing last 50 layers for specialized feature extraction...")
    base_model.trainable = True
    # Freeze the early layers (primitive features like edges/colors)
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr_stage2"]),
        loss='categorical_crossentropy', metrics=metrics
    )
    
    history = model.fit(
        train_ds, validation_data=val_ds, 
        epochs=params["epochs_stage2"], callbacks=[early_stopping, reduce_lr]
    )
    
    return model, history

# --- 3. MLOPS ORCHESTRATION & REGISTRY WORKFLOW ---
def run_full_mlops_lifecycle(train_ds, val_ds, test_ds, lr_stage1, lr_stage2, epochs_stage1, epochs_stage2):
    """
    Main Orchestrator: Training -> Evaluation -> Quality Gate -> Registry Transition.
    """
    params = {
        "lr_stage1": lr_stage1, "lr_stage2": lr_stage2,
        "epochs_stage1": epochs_stage1, "epochs_stage2": epochs_stage2,
        "quality_gate": 0.80,  # 80% accuracy required for Production status
        "model_name": "PlantDisease_Production_Model"
    }

    configure_gpu()
    mlflow.set_experiment("Plant_Disease_Intelligence")

    with mlflow.start_run(run_name="Professional_Production_Run") as run:
        run_id = run.info.run_id
        mlflow.log_params(params)
        logger.info(f"Active MLflow Tracking: {run_id}")

        # [A] Execute Training Pipeline
        model, history = train_plant_model(train_ds, val_ds, params)

        # [B] Comprehensive Model Evaluation
        logger.info("📊 Running final test set evaluation...")
        results = model.evaluate(test_ds)
        test_loss, test_acc, test_precision, test_recall = results[0:4]

        # Calculate F1-Score for balanced performance verification
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
        
        # Log all metrics to MLflow for historical tracking
        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": f1_score
        })

        # [C] Artifact Generation (Visual Reports)
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='train_acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.title('Accuracy Evolution (Head Training + Fine-Tuning)')
        plt.savefig("accuracy_evolution.png")
        mlflow.log_artifact("accuracy_evolution.png")

        # [D] Model Packaging (Python Function Format)
        sample_img = next(iter(test_ds))[0][:1].numpy()
        signature = infer_signature(sample_img, model.predict(sample_img))
        model_path = "plant_disease_final.keras"
        model.save(model_path)
        
        mlflow.pyfunc.log_model(
            artifact_path="plant_model_bundle",
            python_model=PlantDiseaseWrapper(),
            artifacts={"keras_model": model_path},
            signature=signature
        )

        # --- [E] ENTERPRISE REGISTRY WORKFLOW ---
        client = mlflow.tracking.MlflowClient()
        model_uri = f"runs:/{run_id}/plant_model_bundle"
        model_name = params["model_name"]

        # Step 1: Automated Registration
        logger.info(f"📦 Step 1: Registering model in Central Registry: {model_name}")
        model_details = mlflow.register_model(model_uri, model_name)
        version = model_details.version

        # Step 2: Transition to STAGING (Testing phase)
        logger.info(f"🧪 Step 2: Moving Version {version} to STAGING...")
        client.transition_model_version_stage(
            name=model_name, version=version, stage="Staging"
        )

        # Step 3: Deployment Quality Gate (Logical Assessment)
        logger.info(f"⚖️ Step 3: Verifying Quality Gate (Acc >= {params['quality_gate']})")
        
        if test_acc >= params["quality_gate"] and f1_score >= 0.75:
            # Step 4: Full Promotion to PRODUCTION
            logger.info(f"✅ Quality Gate Passed! Accuracy: {test_acc:.2%}")
            logger.info(f"🚀 Step 4: Promoting version {version} to PRODUCTION (LIVE)...")
            
            client.transition_model_version_stage(
                name=model_name, version=version, stage="Production",
                archive_existing_versions=True # Safely archive older production models
            )
            logger.info(f"🌟 Model v{version} is now handling LIVE requests.")
        else:
            logger.warning(f"⚠️ Quality Gate Failed (Acc: {test_acc:.2%}). Model remains in STAGING.")
            
        return run_id
