# 🌿 Plant Disease Intelligence System  
## Production-Grade Computer Vision & MLOps Pipeline (ResNet50 + MLflow + FastAPI)

An end-to-end **Computer Vision system** for multi-class plant disease classification (38 classes) built with a production mindset.

This project goes beyond training a CNN model. It implements a complete **Deep Learning + MLOps lifecycle**, including:

- GPU configuration & memory control
- Exploratory Data Analysis (EDA)
- Severe class imbalance handling
- Transfer Learning with ResNet50
- Two-stage fine-tuning strategy
- MLflow experiment tracking & model registry
- Automated Quality Gate promotion logic
- FastAPI production deployment
- SQL Server logging & monitoring
- Streamlit interactive dashboard

---

# 📌 Problem Statement

Plant diseases significantly impact agricultural productivity.  
The objective of this system is to:

> Automatically classify plant leaf diseases across 38 categories with high reliability, robustness, and production readiness.

Dataset: PlantVillage (38 disease classes)

---

# 🔍 Exploratory Data Analysis (EDA)

### Key Findings:

- ✅ High-quality images (lab-controlled backgrounds)
- ⚠️ Severe Class Imbalance detected  
  - Imbalance Ratio: **36.23 : 1**
- ⚠️ Risk of overfitting due to clean backgrounds

### Strategic Decisions:

1. Heavy Data Augmentation to simulate real-world farm noise
2. Oversampling minority classes via augmentation
3. Two-stage transfer learning for stable convergence

---

# 🧠 Model Architecture

### Backbone:
- **ResNet50 (ImageNet pretrained)**
- Input Size: (224 × 224 × 3)

### Custom Head:
- GlobalAveragePooling2D
- BatchNormalization
- Dense(256, ReLU)
- Dropout(0.4)
- Dense(38, Softmax)

---

# 🚀 Training Strategy (Professional Two-Stage Fine-Tuning)

### Stage 1 – Head Training
- Frozen ResNet50 backbone
- Learning Rate: 1e-4
- Focus: Train classification head

### Stage 2 – Fine-Tuning
- Unfreeze last 50 layers
- Lower learning rate (5e-5)
- Improve domain-specific feature extraction

### Optimization Tools:
- EarlyStopping (patience=10)
- ReduceLROnPlateau (factor=0.3)
- GPU memory growth control
- Prefetch optimization
- Controlled shuffle buffering

---

# 🛠 Data Pipeline Engineering

### Data Splitting
- 80% Training
- 10% Validation
- 10% Testing
- Stratified per class

### Image Preprocessing
- ResNet50 `preprocess_input`
- TensorFlow data pipelines
- Prefetch (2 batches to prevent RAM overflow)

### Heavy Data Augmentation
- RandomFlip
- RandomRotation
- RandomZoom
- RandomTranslation
- RandomContrast
- RandomBrightness

Purpose:
- Handle imbalance
- Improve generalization
- Reduce overfitting on lab-style backgrounds

---

# 📊 Final Model Performance

| Metric | Score |
|--------|--------|
| Test Accuracy | **0.834** |
| Test Precision | **0.84** |
| Test Recall | **0.83** |
| Test F1-Score | **0.83** |

Quality Gate Threshold:
- Accuracy ≥ 0.80
- F1-Score ≥ 0.80

✅ Model Successfully Promoted to Production.

---

# 🔁 Full MLOps Lifecycle (MLflow)

This project implements a complete ML lifecycle:

- Experiment Tracking
- Parameter Logging
- Metric Logging
- Artifact Storage (Accuracy & Precision reports)
- Model Packaging (Custom PyFunc Wrapper)
- Model Registry Workflow:
  1. Register
  2. Transition to Staging
  3. Quality Gate Check
  4. Auto-Promotion to Production

---

# 🌐 Production API (FastAPI)

Features:

- Model pulled directly from MLflow Registry
- GPU Warm-up (Cold Start Mitigation)
- Image validation & preprocessing
- JWT-aware Rate Limiting
- SQL Server logging
- Latency tracking
- Health monitoring endpoint

### Endpoints:
- `/predict`
- `/health`

---

# 🗄 Database & Monitoring

Every prediction logs:

- Filename
- Predicted class
- Confidence score
- Latency
- Timestamp

Stored in:
- SQL Server

Enables:
- Analytics dashboard
- Latency trend tracking
- Disease frequency monitoring

---

# 📊 Interactive Dashboard (Streamlit)

Features:

- Upload leaf image
- Real-time prediction
- Confidence visualization
- Latency display
- Analytics dashboard
- Disease frequency charts
- System performance metrics

---

# ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- ResNet50 (Transfer Learning)
- MLflow
- FastAPI
- Streamlit
- SQL Server
- Pandas / NumPy / Matplotlib
- JWT Authentication
- Rate Limiting (SlowAPI)

---

# 🎯 Engineering Highlights

- Production-grade GPU configuration
- Cold-start mitigation (Model Warm-up)
- Automated model governance
- Modular pipeline architecture
- Registry-based deployment
- Real-time monitoring & logging
- Full-stack AI system (Model + API + DB + UI)
- Observability: Logging, latency monitoring, prediction persistence.

---

# 🚀 Future Improvements

- Test-Time Augmentation (TTA)
- Grad-CAM visualization
- Model Quantization for edge deployment
- CI/CD pipeline integration
- Kubernetes containerization

---

# 👨‍💻 Author Mindset

Built with a **Computer Vision Engineer + MLOps mindset**,  
focusing on:

- Scalability
- Reliability
- Deployment readiness
- Real-world agricultural applicability
- 
## 👨‍💻 Author

**Youssef Mahmoud**
Faculty of Computers & Information
Aspiring **Data Scientist / ML Engineer**

---
URL Linked in : [https://www.linkedin.com/in/youssef-mahmoud-63b243361?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BKBSoRAFOSyucvi6vDlDfbg%3D%3D]
⭐ If you like this project, consider giving it a star on GitHub!
---

🌿 Intelligent Farming Starts with Intelligent Models.
