# 🌿 Plant Disease Intelligence Platform

An end-to-end **computer vision and MLflow platform** for **automated crop disease diagnosis** using **ResNet50 transfer learning, MLflow lifecycle management, and production-oriented model governance**.

This project is designed to turn raw leaf images into **fast, scalable disease predictions** that can support early agricultural intervention, reduce dependence on manual expert review, and improve the speed of disease response in crop monitoring workflows.

---

## 📌 Project Overview

Plant disease diagnosis is still heavily dependent on **manual expert inspection**, which creates serious operational bottlenecks in agriculture:

* diagnosis can take **2–3 days**
* agronomy experts are limited and not always accessible
* delayed detection can significantly worsen crop damage
* rare diseases are often underrepresented and harder to detect
* image conditions vary in practice, making robust classification difficult

To address this, this project builds a **production-grade plant disease diagnosis platform** that classifies leaf images into **38 disease categories** using **ResNet50-based transfer learning**.

The system combines:

* **deep learning for image classification**
* **data augmentation to simulate field variability**
* **imbalance-aware training**
* **MLflow for experiment tracking and model lifecycle management**
* **quality-gated promotion for deployment readiness**

The result is a model that delivers **83.4% accuracy** across **54,305 images** with **sub-3-second inference**, enabling faster diagnosis and more scalable crop monitoring.

---

## 🎯 Problem Statement

Agricultural disease detection is a high-impact but difficult classification problem.

### Key challenges in the domain

* **expert dependency**: diagnosis often requires agronomists or plant pathology specialists
* **slow turnaround**: manual review delays intervention and treatment
* **high economic risk**: late detection can increase crop loss significantly
* **class imbalance**: some diseases appear much less frequently than others
* **visual similarity**: multiple diseases can produce similar leaf symptoms
* **real-world variability**: lighting, leaf angle, blur, and background noise affect generalization

This makes plant disease recognition more than just an image classification task — it is a **decision-support problem** where speed, robustness, and class balance matter.

---

## 💡 Solution

This project introduces a **Plant Disease Intelligence Platform** built around **ResNet50 transfer learning** and a **production-oriented ML workflow**.

The platform is designed to:

* classify leaf images into **38 plant disease classes**
* reduce diagnosis time from **days to seconds**
* handle class imbalance more effectively than naive training
* support reproducible experimentation through **MLflow**
* prepare the model for downstream deployment via **quality gates and registry promotion**

In practice, the system acts as a **first-line disease triage engine**:

* flagging likely disease classes quickly
* reducing expert review workload
* accelerating intervention for affected crops
* making disease screening more scalable

---

## 🧠 System Architecture

```text
Leaf Image
   ↓
Image Preprocessing + Resizing
   ↓
Data Augmentation Pipeline
   ↓
ResNet50 Transfer Learning Model
   ↓
Disease Class Probabilities (38 classes)
   ↓
MLflow Tracking + Quality Gate Validation
   ↓
Model Registry / Deployment Candidate
```

---

## 📊 Dataset

**Dataset:** PlantVillage
**Task Type:** Multi-class image classification
**Domain:** Plant pathology / agricultural disease diagnosis

### Dataset summary

* **54,305 images**
* **38 disease classes**
* **35:1 class imbalance** between the largest and smallest classes

The dataset contains labeled images of plant leaves representing multiple crops and disease categories. It provides a strong benchmark for supervised disease classification, while also introducing a practical challenge: **performance must remain useful across both common and rare diseases**.

---

## 🔍 Data Challenges & Modeling Considerations

This project is valuable from a **Data Science** perspective because the core challenge is not just model training — it is understanding and managing the structure of the data.

### 1) Severe class imbalance

Some disease classes are far more frequent than others, with a class ratio of approximately **35:1**.
Without imbalance-aware training, the model risks over-optimizing for common diseases while underperforming on rare but important cases.

### 2) Inter-class visual similarity

Different plant diseases can produce **visually overlapping symptoms** such as:

* yellowing
* brown spotting
* edge discoloration
* leaf texture damage

This makes the classification task harder than a standard object-recognition problem because the model must learn **fine-grained visual differences**.

### 3) Risk of weak real-world generalization

PlantVillage images are cleaner than many real agricultural images. In practice, field images may include:

* lighting changes
* shadows
* camera blur
* cluttered backgrounds
* partially damaged leaves

To reduce this gap, the training pipeline uses **heavy augmentation** to make the model less dependent on ideal image conditions.

---

## ⚙️ Modeling Approach

### Base Model

**ResNet50** pre-trained on ImageNet

### Why ResNet50?

ResNet50 was selected because it offers a strong balance between:

* high-quality visual feature extraction
* transfer learning effectiveness
* manageable training cost
* production practicality compared with heavier architectures

Rather than training from scratch, the project uses **transfer learning** so the model can leverage rich low-level and mid-level visual features learned from large-scale image data.

---

## 🏗️ Training Strategy

The model is trained in **two stages**, which is a common and effective strategy in transfer learning workflows.

### Stage 1 — Train the classification head

* freeze the ResNet50 backbone
* train the top classification layers
* allow the model to learn task-specific decision boundaries first

### Stage 2 — Fine-tune deeper visual features

* unfreeze the last **50 layers**
* continue training with a lower learning rate
* adapt higher-level visual representations to plant disease patterns

This approach is important because it helps preserve the useful general features of the pretrained backbone while still allowing the model to specialize for disease recognition.

---

## 🧪 Training Pipeline & Optimization Techniques

### Data preprocessing and pipeline

The project uses a **TensorFlow `tf.data` pipeline** to support efficient training and scalable preprocessing.

### Core pipeline components

* image loading and preprocessing
* batching
* prefetching with **AUTOTUNE**
* efficient streaming through the training loop

### Augmentation strategy

Heavy augmentation is used to simulate real-world visual variability, such as:

* rotation
* zoom / scale changes
* shifts / translation
* brightness or orientation variation

This is one of the most important parts of the project because it directly targets the biggest practical weakness of many agricultural image models: **fragility outside clean benchmark conditions**.

### Imbalance handling

The training process also includes explicit handling of class imbalance so that minority classes are not drowned out by dominant classes during optimization.

---

## 📈 Model Performance

| Metric             |           Score |
| ------------------ | --------------: |
| **Accuracy**       |       **83.4%** |
| **Precision**      |         **84%** |
| **Recall**         |         **83%** |
| **F1-Score**       |         **83%** |
| **Inference Time** | **< 3 seconds** |

### Interpretation

The performance is meaningful because it is achieved on a **38-class agricultural classification task** with substantial imbalance and fine-grained visual overlap between classes.

The project is not just optimizing for a single number — it is trying to balance:

* **predictive quality**
* **latency**
* **scalability**
* **deployment readiness**

---

## 🚀 Key Features

* **38-class plant disease classification**
* **ResNet50 transfer learning with fine-tuning**
* **Heavy data augmentation for robustness**
* **Imbalance-aware training strategy**
* **Fast inference (< 3s)**
* **MLflow experiment tracking and lifecycle management**
* **quality-gated model promotion**
* **reproducible training workflow**

---

## 🔬 Why This Project Demonstrates Data Science Skills

This repository is more than a CNN training exercise. It demonstrates several core Data Science competencies:

### 1) Problem framing

The project starts from a real operational problem:

> how to reduce the delay and cost of plant disease diagnosis while maintaining useful predictive quality across many disease classes.

### 2) Data understanding

It explicitly deals with:

* class imbalance
* fine-grained class overlap
* generalization risk between clean benchmark images and field conditions

### 3) Modeling rationale

The README and pipeline justify:

* why **transfer learning** is appropriate
* why **ResNet50** is a practical backbone
* why **two-stage fine-tuning** improves adaptation
* why augmentation is central to the problem, not optional

### 4) Evaluation awareness

The project treats performance as a trade-off between:

* accuracy
* class balance
* inference speed
* real-world usability

### 5) Production mindset

The workflow includes:

* MLflow tracking
* quality gates
* model lifecycle management
* deployment planning

That combination makes the project a **strong applied Data Science + Computer Vision portfolio piece**, not just a notebook experiment.

---

## 🔄 MLflow Lifecycle Management

The project includes an **MLflow-based MLOps workflow** to make experimentation reproducible and promotion decisions more disciplined.

### MLflow capabilities used

* **experiment tracking**
* **parameter logging**
* **metric logging**
* **artifact tracking**
* **model versioning**
* **quality-gated validation**
* **registry-based promotion workflow**

### Quality gate concept

Only models that satisfy the minimum quality criteria should be considered deployment candidates.

Example deployment gate:

* **Accuracy ≥ 80%**
* stable validation behavior
* reproducible tracked run

This is important because it shifts the project from “I trained a model” to “I manage a model lifecycle.”

---

## 🏗️ Project Pipeline

```text
Dataset
   ↓
EDA + Class Distribution Analysis
   ↓
Train / Validation / Test Split (80 / 10 / 10)
   ↓
tf.data Pipeline + Image Preprocessing
   ↓
Data Augmentation
   ↓
ResNet50 Transfer Learning
   ↓
Fine-Tuning Stage
   ↓
Evaluation + Metrics Logging
   ↓
MLflow Tracking
   ↓
Quality Gate Validation
   ↓
Production Registry Candidate
```

---

## 💼 Business & Operational Impact

The practical value of the project is in **diagnostic acceleration and expert-efficiency gains**.

### Expected operational impact

* **diagnosis time:** from **2–3 days → under 3 seconds**
* **expert workload reduction:** the system can act as a first-pass screener before manual review
* **faster intervention:** quicker disease identification can reduce delay in treatment decisions
* **scalable monitoring:** supports larger crop volumes without proportional growth in expert effort

### Example value proposition

A productionized version of this system could be used to:

* prioritize expert review queues
* support agronomists in the field
* provide mobile-assisted diagnosis for farmers
* improve early detection workflows in agricultural support systems

---

## 🛠️ Tech Stack

| Layer                      | Technology             | Purpose                                    |
| -------------------------- | ---------------------- | ------------------------------------------ |
| Deep Learning              | **TensorFlow / Keras** | model training and fine-tuning             |
| Model Backbone             | **ResNet50**           | transfer learning for image classification |
| Data Pipeline              | **tf.data**            | scalable input pipeline                    |
| MLOps                      | **MLflow**             | tracking, versioning, lifecycle management |
| Deployment (planned)       | **FastAPI**            | inference API                              |
| Cloud Inference (planned)  | **AWS / GCP GPU**      | scalable serving                           |
| Mobile Inference (planned) | **TensorFlow Lite**    | lightweight edge deployment                |

---

## 📁 Project Structure

```bash
project/
│
├── data_pipeline.py          # image loading, preprocessing, tf.data workflow
├── eda.py                    # class distribution analysis and visual inspection
├── model.py                  # ResNet50 training and fine-tuning
├── mlflow_lifecycle.py       # experiment tracking and registry logic
│
├── api.py                    # FastAPI inference service (planned / optional)
├── app.py                    # dashboard / UI layer (if included)
│
├── MLproject                 # MLflow project configuration
├── conda.yaml                # reproducible environment
├── README.md
└── docs/
```

> Update the structure section to match the actual filenames in your repository.

---

## 🏁 How to Run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Train the model / run the MLflow workflow

```bash
mlflow run .
```

### 3) If API serving is included, launch FastAPI

```bash
uvicorn api:app --reload
```

---

## 🛣️ Roadmap

### Phase 1 — Completed

* model training
* transfer learning pipeline
* MLflow lifecycle setup
* production-ready model candidate

### Phase 2 — Next

* FastAPI deployment
* cloud GPU inference workflow

### Phase 3 — Planned

* TensorFlow Lite mobile integration
* field-friendly prediction workflow

### Phase 4 — Planned

* real-world field image collection
* drift monitoring and periodic retraining

---

## 🔮 Future Improvements

Potential next steps for strengthening the platform:

* benchmark **EfficientNetV2 / ConvNeXt / ViT** against ResNet50
* add **class-wise error analysis** and confusion reporting
* evaluate **macro metrics per disease class** for rare-class reliability
* integrate **real field images** to reduce benchmark-to-production gap
* add **drift detection and scheduled retraining**
* support **top-k predictions** for expert review workflows
* build a **mobile-first diagnosis interface** for farmers or field agents

---

## 👨‍💻 Author

**Youssef Mahmoud**
AI / Data Science Student

[LinkedIn](https://www.linkedin.com/in/youssef-mahmoud-63b243361)

---

## ⭐ Final Note

This project is not just about classifying leaf images.

It is about building a **production-oriented disease diagnosis system** that combines **computer vision, transfer learning, data-aware training strategy, and MLflow-based lifecycle management** to support faster agricultural decision-making.

In other words, it is a portfolio project designed to show not only that a model can predict plant disease — but that the full workflow can be engineered, evaluated, and prepared for real-world use.
