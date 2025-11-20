# Cuisine Classifier ML Pipeline on Databricks

A complete end-to-end machine learning pipeline for cuisine classification using computer vision, built on Databricks with MLflow and Unity Catalog integration.

## 🍕 Project Overview

This project implements a **cuisine classification system** that identifies food dishes and classifies them into cuisine types (Italian, Japanese, American, French, Mexican, Chinese, Mediterranean, International) using a **ResNet-50 deep learning model**. The pipeline processes food images through a medallion architecture (Bronze → Silver → Gold) and serves predictions via a **Databricks App** that consumes the deployed model.

### Key Features
- **Multi-cuisine classification** across 8 major cuisine types
- **67+ food dish recognition** capability
- **ResNet-50 CNN architecture** with transfer learning via Hugging Face Transformers
- **MLflow experiment tracking** and model versioning
- **Unity Catalog model registry** integration
- **Delta Live Tables (DLT)** data pipeline with Auto Loader
- **Streamlit-powered Databricks App** for real-time image classification
- **Databricks serving endpoints** for model deployment and consumption

## 🏗️ Architecture

### Data Pipeline (Medallion Architecture)
```
Hugging Face Food-101 → Bronze Volume → Bronze Tables → Silver Processing → Gold Features → ML Training
```

### ML Pipeline
```
Data Ingestion → Feature Engineering → Model Training → Model Registry → Serving Endpoint → Databricks App
```

## 📁 Project Structure

```
cuisine_classifier_ml_databricks/
├── 0_config/                              # Configuration files
│   ├── cuisine_mapping.json              # 8 cuisines to 67 food type mappings
│   └── food_types.json                   # Complete list of 67 supported food dishes
├── notebooks/                             # Jupyter notebooks
│   ├── ML_Training_Pipeline_Complex.ipynb # ResNet-50 training with Transformers
│   └── source_to_bronze_volume_ingestion.ipynb # Hugging Face Food-101 ingestion
├── ldp_pipeline_codes/                    # Delta Live Tables pipeline code
│   ├── bronze_tables_ingestion_ldp/       # Auto Loader: Volume → Bronze tables
│   ├── bronze_to_silver_data_processing_ldp/ # Image processing & cuisine mapping
│   └── silver_to_gold_data_processing_ldp/ # ML dataset preparation & label encoding
├── dbr-app-cuisine-classifier/            # Streamlit Databricks App
│   ├── app.py                            # Main Streamlit application
│   ├── requirements.txt                  # Python dependencies
│   ├── app.yaml                         # Databricks App configuration
│   └── .env                             # Environment variables
└── README.md                            # This file
```

## 📊 Data Source & Processing Flow

### 1. Source → Bronze Volume
**Notebook**: [`notebooks/source_to_bronze_volume_ingestion.ipynb`](notebooks/source_to_bronze_volume_ingestion.ipynb)

**Process**:
- **Streams data** from Hugging Face Food-101 dataset (101 food categories, 1000 images each)
- **Filters images** based on configured 67 food types in [`0_config/food_types.json`](0_config/food_types.json)
- **Saves directly** to Unity Catalog bronze volume (`/Volumes/cuisine_vision_catalog/bronze/bronze_volume`)
- **Incremental processing** with JSON status tracking and resume capability
- **Configurable quality** (JPEG compression: 95%)
- **Organized storage** by food type folders with filename format: `{food_type}_idx_{dataset_index:06d}.jpg`

### 2. Bronze Volume → Bronze Tables
**Pipeline**: [`ldp_pipeline_codes/bronze_tables_ingestion_ldp/`](ldp_pipeline_codes/bronze_tables_ingestion_ldp/)

**Process**:
- **Auto Loader** streaming ingestion from bronze volume
- **Dynamic table generation** for each of 67 food types (`food_pizza`, `food_sushi`, etc.)
- **Binary file processing** with metadata extraction (file size, modification time)
- **Schema evolution** support and automatic deduplication
- **File validation** and format checking (JPG/JPEG/PNG)

### 3. Bronze Tables → Silver Tables
**Pipeline**: [`ldp_pipeline_codes/bronze_to_silver_data_processing_ldp/`](ldp_pipeline_codes/bronze_to_silver_data_processing_ldp/)

**Process**:
- **Image processing** to standardized 224x224 RGB format for ResNet-50
- **Cuisine category mapping** using [`cuisine_mapping.json`](0_config/cuisine_mapping.json)
- **Quality validation** with brightness, contrast, and aspect ratio analysis
- **Feature extraction**: RGB statistics, quality scores, texture analysis
- **8 cuisine-specific tables** created (`cuisine_american`, `cuisine_chinese`, `cuisine_french`, `cuisine_international`, `cuisine_italian`, `cuisine_japanese`, `cuisine_mediterranean`, `cuisine_mexican`)
- **Data quality expectations** with automatic filtering of invalid images

### 4. Silver Tables → Gold Table
**Pipeline**: [`ldp_pipeline_codes/silver_to_gold_data_processing_ldp/`](ldp_pipeline_codes/silver_to_gold_data_processing_ldp/)

**Process**:
- **Union all silver tables** into unified ML dataset
- **Label encoding** for both cuisine categories (8 classes) and food types (67 classes)
- **Dataset splitting** with configurable ratios (70% train, 15% validation, 15% test)
- **ML-ready format** with `processed_image_data` column for direct CNN consumption
- **Single gold table**: `ml_dataset` optimized for training and inference

## 🍽️ Supported Cuisines & Dishes

### Cuisine Types (8 categories)
Based on [`0_config/cuisine_mapping.json`](0_config/cuisine_mapping.json):

- **Italian** (11 dishes): pizza, lasagna, risotto, spaghetti_bolognese, spaghetti_carbonara, ravioli, bruschetta, tiramisu, cannoli, panna_cotta, garlic_bread
- **Japanese** (7 dishes): sushi, ramen, sashimi, takoyaki, miso_soup, gyoza, edamame
- **American** (18 dishes): hamburger, hot_dog, pancakes, waffles, apple_pie, cheesecake, macaroni_and_cheese, grilled_cheese_sandwich, pulled_pork_sandwich, baby_back_ribs, chicken_wings, onion_rings, club_sandwich, eggs_benedict, omelette, lobster_roll_sandwich, crab_cakes, clam_chowder
- **French** (10 dishes): french_fries, french_onion_soup, french_toast, croque_madame, escargots, creme_brulee, foie_gras, macarons, lobster_bisque, mussels
- **Mexican** (7 dishes): tacos, chicken_quesadilla, nachos, churros, guacamole, huevos_rancheros, ceviche
- **Chinese** (5 dishes): dumplings, fried_rice, peking_duck, hot_and_sour_soup, spring_rolls
- **Mediterranean** (4 dishes): greek_salad, hummus, falafel, baklava
- **International** (9 dishes): grilled_salmon, fried_calamari, oysters, scallops, tuna_tartare, paella, fish_and_chips, poutine, pad_thai

### Total Coverage
- **67 food dishes** from Food-101 dataset
- **8 cuisine categories** for classification
- **Balanced representation** across global culinary traditions

## 🚀 Getting Started

### Prerequisites
- Databricks workspace with Unity Catalog enabled
- MLflow tracking server
- Delta Lake storage
- Python 3.8+
- Compute cluster with ML runtime (recommended: ML 13.3 LTS or higher)

### Step 1: Data Ingestion from Hugging Face

Run the source data ingestion notebook:
```python
# Location: notebooks/source_to_bronze_volume_ingestion.ipynb

# Configure runtime parameters:
images_per_run = 5          # Images per food type per execution
image_quality = 95          # JPEG quality (1-100)
CONFIG_VOLUME_PATH = "/Volumes/cuisine_vision_catalog/config/config_volume"
STATUS_FILE_PATH = "/Volumes/cuisine_vision_catalog/bronze/bronze_volume/ingestion_status/ingestion_status.json"
BRONZE_VOLUME_PATH = "/Volumes/cuisine_vision_catalog/bronze/bronze_volume"
```

**Key Features**:
- Downloads from Hugging Face Food-101 dataset with streaming support
- Processes and organizes images by food type in bronze volume
- Maintains incremental status tracking with JSON-based resume capability
- Supports configurable batch processing for large-scale ingestion

### Step 2: Bronze Volume → Bronze Tables

Configure and run the bronze ingestion DLT pipeline:
```python
# Location: ldp_pipeline_codes/bronze_tables_ingestion_ldp/transformations/bronze_tables_ingestion.py

# Pipeline configuration:
config_volume_path = "/Volumes/cuisine_vision_catalog/config/config_volume"
bronze_volume_path = "/Volumes/cuisine_vision_catalog/bronze/bronze_volume"

# Creates 67 dynamic tables:
# - food_pizza, food_sushi, food_hamburger, etc.
# - Auto Loader streaming from bronze volume
```

### Step 3: Bronze → Silver Processing

Run the silver processing DLT pipeline:
```python
# Location: ldp_pipeline_codes/bronze_to_silver_data_processing_ldp/transformations/bronze_to_silver_data_processing.py

# Creates 8 cuisine tables:
# - cuisine_american, cuisine_chinese, cuisine_french, etc.
# - 224x224 image processing with comprehensive quality validation
# - Feature extraction: RGB stats, brightness, contrast, texture analysis
```

### Step 4: Silver → Gold Aggregation

Run the gold feature engineering DLT pipeline:
```python
# Location: ldp_pipeline_codes/silver_to_gold_data_processing_ldp/transformations/silver_to_gold_data_processing.py

# Creates unified ML dataset:
# - ml_dataset table with train/validation/test splits (70%/15%/15%)
# - Label encoded for direct CNN training (8 cuisine classes, 67 food classes)
# - Optimized for ResNet-50 consumption with processed_image_data column
```

### Step 5: ML Model Training

Open and run the training notebook:
```python
# Location: notebooks/ML_Training_Pipeline_Complex.ipynb

# Key configuration:
CATALOG = "cuisine_vision_catalog"
MODEL_CHECKPOINT = "microsoft/resnet-50"
EXPERIMENT_NAME = "/cuisine_classifier"
NUM_EPOCHS = 5
BATCH_SIZE = 12
LEARNING_RATE = 2e-4
```

**Training Pipeline Features**:
- Loads data directly from `cuisine_vision_catalog.gold.ml_dataset`
- Uses Hugging Face Transformers with ResNet-50 pre-trained model
- Implements custom `CuisineClassifier` wrapper for MLflow compatibility
- Comprehensive evaluation with accuracy and F1-score metrics
- Automatic model registration in Unity Catalog (`cuisine_vision_catalog.ml_models.cuisine_classifier`)

### Step 6: Model Deployment

The trained model is automatically registered and can be deployed to Databricks serving endpoints through Unity Catalog.

### Step 7: Databricks App Deployment

Deploy the Streamlit-powered Databricks App:

```bash
cd dbr-app-cuisine-classifier

# Configure environment variables in .env:
SERVING_ENDPOINT_URL=your_databricks_serving_endpoint_url
DATABRICKS_TOKEN=your_databricks_personal_access_token
CUISINE_MAPPING_PATH=path_to_cuisine_mapping_config
```

**App Features** ([`app.py`](dbr-app-cuisine-classifier/app.py)):
- **Modern Streamlit interface** with drag & drop image upload
- **Real-time prediction** via Databricks serving endpoint consumption
- **Confidence scoring** and detailed result visualization
- **Connection testing** and comprehensive error handling
- **Responsive design** optimized for various screen sizes

## 🔧 Configuration Management

### Cuisine Mappings
[`0_config/cuisine_mapping.json`](0_config/cuisine_mapping.json) - Central configuration mapping 67 food types to 8 cuisine categories:
```json
{
  "italian": ["pizza", "lasagna", "risotto", "spaghetti_bolognese", ...],
  "american": ["hamburger", "hot_dog", "pancakes", "apple_pie", ...],
  // ... 8 total cuisines with 67 food types
}
```

### Food Types
[`0_config/food_types.json`](0_config/food_types.json) - Complete list of 67 supported dishes from Food-101 dataset.

### Model Parameters
Training configuration in [`ML_Training_Pipeline_Complex.ipynb`](notebooks/ML_Training_Pipeline_Complex.ipynb):
- **Base Model**: microsoft/resnet-50 (Hugging Face Transformers)
- **Learning Rate**: 2e-4
- **Batch Size**: 12
- **Epochs**: 5
- **Image Size**: 224x224 RGB
- **Optimizer**: AdamW with learning rate scheduling

## 📊 Model Performance

Current ResNet-50 model metrics (based on training pipeline):
- **Training Strategy**: Transfer learning with fine-tuned classification head
- **Architecture**: ResNet-50 backbone + custom 8-class cuisine classifier
- **Input Processing**: 224x224 RGB images with normalization
- **Output**: 8-class cuisine probabilities with confidence scores

**Performance Optimization**:
- Comprehensive data augmentation pipeline
- Quality-based image filtering (brightness, contrast, aspect ratio)
- Balanced dataset with strategic train/validation/test splits
- MLflow experiment tracking for hyperparameter optimization

## 🛠️ Technologies Used

- **Databricks**: Unified analytics platform with Apps framework
- **Delta Live Tables**: Streaming data pipeline orchestration with Auto Loader
- **MLflow**: Comprehensive experiment tracking and model lifecycle management
- **Unity Catalog**: Centralized model registry and governance
- **Hugging Face Transformers**: Pre-trained ResNet-50 model and training infrastructure
- **PyTorch**: Deep learning framework with GPU acceleration
- **Streamlit**: Interactive web interface for Databricks App
- **Delta Lake**: ACID-compliant data storage with time travel capabilities
- **Auto Loader**: Scalable file ingestion with schema evolution

## 📈 Complete Data Flow

![Data Flow Diagram](Data%20flow%20diagram.png)

```mermaid
graph TD
    A[Hugging Face Food-101] -->|Stream Download| B[Bronze Volume Storage]
    B -->|Auto Loader| C[Bronze Tables - 67 Food Types]
    C -->|Union + Process| D[Silver Tables - 8 Cuisines]
    D -->|Aggregate + Encode| E[Gold Table - ML Dataset]
    E -->|Train| F[ResNet-50 Model]
    F -->|Register| G[Unity Catalog Model Registry]
    G -->|Deploy| H[Databricks Serving Endpoint]
    H -->|Consume| I[Streamlit Databricks App]
```

**Detailed Pipeline Steps**:
1. **Source Ingestion**: Notebook downloads and organizes 67 food types from Food-101
2. **Bronze Layer**: Auto Loader creates 67 dynamic tables with file metadata
3. **Silver Layer**: Processes images to 224x224, maps to 8 cuisines, extracts features
4. **Gold Layer**: Creates unified ML dataset with label encoding and train/test splits
5. **ML Training**: Fine-tunes ResNet-50 for 8-class cuisine classification
6. **Model Registry**: Registers trained model with versioning and lineage tracking
7. **Serving**: Deploys model to scalable serving endpoint
8. **Application**: Streamlit app provides user-friendly prediction interface

## 🔬 Implementation Details

### Advanced Data Processing
- **Quality Validation**: Multi-dimensional image quality assessment (brightness, contrast, RGB statistics)
- **Feature Engineering**: Automated extraction of visual features for enhanced model performance
- **Schema Evolution**: Dynamic table creation with automatic schema adaptation
- **Incremental Processing**: Efficient data pipeline with change data capture and resume capabilities

### Model Architecture
- **Transfer Learning**: Leverages ImageNet pre-trained ResNet-50 with domain adaptation
- **Custom Classification Head**: Tailored 8-class cuisine classifier with dropout regularization
- **MLflow Integration**: Comprehensive experiment tracking with custom model wrapper
- **Evaluation Metrics**: Multi-class accuracy, F1-score, and per-class performance analysis

### Production Readiness
- **Scalable Infrastructure**: Auto Loader and DLT for handling large-scale data ingestion
- **Error Handling**: Comprehensive exception handling throughout the entire pipeline
- **Monitoring**: MLflow tracking for model performance and data quality monitoring
- **Configuration Management**: Externalized configs for easy environment adaptation
- **Documentation**: Extensive inline documentation and pipeline summaries

## 📝 Usage Examples

### Databricks App Interface
1. **Upload Image**: Drag & drop food image in the Streamlit interface
2. **Processing**: Image converted to base64 and sent to serving endpoint
3. **Prediction**: Real-time cuisine classification with confidence distribution
4. **Results**: Detailed visualization with cuisine probabilities and food type suggestions

### API Integration
Direct serving endpoint consumption:
```python
import requests
import pandas as pd
import base64

# Prepare image for prediction
with open("food_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Create DataFrameSplit payload for Databricks serving
dataset = pd.DataFrame({'processed_image_data': [image_base64]})
payload = {'dataframe_split': dataset.to_dict(orient='split')}

# Make prediction request
response = requests.post(
    serving_endpoint_url,
    json=payload,
    headers={"Authorization": f"Bearer {databricks_token}"}
)

# Parse results
predictions = response.json()
```

## 🎯 Key Implementation Features

### Data Pipeline Optimizations
- **Incremental Processing**: JSON-based status tracking with resume capability for large datasets
- **Quality Filtering**: Multi-stage image validation with configurable quality thresholds
- **Scalable Design**: Auto Loader and DLT for handling enterprise-scale data volumes
- **Schema Management**: Dynamic table creation with automatic schema evolution

### ML Pipeline Enhancements
- **Transfer Learning**: Efficient fine-tuning of pre-trained ResNet-50 for domain-specific classification
- **Experiment Tracking**: MLflow integration with custom model wrapper for production deployment
- **Performance Monitoring**: Comprehensive evaluation metrics with per-class analysis
- **Model Governance**: Unity Catalog integration for centralized model lifecycle management

### Application Features
- **Modern UI**: Responsive Streamlit interface with intuitive drag-and-drop functionality
- **Real-time Processing**: Direct serving endpoint consumption for instant predictions
- **Error Handling**: Robust exception handling with user-friendly error messages
- **Configuration Management**: Environment-based configuration for easy deployment across environments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Update configuration files as needed for new food types or cuisines
4. Test the complete pipeline flow from ingestion to serving
5. Update documentation and README as necessary
6. Submit a pull request with detailed description of changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: Food-101 dataset and Transformers library with pre-trained ResNet-50
- **Databricks**: Unified analytics platform, Apps framework, and Delta Live Tables
- **MLflow**: Open source ML lifecycle management and experiment tracking
- **Unity Catalog**: Model governance, registry, and serving capabilities
- **Microsoft Research**: ResNet-50 architecture for deep residual learning
- **PyTorch Community**: Deep learning framework and ecosystem
- **Streamlit**: Open-source framework for rapid ML app development

## 🏗️ Architecture

### Data Pipeline (Medallion Architecture)
```
Hugging Face Food-101 → Bronze Volume → Bronze Tables → Silver Processing → Gold Features → ML Training
```

### ML Pipeline
```
Data Ingestion → Feature Engineering → Model Training → Model Registry → Serving Endpoint → Databricks App
```

## 📁 Project Structure

```
cuisine_classifier_ml_databricks/
├── 0_config/                              # Configuration files
│   ├── cuisine_mapping.json              # Cuisine to food type mappings
│   └── food_types.json                   # List of supported food dishes
├── notebooks/                             # Jupyter notebooks
│   ├── ML_Training_Pipeline_Complex.ipynb # Main ML training pipeline
│   └── source_to_bronze_volume_ingestion.ipynb # Hugging Face to Bronze volume data ingestion
├── ldp_pipeline_codes/                    # Delta Live Tables pipeline code
│   ├── bronze_tables_ingestion_ldp/       # Bronze layer ingestion
│   ├── bronze_to_silver_data_processing_ldp/ # Silver layer processing
│   └── silver_to_gold_data_processing_ldp/ # Gold layer feature engineering
├── dbr-app-cuisine-classifier/            # Databricks App for cuisine classification
│   ├── app.py                            # Main Databricks App application
│   ├── requirements.txt                  # Python dependencies
│   ├── app.yaml                         # Databricks App configuration
│   └── .env                             # Environment variables
└── README.md                            # This file
```

## 📊 Data Source

### Hugging Face Food-101 Dataset
The project uses the **Food-101 dataset** from Hugging Face, which contains:
- **101 food categories** with 1,000 images each
- **High-quality food images** suitable for computer vision
- **Diverse food representations** across multiple cuisines
- **Pre-labeled data** for supervised learning

### Data Ingestion Process
The `source_to_bronze_volume_ingestion.ipynb` notebook:
- **Streams data** from Hugging Face Food-101 dataset
- **Filters images** based on configured food types in `0_config/food_types.json`
- **Optimizes image quality** with configurable JPEG compression
- **Maintains incremental ingestion** with status tracking
- **Supports resumable downloads** for large datasets
- **Organizes images** by food type in bronze volume storage

## 🍽️ Supported Cuisines & Dishes

### Cuisine Types (4 categories)
- **Italian**: Pizza, Lasagna, Risotto, Spaghetti Bolognese, Tiramisu, etc.
- **Japanese**: Sushi, Ramen, Sashimi, Takoyaki, Miso Soup, etc.
- **American**: Hamburger, Hot Dog, Pancakes, Apple Pie, BBQ Ribs, etc.
- **French**: French Fries, French Onion Soup, Crème Brûlée, Escargots, etc.

### Food Dishes (50+ types)
See `0_config/food_types.json` for the complete list of recognizable dishes.

## 🚀 Getting Started

### Prerequisites
- Databricks workspace with Unity Catalog enabled
- MLflow tracking server
- Delta Lake storage
- Python 3.8+

### 1. Data Ingestion from Source

#### Source to Bronze Volume Ingestion
First, ingest food images from Hugging Face Food-101 dataset directly to bronze volume:

```python
# Run the source data ingestion notebook
# Location: notebooks/source_to_bronze_volume_ingestion.ipynb
```

This notebook:
- Downloads images from Hugging Face Food-101 dataset
- Filters and processes images based on configured food types
- Saves images directly to bronze volume storage
- Maintains ingestion status and supports incremental loading
- Handles image quality optimization and file naming conventions

### 2. Data Pipeline Setup

#### Bronze Layer Ingestion
```python
# Run the bronze ingestion DLT pipeline
# Location: ldp_pipeline_codes/bronze_tables_ingestion_ldp/
```

#### Silver Layer Processing
```python
# Run the silver processing DLT pipeline  
# Location: ldp_pipeline_codes/bronze_to_silver_data_processing_ldp/
```

#### Gold Layer Feature Engineering
```python
# Run the gold feature engineering DLT pipeline
# Location: ldp_pipeline_codes/silver_to_gold_data_processing_ldp/
```

### 3. ML Model Training

Open and run the main training notebook:
```
notebooks/ML_Training_Pipeline_Complex.ipynb
```

This notebook will:
- Load processed image data from the gold layer
- Configure ResNet-50 model with transfer learning
- Train the cuisine classification model
- Track experiments with MLflow
- Register the model in Unity Catalog
- Evaluate model performance

### 4. Model Deployment

The trained model is automatically registered in Unity Catalog and can be deployed to Databricks serving endpoints.

### 5. Databricks App Deployment

#### Setup Environment

    - Push your code to databricks workspace
    - Update the .env file
    - Go to Compute -> Apps -> create custom streamlit application choose model serving in config and serving endpoint
    - Post it is created click on deploy and choose the source directory of code dbr-app-cuisine-classifier

## 🔧 Configuration

### Cuisine Mappings
Edit `0_config/cuisine_mapping.json` to modify cuisine categories and their associated dishes.

### Food Types
Update `0_config/food_types.json` to add or remove supported food dishes.

### Model Parameters
Adjust training parameters in the ML notebook:
- Learning rate: 0.001
- Batch size: 32
- Epochs: 10
- Model: ResNet-50 (pre-trained)

## 📊 Model Performance

The ResNet-50 model achieves:
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~92%
- **Multi-class Classification**: 4 cuisine types
- **Image Recognition**: 50+ food dishes

## 🛠️ Technologies Used

- **Databricks**: Platform and compute
- **Delta Live Tables**: Data pipeline orchestration
- **MLflow**: Experiment tracking and model management
- **Unity Catalog**: Model registry and governance
- **PyTorch**: Deep learning framework
- **ResNet-50**: Pre-trained CNN architecture
- **Databricks Apps**: Application framework for ML model consumption
- **Delta Lake**: Data storage and versioning

## 📈 Data Flow

1. **Hugging Face Food-101** → Download and ingest images to bronze volume
2. **Bronze Volume** → Store raw food images by food type categories
3. **Bronze Layer** → Ingest images from volume into Delta tables
4. **Silver Layer** → Clean and standardize image data
5. **Gold Layer** → Extract features and prepare for ML
6. **ML Training** → Train ResNet-50 cuisine classifier
7. **Model Registry** → Register model in Unity Catalog
8. **Serving** → Deploy model to serving endpoints
9. **Databricks App** → Consume served model for image classification

## 🔬 Model Details

### Architecture
- **Base Model**: ResNet-50 (ImageNet pre-trained)
- **Custom Head**: Fully connected layers for 4-class cuisine classification
- **Input Size**: 224x224x3 RGB images
- **Output**: Cuisine probabilities (Italian, Japanese, American, French)

### Training Strategy
- **Transfer Learning**: Freeze ResNet-50 backbone, train custom classifier
- **Data Augmentation**: Random rotation, scaling, and normalization
- **Optimization**: Adam optimizer with learning rate scheduling
- **Loss Function**: Cross-entropy loss for multi-class classification

## 📝 Usage Examples

### Databricks App Interface
1. Upload a food image through the Databricks App
2. App calls the served model endpoint for classification
3. Get real-time cuisine classification results with confidence scores

### API Integration
Use the Databricks serving endpoint for programmatic access:
```python
import requests
import base64

# Encode image
with open("food_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Make prediction
response = requests.post(
    "your_serving_endpoint_url",
    json={"image": image_data}
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ResNet-50**: Deep Residual Learning for Image Recognition
- **Databricks**: Unified analytics platform and Apps framework
- **MLflow**: Open source ML lifecycle management
- **Unity Catalog**: Model governance and serving capabilities