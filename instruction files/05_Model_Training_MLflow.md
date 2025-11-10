# 05. Simple ML Training Pipeline with MLflow Integration

**Project:** Cuisine Vision Pipeline (CVP) - ML Engineering  
**Focus:** Direct CNN training with HuggingFace Transformers and MLflow tracking  
**Architecture:** Fine-tuned ResNet-50 (or other models) with simple preprocessing  
**Responsibility:** ML Engineering pipeline using processed images from gold layer

## 1. Environment Setup and Dependencies

### 1.1 Install Required Libraries
```python
# Install essential packages for ML training
%pip install torch torchvision transformers datasets mlflow scikit-learn
```

### 1.2 Restart Python Environment
```python
# Restart Python to ensure clean environment
dbutils.library.restartPython()
```

### 1.3 Import Libraries
```python
# Simple imports - clean and minimal
import mlflow
import torch
import pandas as pd
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from PIL import Image
import io
from torchvision.transforms import Compose, Normalize, ToTensor, Lambda
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import os

print("✅ Simple imports loaded successfully")
```

## 2. Configuration and Data Loading

### 2.1 Training Configuration
```python
# Simple configuration - easily adjustable
CATALOG = "cuisine_vision_catalog"
MODEL_CHECKPOINT = "microsoft/resnet-50"  # Can be changed to other models
EXPERIMENT_NAME = "/cuisine_classifier"
NUM_EPOCHS = 5
BATCH_SIZE = 12
LEARNING_RATE = 2e-4

print(f"🔧 Configuration:")
print(f"   📊 Catalog: {CATALOG}")
print(f"   🧠 Model: {MODEL_CHECKPOINT}")
print(f"   🔄 Epochs: {NUM_EPOCHS}")
print(f"   📦 Batch Size: {BATCH_SIZE}")
print(f"   📈 Learning Rate: {LEARNING_RATE}")
```

### 2.2 Data Loading from Gold Layer
```python
# Simple data loading - direct from gold table
print("📊 Loading data from gold layer...")

# Load data directly - no complex joins needed
dataset_df = (
    spark.table(f"{CATALOG}.gold.ml_dataset")
    .select("processed_image_data", "cuisine_category")
    .filter("processed_image_data IS NOT NULL")
    .toPandas()
)

print(f"✅ Loaded {len(dataset_df)} samples")
print(f"   �️ Cuisines: {sorted(dataset_df['cuisine_category'].unique())}")

# Create HuggingFace dataset - simple rename for compatibility
dataset = Dataset.from_pandas(
    dataset_df.rename(columns={
        "processed_image_data": "image", 
        "cuisine_category": "label"
    })
)

# Simple train/test split
splits = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = splits['train']
val_ds = splits['test']

print(f"✅ Data splits:")
print(f"   🏋️ Training: {len(train_ds)} samples")
print(f"   ✅ Validation: {len(val_ds)} samples")
```

## 3. Data Preprocessing and Model Setup

### 3.1 Image Preprocessing Pipeline
```python
# Simple preprocessing - exactly like reference notebook patterns
print("🔄 Setting up simple preprocessing...")

# Load image processor for the selected model
image_processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

# Simple transform pipeline using standard torchvision transforms
transforms = Compose([
    Lambda(lambda b: Image.open(io.BytesIO(b)).convert("RGB")),
    ToTensor(),
    Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

def preprocess(batch):
    """Simple preprocessing function"""
    batch["image"] = [transforms(image) for image in batch["image"]]
    return batch

# Apply transforms to datasets
train_ds.set_transform(preprocess)
val_ds.set_transform(preprocess)

print("✅ Simple preprocessing setup complete")
```

### 3.2 Model Configuration and Loading
```python
# Simple model setup - no complex wrappers
print("🧠 Setting up simple model...")

# Create simple label mappings from data
unique_labels = sorted(set(dataset['label']))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)

print(f"✅ Labels: {id2label}")

# Load model - simple and direct using AutoModel
model = AutoModelForImageClassification.from_pretrained(
    MODEL_CHECKPOINT,
    label2id=label2id,
    id2label=id2label,
    num_labels=num_labels,
    ignore_mismatched_sizes=True  # Allows different number of classes
)

print(f"✅ Model loaded with {num_labels} classes")
```

### 3.3 Performance Optimization Setup
```python
# Optimize training performance and eliminate warnings
import os

print("🔧 Optimizing training performance...")

# Set threading for better CPU utilization
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

# Configure PyTorch for optimal performance
torch.set_num_threads(8)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   🖥️ Training device: {device}")
print(f"   🧵 CPU threads: 8")
print("✅ Performance optimizations applied")
```

## 4. Model Training with MLflow Integration

### 4.1 Training Pipeline Setup
```python
# Simple training - using standard Transformers Trainer
print("🏋️ Starting training...")

# Setup MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run() as run:
    print(f"🔄 MLflow run: {run.info.run_id}")
    
    # Training arguments with performance optimizations
    args = TrainingArguments(
        output_dir=f"/dbfs/tmp/cuisine-classifier-simple",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to=[],
        # Performance optimizations to eliminate warnings:
        dataloader_pin_memory=False,  # Fix pin_memory warning
        ddp_find_unused_parameters=False,  # Fix DDP warning
        use_cpu=not torch.cuda.is_available(),  # Optimize for CPU if no GPU
    )
    
    # Simple data collator - handles batching
    def collate_fn(examples):
        pixel_values = torch.stack([e["image"] for e in examples])
        labels = torch.tensor([label2id[e["label"]] for e in examples], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}
    
    # Simple metrics computation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        return {'accuracy': accuracy, 'f1': f1}

    # Create Trainer - standard Transformers approach
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=train_ds, 
        eval_dataset=val_ds, 
        processing_class=image_processor,  # Use processing_class (not tokenizer)
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    print("🚀 Training started...")
    trainer.train()
    print("✅ Training completed!")
    
    # Evaluate final performance
    print("📊 Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"✅ Final metrics: {eval_results}")
    
    # Log parameters to MLflow
    mlflow.log_param("model_checkpoint", MODEL_CHECKPOINT)
    mlflow.log_param("num_epochs", NUM_EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("num_labels", num_labels)
    
    # Log metrics to MLflow
    for key, value in eval_results.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)
```

## 5. Model Wrapper and MLflow Registration

### 5.1 Create Model Wrapper for Deployment
```python
# Simple model wrapper for MLflow - deployment ready
print("📦 Creating simple model wrapper...")

from transformers import pipeline

# Create pipeline from trained model
classifier = pipeline(
    "image-classification", 
    model=trainer.model, 
    feature_extractor=image_processor
)

class SimpleCuisineClassifier(mlflow.pyfunc.PythonModel):
    """Simple wrapper for cuisine classification - deployment ready"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.pipeline.model.eval()
    
    def predict(self, context, model_input):
        """Simple prediction method"""
        # Handle DataFrame input (batch prediction)
        if isinstance(model_input, pd.DataFrame):
            # Convert bytes to PIL images
            images = model_input['processed_image_data'].apply(
                lambda b: Image.open(io.BytesIO(b)).convert("RGB")
            ).tolist()
            
            # Get predictions
            with torch.no_grad():
                predictions = self.pipeline(images)
            
            # Return top prediction for each image
            return pd.DataFrame([
                max(pred, key=lambda x: x['score']) 
                for pred in predictions
            ])
        
        # Handle single image bytes
        else:
            image = Image.open(io.BytesIO(model_input)).convert("RGB")
            with torch.no_grad():
                prediction = self.pipeline(image)
            return max(prediction, key=lambda x: x['score'])

# Create wrapped model
wrapped_model = SimpleCuisineClassifier(classifier)
print("✅ Simple model wrapper created")
```

### 5.2 MLflow Model Registration
```python
# Simple MLflow logging and registration
print("📊 Logging model to MLflow...")

# Import signature utilities for Unity Catalog compatibility
from mlflow.models.signature import infer_signature

with mlflow.start_run(run_id=run.info.run_id):
    # Test model with sample data and create signature
    test_df = dataset_df[['processed_image_data']].head(3)
    test_predictions = wrapped_model.predict(None, test_df)
    print(f"✅ Test predictions: {test_predictions}")
    
    # Create model signature - required for Unity Catalog
    signature = infer_signature(test_df, test_predictions)
    print(f"✅ Model signature created: {signature}")
    
    # Log model with signature - required for Unity Catalog
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wrapped_model,
        signature=signature,  # Required for Unity Catalog
        pip_requirements=[
            "torch", 
            "transformers", 
            "pillow", 
            "pandas",
            "numpy"
        ]
    )
    
    print(f"✅ Model logged with signature: {model_info.model_uri}")

# Register to Unity Catalog - simple registration
full_model_name = f"{CATALOG}.ml_models.cuisine_classifier"
registered_model = mlflow.register_model(
    model_uri=model_info.model_uri, 
    name=full_model_name,
    tags={
        "stage": "development",
        "task": "image_classification", 
        "architecture": "ResNet-50",  # Update based on MODEL_CHECKPOINT
        "approach": "simple"
    }
)

print(f"🎉 Model registered successfully!")
print(f"   📦 Model: {full_model_name}")
print(f"   🏷️ Version: {registered_model.version}")
```

## 6. Model Testing and Performance Analysis

### 6.1 Simple Model Testing
```python
# Simple testing - verify everything works
print("🧪 Final testing...")

# Test with a few samples
test_samples = dataset_df.sample(n=4)
for idx, row in test_samples.iterrows():
    true_label = row['cuisine_category']
    image_bytes = row['processed_image_data']
    
    # Make prediction
    prediction = wrapped_model.predict(None, image_bytes)
    
    print(f"Sample {idx}:")
    print(f"   ✅ True: {true_label}")
    print(f"   🎯 Predicted: {prediction['label']} (score: {prediction['score']:.3f})")
    print()

print("🎉 Simple pipeline completed successfully!")
print("\n📋 Summary:")
print(f"   📊 Total samples: {len(dataset_df)}")
print(f"   🏷️ Classes: {num_labels}")
print(f"   🔄 Epochs: {NUM_EPOCHS}")
print(f"   📦 Model: {full_model_name} v{registered_model.version}")
```

### 6.2 Dataset Analysis (Optional)
```python
# Dataset Analysis - Check for common issues
print("🔍 Dataset Analysis:")
print(f"📊 Total samples: {len(dataset_df)}")

# Check class distribution
class_counts = dataset_df['cuisine_category'].value_counts()
print(f"\n�️ Class Distribution:")
for cuisine, count in class_counts.items():
    percentage = (count / len(dataset_df)) * 100
    print(f"   {cuisine}: {count} samples ({percentage:.1f}%)")

# Check for class imbalance
min_samples = class_counts.min()
max_samples = class_counts.max()
imbalance_ratio = max_samples / min_samples
print(f"\n⚖️ Class Imbalance Analysis:")
print(f"   Min class size: {min_samples} samples")
print(f"   Max class size: {max_samples} samples") 
print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")

# Identify potential issues
print(f"\n⚠️ Potential Issues Detected:")
if imbalance_ratio > 3:
    print("   🚨 SIGNIFICANT CLASS IMBALANCE! Some classes have 3x+ more samples than others")
    print("      → Solution: Use class weights or data augmentation")

if min_samples < 50:
    print("   🚨 VERY SMALL DATASET! Some classes have <50 samples")
    print("      → Solution: Collect more data or use data augmentation")

if len(dataset_df) < 500:
    print("   � SMALL TOTAL DATASET! Less than 500 samples for deep learning")
    print("      → Solution: Collect significantly more data")

print(f"\n📈 Recommendations:")
print(f"   • Ideal dataset size: 1000+ samples per class")
print(f"   • Current average: {len(dataset_df) / num_labels:.0f} samples per class")
print(f"   • Minimum recommended: 200+ samples per class")
```

### 6.3 Performance Analysis (Optional)
```python
# Training Performance Analysis
print("📊 Training Performance Analysis:")

# Analyze final training metrics
if 'eval_results' in locals():
    print("\n✅ Final Evaluation Metrics:")
    for metric, value in eval_results.items():
        if isinstance(value, (int, float)):
            print(f"   {metric}: {value:.4f}")
    
    # Interpret the metrics
    eval_acc = eval_results.get('eval_accuracy', 0)
    eval_loss = eval_results.get('eval_loss', float('inf'))
    
    print(f"\n🎯 Performance Interpretation:")
    if eval_acc < 0.3:
        print("   🔴 CRITICAL: Very low accuracy (<30%) - model is barely learning")
        print("      → Likely causes: insufficient data, too few epochs, or data quality issues")
    elif eval_acc < 0.5:
        print("   🟡 POOR: Low accuracy (<50%) - significant improvement needed")
        print("      → Likely causes: class imbalance, insufficient training, or weak features")
    elif eval_acc < 0.7:
        print("   🟠 FAIR: Moderate accuracy (<70%) - room for improvement")
        print("      → Solutions: more training, data augmentation, or hyperparameter tuning")
    elif eval_acc < 0.85:
        print("   🟢 GOOD: Solid accuracy (70-85%) - decent performance")
        print("      → Can improve with more data or fine-tuning")
    else:
        print("   🟢 EXCELLENT: High accuracy (>85%) - great performance!")
```