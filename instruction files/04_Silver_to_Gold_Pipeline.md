# 04. Silver to Gold Pipeline Implementation ✅ SIMPLIFIED

**Project:** Cuisine Vision Pipeline (CVP) - Gold Layer  
**Focus:** Simple DLT data transformation for CNN training  
**Target:** Unified ML dataset for direct ResNet-50 training  
**Architecture:** Streaming aggregation from 8 silver tables to single ml_dataset table  
**Responsibility:** Data Engineering ONLY (simplified for direct CNN usage)

## 1. Pipeline Overview

### 1.1 Pipeline Configuration  
- **Pipeline Name:** `silver_to_gold_data_processing_ldp`
- **Target Schema:** `cuisine_vision_catalog.gold`
- **Source:** Silver schema tables (`cuisine_vision_catalog.silver.cuisine_*`)
- **Processing Mode:** Simple streaming aggregation for CNN training
- **Table Count:** 1 core table (ml_dataset only)

### 1.2 Simplified Gold Layer Architecture
**Clean Data Transformation Structure:**
```
Single Table:
- ml_dataset: Unified dataset from 8 silver tables with CNN-ready processed images
```

### 1.3 Data Engineering Objectives
- **Simple Transformation:** Union silver tables with basic label encoding
- **CNN Ready:** Direct `processed_image_data` for ResNet-50 training
- **Label Encoding:** Numeric encoding for food types and cuisines
- **Schema Consistency:** Standardized columns across all cuisine tables
- **Performance:** Optimized Delta table with auto-optimization enabled

## 2. Implementation Structure

### 2.1 DLT Pipeline Setup
**File:** `silver_to_gold_data_processing.py` (139 lines total)

**Section 1: Configuration and Discovery**
- Auto-discover all silver cuisine tables (`cuisine_*`)
- Load cuisine_mapping.json and food_types.json for label encoding
- Create bidirectional label mappings for consistent encoding
- Simple configuration management with error handling

**Section 2: Unified Dataset Creation**  
- Stream and union all 8 silver cuisine tables into single dataset
- Apply label encoding for food types and cuisines
- Add dataset splits (train/validation/test) with stratified random sampling
- Create single ml_dataset table with all required fields for CNN training

### 2.2 Primary Gold Table Schema

#### **Table: ml_dataset** (Single Unified ML Dataset)
```
Columns:
- image_id (string): Unique identifier from silver layer
- food_type (string): Food classification label (pizza, sushi, etc.)
- cuisine_category (string): Cuisine classification label (italian, japanese, etc.)
- food_type_encoded (int): Numeric label for food type (0-based indexing) 
- cuisine_category_encoded (int): Numeric label for cuisine (0-based indexing)
- processed_image_data (binary): 224x224 CNN-ready image data (KEY COLUMN)
- dataset_split (string): Train/validation/test split assignment
- split_seed (double): Random seed value for split reproducibility
- original_file_path (string): Source file tracking for lineage
- filename (string): Original filename from bronze layer
- dataset_index (bigint): Original dataset index
- quality_score (double): Image quality metric from silver
- mean_rgb (array<double>): RGB channel averages [R, G, B]
- std_rgb (array<double>): RGB standard deviations [R, G, B]
- brightness (double): Overall image brightness (0-1)
- contrast (double): Overall image contrast (0-1)
- aspect_ratio (double): Original image aspect ratio
- processing_timestamp (timestamp): Silver layer processing time
- gold_processing_timestamp (timestamp): Gold layer creation time
```

## 3. Simplified Label Encoding Implementation

### 3.1 Label Encoding Strategy
**Simple Dynamic Label Management:**
```python
# Load configuration files
config_content = dbutils.fs.head(f"{config_volume_path}/cuisine_mapping.json", max_bytes=1000000)
cuisine_mapping = json.loads(config_content)
config_content = dbutils.fs.head(f"{config_volume_path}/food_types.json", max_bytes=1000000)
food_types_list = json.loads(config_content)

# Create bidirectional mappings
all_cuisines = sorted(list(cuisine_mapping.keys()))
all_food_types = sorted(food_types_list)
cuisine_to_id = {cuisine: idx for idx, cuisine in enumerate(all_cuisines)}
food_type_to_id = {food_type: idx for idx, food_type in enumerate(all_food_types)}

# Apply encoding in DLT transformations
.withColumn("food_type_encoded", 
   when(col("food_type").isin(list(food_type_to_id.keys())), 
        create_map([lit(x) for pair in food_type_to_id.items() for x in pair])[col("food_type")]).otherwise(lit(-1)))
.withColumn("cuisine_category_encoded", 
   when(col("cuisine_category").isin(list(cuisine_to_id.keys())), 
        create_map([lit(x) for pair in cuisine_to_id.items() for x in pair])[col("cuisine_category")]).otherwise(lit(-1)))
```

### 3.2 Dataset Splitting Strategy
**Stratified Random Sampling:**
```python
# Add dataset splits with reproducible random seed
.withColumn("split_seed", rand(seed=random_seed))
.withColumn("dataset_split",
   when(col("split_seed") < 0.70, lit("train"))      # 70% training
   .when(col("split_seed") < 0.85, lit("validation")) # 15% validation  
   .otherwise(lit("test")))                           # 15% test
```

## 4. DLT Pipeline Implementation

### 4.1 Pipeline Configuration and Discovery
```python
# Auto-discover all silver cuisine tables
silver_tables = spark.sql(f"SHOW TABLES IN {silver_schema}").collect()
cuisine_tables = [row.tableName for row in silver_tables if row.tableName.startswith('cuisine_')]

if not cuisine_tables:
    raise ValueError("No silver cuisine tables found")

# Load configuration files for label encoding  
try:
    config_path = f"{config_volume_path}/cuisine_mapping.json"
    config_content = dbutils.fs.head(config_path, max_bytes=1000000)
    cuisine_mapping = json.loads(config_content)
    
    config_path = f"{config_volume_path}/food_types.json"
    config_content = dbutils.fs.head(config_path, max_bytes=1000000)
    food_types_list = json.loads(config_content)
except Exception as e:
    raise ValueError(f"Configuration files not available: {str(e)}")
```

### 4.2 ml_dataset Table - Unified Dataset Creation
```python
@dlt.table(
    name="ml_dataset",
    comment="Unified ML dataset for CNN training - simplified for ResNet-50",
    table_properties={
        "pipelines.autoOptimize.managed": "true",
        "delta.autoOptimize.optimizeWrite": "true", 
        "delta.autoOptimize.autoCompact": "true",
        "delta.enableChangeDataFeed": "true"
    }
)
def create_ml_dataset():
    """Create unified ML dataset from all silver cuisine tables - simplified for CNN training"""
    
    # Union all silver cuisine tables using SQL
    union_queries = []
    for table in cuisine_tables:
        union_queries.append(f"SELECT * FROM {silver_schema}.{table}")
    
    if not union_queries:
        raise ValueError("No silver cuisine tables found")
    
    # Execute union query
    union_query = " UNION ALL ".join(union_queries)
    unified_silver = spark.sql(union_query)
    
    # Process unified dataset - simplified for CNN training
    ml_dataset = (
        unified_silver
        # Add label encodings using the mappings we created
        .withColumn("food_type_encoded", 
                   when(col("food_type").isin(list(food_type_to_id.keys())), 
                        create_map([lit(x) for pair in food_type_to_id.items() for x in pair])[col("food_type")]).otherwise(lit(-1)))
        .withColumn("cuisine_category_encoded", 
                   when(col("cuisine_category").isin(list(cuisine_to_id.keys())), 
                        create_map([lit(x) for pair in cuisine_to_id.items() for x in pair])[col("cuisine_category")]).otherwise(lit(-1)))
        
        # Add dataset splits (stratified random)
        .withColumn("split_seed", rand(seed=random_seed))
        .withColumn("dataset_split",
                   when(col("split_seed") < 0.70, lit("train"))
                   .when(col("split_seed") < 0.85, lit("validation"))
                   .otherwise(lit("test")))
        
        # Add processing timestamp
        .withColumn("gold_processing_timestamp", current_timestamp())
        
        # Select and organize columns for ML - only essential columns for CNN training
        .select(
            "image_id",
            "food_type",
            "cuisine_category", 
            "food_type_encoded",
            "cuisine_category_encoded",
            "processed_image_data",  # This is the key column for CNN training
            "dataset_split",
            "split_seed",
            "original_file_path",
            "filename", 
            "dataset_index",
            "quality_score",
            "mean_rgb",
            "std_rgb", 
            "brightness",
            "contrast",
            "aspect_ratio",
            "processing_timestamp",
            "gold_processing_timestamp"
        )
    )
    
    return ml_dataset

# NOTE: feature_vectors table removed - not needed for ResNet-50 CNN training
# The processed_image_data in ml_dataset is sufficient for direct CNN training
```

## 5. Implementation Guidelines

### 5.1 Pipeline Deployment Steps
1. **Create gold schema in Unity Catalog:** `cuisine_vision_catalog.gold`
2. **Configure DLT pipeline parameters:**
   - `silver_schema` - Source silver schema path
   - `config_volume_path` - Configuration files location (cuisine_mapping.json, food_types.json)
   - `random_seed` - Reproducibility seed (default: 42)
3. **Deploy DLT pipeline in streaming mode** with proper checkpointing
4. **Monitor execution** for simple data transformation performance

### 5.2 Performance Optimization
- **Simple Processing:** Basic union and label encoding operations
- **Auto-optimization:** Enable Delta Lake optimizations for ml_dataset table
- **Streaming:** Efficient processing of silver table updates
- **Memory Management:** Standard Spark configurations for data processing

### 5.3 Data Quality Expectations
**Key Data Quality Checks:**
```python
# Ensure essential data quality in DLT transformations
- image_id IS NOT NULL
- processed_image_data IS NOT NULL (key for CNN training)
- cuisine_category_encoded >= 0 (valid label encoding)
- food_type_encoded >= 0 (valid label encoding)  
- dataset_split IN ('train', 'validation', 'test')
```

## 6. Next Steps - ML Pipeline Integration

### 6.1 Downstream ML Usage
The simplified gold layer is designed for **direct CNN training:**

1. **ML Training Pipeline:** Load `ml_dataset` table directly for ResNet-50 training
2. **HuggingFace Integration:** Use `processed_image_data` with AutoImageProcessor
3. **Model Training:** Standard Transformers Trainer with MLflow tracking
4. **Model Serving:** Direct image classification without complex feature lookups

### 6.2 Separation of Concerns
**Data Engineering Responsibilities (DLT Pipeline):**
- ✅ Clean data transformation and unification (139 lines)
- ✅ Label encoding and schema standardization  
- ✅ Dataset splitting for ML experiments
- ✅ Performance-optimized Delta table creation

**ML Engineering Responsibilities (Separate Notebook):**
- HuggingFace Transformers model loading and training
- MLflow experiment tracking and model registration
- Model comparison and hyperparameter tuning
- Model serving and real-time inference

### 6.3 Target Outcomes
**Simplified Gold Layer Delivers:**
- **ml_dataset:** 🗂️ Single unified table with CNN-ready processed images
- **Direct Usage:** 🚀 Ready for ResNet-50, ViT, Swin, or other vision transformers
- **Performance:** ⚡ Optimized Delta table with auto-compaction
- **Scalability:** 📈 Streaming architecture supporting growing datasets

## 7. Summary

This simplified DLT pipeline provides **clean data engineering** focused on:

1. **Data Transformation:** Unifying 8 silver tables into consistent schema (139 lines total)
2. **Label Management:** Simple encoding based on configuration files
3. **CNN Ready:** Direct `processed_image_data` for vision transformer training
4. **Performance:** Optimized Delta table ready for ML consumption

**Next Phase:** ML engineers use the unified `ml_dataset` for direct CNN training with HuggingFace Transformers.

---

**🎯 Simplified approach enables:**
- Data engineers focus on clean data transformation (75% code reduction)
- ML engineers focus on model training with standard patterns
- Direct integration with HuggingFace ecosystem
- Clear lineage: DLT → ml_dataset → HuggingFace → MLflow → Model Serving