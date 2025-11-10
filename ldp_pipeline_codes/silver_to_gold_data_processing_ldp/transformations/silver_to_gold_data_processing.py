# Silver to Gold Data Processing Pipeline - DLT Implementation (SIMPLIFIED)
"""
Project: Cuisine Vision Pipeline (CVP) - Gold Layer  
Target: Unified ML dataset for CNN training and inference
Source: Silver tables (cuisine_vision_catalog.silver.cuisine_*)
Architecture: Streaming aggregation from 8 silver tables to unified ML-ready structure

Simplified version for direct CNN training (ResNet-50) - feature extraction removed.
Only creates ml_dataset table with processed images and labels.
"""

import dlt
import json
from pyspark.sql.functions import (
    col, lit, current_timestamp, when, rand, create_map
)

# PIPELINE CONFIGURATION
# ============================================================================

# Get pipeline parameters
config_volume_path = spark.conf.get("config_volume_path")
silver_schema = spark.conf.get("silver_schema", "cuisine_vision_catalog.silver")
random_seed = int(spark.conf.get("random_seed", "42"))

# Load configuration files
try:
    # Load cuisine mapping
    config_path = f"{config_volume_path}/cuisine_mapping.json"
    config_content = dbutils.fs.head(config_path, max_bytes=1000000)
    cuisine_mapping = json.loads(config_content)
    
    # Load food types
    config_path = f"{config_volume_path}/food_types.json"
    config_content = dbutils.fs.head(config_path, max_bytes=1000000)
    food_types_list = json.loads(config_content)
    
except Exception as e:
    raise ValueError(f"Configuration files not available: {str(e)}")

# Create label mappings for ML
all_cuisines = sorted(list(cuisine_mapping.keys()))
all_food_types = sorted(food_types_list)

# Create bidirectional mappings
cuisine_to_id = {cuisine: idx for idx, cuisine in enumerate(all_cuisines)}
id_to_cuisine = {idx: cuisine for idx, cuisine in enumerate(all_cuisines)}
food_type_to_id = {food_type: idx for idx, food_type in enumerate(all_food_types)}
id_to_food_type = {idx: food_type for idx, food_type in enumerate(all_food_types)}

# AUTO-DISCOVER SILVER TABLES
# ============================================================================

# Get all tables in silver schema
silver_tables = spark.sql(f"SHOW TABLES IN {silver_schema}").collect()
cuisine_tables = [row.tableName for row in silver_tables if row.tableName.startswith('cuisine_')]

if not cuisine_tables:
    raise ValueError("No silver cuisine tables found")

# GOLD TABLE DEFINITIONS (SIMPLIFIED)
# ============================================================================

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
    
    # Union all silver cuisine tables
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

# End of simplified pipeline