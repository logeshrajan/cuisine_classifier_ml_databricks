# Bronze Volume to Bronze Tables Pipeline - DLT Implementation
"""
Project: Cuisine Vision Pipeline (CVP) - Bronze Layer  
Target: Dynamic bronze tables with Cloud Files Auto Loader from bronze volume
Schema: cuisine_vision_catalog.bronze
"""

import dlt
from datetime import datetime

# Note: 'spark' and 'dbutils' are automatically available in Databricks DLT environment

# Import utility functions
from utilities.utils import (
    load_config_file,
    create_bronze_table_for_food_type,
    print_pipeline_summary,
    get_pipeline_params
)




# ============================================================================
# PIPELINE IMPLEMENTATION
# ============================================================================

# Get runtime parameters from DLT pipeline configuration
pipeline_params = get_pipeline_params(spark)

# Extract commonly used configuration values
config_volume_path = pipeline_params["config_volume_path"]
bronze_volume_path = pipeline_params["bronze_volume_path"]  # Direct read from bronze volume

# Load cuisine mapping directly
cuisine_mapping = load_config_file("cuisine_mapping.json", config_volume_path, dbutils)
if not cuisine_mapping:
    raise ValueError("Cuisine mapping configuration not available")

# Create food-to-cuisine mapping
food_to_cuisine_map = {}
for cuisine, foods in cuisine_mapping.items():
    for food in foods:
        food_to_cuisine_map[food] = cuisine

# Extract commonly used counts
total_food_types = len(food_to_cuisine_map)
total_cuisines = len(cuisine_mapping)

# Load food types list directly  
food_types_list = load_config_file("food_types.json", config_volume_path, dbutils)
if not food_types_list:
    raise ValueError("Food types configuration not available")

# Generate bronze tables for all food types
created_tables = []
for food_type in sorted(food_to_cuisine_map.keys()):
    try:
        table_function = create_bronze_table_for_food_type(food_type, food_to_cuisine_map, pipeline_params, spark)
        created_tables.append(f"food_{food_type}")
    except Exception as e:
        # Log error but continue with other tables
        pass

# Print pipeline summary
print_pipeline_summary(food_to_cuisine_map, created_tables, cuisine_mapping)
