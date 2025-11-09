# Bronze Pipeline Utility Functions
"""
Utility functions for the Bronze Layer DLT Pipeline
Contains configuration loading, parameter management, and helper functions
"""

import json
from pyspark.sql.functions import col, lit, current_timestamp, when, regexp_extract, concat, udf
from pyspark.sql.types import StringType
import dlt

def get_pipeline_params(spark):
    """Get pipeline parameters with defaults"""
    return {
        # Essential path parameters for bronze volume approach
        "config_volume_path": spark.conf.get("config_volume_path"),  # For JSON config files only
        "bronze_volume_path": spark.conf.get("bronze_volume_path")   # Source data and schema location
    }


def load_config_file(filename, config_volume_path, dbutils):
    """Load configuration file from config volume using dbutils"""
    try:
        config_path = f"{config_volume_path}/{filename}"
        config_content = dbutils.fs.head(config_path, max_bytes=1000000)
        return json.loads(config_content)
    except Exception as e:
        return None


def extract_dataset_index(filename):
    """Extract dataset index from filename"""
    # Expected format: food_type_idx_xxxxxx.jpg
    try:
        if "_idx_" in filename:
            index_part = filename.split("_idx_")[1].split(".")[0]
            return int(index_part)
    except:
        pass
    return None


def create_bronze_table_for_food_type(food_type, food_to_cuisine_map, pipeline_params, spark):
    """Create a bronze table for a specific food type using Auto Loader to ingest files from bronze volume"""
    
    bronze_volume_path = pipeline_params['bronze_volume_path']  # Reading directly from bronze volume
    
    source_path = f"{bronze_volume_path}/{food_type}"  # Source is bronze volume
    table_name = f"food_{food_type}"
    schema_location = f"{bronze_volume_path}/autoloader_schemas/{table_name}"  # Schema co-located with data
    
    # Configure table properties for performance
    table_properties = {
        "pipelines.autoOptimize.managed": "true",
        "delta.autoOptimize.optimizeWrite": "true",
        "delta.autoOptimize.autoCompact": "true"
    }
    
    @dlt.table(
        name=table_name,
        comment=f"Bronze table for {food_type} images - Auto Loader file ingestion from bronze volume",
        table_properties=table_properties
    )
    def bronze_food_table():
        return (
            spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "binaryFile")                    # Binary file format for direct ingestion
            .option("cloudFiles.schemaLocation", schema_location)         # Schema evolution location
            .option("cloudFiles.includeExistingFiles", "true")           # Include existing files on first run
            .option("cloudFiles.allowOverwrites", "false")               # Prevent reprocessing
            .option("cloudFiles.maxFilesPerTrigger", "100")              # Batch size control
            .load(source_path)
            .withColumn("ingested_at", current_timestamp())               # Add ingestion timestamp
            .withColumn("filename", regexp_extract(col("path"), r"([^/]+)$", 1))  # Extract filename from path
            .withColumn("file_format", regexp_extract(col("filename"), r"\.(\w+)$", 1))  # Extract file extension
            .withColumn("dataset_index", 
                when(col("filename").contains("_idx_"), 
                     regexp_extract(col("filename"), r"_idx_(\d+)", 1).cast("bigint"))
                .otherwise(lit(None).cast("bigint")))                     # Extract dataset index
            .withColumn("food_type", lit(food_type))                      # Add food type as literal
            # Auto Loader base columns: path, modificationTime, length, content
            # Enhanced metadata: ingested_at, filename, file_format, dataset_index, food_type
            # No cuisine category - will be derived in silver layer from food_type
        )
    
    return bronze_food_table


def print_pipeline_summary(food_to_cuisine_map, created_tables, cuisine_mapping):
    """Print a summary of the pipeline generation results"""
    print(f"\n📊 Summary:")
    print(f"  - Total food types: {len(food_to_cuisine_map)}")
    print(f"  - Tables created: {len(created_tables)}")
    print(f"  - Cuisines covered: {len(cuisine_mapping)}")


# Note: Auto Loader reads from bronze volume and creates bronze tables
# Note: No file movement or archival needed - files stay in permanent bronze location  
# Note: Schema files co-located with data in bronze volume for operational simplicity
# Note: Bronze layer contains raw data + essential metadata (no business logic like cuisine mapping)

