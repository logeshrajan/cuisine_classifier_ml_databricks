# Bronze to Silver Data Processing Pipeline - DLT Implementation (Simplified)
"""
Project: Cuisine Vision Pipeline (CVP) - Silver Layer  
Target: 8 cuisine-based silver tables with processed 224x224 images
Source: Bronze tables (cuisine_vision_catalog.bronze.food_*)
Architecture: Streaming image processing with cuisine derivation

This version embeds all functionality to avoid module serialization issues.
"""

import dlt
import json
import io
import uuid
from PIL import Image, ImageStat
from pyspark.sql.functions import (
    col, lit, current_timestamp, when, length, 
    udf
)
from pyspark.sql.types import (
    BinaryType, StructType, StructField, ArrayType, 
    DoubleType, IntegerType, StringType, BooleanType
)

# EMBEDDED UTILITY FUNCTIONS
# ============================================================================

def validate_image_quality(image_binary):
    """Validate if image binary data is valid and meets quality requirements"""
    try:
        if not image_binary:
            return False
            
        # Convert bytes to PIL Image
        image_stream = io.BytesIO(image_binary)
        img = Image.open(image_stream)
        
        # Basic validation checks
        if img.size[0] < 32 or img.size[1] < 32:  # Too small
            return False
        if img.size[0] > 4096 or img.size[1] > 4096:  # Too large
            return False
        if img.mode not in ['RGB', 'RGBA', 'L']:  # Invalid color mode
            return False
            
        return True
        
    except Exception as e:
        return False

def process_image_to_224x224(image_binary):
    """Process image binary to 224x224 JPEG format"""
    try:
        if not image_binary:
            return None
            
        # Open image from binary data
        image_stream = io.BytesIO(image_binary)
        img = Image.open(image_stream)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate resize dimensions maintaining aspect ratio
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height
        
        if aspect_ratio > 1:  # Wider than tall
            new_width = 224
            new_height = int(224 / aspect_ratio)
        else:  # Taller than wide or square
            new_height = 224
            new_width = int(224 * aspect_ratio)
        
        # Resize image
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create 224x224 canvas with black padding
        canvas = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Calculate position to center the image
        paste_x = (224 - new_width) // 2
        paste_y = (224 - new_height) // 2
        
        # Paste resized image onto canvas
        canvas.paste(img_resized, (paste_x, paste_y))
        
        # Convert to JPEG binary
        output_stream = io.BytesIO()
        canvas.save(output_stream, format='JPEG', quality=95)
        output_binary = output_stream.getvalue()
        
        return output_binary
        
    except Exception as e:
        return None

def extract_image_features(image_binary):
    """Extract image features: RGB means, quality score, dimensions, histogram, texture features"""
    try:
        if not image_binary:
            return {
                "mean_rgb": [0.0, 0.0, 0.0], 
                "std_rgb": [0.0, 0.0, 0.0],
                "quality_score": 0.0, 
                "brightness": 0.0,
                "contrast": 0.0,
                "original_width": 0, 
                "original_height": 0,
                "aspect_ratio": 0.0
            }
            
        # Open image from binary data
        image_stream = io.BytesIO(image_binary)
        img = Image.open(image_stream)
        
        # Get original dimensions
        original_width, original_height = img.size
        aspect_ratio = original_width / original_height
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate RGB statistics
        stat = ImageStat.Stat(img)
        mean_rgb = stat.mean[:3]  # R, G, B means
        std_rgb = stat.stddev[:3] if len(stat.stddev) >= 3 else [0.0, 0.0, 0.0]
        
        # Convert to simple brightness calculation using PIL
        # Convert to grayscale using PIL built-in method
        gray_img = img.convert('L')  # Convert to grayscale
        gray_stat = ImageStat.Stat(gray_img)
        
        # Calculate brightness and contrast using PIL statistics
        brightness = gray_stat.mean[0] / 255.0  # Normalize to 0-1
        contrast = gray_stat.stddev[0] / 255.0 if gray_stat.stddev else 0.0  # Normalize to 0-1
        
        # Simple quality score based on contrast (higher contrast = better quality)
        quality_score = min(1.0, contrast * 2.0)  # Scale contrast as quality indicator
        
        return {
            "mean_rgb": list(mean_rgb),  # Ensure it's a list, not numpy array
            "std_rgb": list(std_rgb),    # Ensure it's a list, not numpy array
            "quality_score": float(quality_score),
            "brightness": float(brightness),
            "contrast": float(contrast),
            "original_width": int(original_width),
            "original_height": int(original_height),
            "aspect_ratio": float(aspect_ratio)
        }
        
    except Exception as e:
        return {
            "mean_rgb": [0.0, 0.0, 0.0],
            "std_rgb": [0.0, 0.0, 0.0], 
            "quality_score": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "original_width": 0, 
            "original_height": 0,
            "aspect_ratio": 0.0
        }

def generate_image_id(food_type, dataset_index):
    """Generate unique image ID combining food type and dataset index"""
    try:
        if dataset_index is not None:
            return f"{food_type}_{int(dataset_index):06d}"
        else:
            # Fallback to UUID if no dataset index
            return f"{food_type}_{str(uuid.uuid4())[:8]}"
    except:
        return f"{food_type}_{str(uuid.uuid4())[:8]}"

# REGISTER UDFs
# ============================================================================
process_image_udf = udf(process_image_to_224x224, BinaryType())
extract_features_udf = udf(extract_image_features, StructType([
    StructField("mean_rgb", ArrayType(DoubleType()), True),
    StructField("std_rgb", ArrayType(DoubleType()), True),
    StructField("quality_score", DoubleType(), True),
    StructField("brightness", DoubleType(), True),
    StructField("contrast", DoubleType(), True),
    StructField("original_width", IntegerType(), True),
    StructField("original_height", IntegerType(), True),
    StructField("aspect_ratio", DoubleType(), True)
]))
validate_quality_udf = udf(validate_image_quality, BooleanType())
generate_id_udf = udf(generate_image_id, StringType())

# PIPELINE CONFIGURATION
# ============================================================================

# Get pipeline parameters
config_volume_path = spark.conf.get("config_volume_path")
bronze_schema = spark.conf.get("bronze_schema", "cuisine_vision_catalog.bronze")

# Load cuisine mapping configuration
try:
    config_path = f"{config_volume_path}/cuisine_mapping.json"
    config_content = dbutils.fs.head(config_path, max_bytes=1000000)
    cuisine_mapping = json.loads(config_content)
except Exception as e:
    raise ValueError(f"Cuisine mapping configuration not available: {str(e)}")

# Load food types list for validation
try:
    config_path = f"{config_volume_path}/food_types.json"
    config_content = dbutils.fs.head(config_path, max_bytes=1000000)
    food_types_list = json.loads(config_content)
except Exception as e:
    raise ValueError(f"Food types configuration not available: {str(e)}")

# Create food_type -> cuisine mapping for efficient lookups
food_to_cuisine_map = {}
for cuisine, foods in cuisine_mapping.items():
    for food in foods:
        food_to_cuisine_map[food] = cuisine

total_cuisines = len(cuisine_mapping)
total_food_types = len(food_to_cuisine_map)

# SILVER TABLE GENERATION FACTORY
# ============================================================================

def create_silver_table_factory(cuisine_name, food_list, bronze_schema):
    """Factory function to create DLT table for a specific cuisine"""
    
    # Capture the values in the closure properly
    _cuisine_name = cuisine_name
    _food_list = list(food_list)  # Make a copy
    _bronze_schema = bronze_schema
    
    # Add image metadata to enable image preview in Databricks
    image_meta = {"spark.contentAnnotation": '{"mimeType": "image/jpeg"}'}
    
    @dlt.table(
        name=f"cuisine_{_cuisine_name}",
        comment=f"Silver table for {_cuisine_name} cuisine with processed 224x224 images",
        table_properties={
            "pipelines.autoOptimize.managed": "true",
            "delta.autoOptimize.optimizeWrite": "true",
            "delta.autoOptimize.autoCompact": "true",
            "delta.enableChangeDataFeed": "true"
        }
    )
    @dlt.expect_all_or_drop({
        "valid_image_data": "processed_image_data IS NOT NULL",
        "valid_food_type": "food_type IS NOT NULL",
        "valid_cuisine": "cuisine_category IS NOT NULL", 
        "valid_dimensions": "processed_width = 224 AND processed_height = 224"
    })
    @dlt.expect( # Just warn for quality - don't drop
        "good_quality", "quality_score >= 0.3"  # Higher threshold for warning
    )
    def silver_cuisine_table():
        # Create base query to union all bronze tables for this cuisine
        bronze_tables = []
        for food_type in _food_list:
            bronze_table_name = f"{_bronze_schema}.food_{food_type}"
            bronze_tables.append(f"SELECT * FROM {bronze_table_name}")
        
        # Union all bronze tables for this cuisine
        union_query = " UNION ALL ".join(bronze_tables)
        bronze_data = spark.sql(union_query)
        
        # Process the unified bronze data
        processed_data = (
            bronze_data
            # Filter valid images only
            .filter(col("content").isNotNull())
            .filter(col("file_format").isin(["jpg", "jpeg", "png"]))
            
            # Validate image quality
            .filter(validate_quality_udf(col("content")))
            
            # Generate image ID
            .withColumn("image_id", 
                       generate_id_udf(col("food_type"), col("dataset_index")))
            
            # Add cuisine category
            .withColumn("cuisine_category", lit(_cuisine_name))
            
            # Process image to 224x224 with metadata for display
            .withColumn("processed_image_data", 
                       process_image_udf(col("content")).alias("processed_image_data", metadata=image_meta))
            
            # Extract image features
            .withColumn("features", extract_features_udf(col("content")))
            
            # Expand features into separate columns
            .withColumn("mean_rgb", col("features.mean_rgb"))
            .withColumn("std_rgb", col("features.std_rgb"))
            .withColumn("quality_score", col("features.quality_score"))
            .withColumn("brightness", col("features.brightness"))
            .withColumn("contrast", col("features.contrast"))
            .withColumn("original_width", col("features.original_width"))
            .withColumn("original_height", col("features.original_height"))
            .withColumn("aspect_ratio", col("features.aspect_ratio"))
            
            # Add processed dimensions and file size
            .withColumn("processed_width", lit(224))
            .withColumn("processed_height", lit(224))
            .withColumn("processed_file_size", 
                       when(col("processed_image_data").isNotNull(), 
                            length(col("processed_image_data")))
                       .otherwise(lit(0)))
            
            # Rename columns for clarity
            .withColumn("original_file_path", col("path"))
            .withColumn("original_file_size", col("length"))
            .withColumn("processing_timestamp", current_timestamp())
            
            # Select final columns for silver table
            .select(
                "image_id",
                "food_type", 
                "cuisine_category",
                "processed_image_data",
                "original_file_path",
                "filename",
                "file_format", 
                "dataset_index",
                "original_file_size",
                "processed_file_size",
                "original_width",
                "original_height", 
                "processed_width",
                "processed_height",
                "mean_rgb",
                "std_rgb",
                "quality_score",
                "brightness",
                "contrast",
                "aspect_ratio",
                "ingested_at",
                "processing_timestamp"
            )
            
            # Filter out failed image processing
            .filter(col("processed_image_data").isNotNull())
        )
        
        return processed_data
    
    return silver_cuisine_table

# SILVER TABLE GENERATION
# ============================================================================

# Generate silver tables for each cuisine using factory pattern
created_tables = []
for cuisine_name, food_list in cuisine_mapping.items():
    # Create the table function using factory
    table_func = create_silver_table_factory(cuisine_name, food_list, bronze_schema)
    created_tables.append(f"cuisine_{cuisine_name}")
    
    # Register the function in globals so DLT can find it
    globals()[f"cuisine_{cuisine_name}_table"] = table_func

# End of pipeline