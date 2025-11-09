# Silver to Gold Data Processing Pipeline - DLT Implementation
"""
Project: Cuisine Vision Pipeline (CVP) - Gold Layer  
Target: Unified ML dataset with advanced features for CNN training and inference
Source: Silver tables (cuisine_vision_catalog.silver.cuisine_*)
Architecture: Streaming aggregation from 8 silver tables to unified ML-ready structure

This version creates ML-ready datasets with advanced feature engineering.
"""

import dlt
import json
import io
import numpy as np
from PIL import Image
from pyspark.sql.functions import (
    col, lit, current_timestamp, when, udf, struct, rand, create_map
)
from pyspark.sql.types import (
    StructType, StructField, ArrayType, DoubleType, 
    IntegerType, StringType
)

# EMBEDDED ML FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_advanced_color_features(image_binary):
    """Extract advanced color features for ML training"""
    try:
        if not image_binary:
            return {
                "color_histogram": [0.0] * 64,
                "dominant_colors": [[0, 0, 0]] * 5,
                "color_percentages": [0.0] * 5,
                "color_temperature": 0.0,
                "saturation_mean": 0.0,
                "hue_distribution": [0.0] * 12
            }
            
        # Open image from binary data
        image_stream = io.BytesIO(image_binary)
        img = Image.open(image_stream)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get image data as numpy array for processing
        img_array = np.array(img)
        pixels = img_array.reshape(-1, 3)
        
        # 1. Color Histogram (64 bins - compress RGB to 4x4x4)
        hist_r = np.histogram(pixels[:, 0], bins=16, range=(0, 256))[0]
        hist_g = np.histogram(pixels[:, 1], bins=16, range=(0, 256))[0]
        hist_b = np.histogram(pixels[:, 2], bins=16, range=(0, 256))[0]
        
        # Combine histograms and normalize
        color_histogram = np.concatenate([hist_r, hist_g, hist_b]).astype(float)
        if color_histogram.sum() > 0:
            color_histogram = color_histogram / color_histogram.sum()
        color_histogram = color_histogram.tolist()[:64]  # Ensure 64 dimensions
        
        # 2. Dominant Colors using simple clustering (k-means approximation)
        # Use simple method: find most frequent colors
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Sort by frequency and get top 5
        top_indices = np.argsort(counts)[-5:][::-1]
        dominant_colors = []
        color_percentages = []
        total_pixels = len(pixels)
        
        for i in range(5):
            if i < len(top_indices):
                color = unique_colors[top_indices[i]].tolist()
                percentage = float(counts[top_indices[i]]) / total_pixels
            else:
                color = [0, 0, 0]
                percentage = 0.0
            dominant_colors.append(color)
            color_percentages.append(percentage)
        
        # 3. Color Temperature (warm vs cool)
        # Calculate average color and determine temperature
        avg_color = np.mean(pixels, axis=0)
        # Simple heuristic: warm colors have more red/yellow, cool have more blue
        color_temperature = float((avg_color[0] + avg_color[1] - avg_color[2] * 2) / 255.0)
        color_temperature = max(-1.0, min(1.0, color_temperature))  # Clamp to [-1, 1]
        
        # 4. Saturation (convert to HSV for saturation calculation)
        img_hsv = img.convert('HSV')
        hsv_array = np.array(img_hsv)
        saturation_mean = float(np.mean(hsv_array[:, :, 1]) / 255.0)
        
        # 5. Hue Distribution (12 bins for hue)
        hue_values = hsv_array[:, :, 0].flatten()
        hue_hist = np.histogram(hue_values, bins=12, range=(0, 256))[0].astype(float)
        if hue_hist.sum() > 0:
            hue_hist = hue_hist / hue_hist.sum()
        hue_distribution = hue_hist.tolist()
        
        return {
            "color_histogram": color_histogram,
            "dominant_colors": dominant_colors,
            "color_percentages": color_percentages,
            "color_temperature": color_temperature,
            "saturation_mean": saturation_mean,
            "hue_distribution": hue_distribution
        }
        
    except Exception as e:
        # Return default values on error
        return {
            "color_histogram": [0.0] * 64,
            "dominant_colors": [[0, 0, 0]] * 5,
            "color_percentages": [0.0] * 5,
            "color_temperature": 0.0,
            "saturation_mean": 0.0,
            "hue_distribution": [0.0] * 12
        }

def extract_texture_features(image_binary):
    """Extract texture features using edge detection and basic patterns"""
    try:
        if not image_binary:
            return {
                "edge_density": 0.0,
                "texture_contrast": 0.0,
                "texture_energy": 0.0,
                "gradient_magnitude": 0.0,
                "smoothness": 0.0
            }
            
        # Open image from binary data
        image_stream = io.BytesIO(image_binary)
        img = Image.open(image_stream)
        
        # Convert to grayscale for texture analysis
        gray_img = img.convert('L')
        gray_array = np.array(gray_img)
        
        # 1. Edge Density using simple edge detection
        # Apply basic edge filters
        edges_h = np.abs(np.diff(gray_array, axis=0))
        edges_v = np.abs(np.diff(gray_array, axis=1))
        
        # Calculate edge density
        edge_pixels = np.sum(edges_h > 30) + np.sum(edges_v > 30)
        total_pixels = gray_array.size
        edge_density = float(edge_pixels) / total_pixels
        
        # 2. Texture Contrast (standard deviation)
        texture_contrast = float(np.std(gray_array) / 255.0)
        
        # 3. Texture Energy (measure of uniformity)
        # Use histogram to calculate energy
        hist = np.histogram(gray_array, bins=256, range=(0, 256))[0]
        hist_norm = hist / hist.sum() if hist.sum() > 0 else hist
        texture_energy = float(np.sum(hist_norm ** 2))
        
        # 4. Gradient Magnitude (average gradient strength)
        grad_magnitude = np.sqrt(edges_h[:-1, :] ** 2 + edges_v[:, :-1] ** 2)
        gradient_magnitude = float(np.mean(grad_magnitude) / 255.0)
        
        # 5. Smoothness (inverse of variance)
        variance = np.var(gray_array)
        smoothness = float(1.0 / (1.0 + variance / (255.0 ** 2)))
        
        return {
            "edge_density": edge_density,
            "texture_contrast": texture_contrast,
            "texture_energy": texture_energy,
            "gradient_magnitude": gradient_magnitude,
            "smoothness": smoothness
        }
        
    except Exception as e:
        return {
            "edge_density": 0.0,
            "texture_contrast": 0.0,
            "texture_energy": 0.0,
            "gradient_magnitude": 0.0,
            "smoothness": 0.0
        }

def extract_shape_features(image_binary):
    """Extract basic shape and structural features"""
    try:
        if not image_binary:
            return {
                "symmetry_score": 0.0,
                "structural_complexity": 0.0,
                "aspect_ratio_normalized": 0.5
            }
            
        # Open image from binary data
        image_stream = io.BytesIO(image_binary)
        img = Image.open(image_stream)
        
        # Convert to grayscale
        gray_img = img.convert('L')
        gray_array = np.array(gray_img)
        
        # 1. Symmetry Score (horizontal symmetry)
        left_half = gray_array[:, :gray_array.shape[1]//2]
        right_half = gray_array[:, gray_array.shape[1]//2:]
        right_half_flipped = np.flip(right_half, axis=1)
        
        # Ensure same dimensions
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]
        
        # Calculate symmetry as normalized cross-correlation
        if left_half.size > 0 and right_half_flipped.size > 0:
            diff = np.abs(left_half.astype(float) - right_half_flipped.astype(float))
            symmetry_score = float(1.0 - np.mean(diff) / 255.0)
        else:
            symmetry_score = 0.0
        
        # 2. Structural Complexity (edge distribution entropy)
        edges = np.abs(np.diff(gray_array, axis=0)) + np.abs(np.diff(gray_array, axis=1)[:, :-1])
        if edges.size > 0:
            # Calculate entropy of edge distribution
            edge_hist = np.histogram(edges, bins=16)[0]
            edge_hist_norm = edge_hist / edge_hist.sum() if edge_hist.sum() > 0 else edge_hist
            entropy = -np.sum(edge_hist_norm * np.log(edge_hist_norm + 1e-10))
            structural_complexity = float(entropy / np.log(16))  # Normalize by max entropy
        else:
            structural_complexity = 0.0
        
        # 3. Aspect Ratio (normalized to 0-1)
        height, width = gray_array.shape
        aspect_ratio = width / height if height > 0 else 1.0
        # Normalize aspect ratio to 0-1 range (0.5 = square, 0 = very tall, 1 = very wide)
        aspect_ratio_normalized = float(min(aspect_ratio, 2.0) / 2.0)
        
        return {
            "symmetry_score": max(0.0, min(1.0, symmetry_score)),
            "structural_complexity": max(0.0, min(1.0, structural_complexity)),
            "aspect_ratio_normalized": aspect_ratio_normalized
        }
        
    except Exception as e:
        return {
            "symmetry_score": 0.0,
            "structural_complexity": 0.0,
            "aspect_ratio_normalized": 0.5
        }

def generate_feature_vector_v1(color_features, texture_features, shape_features, basic_features):
    """Generate 100-dimensional feature vector for ML"""
    try:
        # Combine all features into 100-dimensional vector
        feature_vector = []
        
        # Color features (64 dimensions)
        feature_vector.extend(color_features.get("color_histogram", [0.0] * 64)[:64])
        
        # Advanced color features (16 dimensions)
        feature_vector.extend([
            color_features.get("color_temperature", 0.0),
            color_features.get("saturation_mean", 0.0)
        ])
        # Add dominant color percentages (top 5)
        percentages = color_features.get("color_percentages", [0.0] * 5)[:5]
        feature_vector.extend(percentages)
        # Add hue distribution (truncated to 9)
        hue_dist = color_features.get("hue_distribution", [0.0] * 12)[:9]
        feature_vector.extend(hue_dist)
        
        # Texture features (10 dimensions)
        feature_vector.extend([
            texture_features.get("edge_density", 0.0),
            texture_features.get("texture_contrast", 0.0),
            texture_features.get("texture_energy", 0.0),
            texture_features.get("gradient_magnitude", 0.0),
            texture_features.get("smoothness", 0.0)
        ])
        
        # Shape features (3 dimensions)
        feature_vector.extend([
            shape_features.get("symmetry_score", 0.0),
            shape_features.get("structural_complexity", 0.0),
            shape_features.get("aspect_ratio_normalized", 0.5)
        ])
        
        # Basic features from silver layer (10 dimensions)
        mean_rgb = basic_features.get("mean_rgb", [0.0, 0.0, 0.0])[:3]
        std_rgb = basic_features.get("std_rgb", [0.0, 0.0, 0.0])[:3]
        feature_vector.extend(mean_rgb)
        feature_vector.extend(std_rgb)
        feature_vector.extend([
            basic_features.get("brightness", 0.0),
            basic_features.get("contrast", 0.0),
            basic_features.get("quality_score", 0.0),
            basic_features.get("aspect_ratio", 1.0)
        ])
        
        # Padding to ensure exactly 100 dimensions
        while len(feature_vector) < 100:
            feature_vector.append(0.0)
        
        # Truncate to exactly 100 dimensions
        feature_vector = feature_vector[:100]
        
        # Convert to list of floats
        return [float(x) for x in feature_vector]
        
    except Exception as e:
        # Return zero vector on error
        return [0.0] * 100

def generate_perceptual_hash(image_binary):
    """Generate perceptual hash for image similarity detection"""
    try:
        if not image_binary:
            return "0" * 64
            
        # Open image and resize to 8x8 for hashing
        image_stream = io.BytesIO(image_binary)
        img = Image.open(image_stream)
        img = img.convert('L').resize((8, 8))
        
        # Convert to array and calculate average
        pixels = np.array(img)
        avg = np.mean(pixels)
        
        # Create hash based on pixel values vs average
        hash_bits = (pixels > avg).flatten()
        hash_string = ''.join(['1' if bit else '0' for bit in hash_bits])
        
        return hash_string
        
    except Exception as e:
        return "0" * 64

# REGISTER UDFs
# ============================================================================

# Define UDF return types
color_features_schema = StructType([
    StructField("color_histogram", ArrayType(DoubleType()), True),
    StructField("dominant_colors", ArrayType(ArrayType(IntegerType())), True),
    StructField("color_percentages", ArrayType(DoubleType()), True),
    StructField("color_temperature", DoubleType(), True),
    StructField("saturation_mean", DoubleType(), True),
    StructField("hue_distribution", ArrayType(DoubleType()), True)
])

texture_features_schema = StructType([
    StructField("edge_density", DoubleType(), True),
    StructField("texture_contrast", DoubleType(), True),
    StructField("texture_energy", DoubleType(), True),
    StructField("gradient_magnitude", DoubleType(), True),
    StructField("smoothness", DoubleType(), True)
])

shape_features_schema = StructType([
    StructField("symmetry_score", DoubleType(), True),
    StructField("structural_complexity", DoubleType(), True),
    StructField("aspect_ratio_normalized", DoubleType(), True)
])

# Register UDFs
extract_color_features_udf = udf(extract_advanced_color_features, color_features_schema)
extract_texture_features_udf = udf(extract_texture_features, texture_features_schema)
extract_shape_features_udf = udf(extract_shape_features, shape_features_schema)
generate_feature_vector_udf = udf(lambda cf, tf, sf, bf: generate_feature_vector_v1(cf, tf, sf, bf), ArrayType(DoubleType()))
generate_perceptual_hash_udf = udf(generate_perceptual_hash, StringType())

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

# GOLD TABLE DEFINITIONS
# ============================================================================

@dlt.table(
    name="ml_dataset",
    comment="Unified ML dataset with all cuisines for training and inference",
    table_properties={
        "pipelines.autoOptimize.managed": "true",
        "delta.autoOptimize.optimizeWrite": "true", 
        "delta.autoOptimize.autoCompact": "true",
        "delta.enableChangeDataFeed": "true"
    }
)
def create_ml_dataset():
    """Create unified ML dataset from all silver cuisine tables"""
    
    # Union all silver cuisine tables
    union_queries = []
    for table in cuisine_tables:
        union_queries.append(f"SELECT * FROM {silver_schema}.{table}")
    
    if not union_queries:
        raise ValueError("No silver cuisine tables found")
    
    # Execute union query
    union_query = " UNION ALL ".join(union_queries)
    unified_silver = spark.sql(union_query)
    
    # Process unified dataset
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
        
        # Select and organize columns for ML
        .select(
            "image_id",
            "food_type",
            "cuisine_category", 
            "food_type_encoded",
            "cuisine_category_encoded",
            "processed_image_data",
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

@dlt.table(
    name="feature_vectors", 
    comment="Advanced feature vectors and embeddings for enhanced ML training",
    table_properties={
        "pipelines.autoOptimize.managed": "true",
        "delta.autoOptimize.optimizeWrite": "true",
        "delta.autoOptimize.autoCompact": "true"
    }
)
def create_feature_vectors():
    """Create advanced feature vectors from ML dataset"""
    
    # Read from ml_dataset
    ml_data = dlt.read("ml_dataset")
    
    feature_vectors = (
        ml_data
        # Extract advanced color features
        .withColumn("color_features", extract_color_features_udf(col("processed_image_data")))
        
        # Extract texture features  
        .withColumn("texture_features", extract_texture_features_udf(col("processed_image_data")))
        
        # Extract shape features
        .withColumn("shape_features", extract_shape_features_udf(col("processed_image_data")))
        
        # Combine basic features for ML vector
        .withColumn("basic_features", 
                   struct(
                       col("mean_rgb").alias("mean_rgb"),
                       col("std_rgb").alias("std_rgb"), 
                       col("brightness").alias("brightness"),
                       col("contrast").alias("contrast"),
                       col("quality_score").alias("quality_score"),
                       col("aspect_ratio").alias("aspect_ratio")
                   ))
        
        # Generate 100-dimensional feature vector
        .withColumn("feature_vector_v1", 
                   generate_feature_vector_udf(
                       col("color_features"),
                       col("texture_features"), 
                       col("shape_features"),
                       col("basic_features")
                   ))
        
        # Generate perceptual hash
        .withColumn("perceptual_hash", generate_perceptual_hash_udf(col("processed_image_data")))
        
        # Add timestamp
        .withColumn("feature_extraction_timestamp", current_timestamp())
        
        # Select final columns
        .select(
            "image_id",
            col("color_features.color_histogram").alias("color_histogram"),
            col("color_features.dominant_colors").alias("dominant_colors"),
            col("color_features.color_percentages").alias("color_percentages"),
            col("color_features.color_temperature").alias("color_temperature"),
            col("color_features.saturation_mean").alias("saturation_mean"),
            col("color_features.hue_distribution").alias("hue_distribution"),
            col("texture_features.edge_density").alias("edge_density"),
            col("texture_features.texture_contrast").alias("texture_contrast"),
            col("texture_features.texture_energy").alias("texture_energy"),
            col("texture_features.gradient_magnitude").alias("gradient_magnitude"),
            col("texture_features.smoothness").alias("smoothness"),
            col("shape_features.symmetry_score").alias("symmetry_score"),
            col("shape_features.structural_complexity").alias("structural_complexity"),
            col("shape_features.aspect_ratio_normalized").alias("aspect_ratio_normalized"),
            "feature_vector_v1",
            "perceptual_hash",
            "feature_extraction_timestamp"
        )
    )
    
    return feature_vectors

# End of pipeline