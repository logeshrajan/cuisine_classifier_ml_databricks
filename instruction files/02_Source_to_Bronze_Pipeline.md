# 02. Source to Bronze Pipeline Implementation

**Project:** Cuisine Vision Pipeline (CVP) - Clean Bronze Layer  
**Focus:** Raw data ingestion with clean separation from ML engineering  
**Target:** 71 bronze tables (one per food type) with image binary data  
**Architecture:** Direct source-to-bronze with Unity Catalog volumes and Delta Live Tables
**Scope:** Data engineering only - pure data transformation and ingestion

## 1. Pipeline Overview

### 1.1 Streamlined Architecture
- **Flow:** HuggingFace → Bronze Volume → Bronze Tables → Silver → Gold
- **Eliminated:** Landing layer complexity and archival management
- **Bronze Volume:** Direct storage location for raw image files
- **Bronze Tables:** 71 DLT-managed tables for structured access

### 1.2 Core Components
**Notebook:** `source_to_bronze_volume_ingestion.ipynb`
- Direct Hugging Face to bronze volume ingestion
- Food dataset with strategic multi-cuisine selection
- Index-based tracking and duplicate prevention

**DLT Pipeline:** `bronze_tables_ingestion.py`
- Auto Loader reading from bronze volume
- Dynamic table creation for all configured food types
- Pure data engineering - no ML logic or Feature Store integration
- Optimized for continuous streaming processing

**Configuration Management:**
- `cuisine_mapping.json` - Maps food types to cuisine categories
- `food_types.json` - Defines all supported food types
- Stored in Unity Catalog config volume for centralized access

### 1.3 Food Selection Strategy (Configuration-Driven)
Strategic selection across traditional cuisine categories defined in `food_types.json`:
- **Italian:** pizza, lasagna, risotto, spaghetti_bolognese, spaghetti_carbonara, ravioli, bruschetta, tiramisu, cannoli, panna_cotta, garlic_bread
- **Japanese:** sushi, ramen, sashimi, takoyaki, miso_soup, gyoza, edamame  
- **American (18):** hamburger, hot_dog, pancakes, waffles, apple_pie, cheesecake, macaroni_and_cheese, grilled_cheese_sandwich, pulled_pork_sandwich, baby_back_ribs, chicken_wings, onion_rings, club_sandwich, eggs_benedict, omelette, lobster_roll_sandwich, crab_cakes, clam_chowder
- **French (10):** french_fries, french_onion_soup, french_toast, croque_madame, escargots, creme_brulee, foie_gras, macarons, lobster_bisque, mussels
- **Mexican (7):** tacos, chicken_quesadilla, nachos, churros, guacamole, huevos_rancheros, ceviche
- **Chinese (5):** dumplings, fried_rice, peking_duck, hot_and_sour_soup, spring_rolls
- **Mediterranean (10):** greek_salad, hummus, falafel, baklava, grilled_salmon, fried_calamari, oysters, scallops, tuna_tartare, paella
- **British_Canadian (3):** fish_and_chips, poutine, pad_thai

## 2. Bronze Volume Ingestion

### 2.1 Direct Ingestion Implementation
**File:** `source_to_bronze_volume_ingestion.ipynb`

**Key Function:** `ingest_food_images_to_bronze()`
- Direct save to bronze volume (no landing layer)
- Index-based tracking with `{food_type}_idx_{dataset_index:06d}.jpg` naming
- Incremental ingestion with duplicate prevention
- Status tracking via `ingestion_status.json`

**Volume Structure:**
```
/Volumes/cuisine_vision_catalog/bronze/bronze_volume/food-101/
├── pizza/
│   ├── pizza_idx_000001.jpg
│   ├── pizza_idx_000015.jpg
│   └── ...
├── sushi/
│   ├── sushi_idx_000003.jpg
│   ├── sushi_idx_000028.jpg
│   └── ...
└── ...configured food directories (71 total food types)
```

**Configuration Volume Structure:**
```
/Volumes/cuisine_vision_catalog/config/config_volume/
├── cuisine_mapping.json      # Food type → cuisine mapping
└── food_types.json          # List of all supported food types
```

### 2.2 Configuration Parameters
**Essential Paths:**
- `BRONZE_VOLUME_PATH` - Target bronze volume path: `/Volumes/cuisine_vision_catalog/bronze/bronze_volume`
- `CONFIG_VOLUME_PATH` - Configuration files location: `/Volumes/cuisine_vision_catalog/config/config_volume`
- `STATUS_FILE_PATH` - Ingestion status tracking in config volume

**Clean Separation Design:**
- **Data Engineering Focus**: Pure data ingestion and organization
- **No ML Logic**: No Feature Store, model training, or ML-specific transformations
- **Configuration-Driven**: All business logic defined in JSON configuration files

**Status Tracking Format:**
```json
{
  "food_type": {
    "count": 25,
    "last_index": 150234,
    "last_filename": "pizza_idx_150234.jpg"
  }
}
```

### 2.3 Execution Workflow
1. **Setup Phase:** Load food types from configuration
2. **Incremental Ingestion:** 10 images per food type per run
3. **Progress Monitoring:** Track progress by cuisine groups
4. **Target Achievement:** 50+ images per configured food type

## 3. Bronze Tables Creation

### 3.1 DLT Pipeline Configuration
**File:** `bronze_tables_ingestion.py`
**Pipeline Name:** `bronze_tables_ingestion_ldp`
**Target Schema:** `cuisine_vision_catalog.bronze`
**Source:** Bronze volume files
**Processing Mode:** Continuous streaming with Auto Loader

### 3.2 Dynamic Table Generation
**Configuration-Driven Approach:**
- **JSON Config:** `cuisine_mapping.json` and `food_types.json`
- **Table Creation:** Dynamic generation for all configured food types
- **Naming Convention:** `food_{food_type}` (e.g., `food_pizza`, `food_sushi`)

### 3.3 Bronze Table Schema (Clean Data Engineering)
**Streamlined 10-column schema for pure data engineering:**
```sql
Essential Data Engineering Columns:
- file_path (string): Full path to source image file in bronze volume
- filename (string): Extracted filename for easy reference
- food_type (string): Food type name (lowercase) - from directory structure
- cuisine_category (string): Cuisine classification from configuration mapping
- image_data (binary): Raw image binary data stored in table
- file_size_bytes (bigint): File size in bytes
- file_timestamp (timestamp): File modification time
- ingested_at (timestamp): Processing timestamp
- dataset_index (bigint): Original dataset index from filename
- file_format (string): Image format (jpg, png, etc.)
```

**Key Design Principles:**
- **Raw data focus**: No image processing or ML-specific transformations
- **Business logic separation**: Cuisine mapping via configuration, not hardcoded
- **Clean architecture**: Pure data engineering, no ML engineering concerns
- **Downstream ready**: Provides clean foundation for silver layer processing

### 3.4 Auto Loader Configuration
**Optimized Settings:**
```python
.format("cloudFiles")
.option("cloudFiles.format", "binaryFile")
.option("cloudFiles.schemaLocation", schema_location)
.option("cloudFiles.includeExistingFiles", "true")
.option("cloudFiles.allowOverwrites", "false")
.option("cloudFiles.maxFilesPerTrigger", "100")
```

**Key Benefits:**
- **True incremental processing:** No reprocessing of existing files
- **Binary data handling:** Efficient image processing
- **Controlled batching:** Prevents cluster overload
- **Schema evolution:** Automatic schema management

## 4. Implementation Structure

### 4.1 Directory Layout
```
Cuisine_Classifier/
├── notebook codes/
│   └── source_to_bronze_volume_ingestion.ipynb      # Data ingestion notebook
├── dlt_pipelines/
│   └── bronze_tables_ingestion_ldp/                 # Bronze layer DLT pipeline (data engineering)
│       ├── transformations/
│       │   └── bronze_tables_ingestion.py           # Pure data transformation
│       └── utilities/
│           └── utils.py                              # Data engineering utilities only
├── ml_pipelines/                                    # Separate ML engineering layer
│   ├── feature_store_training.py                   # ML pipeline with Feature Store integration
│   └── model_serving_deployment.py                 # Model serving with Feature Store
└── config/                                          # Configuration files
    ├── cuisine_mapping.json                        # Food type → cuisine mapping
    └── food_types.json                             # All supported food types
```

### 4.2 Core Functions

**Ingestion Functions (`notebook`):**
- `load_ingestion_status()` - Load status from Unity Catalog
- `save_ingestion_status()` - Atomic status updates
- `ingest_food_images_to_bronze()` - Main ingestion function
- `check_status()` - Progress monitoring

**DLT Functions (`utils.py`):**
- `create_bronze_table_for_food_type()` - Dynamic table creation
- `load_config_file()` - Configuration management
- `get_pipeline_params()` - Pipeline parameter extraction

### 4.3 Performance Optimizations
**Built-in Optimizations:**
- Delta table auto-optimize settings
- Controlled parallel processing (71 tables)
- Efficient binary data handling
- Optimized Auto Loader configuration

**Table Properties:**
```python
table_properties = {
    "pipelines.autoOptimize.managed": "true",
    "delta.autoOptimize.optimizeWrite": "true", 
    "delta.autoOptimize.autoCompact": "true"
}
```

## 5. Deployment and Operations

### 5.1 Prerequisites
1. **Unity Catalog Setup:**
   - `cuisine_vision_catalog.bronze` schema
   - Bronze volume with appropriate permissions
   - Config volume for JSON files

2. **Configuration Files:**
   - Upload `cuisine_mapping.json` to config volume
   - Upload `food_types.json` to config volume

3. **DLT Pipeline:**
   - Create pipeline targeting bronze schema
   - Configure cluster for file processing workload

### 5.2 Deployment Steps

**Step 1: Data Ingestion**
1. Run `source_to_bronze_volume_ingestion.ipynb`
2. Execute incremental ingestion (10 images per food type per run)
3. Monitor progress until target achieved (50+ images per food)

**Step 2: Bronze Table Creation**
1. Deploy DLT pipeline `bronze_tables_ingestion_ldp`
2. Configure pipeline parameters:
   - `config_volume_path` - Configuration files location
   - `bronze_volume_path` - Bronze volume source path
3. Start pipeline in continuous mode

**Step 3: Validation**
1. Verify 71 bronze tables created
2. Validate image data availability
3. Check incremental processing behavior

### 5.3 Monitoring Strategy
**Key Metrics:**
- **Data Quality:** Row counts per food type table
- **Processing Status:** DLT pipeline execution health
- **Incremental Behavior:** New file processing without reprocessing
- **Performance:** Processing rates and cluster utilization

**Validation Queries:**
```sql
-- Check table creation
SHOW TABLES IN cuisine_vision_catalog.bronze;

-- Verify data availability
SELECT food_type, COUNT(*) as image_count 
FROM cuisine_vision_catalog.bronze.food_pizza 
GROUP BY food_type;

-- Check binary data
SELECT filename, LENGTH(image_data) as size_bytes 
FROM cuisine_vision_catalog.bronze.food_pizza 
LIMIT 5;
```

## 6. Expected Outcomes

### 6.1 Bronze Layer Results
**Achievements:**
- **71 bronze tables** with optimized schema
- **Image binary data** stored for ML processing
- **Incremental processing** with no reprocessing
- **Scalable architecture** ready for production
- **Simplified operations** with minimal complexity

### 6.2 Success Metrics
**Performance Indicators:**
- Zero data loss during ingestion and processing
- Efficient incremental behavior (new files only)
- Image data accessibility for downstream processing
- Operational simplicity with 2-parameter configuration
- Production-ready performance optimizations

### 6.3 Next Phase Preparation
**Silver Layer Ready (Clean Separation):**
- Bronze tables serve as optimized raw data source
- Binary image data available for feature extraction in silver layer
- Clean separation enables independent scaling of data vs ML engineering
- Configuration-driven approach allows business logic changes without code changes
- Foundation established for DLT silver layer (data engineering) and ML pipeline (ML engineering)

---

## 📋 Implementation Checklist

**Phase 1: Bronze Volume Ingestion**
- [ ] Run notebook setup and configuration
- [ ] Execute strategic food selection setup
- [ ] Perform incremental ingestion (target: 50+ images per food)
- [ ] Validate bronze volume structure

**Phase 2: Bronze Table Creation**
- [ ] Deploy DLT pipeline with correct parameters
- [ ] Verify 71 tables created successfully
- [ ] Validate image binary data storage
- [ ] Confirm incremental processing behavior

**Phase 3: Validation and Monitoring**
- [ ] Execute validation queries
- [ ] Monitor processing performance
- [ ] Verify data quality across all food types
- [ ] Document operational procedures

**Key Configuration Parameters:**
- `config_volume_path` - Configuration files location
- `bronze_volume_path` - Bronze volume source path

**Success Criteria:**
- 71 bronze tables with raw image binary data
- Clean separation: data engineering only (no ML logic)  
- Incremental processing without reprocessing
- Configuration-driven business logic (cuisine mappings)
- Scalable foundation for both silver layer (DLT) and ML pipeline (Feature Store)