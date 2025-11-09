# 01. Databricks Azure Setup & Unity Catalog Configuration

**Project:** Cuisine Vision Pipeline (CVP) - Demo Scale  
**Platform:** Microsoft Azure  
**Target Users:** 1 (Demo/Development)  

## 1. Azure Databricks Workspace Setup ✅

### 1.1 Prerequisites
- Azure subscription with appropriate permissions
- Resource group for Databricks resources
- Azure Storage Account (ADLS Gen2) for data storage

### 1.2 Create Azure Databricks Workspace via Azure Portal

#### Step-by-Step Portal Instructions:

1. **Sign in to Azure Portal**
   - Go to [https://portal.azure.com](https://portal.azure.com)
   - Sign in with your Azure account

2. **Create Resource Group (if not exists)**
   - Click "Resource groups" in the left menu
   - Click "Create" or "+ Add"
   - Fill in the details:
     - **Subscription:** Select your Azure subscription
     - **Resource group name:** `rg-cuisine-vision-demo`
     - **Region:** East US (or your preferred region)
   - Click "Review + Create" → "Create"

3. **Create Databricks Workspace**
   - In the Azure Portal search bar, type "Azure Databricks"
   - Click "Azure Databricks" from the results
   - Click "+ Create" or "Create Azure Databricks workspace"

4. **Configure Workspace Settings**
   Fill in the following details:

   **Basics Tab:**
   - **Subscription:** Select your Azure subscription
   - **Resource group:** `rg-cuisine-vision-demo` (created above)
   - **Workspace name:** `ws-databricks-cuisine-vision`
   - **Region:** East US (same as resource group)
   - **Pricing tier:** Premium (required for Unity Catalog)

   **Networking Tab:** (Optional - use defaults for demo)
   - **Deploy Azure Databricks workspace in your Virtual Network:** No
   - **Enable No Public IP:** No (keep disabled for demo)

   **Advanced Tab:** (Optional)
   - **Managed Resource Group Name:** Leave default or specify custom name rg-dbr-managed-cuisine-vision

5. **Review and Create**
   - Click "Review + Create"
   - Review all settings:
     - Workspace name: `ws-databricks-cuisine-vision`
     - Region: East US
     - Pricing tier: Premium ✅
     - Resource group: `rg-cuisine-vision-demo`
   - Click "Create"

6. **Wait for Deployment**
   - Deployment typically takes 5-10 minutes
   - You'll see "Deployment in progress..." message
   - Wait for "Your deployment is complete" confirmation

7. **Access Your Workspace**
   - Click "Go to resource" when deployment completes
   - Or navigate to: Resource groups → `rg-cuisine-vision-demo` → `databricks-cuisine-vision`
   - Click "Launch Workspace" button
   - This will open the Databricks workspace in a new tab

#### Alternative: Using Azure CLI
```bash
# If you prefer CLI method
az databricks workspace create \
  --resource-group "rg-cuisine-vision-demo" \
  --name "databricks-cuisine-vision" \
  --location "East US" \
  --sku "premium"
```

### 1.3 Workspace Configuration Verification

After creating your workspace, verify these settings:
- **Tier:** Premium ✅ (required for Unity Catalog)
- **Region:** East US (or your selected region)
- **Networking:** Standard deployment (no private endpoints for demo)
- **Status:** Running (may take a few minutes after creation)

### 1.4 Common Issues and Troubleshooting

#### Issue 1: "Premium tier not available"
- **Solution:** Ensure your Azure subscription supports Premium Databricks
- **Alternative:** Contact Azure support to enable Premium tier

#### Issue 2: "Insufficient permissions to create workspace"
- **Required Permissions:** Contributor role on the subscription or resource group
- **Solution:** Ask your Azure admin to grant Contributor access

#### Issue 3: "Region not available"
- **Available Regions for Databricks:** East US, East US 2, West US, West US 2, Central US, North Central US, South Central US, West Central US, Canada Central, Canada East, Brazil South, UK South, UK West, West Europe, North Europe, France Central, Germany West Central, Switzerland North, UAE North, South Africa North, Australia East, Australia Southeast, Central India, South India, West India, Japan East, Japan West, Korea Central, Southeast Asia, East Asia
- **Solution:** Select from available regions listed above

#### Issue 4: "Deployment failed"
- **Common causes:** Resource name conflicts, quota limits, region capacity
- **Solution:** 
  - Use a unique workspace name
  - Try a different region
  - Check Azure service health status

#### Issue 5: "Cannot access workspace after creation"
- **Wait Time:** Workspace initialization can take 10-15 minutes
- **Check Status:** Go to resource → Overview → check if status is "Running"
- **Browser Issues:** Try incognito mode or clear browser cache

### 1.5 Post-Creation Checklist

After successfully creating your Databricks workspace, verify:

- ☐ **Resource Group Created:** `rg-cuisine-vision-demo` exists
- ☐ **Workspace Created:** `databricks-cuisine-vision` appears in resource group
- ☐ **Pricing Tier:** Shows "Premium" in the Overview page
- ☐ **Status:** Shows "Running" (not "Creating" or "Failed")
- ☐ **Launch Button:** "Launch Workspace" button is clickable
- ☐ **Workspace Access:** Can successfully open Databricks workspace UI
- ☐ **Authentication:** Can sign in to the Databricks workspace

**Expected Timeline:**
- Resource group creation: 1-2 minutes
- Databricks workspace creation: 5-10 minutes  
- Workspace initialization: 5-15 minutes
- **Total time:** 15-30 minutes

## 2. Unity Catalog Setup ✅

### 2.1 Enable Unity Catalog
Once your workspace is running:

1. **Access Databricks Workspace**
   - From Azure Portal → Your Resource Group → Databricks workspace
   - Click "Launch Workspace" 
   - Sign in when prompted

2. **Navigate to Unity Catalog**
   - In Databricks workspace, look for "Data" in the left sidebar
   - Click "Data" → "Unity Catalog" 
   - If Unity Catalog option is not visible, your workspace may still be initializing

3. **Enable Unity Catalog**
   - Click "Enable Unity Catalog" (if prompted)
   - Follow the setup wizard
   - This may require admin permissions

4. **Create Metastore** (if not exists)
   - Unity Catalog will prompt you to create a metastore
   - Choose your region (should match your workspace region)
   - Accept default settings for demo

### 2.2 Create Catalog Structure
```sql
-- Create main catalog
CREATE CATALOG IF NOT EXISTS cuisine_vision_catalog;
USE CATALOG cuisine_vision_catalog;

-- Create config schema for configuration files and volumes
CREATE SCHEMA IF NOT EXISTS config 
COMMENT 'Configuration files, mappings, and shared utilities for the pipeline';

-- Create bronze schema for raw ingested data
CREATE SCHEMA IF NOT EXISTS bronze 
COMMENT 'Raw ingested data from Food-101 dataset with dynamic food type tables';

-- Create silver schema for cleaned and processed data  
CREATE SCHEMA IF NOT EXISTS silver 
COMMENT 'Cleaned and validated data with cuisine mappings and CNN-ready images';

-- Create gold schema for ML-ready datasets
CREATE SCHEMA IF NOT EXISTS gold 
COMMENT 'ML-ready unified datasets and 100D feature vectors for training';

-- Create models schema for ML artifacts
CREATE SCHEMA IF NOT EXISTS models 
COMMENT 'ML models, model artifacts, and Feature Store tables';

-- Create Unity Catalog Volumes for different purposes
-- Bronze volume for raw data ingestion
CREATE VOLUME IF NOT EXISTS bronze.bronze_volume
COMMENT 'Raw data storage for Food-101 images organized by food type';

-- Config volume for configuration files
CREATE VOLUME IF NOT EXISTS config.config_volume
COMMENT 'Configuration files: cuisine_mapping.json, food_types.json, etc.';
```

### 2.3 Set Permissions (Demo - Single User)
```sql
-- Grant permissions to current user
GRANT ALL PRIVILEGES ON CATALOG cuisine_vision_catalog TO `<your-email@domain.com>`;
GRANT ALL PRIVILEGES ON SCHEMA cuisine_vision_catalog.config TO `<your-email@domain.com>`;
GRANT ALL PRIVILEGES ON SCHEMA cuisine_vision_catalog.bronze TO `<your-email@domain.com>`;
GRANT ALL PRIVILEGES ON SCHEMA cuisine_vision_catalog.silver TO `<your-email@domain.com>`;
GRANT ALL PRIVILEGES ON SCHEMA cuisine_vision_catalog.gold TO `<your-email@domain.com>`;
GRANT ALL PRIVILEGES ON SCHEMA cuisine_vision_catalog.models TO `<your-email@domain.com>`;

-- Grant volume permissions
GRANT ALL PRIVILEGES ON VOLUME cuisine_vision_catalog.bronze.bronze_volume TO `<your-email@domain.com>`;
GRANT ALL PRIVILEGES ON VOLUME cuisine_vision_catalog.config.config_volume TO `<your-email@domain.com>`;
```

## 3. Storage Architecture ✅

### 3.1 Unity Catalog Storage Strategy ✅
Unity Catalog uses its **managed storage account** for all data. We'll leverage Unity Catalog Volumes for efficient data organization with clean separation:

- **Managed tables** (Bronze/Silver/Gold): Unity Catalog default managed storage paths
- **Bronze Volume** (`bronze.bronze_volume`): Raw data storage for Food-101 images organized by food type  
- **Config Volume** (`config.config_volume`): Configuration files (cuisine_mapping.json, food_types.json)

### 3.2 Clean Architecture File Organization
```
Unity Catalog Managed Storage
├── tables/                                    # Managed tables with clean separation
│   ├── bronze/                               # Raw data tables (71 food_* tables)
│   │   ├── food_pizza/
│   │   ├── food_sushi/
│   │   └── ... (71 food type tables)
│   ├── silver/                               # Processed data (8 cuisine tables)
│   │   ├── cuisine_italian/
│   │   ├── cuisine_japanese/
│   │   └── ... (8 cuisine tables)
│   ├── gold/                                 # ML-ready datasets
│   │   ├── ml_dataset/                       # Unified training dataset
│   │   └── feature_vectors/                  # 100D feature vectors
│   └── models/                               # ML artifacts with Feature Store
│       ├── feature_tables/                   # Feature Store tables
│       ├── model_registry/                   # Registered models
│       └── model_artifacts/                  # Model files and metadata
└── volumes/                                  # Unity Catalog Volumes
    └── cuisine_vision_catalog/
        ├── bronze/
        │   └── bronze_volume/                # Raw Food-101 images by food type
        │       └── food-101/
        │           ├── pizza/
        │           ├── sushi/
        │           └── ... (71 food types)
        └── config/
            └── config_volume/                # Configuration files
                ├── cuisine_mapping.json      # Food type → cuisine mapping
                └── food_types.json          # List of all food types
```

### 3.3 Clean Separation Data Flow Architecture
```
Food-101 Dataset Ingestion
         ↓
cuisine_vision_catalog.bronze.bronze_volume (Unity Catalog Volume)
         ↓ 
cuisine_vision_catalog.bronze.food_* (71 DLT tables - data engineering)
         ↓
cuisine_vision_catalog.silver.cuisine_* (8 DLT tables - data engineering)
         ↓
cuisine_vision_catalog.gold.* (2 DLT tables - data engineering)
         ↓
ML Pipeline with Feature Store (ML engineering)
         ↓
cuisine_vision_catalog.models.feature_tables (Feature Store tables)
         ↓
Model Training & Serving (ML engineering)
```

**Benefits:**
- ✅ **Single storage account** - No additional ADLS Gen2 needed
- ✅ **Clean separation** - Data engineering (DLT) vs ML engineering (Feature Store)
- ✅ **Unity Catalog governance** - All data under UC management
- ✅ **Cost effective** - Leverages existing UC storage
- ✅ **Simple architecture** - Clear separation of concerns

## 4. Compute Cluster Configuration  ✅

### 4.1 Create ML Runtime Cluster
- **Cluster Name:** `cuisine-vision-ml-cluster`
- **Databricks Runtime:** 13.3 LTS ML (or latest ML runtime)
- **Node Type:** Standard_D4s_v3 (demo scale - 4 cores, 16GB RAM)
- **Workers:** 0 (Single Node for demo)
- **Spot Instances:** Enabled (cost optimization for demo)

### 4.2 Install Required Libraries

#### **Notebook-Level Installation (Recommended for Demo)**
Since kernel crashes can occur with cluster-level installations, use notebook-level installation:

```python
# Install in notebook cells - run this at the start of each notebook
%pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
%pip install mlflow[extras]
%pip install opencv-python-headless
%pip install pillow
%pip install pyyaml
%pip install pandas
%pip install numpy
%pip install matplotlib
%pip install seaborn

# Restart Python to use newly installed packages
dbutils.library.restartPython()
```

#### **Alternative: Cluster-Level Libraries**
If your kernel is stable, you can install at cluster level:
- Go to **Compute** → Your cluster → **Libraries** tab
- Install packages one by one to avoid memory issues

**Note:** Notebook-level installation is more reliable for demos and avoids kernel crashes.

## 5. Validation Steps ✅

### 5.1 Test Unity Catalog Connection ✅
```sql
-- Verify catalog access
SHOW CATALOGS;
USE CATALOG cuisine_vision_catalog;
SHOW SCHEMAS;
```

### 5.2 Test Unity Catalog Managed Storage ✅
```python
# Test Unity Catalog managed storage access
spark.sql("SHOW TABLES IN cuisine_vision_catalog.bronze")
spark.sql("SHOW TABLES IN cuisine_vision_catalog.silver") 
spark.sql("SHOW TABLES IN cuisine_vision_catalog.gold")
```

### 5.3 Test Cluster Libraries ✅
```python
# Verify installations
import torch
import torchvision
import mlflow
import cv2
from PIL import Image
import yaml

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"MLflow version: {mlflow.__version__}")
print("All libraries imported successfully!")
```

## 7. Next Steps

After completing this setup:
1. ✅ Azure Databricks workspace is ready
2. ✅ Unity Catalog is configured with clean separation architecture
3. ✅ Storage volumes and schemas created for data engineering and ML engineering separation  
4. ✅ Compute cluster with ML libraries installed
5. ➡️ **Next:** Proceed to "02_Source_to_Bronze_Pipeline.md"

## 8. Clean Architecture Summary

### 8.1 Data Engineering Layer (DLT Pipelines)
- **Bronze Layer**: Raw data ingestion from Food-101 dataset (71 food_* tables)
- **Silver Layer**: Image processing and cuisine categorization (8 cuisine_* tables)  
- **Gold Layer**: ML-ready unified datasets and feature extraction (2 tables)
- **Focus**: Data transformation, cleansing, and preparation

### 8.2 ML Engineering Layer (Feature Store + MLflow)
- **Feature Store**: Enhanced training with 100D feature vectors
- **Model Training**: PyTorch ResNet-50 with MLflow tracking
- **Model Registry**: Unity Catalog model registration and versioning
- **Model Serving**: Production deployment with real-time feature lookups
- **Focus**: Machine learning workflows, experimentation, and serving

## Cost Optimization for Demo

### Resource Sizing
- **Single Node Cluster:** Reduces costs significantly
- **Spot Instances:** Up to 80% savings on compute
- **Standard Storage:** LRS for demo data
- **Auto-termination:** Set cluster to terminate after 30 minutes of inactivity

### Estimated Monthly Cost (Demo Scale)
- Databricks Premium: ~$50-100/month (minimal usage)
- Azure Storage (ADLS Gen2): ~$5-10/month (10GB data)
- Compute (D4s_v3 Spot): ~$20-40/month (intermittent usage)
- **Total Estimated:** ~$75-150/month for demo environment