import os
import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io
import base64
import json
import logging

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Serving Endpoint Configuration
SERVING_ENDPOINT_URL = "https://adb-2867553723712000.0.azuredatabricks.net/serving-endpoints/cuisine-classifier/invocations"
# Your actual endpoint URL from Databricks

# Authentication - you'll need to set this up
# Option 1: Personal Access Token (for development)
DATABRICKS_TOKEN = 'token'  # Replace with your token

# Cuisine Mapping Configuration
CUISINE_MAPPING_PATH = "/Volumes/cuisine_vision_catalog/config/config_volume/cuisine_mapping.json"

# Option 2: Service Principal (for production) - commented out
# CLIENT_ID = "your-service-principal-client-id"
# CLIENT_SECRET = "your-service-principal-secret"

# ============================================================================
# STREAMLIT APP
# ============================================================================

# Set page config
st.set_page_config(
    page_title="🍽️ Cuisine Classifier",
    page_icon="🍽️",
    layout="wide",  # Changed to wide for better modern layout
    initial_sidebar_state="collapsed"  # Start with clean main view
)

# Custom CSS for modern, professional styling
st.markdown("""
<style>
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .hero-section {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 3rem;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .upload-zone {
        background: #f8fafc;
        border: 2px dashed #e2e8f0;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #667eea;
        background: #f1f5f9;
    }
    
    .prediction-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .result-cuisine {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin: 1.5rem 0;
        color: #1a202c;
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.8rem 1.5rem;
        border-radius: 50px;
        font-weight: bold;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    
    .confidence-high {
        background: #10b981;
        color: white;
    }
    
    .confidence-medium {
        background: #f59e0b;
        color: white;
    }
    
    .confidence-low {
        background: #ef4444;
        color: white;
    }
    
    .image-preview {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 1rem 0;
    }
    
    .cuisine-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .cuisine-item {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .status-connected {
        background: #dcfce7;
        color: #166534;
    }
    
    .status-error {
        background: #fef2f2;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)

# Modern Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">🍽️ AI Cuisine Classifier</div>
    <div class="hero-subtitle">Discover the cuisine of any dish with advanced AI technology</div>
    <div style="opacity: 0.8; font-size: 1rem; margin-top: 1rem;">
        Powered by Databricks Model Serving & Streamlit
    </div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_cuisine_mapping():
    """Load cuisine mapping from config volume"""
    try:
        logger.info("🔄 Loading cuisine mapping from config volume...")
        
        # For local testing, create a sample mapping
        # In production, you'd read from the actual volume path
        sample_mapping = {
            "chinese": ["noodles", "fried rice", "dumpling", "spring roll", "sweet and sour", "kung pao"],
            "italian": ["pizza", "pasta", "lasagna", "spaghetti", "risotto", "gelato"],
            "japanese": ["sushi", "ramen", "tempura", "miso soup", "katsu", "onigiri"],
            "indian": ["curry", "naan", "biryani", "samosa", "tandoori", "dal"],
            "mexican": ["taco", "burrito", "quesadilla", "enchilada", "guacamole", "salsa"],
            "thai": ["pad thai", "tom yum", "green curry", "mango sticky rice", "som tam", "larb"],
            "american": ["burger", "hot dog", "barbecue", "mac and cheese", "fried chicken", "apple pie"],
            "korean": ["kimchi", "bulgogi", "bibimbap", "korean bbq", "japchae", "tteokbokki"]
        }
        
        logger.info(f"✅ Cuisine mapping loaded: {len(sample_mapping)} cuisines")
        return sample_mapping
        
    except Exception as e:
        logger.error(f"❌ Failed to load cuisine mapping: {str(e)}")
        return {}

def display_cuisine_mapping(cuisine_mapping):
    """Display cuisine to food mapping in a compact sidebar format"""
    if not cuisine_mapping:
        st.warning("⚠️ Could not load cuisine mapping")
        return
    
    # Create a compact display for sidebar
    for cuisine, foods in cuisine_mapping.items():
        with st.expander(f"{cuisine.title()}", expanded=False):
            # Display foods as a simple list
            for food in foods:
                st.write(f"• {food}")
        
        # Small spacing
        st.markdown("")

def predict_cuisine_via_endpoint(image):
    """Make prediction using Databricks serving endpoint - Updated to match Databricks format"""
    try:
        logger.info("🔄 Starting image prediction process...")
        
        # Log image details
        logger.info(f"📸 Image mode: {image.mode}, Size: {image.size}")
        
        # Handle different image modes (fix RGBA issue)
        if image.mode in ("RGBA", "LA", "P"):
            logger.info(f"🔄 Converting image from {image.mode} to RGB...")
            # Convert RGBA/LA/P to RGB by adding white background
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode == "P":
                image = image.convert("RGBA")
            background.paste(image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None)
            image = background
            logger.info("✅ Image successfully converted to RGB")
        elif image.mode != "RGB":
            logger.info(f"🔄 Converting image from {image.mode} to RGB...")
            image = image.convert("RGB")
            logger.info("✅ Image successfully converted to RGB")
        
        # Convert PIL image to bytes
        logger.info("🔄 Converting image to bytes...")
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG', quality=95)
        image_bytes = img_buffer.getvalue()
        logger.info(f"✅ Image converted to bytes. Size: {len(image_bytes)} bytes")
        
        # Convert bytes to base64 string for JSON serialization
        logger.info("🔄 Encoding image to base64...")
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        logger.info(f"✅ Image encoded to base64. Length: {len(image_base64)} characters")
        
        # Create DataFrame in the exact format expected by serving endpoint
        logger.info("🔄 Creating payload for serving endpoint...")
        dataset = pd.DataFrame({
            'processed_image_data': [image_base64]
        })
        
        # Use the exact format from Databricks sample code
        payload = {
            'dataframe_split': dataset.to_dict(orient='split')
        }
        logger.info("✅ Payload created successfully")
        
        # Headers for authentication
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN[:10]}***",  # Log only first 10 chars for security
            "Content-Type": "application/json"
        }
        logger.info("🔄 Making request to serving endpoint...")
        
        # Make request using the exact format from Databricks sample
        data_json = json.dumps(payload, allow_nan=True)
        response = requests.request(
            method='POST',
            headers={
                "Authorization": f"Bearer {DATABRICKS_TOKEN}",
                "Content-Type": "application/json"
            },
            url=SERVING_ENDPOINT_URL,
            data=data_json,
            timeout=30
        )
        
        logger.info(f"📡 Response status: {response.status_code}")
        
        # Check if request was successful
        if response.status_code == 200:
            logger.info("✅ Successful response from serving endpoint")
            result = response.json()
            logger.info(f"📊 Response data: {str(result)[:200]}...")  # Log first 200 chars
            
            # Extract prediction from response
            # The response format may vary - adjust based on your model output
            if "predictions" in result:
                prediction = result["predictions"][0]
                logger.info("✅ Found predictions in response")
            elif isinstance(result, list) and len(result) > 0:
                prediction = result[0]
                logger.info("✅ Using first item from list response")
            else:
                prediction = result
                logger.info("✅ Using direct response as prediction")
            
            final_result = {
                'cuisine': prediction.get('label', 'Unknown'),
                'confidence': prediction.get('score', 0.0)
            }
            logger.info(f"🎯 Final prediction: {final_result}")
            return final_result
            
        else:
            error_msg = f"Endpoint error: {response.status_code} - {response.text}"
            logger.error(f"❌ {error_msg}")
            st.error(f"❌ {error_msg}")
            return {'cuisine': 'Error', 'confidence': 0.0}
            
    except requests.exceptions.Timeout as e:
        error_msg = "Request timeout. The serving endpoint is taking too long to respond."
        logger.error(f"❌ {error_msg}: {str(e)}")
        st.error(f"❌ {error_msg}")
        return {'cuisine': 'Timeout', 'confidence': 0.0}
    except requests.exceptions.ConnectionError as e:
        error_msg = "Connection error. Cannot reach the serving endpoint."
        logger.error(f"❌ {error_msg}: {str(e)}")
        st.error(f"❌ {error_msg}")
        return {'cuisine': 'Connection Error', 'confidence': 0.0}
    except requests.exceptions.RequestException as e:
        error_msg = f"Request error: {str(e)}"
        logger.error(f"❌ {error_msg}")
        st.error(f"❌ {error_msg}")
        return {'cuisine': 'Request Error', 'confidence': 0.0}
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON response: {str(e)}"
        logger.error(f"❌ {error_msg}")
        st.error(f"❌ {error_msg}")
        return {'cuisine': 'JSON Error', 'confidence': 0.0}
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.exception("Full exception details:")  # This logs the full stack trace
        st.error(f"❌ {error_msg}")
        return {'cuisine': 'Error', 'confidence': 0.0}

def test_endpoint_connection():
    """Test if the serving endpoint is reachable"""
    try:
        logger.info("🔄 Testing serving endpoint connection...")
        
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Simple test payload using Databricks format
        dataset = pd.DataFrame({'processed_image_data': []})  # Empty test
        payload = {
            'dataframe_split': dataset.to_dict(orient='split')
        }
        
        data_json = json.dumps(payload, allow_nan=True)
        response = requests.request(
            method='POST',
            headers=headers,
            url=SERVING_ENDPOINT_URL,
            data=data_json,
            timeout=10
        )
        
        is_reachable = response.status_code in [200, 400]  # 400 is ok for empty payload
        
        if is_reachable:
            logger.info(f"✅ Endpoint is reachable. Status: {response.status_code}")
        else:
            logger.warning(f"⚠️ Endpoint returned status {response.status_code}: {response.text}")
        
        return is_reachable
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to endpoint: {str(e)}")
        return False

# ============================================================================
# MAIN APP INTERFACE
# ============================================================================

# Load cuisine mapping
cuisine_mapping = load_cuisine_mapping()

# Test endpoint connection on startup (do this once, quietly)
if 'endpoint_tested' not in st.session_state:
    st.session_state.endpoint_tested = False

if not st.session_state.endpoint_tested:
    with st.spinner("🔄 Testing connection..."):
        if test_endpoint_connection():
            st.session_state.connection_status = "✅ Connected"
        else:
            st.session_state.connection_status = "❌ Connection Error"
    st.session_state.endpoint_tested = True

# Show connection status at top of main content
status_text = st.session_state.get('connection_status', 'Checking...')
if 'Connected' in status_text:
    st.success(f"� AI Model {status_text}")
else:
    st.error(f"🔴 {status_text}")

# Main upload area with modern design
uploaded_file = st.file_uploader(
    "",
    type=['jpg', 'jpeg', 'png'],
    help="Upload JPG, JPEG, or PNG files up to 200MB",
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        # Modern two-section layout
        st.markdown("---")
        
        # Image section
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="image-preview">', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption(f"📁 {uploaded_file.name}")
        
        # Prediction section
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        
        # Centered predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("� Analyze Cuisine", type="primary", use_container_width=True):
                with st.spinner("� AI is analyzing your dish..."):
                    result = predict_cuisine_via_endpoint(image)
                    
                    if result['cuisine'] not in ['Error', 'Timeout', 'Connection Error']:
                        confidence_pct = result['confidence'] * 100
                        
                        # Modern results display
                        st.markdown(f'<div class="result-cuisine">{result["cuisine"].title()}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Confidence badge with proper styling
                        if confidence_pct >= 70:
                            badge_class = "confidence-high"
                            confidence_text = f"{confidence_pct:.1f}% Confidence - Excellent"
                        elif confidence_pct >= 50:
                            badge_class = "confidence-medium" 
                            confidence_text = f"{confidence_pct:.1f}% Confidence - Good"
                        else:
                            badge_class = "confidence-low"
                            confidence_text = f"{confidence_pct:.1f}% Confidence - Uncertain"
                        
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <span class="confidence-badge {badge_class}">{confidence_text}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Modern progress bar
                        st.progress(result['confidence'])
                        
                        # Success message
                        if confidence_pct >= 80:
                            st.success("🎉 This looks like a perfect match!")
                        elif confidence_pct >= 60:
                            st.info("👍 Good prediction with solid confidence")
                        else:
                            st.warning("🤔 The AI is uncertain. Try a clearer image for better results.")
                    else:
                        st.error(f"❌ Analysis failed: {result['cuisine']}")
                        st.info("💡 Please check your connection and try again")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close prediction card
                    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("💡 Please try uploading a different image")

# Cuisine Guide Section
st.markdown("---")
st.markdown("## 🗺️ Supported Cuisines & Examples")

# Create cuisine grid
cuisine_items = list(cuisine_mapping.items())
cols = st.columns(4)

for idx, (cuisine, foods) in enumerate(cuisine_items):
    col_idx = idx % 4
    with cols[col_idx]:
        st.markdown(f"""
        <div class="cuisine-item">
            <h4>{cuisine.title()}</h4>
            <ul>
        """, unsafe_allow_html=True)
        for food in foods[:3]:  # Show only first 3 foods
            st.markdown(f"• {food.title()}")
        st.markdown("</div>", unsafe_allow_html=True)

# Close main container
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <div style="margin-bottom: 1rem;">
        <strong>🎯 How It Works</strong><br>
        Upload → AI Analysis → Instant Results
    </div>
    <div style="margin-bottom: 1rem;">
        <strong>🔧 Technology</strong><br>
        Databricks Model Serving • Advanced Computer Vision • Real-time Predictions
    </div>
    <div style="font-size: 0.9rem; opacity: 0.7;">
        Powered by Databricks Model Serving & Streamlit
    </div>
</div>
""", unsafe_allow_html=True)