import os
import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io
import base64
import json
import logging
import os
from dotenv import load_dotenv 

load_dotenv()
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
SERVING_ENDPOINT_URL = os.environ.get("SERVING_ENDPOINT_URL")
print(SERVING_ENDPOINT_URL)
# Your actual endpoint URL from Databricks

# Authentication - you'll need to set this up
# Option 1: Personal Access Token (for development)
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN', None)

# Cuisine Mapping Configuration
CUISINE_MAPPING_PATH = os.environ.get("CUISINE_MAPPING_PATH")

# Option 2: Service Principal (for production) - commented out
# CLIENT_ID = "your-service-principal-client-id"
# CLIENT_SECRET = "your-service-principal-secret"

# ============================================================================
# STREAMLIT APP
# ============================================================================

# Set page config
st.set_page_config(
    page_title="Cuisine Classifier",
    page_icon="",
    layout="wide",  # Changed to wide for better modern layout
    initial_sidebar_state="collapsed"  # Start with clean main view
)

# Custom CSS for ultra-compact styling
st.markdown("""
<style>
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0.2rem;
    }
    
    .hero-section {
        text-align: center;
        padding: 0.8rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 0.9rem;
        opacity: 0.95;
        margin: 0;
    }
    
    .result-cuisine {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
        color: #1a202c;
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.75rem;
        margin: 0.3rem 0;
    }
    
    .confidence-high { background: #10b981; color: white; }
    .confidence-medium { background: #f59e0b; color: white; }
    .confidence-low { background: #ef4444; color: white; }
    
    .cuisine-item {
        background: #f8f9fa;
        padding: 0.3rem;
        border-radius: 4px;
        border-left: 2px solid #667eea;
        font-size: 0.65rem;
        margin: 0.1rem 0;
    }
    
    .stButton > button {
        height: 2rem;
        font-size: 0.8rem;
    }
    
    /* Hide image expand button */
    button[title="View fullscreen"] {
        display: none !important;
    }
    
    .stImage > div > div > div > button {
        display: none !important;
    }
    
    .stImage button[kind="secondary"] {
        display: none !important;
    }
    
    /* Hide any expand icons on images */
    .stImage [data-testid="baseButton-secondary"] {
        display: none !important;
    }
    
    h1, h2, h3 {
        margin: 0.3rem 0 !important;
        font-size: 1rem !important;
    }
    
    .stApp > div:first-child {
        padding-top: 0.5rem !important;
    }
    
    .main .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Modern Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">🍽️ AI Cuisine Classifier</div>
    <div class="hero-subtitle">Upload a food image to identify its cuisine type with AI precision</div>
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
            timeout=60
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
            if final_result["confidence"] <= 0.3:
                final_result["cuisine"] = 'Not a food / Unknown image data'
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
            timeout=30
        )
        
        is_reachable = response.status_code == 200  # Only 200 is success
        
        if is_reachable:
            logger.info(f"✅ Endpoint is reachable. Status: {response.status_code}")
        else:
            logger.warning(f"⚠️ Endpoint returned status {response.status_code}: {response.text}")

        return is_reachable, response, None
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to endpoint: {str(e)}")
        return False, None, e

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
        is_reachable, response, error = test_endpoint_connection()
        if is_reachable:
            st.session_state.connection_status = "✅ Connected"
        else:
            # parse the response and fetch the error message
            if not error:
                err_msg = response.json().get('message', 'Unknown error')
                st.session_state.connection_status = err_msg
            else:
                st.session_state.connection_status = error
    st.session_state.endpoint_tested = True

# Show connection status at top of main content
status_text = st.session_state.get('connection_status', 'Checking...')
if 'Connected' in status_text:
    st.success(f"{status_text} to 'cuisine_classifier' model")
else:
    st.error(f"❌ Unable to connect to 'cuisine_classifier' model: {status_text}")

# Enhanced upload area with custom styling
st.markdown("""
<style>
.upload-zone {
    background: linear-gradient(145deg, #f8fafc 0%, #e2e8f0 100%);
    border: 2px dashed #667eea;
    border-radius: 15px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
    transition: all 0.3s ease;
}
.upload-zone:hover {
    border-color: #4f46e5;
    background: linear-gradient(145deg, #f1f5f9 0%, #e2e8f0 100%);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# Main upload area with modern design - only show if no file uploaded
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'upload_key' not in st.session_state:
    st.session_state.upload_key = 0

# Only show uploader if no file is currently uploaded
if st.session_state.uploaded_file is None:
    # Create container to force refresh
    upload_container = st.container()
    with upload_container:
        uploaded_file = st.file_uploader(
            "Drag and drop your food image here",
            type=['jpg', 'jpeg', 'png'],
            help="Upload JPG, JPEG, or PNG files up to 200MB",
            label_visibility="visible",
            key=f"file_uploader_{st.session_state.upload_key}"
        )
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.rerun()
else:
    uploaded_file = st.session_state.uploaded_file

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        # Compact side-by-side layout: Image left, Results right
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("**📸 Uploaded Image**")
            # Reduce image display to max width 180px for more compact view
            st.image(image, width=180)
            st.caption(f"📁 {uploaded_file.name}")
            
            # Add option to upload new image
            if st.button("📷 Upload New Image", type="secondary"):
                # Clear the uploaded file from session state
                st.session_state.uploaded_file = None
                # Increment key to force new uploader
                st.session_state.upload_key += 1
                st.rerun()
        
        with col2:
            st.markdown("**🎯 Analysis Results**")
            
            # Compact predict button
            if st.button("🚀 Analyze Cuisine", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing..."):
                    result = predict_cuisine_via_endpoint(image)
                    
                    if result['cuisine'] not in ['Error', 'Timeout', 'Connection Error']:
                        confidence_pct = result['confidence'] * 100
                        
                        # Compact results display
                        st.markdown(f'<div class="result-cuisine">{result["cuisine"].title()}</div>', 
                                  unsafe_allow_html=True)
                        
                        # Compact confidence badge
                        if confidence_pct >= 70:
                            badge_class = "confidence-high"
                            confidence_text = f"{confidence_pct:.1f}% - Excellent"
                        elif confidence_pct >= 50:
                            badge_class = "confidence-medium" 
                            confidence_text = f"{confidence_pct:.1f}% - Good"
                        else:
                            badge_class = "confidence-low"
                            confidence_text = f"{confidence_pct:.1f}% - Uncertain"
                        
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <span class="confidence-badge {badge_class}">{confidence_text}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Compact progress bar
                        st.progress(result['confidence'])
                    else:
                        st.error(f"❌ Analysis failed: {result['cuisine']}")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Compact Cuisine Guide Section  
st.markdown("### Cuisine - Food Type Mapping")

# Ultra-compact cuisine grid
cols = st.columns(8)

for idx, (cuisine, foods) in enumerate(cuisine_mapping.items()):
    col_idx = idx % 8
    with cols[col_idx]:
        # Show all foods in each cuisine
        food_list = "".join([f"• {food}<br>" for food in foods[:4]])  # Show max 4 items to keep compact
        st.markdown(f"""
        <div class="cuisine-item">
            <strong style="font-size: 0.7rem;">{cuisine.title()}</strong>
            <div style="font-size: 0.6rem; margin-top: 0.1rem; line-height: 1.2;">
                {food_list}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Close main container
st.markdown('</div>', unsafe_allow_html=True)