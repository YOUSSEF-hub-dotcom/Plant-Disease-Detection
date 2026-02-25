import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from PIL import Image
from sqlalchemy import create_engine
import io
from config import settings # هنستفيد من الـ DATABASE_URL والـ CLASS_NAMES

# --- 1. CONFIGURATION ---
API_URL = "http://localhost:8000"
engine = create_engine(settings.DATABASE_URL)

st.set_page_config(page_title="PlantDoc Dashboard", layout="wide", page_icon="🌿")

# Custom CSS لتحسين الشكل
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2e7d32; color: white; }
    .result-card { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR (Navigation & Status) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100) # Logo افتراضي
    st.title("Plant Intelligence")
    st.markdown("---")
    
    menu = st.selectbox("Select Page", ["🏠 Home & Diagnosis", "📊 Analytics Dashboard", "❓ Help & Info"])
    
    st.markdown("---")
    # API Health Check
    try:
        health = requests.get(f"{API_URL}/health").json()
        st.success("API Status: Connected ✅")
        st.caption(f"Model: {health['model'].split('/')[-2]}")
    except:
        st.error("API Status: Disconnected ❌")

# --- 3. MAIN CONTENT: DIAGNOSIS ---
if menu == "🏠 Home & Diagnosis":
    st.header("🌿 Plant Disease Diagnosis")
    st.write("Upload a high-quality photo of the plant leaf for accurate detection.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image Preview", use_container_width=True)

    with col2:
        if uploaded_file:
            st.subheader("Action Center")
            if st.button("🚀 Start Analysis"):
                with st.spinner("Model is thinking..."):
                    # إرسال الصورة للـ API
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    try:
                        response = requests.post(f"{API_URL}/predict", files=files)
                        if response.status_code == 200:
                            data = response.json()
                            
                            # عرض النتيجة في كارت احترافي
                            st.markdown(f"""
                            <div class="result-card">
                                <h3 style='color: #2e7d32;'>Diagnosis Result:</h3>
                                <h2 style='text-align: center;'>{data['prediction'].replace('___', ' - ')}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.write("")
                            conf = float(data['confidence'])
                            st.write(f"**Confidence Score:** {conf:.2%}")
                            st.progress(conf)
                            
                            st.info(f"⏱️ Processing Latency: {data['latency']}")
                        else:
                            st.error(f"Error: {response.json().get('detail')}")
                    except Exception as e:
                        st.error(f"Connection Error: {str(e)}")
        else:
            st.info("Please upload an image to enable analysis.")

# --- 4. ANALYTICS DASHBOARD ---
elif menu == "📊 Analytics Dashboard":
    st.header("📈 Real-time System Analytics")
    
    # سحب البيانات من SQL Server
    try:
        df = pd.read_sql("SELECT * FROM prediction_logs", engine)
        
        if not df.empty:
            # Row 1: Key Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Diagnoses", len(df))
            m2.metric("Avg Confidence", f"{df['confidence'].mean():.2%}")
            m3.metric("Avg Latency", f"{df['latency'].mean():.3f} sec")
            
            st.markdown("---")
            
            # Row 2: Charts
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Top Detected Diseases")
                fig_disease = px.bar(df['prediction'].value_counts().reset_index(), 
                                   x='prediction', y='count', color='count',
                                   color_continuous_scale='Greens')
                st.plotly_chart(fig_disease, use_container_width=True)
                
            with c2:
                st.subheader("System Latency Trend")
                fig_latency = px.line(df, x='created_at', y='latency', 
                                    title="Latency over time")
                st.plotly_chart(fig_latency, use_container_width=True)
        else:
            st.warning("No data found in logs yet. Start diagnosing to see analytics!")
    except Exception as e:
        st.error(f"DB Error: {e}")

# --- 5. HELP & INFO PAGE ---
elif menu == "❓ Help & Info":
    st.header("❓ Help & System Information")
    
    st.markdown("""
    ### 🌟 How to use PlantDoc
    1. **Upload**: Go to the 'Home' page and upload a clear photo of a plant leaf.
    2. **Analyze**: Click the 'Start Analysis' button.
    3. **Result**: The model will identify the disease and give you a confidence score.
    
    ---
    ### 💡 Tips for Better Accuracy
    * **Lighting**: Use bright, natural light (avoid harsh shadows).
    * **Focus**: Make sure the leaf is in sharp focus.
    * **Background**: A plain or neutral background helps the model focus on the leaf patterns.
    
    ---
    ### 🛠️ Technical Stack
    This system is built using high-end MLOps tools:
    * **Model:** ResNet50 (Convolutional Neural Network)
    * **Backend:** FastAPI for high-performance inference.
    * **Tracking:** MLflow for model versioning and management.
    * **Database:** SQL Server for persistence and analytics.
    """)
    
    st.success("System is fully operational. Happy Farming! 🚜")

# --- 5. FOOTER ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey;'>
        <small>PlantDoc MLOps System v1.0 | Built with FastAPI, Streamlit & MLflow</small>
    </div>
    """, unsafe_allow_html=True
)