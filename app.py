import streamlit as st
import requests
import pd as pd
import plotly.express as px
from PIL import Image
from sqlalchemy import create_engine
import io
from config import settings

# --- 1. DASHBOARD CONFIGURATION ---
API_URL = "http://localhost:8000"
engine = create_engine(settings.DATABASE_URL)

st.set_page_config(
    page_title="PlantDoc | Intelligence Platform", 
    layout="wide", 
    page_icon="🌿"
)

# Professional UI Styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3.5em; 
        background-color: #2e7d32; 
        color: white; 
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover { background-color: #1b5e20; color: white; }
    .result-card { 
        padding: 25px; 
        border-radius: 15px; 
        background-color: white; 
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border-left: 5px solid #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR NAVIGATION & MONITORING ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.title("Plant Intel AI")
    st.markdown("---")
    
    menu = st.selectbox(
        "Navigation", 
        ["🏠 Home & Diagnosis", "📊 Analytics Dashboard", "❓ Help & Info"]
    )
    
    st.markdown("---")
    # Real-time API Health Check
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            health = health_resp.json()
            st.success("API Status: Connected ✅")
            st.caption(f"🚀 Registry Model: {health['model_version'].split('/')[-2]}")
        else:
            st.warning("API Status: Warning ⚠️")
    except:
        st.error("API Status: Offline ❌")

# --- 3. PAGE: DIAGNOSIS INTERFACE ---
if menu == "🏠 Home & Diagnosis":
    st.header("🌿 Intelligent Plant Pathologist")
    st.write("Upload a clear image of the infected leaf for instantaneous disease identification.")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📸 Image Upload")
        uploaded_file = st.file_uploader("Drop leaf photo here...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Leaf Preview", use_container_width=True)

    with col2:
        st.subheader("⚡ Analysis Center")
        if uploaded_file:
            if st.button("🚀 Analyze Plant Health"):
                with st.spinner("Deep Learning Model is scanning the leaf..."):
                    # Prepare file for API transmission
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    try:
                        response = requests.post(f"{API_URL}/predict", files=files)
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Professional Result Card Display
                            clean_label = data['prediction'].replace('___', ' - ').replace('_', ' ')
                            st.markdown(f"""
                            <div class="result-card">
                                <p style='color: grey; margin-bottom: 5px;'>DIAGNOSIS RESULT:</p>
                                <h2 style='color: #2e7d32; margin-top: 0;'>{clean_label}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.write("")
                            confidence = float(data['confidence'])
                            st.metric("Confidence Score", f"{confidence:.2%}")
                            st.progress(confidence)
                            
                            st.info(f"⏱️ Model Inference Latency: {data['latency']}")
                        else:
                            st.error(f"Inference Error: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Failed to connect to backend: {str(e)}")
        else:
            st.info("Waiting for image upload to initialize analysis...")

# --- 4. PAGE: ANALYTICS DASHBOARD ---
elif menu == "📊 Analytics Dashboard":
    st.header("📈 Production System Analytics")
    st.write("Real-time insights from the central prediction database.")
    
    try:
        # Fetching historical logs from SQL Server
        df = pd.read_sql("SELECT * FROM prediction_logs", engine)
        
        if not df.empty:
            # High-level Performance Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Requests", len(df))
            m2.metric("Mean Confidence", f"{df['confidence'].mean():.2%}")
            m3.metric("Avg Latency", f"{df['latency'].mean():.3f}s")
            
            st.divider()
            
            # Visualization Row
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Disease Prevalence")
                disease_counts = df['prediction'].value_counts().reset_index()
                disease_counts.columns = ['Disease', 'Cases']
                fig_disease = px.bar(
                    disease_counts, x='Disease', y='Cases', 
                    color='Cases', color_continuous_scale='Greens',
                    template="plotly_white"
                )
                st.plotly_chart(fig_disease, use_container_width=True)
                
            with c2:
                st.subheader("Inference Performance Trend")
                df['created_at'] = pd.to_datetime(df['created_at'])
                fig_latency = px.line(
                    df, x='created_at', y='latency',
                    labels={'latency': 'Latency (sec)', 'created_at': 'Time'},
                    template="plotly_white"
                )
                fig_latency.update_traces(line_color='#2e7d32')
                st.plotly_chart(fig_latency, use_container_width=True)
        else:
            st.warning("Database logs are currently empty. Start analyzing images to populate dashboard.")
    except Exception as e:
        st.error(f"Database Query Error: {e}")

# --- 5. PAGE: HELP & SYSTEM INFO ---
elif menu == "❓ Help & Info":
    st.header("📖 System Guide & Technical Stack")
    
    st.markdown("""
    ### 🌟 Operational Workflow
    1. **Upload**: Capture or select a high-resolution leaf image.
    2. **Analyze**: Trigger the ResNet50 vision engine.
    3. **Action**: Review the diagnosis and confidence level to determine treatment.
    
    ---
    ### 🔬 Scientific Accuracy Guidelines
    * **Clarity**: Ensure the infected area is the primary focus of the frame.
    * **Lighting**: Natural daylight yields the highest diagnostic precision.
    * **Angle**: Capture the leaf top-down (perpendicular to the camera).
    
    ---
    ### 🛠️ Production Architecture (MLOps)
    This platform leverages enterprise-grade technology:
    * **Core Vision:** ResNet50 (Transfer Learning) with 38 disease classes.
    * **Backend:** FastAPI for high-throughput concurrency.
    * **Tracking:** MLflow for experiment reproducibility & Model Registry.
    * **Persistence:** SQL Server for audit logs and system observability.
    """)
    
    st.success("Platform status: Nominal. Ready for deployment. 🚜")

# --- 6. FOOTER ---
st.divider()
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 0.8em;'>"
    "PlantDoc Platform v1.2 | Integrated MLOps Solution | Built by Youssef Mahmoud"
    "</div>", 
    unsafe_allow_html=True
)
