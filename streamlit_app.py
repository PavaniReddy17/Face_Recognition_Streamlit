"""
Face Recognition Streamlit App - Pavani Detection
Powered by Roboflow AI
"""

import streamlit as st
from pathlib import Path
import tempfile
import time
import os
import cv2
import numpy as np
from roboflow import Roboflow
import requests

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Pavani Face Recognition", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="🎥"
)

# ---------------------------
# PRO UI/CSS
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="st-"], [class*="css-"] {
    font-family: 'Inter', sans-serif;
}

:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --background-color: #0E1117;
    --secondary-background-color: #1A1E29;
    --text-color: #FAFAFA;
    --secondary-text-color: #ADB5BD;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0E1117 0%, #1A1E29 100%);
    color: var(--text-color);
}

header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.8rem;
    padding-bottom: 0.5rem;
}

h3 {
    color: var(--text-color);
    font-weight: 600;
}

h5 {
    color: var(--primary-color);
    font-weight: 600;
    margin-bottom: 0.5rem;
}

[data-testid="column"] {
    background-color: var(--secondary-background-color);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    border: 1px solid #2C334A;
    transition: all 0.3s ease-in-out;
}

[data-testid="column"]:hover {
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);
    border-color: #667eea;
}

[data-testid="stButton"] button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-size: 1.1rem;
    width: 100%;
    transition: all 0.3s ease;
}

[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    transform: translateY(-2px);
}

[data-testid="stDownloadButton"] button {
    background-color: #2C334A;
    color: var(--text-color);
    border: 2px solid var(--primary-color);
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #FFFFFF;
    transform: translateY(-2px);
}

[data-testid="stMetric"] {
    background-color: #2C334A;
    border-radius: 8px;
    padding: 1rem;
    border-left: 5px solid var(--primary-color);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}

[data-testid="stMetricValue"] {
    color: var(--primary-color);
    font-size: 2rem;
    font-weight: 700;
}

[data-testid="stFileUploader"] {
    background-color: #2C334A;
    border: 2px dashed #667eea;
    border-radius: 8px;
    padding: 1.5rem;
}

video {
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    width: 100%;
}

.stProgress > div > div > div > div {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

hr {
    border-top: 2px solid #2C334A;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Roboflow Initialization
# ---------------------------
@st.cache_resource
def init_roboflow():
    """Initialize Roboflow model with HARDCODED keys for easy deployment"""
    try:
        # ---------------------------------------------------------
        # HARDCODED API KEYS (No secrets.toml required)
        # ---------------------------------------------------------
        api_key = "lYQhNaqU50FdyzkR0Gq5"
        workspace = "project-nhn9q"
        project = "face-detection-w9fbh"
        version = 3
        person_name = "Pavani"
        
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        model = proj.version(version).model
        
        return model, workspace, person_name
    except Exception as e:
        st.error(f"❌ Failed to initialize Roboflow: {str(e)}")
        return None, None, None

# ---------------------------
# Video Processing Class
# ---------------------------
class FaceDetectionProcessor:
    def __init__(self, model, person_name, confidence_threshold=0.4):
        self.model = model
        self.person_name = person_name
        self.confidence_threshold = confidence_threshold
    
    def process_video(self, input_path, output_path, progress_callback=None):
        """Process video and detect faces"""
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        # NOTE: 'mp4v' is used for compatibility. 
        # If video doesn't play in browser, try downloading it.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError("Failed to create output video")
        
        # Statistics
        frame_count = 0
        detection_count = 0
        total_confidence = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            if progress_callback and total_frames > 0:
                progress_callback(frame_count / total_frames)
            
            # ---------------------------------------------------------
            # FIX: Create temp file and CLOSE it immediately to release 
            # the Windows file lock.
            # ---------------------------------------------------------
            temp_frame = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_frame.close() # <--- THIS LINE FIXES THE PERMISSION ERROR
            
            cv2.imwrite(temp_frame.name, frame)
            
            try:
                # Get predictions from Roboflow
                predictions = self.model.predict(
                    temp_frame.name,
                    confidence=int(self.confidence_threshold * 100)
                ).json()
                
                # Draw detections
                for pred in predictions['predictions']:
                    x_center = pred['x']
                    y_center = pred['y']
                    w = pred['width']
                    h = pred['height']
                    confidence = pred['confidence']
                    
                    # Calculate box coordinates
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)
                    
                    # Draw rectangle (green)
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label with background
                    label = f"{self.person_name} {confidence*100:.1f}%"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    label_w, label_h = label_size
                    
                    # Background for text
                    cv2.rectangle(frame, (x1, y1 - label_h - 10), 
                                (x1 + label_w, y1), color, -1)
                    
                    # Text
                    cv2.putText(frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Update statistics
                    detection_count += 1
                    total_confidence += confidence
                    
            except Exception as e:
                pass  # Skip frame on error
            
            # Clean up temp frame
            try:
                if os.path.exists(temp_frame.name):
                    os.unlink(temp_frame.name)
            except Exception as e:
                pass
            
            # Write frame
            out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Calculate statistics
        processing_time = time.time() - start_time
        avg_confidence = (total_confidence / detection_count * 100) if detection_count > 0 else 0
        
        stats = {
            'total_frames': frame_count,
            'detections': detection_count,
            'avg_confidence': avg_confidence,
            'processing_time': processing_time,
            'fps': fps
        }
        
        return stats

# ---------------------------
# Main App UI
# ---------------------------
st.title("🎥 Face Recognition System")
st.markdown(f"**Detecting:** Pavani | **Powered by:** Roboflow AI")

# Initialize model
with st.spinner("🔄 Initializing Roboflow model..."):
    model, workspace, person_name = init_roboflow()

if model is None:
    st.stop()

st.success(f"✅ Model loaded successfully! Detecting: **{person_name}**")

# Main Layout
st.markdown("### 1. Upload & Configure")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("##### Upload Video")
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed"
    )
    
    if uploaded_video:
        st.video(uploaded_video)
        file_size = uploaded_video.size / (1024 * 1024)
        st.caption(f"📊 File: {uploaded_video.name} | Size: {file_size:.2f} MB")

with col2:
    st.markdown("##### Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.2,
        max_value=0.95,
        value=0.4,
        step=0.05,
        help="Higher values = fewer but more confident detections"
    ) 
    
    st.markdown("---")
    st.markdown("##### Model Info")
    st.info(f"""
    **Person:** {person_name}  
    **Confidence:** {int(confidence_threshold * 100)}%  
    **Workspace:** {workspace}
    """)

st.markdown("---")

# Process Button
if st.button("🚀 Start Face Detection", use_container_width=True):
    if uploaded_video:
        with st.spinner("Processing video... This may take a few minutes."):
            # Create temp directory
            tmp_dir = Path(tempfile.gettempdir()) / "face_recognition"
            tmp_dir.mkdir(exist_ok=True)
            
            # Save input video
            in_path = tmp_dir / f"input_{int(time.time())}.mp4"
            with open(in_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            
            # Output path
            out_path = tmp_dir / f"output_{int(time.time())}.mp4"
            
            # Progress tracking
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def update_progress(fraction):
                progress_bar.progress(min(1.0, fraction))
                status_text.text(f"Processing: {fraction*100:.1f}%")
            
            try:
                # Process video
                processor = FaceDetectionProcessor(
                    model=model,
                    person_name=person_name,
                    confidence_threshold=confidence_threshold
                )
                
                stats = processor.process_video(
                    input_path=in_path,
                    output_path=out_path,
                    progress_callback=update_progress
                )
                
                progress_bar.progress(1.0)
                status_text.empty()
                st.toast("✅ Processing complete!", icon="🎉")
                
                # Display Results
                st.markdown("---")
                st.markdown("### 2. Results")
                
                res_col1, res_col2 = st.columns([2, 1])
                
                with res_col1:
                    st.markdown("##### Processed Video")
                    if os.path.exists(out_path):
                        with open(out_path, "rb") as f:
                            video_bytes = f.read()
                        
                        # ----------------------------------------------------
                        # This line displays the video directly in the website
                        # ----------------------------------------------------
                        st.video(video_bytes)
                        
                        st.download_button(
                            "⬇️ Download Processed Video",
                            video_bytes,
                            file_name=f"{person_name}_detection_{int(time.time())}.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                    else:
                        st.error("⚠️ Output file not found!")
                
                with res_col2:
                    st.markdown("##### Statistics")
                    st.metric("Total Frames", f"{stats['total_frames']}")
                    st.metric("Face Detections", f"{stats['detections']}")
                    st.metric("Avg Confidence", f"{stats['avg_confidence']:.1f}%")
                    st.metric("Processing Time", f"{stats['processing_time']:.1f}s")
                    st.metric("Video FPS", f"{stats['fps']}")
                    
                    # Detection rate
                    if stats['total_frames'] > 0:
                        detection_rate = (stats['detections'] / stats['total_frames']) * 100
                        st.metric("Detection Rate", f"{detection_rate:.1f}%")
                
                # Cleanup temp files
                try:
                    if os.path.exists(in_path):
                        os.unlink(in_path)
                except:
                    pass
                
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                st.exception(e)
    else:
        st.warning("⚠️ Please upload a video first!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Built with ❤️ using Streamlit & Roboflow</p>",
    unsafe_allow_html=True
)
