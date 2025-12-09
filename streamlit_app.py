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
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0E1117 0%, #1A1E29 100%);
    color: var(--text-color);
}

h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.5rem;
}

[data-testid="stButton"] button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s ease;
}

[data-testid="stButton"] button:hover {
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Roboflow Initialization
# ---------------------------
@st.cache_resource
def init_roboflow():
    """Initialize Roboflow model"""
    try:
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
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {input_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        # Try AVC1 (H.264) first for better browser support, fallback to mp4v
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if not out.isOpened():
                raise Exception("avc1 failed")
        except:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError("Failed to create output video writer")
        
        frame_count = 0
        detection_count = 0
        total_confidence = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_count / total_frames)
            
            # Save current frame to temp file for Roboflow
            # Close immediately to avoid Windows permission errors
            temp_frame = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_frame.close() 
            
            cv2.imwrite(temp_frame.name, frame)
            
            try:
                # Predict
                predictions = self.model.predict(
                    temp_frame.name,
                    confidence=int(self.confidence_threshold * 100)
                ).json()
                
                # Draw
                for pred in predictions['predictions']:
                    x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                    conf = pred['confidence']
                    
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    
                    # Draw Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Draw Label
                    label = f"{self.person_name} {conf*100:.1f}%"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    detection_count += 1
                    total_confidence += conf
                    
            except Exception:
                pass
            
            # Clean up temp frame
            if os.path.exists(temp_frame.name):
                os.unlink(temp_frame.name)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        return {
            'total_frames': frame_count,
            'detections': detection_count,
            'avg_confidence': (total_confidence/detection_count*100) if detection_count > 0 else 0,
            'processing_time': time.time() - start_time,
            'fps': fps
        }

# ---------------------------
# Main App UI
# ---------------------------
st.title("🎥 Face Recognition System")
st.markdown(f"**Detecting:** Pavani | **Powered by:** Roboflow AI")

# Init Model
with st.spinner("🔄 Initializing Model..."):
    model, workspace, person_name = init_roboflow()

if not model:
    st.stop()

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("##### 1. Upload Video")
    # ADDED KEY HERE TO FIX RELOAD ISSUE
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_upload" 
    )
    
    if uploaded_video:
        # Display file info to confirm receipt
        st.info(f"✅ File Received: {uploaded_video.name} ({uploaded_video.size/1024:.0f} KB)")
        st.video(uploaded_video)

with col2:
    st.markdown("##### 2. Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.2, 0.95, 0.4, 0.05)
    
    st.divider()
    
    # Process Button
    start_btn = st.button("🚀 Start Face Detection", use_container_width=True)

if start_btn and uploaded_video:
    with st.spinner("Processing... Do not refresh page."):
        try:
            # 1. Setup Temp Paths
            tmp_dir = Path(tempfile.gettempdir()) / "face_recognition"
            tmp_dir.mkdir(exist_ok=True)
            
            in_path = tmp_dir / f"input_{int(time.time())}.mp4"
            out_path = tmp_dir / f"output_{int(time.time())}.mp4"
            
            # 2. Robust File Save
            uploaded_video.seek(0) # Reset pointer to start
            with open(in_path, "wb") as f:
                f.write(uploaded_video.read())
                
            # 3. Process
            processor = FaceDetectionProcessor(model, person_name, confidence_threshold)
            
            progress_bar = st.progress(0)
            status = st.empty()
            
            def update_bar(p):
                progress_bar.progress(p)
                status.text(f"Processing: {int(p*100)}%")
                
            stats = processor.process_video(in_path, out_path, update_bar)
            
            progress_bar.progress(100)
            status.text("✅ Complete!")
            
            # 4. Show Results
            st.divider()
            st.markdown("### Results")
            
            r_col1, r_col2 = st.columns([2, 1])
            
            with r_col1:
                if os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        video_bytes = f.read()
                    
                    st.success("Video Processed Successfully!")
                    st.video(video_bytes) # Try to play
                    
                    # Download button is critical if browser fails to play codec
                    st.download_button(
                        label="⬇️ Download Result (Recommended)",
                        data=video_bytes,
                        file_name="pavani_detected.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
            
            with r_col2:
                st.metric("Total Detections", stats['detections'])
                st.metric("Avg Confidence", f"{stats['avg_confidence']:.1f}%")
                st.metric("FPS", stats['fps'])

            # Cleanup
            if os.path.exists(in_path):
                os.unlink(in_path)

        except Exception as e:
            st.error(f"❌ Error during processing: {e}")

elif start_btn and not uploaded_video:
    st.warning("⚠️ Please upload a video first.")
