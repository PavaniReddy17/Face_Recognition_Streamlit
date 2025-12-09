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
# CSS Styles
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="st-"], [class*="css-"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0E1117 0%, #1A1E29 100%);
    color: #FAFAFA;
}

h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

[data-testid="stButton"] button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    width: 100%;
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
        # Your specific keys
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
# Video Processing Logic
# ---------------------------
class FaceDetectionProcessor:
    def __init__(self, model, person_name, confidence_threshold=0.4):
        self.model = model
        self.person_name = person_name
        self.confidence_threshold = confidence_threshold
    
    def process_video(self, input_path, output_path, progress_callback=None):
        """Reads video, detects faces, writes output."""
        
        # Open Input
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {input_path}")
        
        # Get Video Properties
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        # Setup Output Writer (Try H.264 first, then MP4V)
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
        
        # Define a static path for the frame being processed
        # This fixes the Axios 400 error by avoiding tempfile locks
        temp_frame_path = "current_processing_frame.jpg"
        
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
            
            # 1. Save frame to disk locally
            cv2.imwrite(temp_frame_path, frame)
            
            try:
                # 2. Send that local file to Roboflow
                predictions = self.model.predict(
                    temp_frame_path,
                    confidence=int(self.confidence_threshold * 100)
                ).json()
                
                # 3. Draw Detections
                for pred in predictions['predictions']:
                    x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                    conf = pred['confidence']
                    
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    
                    # Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    # Label
                    label = f"{self.person_name} {conf*100:.0f}%"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    detection_count += 1
                    total_confidence += conf
                    
            except Exception as e:
                # Log error but keep processing video
                print(f"Frame {frame_count} prediction error: {e}")
                pass
            
            # Write frame to output video
            out.write(frame)
        
        # Cleanup Resources
        cap.release()
        out.release()
        
        # Delete the temp image
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
        
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

# Initialize Model
with st.spinner("🔄 Initializing Model..."):
    model, workspace, person_name = init_roboflow()

if not model:
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("##### 1. Upload Video")
    # KEY="video_uploader" fixes the reload issue
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_uploader"
    )
    
    if uploaded_video:
        st.info(f"✅ Ready to process: {uploaded_video.name}")
        st.video(uploaded_video)

with col2:
    st.markdown("##### 2. Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.2, 0.95, 0.4, 0.05)
    st.divider()
    start_btn = st.button("🚀 Start Face Detection", use_container_width=True)

# ---------------------------
# Execution Logic
# ---------------------------
if start_btn and uploaded_video:
    with st.spinner("Processing video... This may take a moment."):
        try:
            # 1. Setup Temporary Files
            tmp_dir = Path(tempfile.gettempdir()) / "face_recognition"
            tmp_dir.mkdir(exist_ok=True)
            
            in_path = tmp_dir / f"input_{int(time.time())}.mp4"
            out_path = tmp_dir / f"output_{int(time.time())}.mp4"
            
            # 2. Save Uploaded File to Disk (Reset pointer first)
            uploaded_video.seek(0)
            with open(in_path, "wb") as f:
                f.write(uploaded_video.read())
            
            # 3. Run Processing
            processor = FaceDetectionProcessor(model, person_name, confidence_threshold)
            
            # UI Progress Bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_ui(p):
                progress_bar.progress(p)
                status_text.text(f"Processing: {int(p*100)}%")
                
            stats = processor.process_video(in_path, out_path, update_ui)
            
            # 4. Success & Results
            progress_bar.progress(100)
            status_text.text("✅ Processing Complete!")
            
            st.divider()
            st.markdown("### 🎯 Results")
            
            res_c1, res_c2 = st.columns([2, 1])
            
            with res_c1:
                if os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        video_bytes = f.read()
                    
                    st.success("Video processed!")
                    st.video(video_bytes)
                    
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=video_bytes,
                        file_name=f"detected_pavani.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
            
            with res_c2:
                st.metric("Total Detections", stats['detections'])
                st.metric("Avg Confidence", f"{stats['avg_confidence']:.1f}%")
                st.metric("Processing Time", f"{stats['processing_time']:.1f}s")
                st.metric("FPS", stats['fps'])

            # Clean input file
            if os.path.exists(in_path):
                os.unlink(in_path)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)

elif start_btn and not uploaded_video:
    st.warning("⚠️ Please upload a video first!")
