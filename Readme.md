# 🎥 Pavani Face Recognition System

A professional face recognition system built with **Streamlit** and **Roboflow AI** to detect and track Pavani in video footage.

## ✨ Features

- 🎯 **Real-time Face Detection** - Powered by custom-trained Roboflow model
- 📊 **Advanced Analytics** - Detection statistics and confidence scores
- 🎨 **Modern UI** - Professional dark-themed interface
- 📹 **Video Processing** - Supports MP4, MOV, AVI, MKV formats
- ⬇️ **Easy Download** - Download processed videos instantly
- ☁️ **Cloud Deployment** - Ready for Streamlit Cloud

## 🚀 Quick Start

### Local Installation

\`\`\`bash
# Clone repository
git clone <your-repo-url>
cd face-recognition-pavani

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
\`\`\`

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets in the dashboard:
   - `ROBOFLOW_API_KEY`
   - `ROBOFLOW_WORKSPACE`
   - `ROBOFLOW_PROJECT`
   - `ROBOFLOW_VERSION`
   - `PERSON_NAME`
5. Deploy! 🚀

## 📋 Configuration

Edit `.streamlit/secrets.toml` for local development:

\`\`\`toml
ROBOFLOW_API_KEY = "your-api-key"
ROBOFLOW_WORKSPACE = "your-workspace"
ROBOFLOW_PROJECT = "your-project"
ROBOFLOW_VERSION = "3"
PERSON_NAME = "Pavani"
\`\`\`

## 🎯 Usage

1. Upload a video file
2. Adjust confidence threshold (20-95%)
3. Click "Start Face Detection"
4. View results and download processed video

## 📊 Output Statistics

- Total frames processed
- Number of face detections
- Average confidence score
- Processing time
- Detection rate

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Computer Vision:** OpenCV
- **AI Model:** Roboflow
- **Processing:** NumPy

## 📝 License

MIT License

## 👨‍💻 Author

Your Name

---

Built with ❤️ using Streamlit & Roboflow
