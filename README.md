# RT-DETR v2 Object Detection Web App

A professional, full-stack object detection application utilizing the state-of-the-art **RT-DETR v2** (Real-Time Detection Transformer). Built with a lightweight FastAPI backend and a beautiful, vanilla HTML/CSS/JS frontend featuring a glassmorphism design.

## Features
- **Dual Model Support**: Switch seamlessly between `ResNet-50` (Fast) and `ResNet-101` (Accurate) models dynamically.
- **Live Webcam Inference**: Built-in WebRTC support to stream video from your local camera and run object detection in real-time.
- **Image Upload**: Drag-and-drop file upload for static image analysis.
- **Dynamic Thresholding**: Adjust confidence thresholds on the fly to filter detections.
- **CPU Optimized**: The backend is configured to reliably run inference on CPU, avoiding Apple Silicon (`float64`) MPS compatibility issues with Hugging Face transformers.
- **Commercially Safe**: All dependencies and models (Apache 2.0) are open-source and permissible for commercial use.

## Architecture
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (No frontend frameworks required)
- **Backend**: Python 3, FastAPI, Uvicorn
- **Machine Learning**: PyTorch, Hugging Face `transformers`

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/pranaysb/rtdetr-detector.git
cd rtdetr-detector
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```
> Note: The first time you run the server, it will download the RT-DETR v2 ResNet-50 model weights from Hugging Face (~150MB).

### 5. Access the Web App
Open your browser and navigate to:
[http://localhost:8000](http://localhost:8000)

## Acknowledgements
- [RT-DETR v2 by Peking University](https://github.com/lyuwenyu/RT-DETR)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/rt_detr_v2)
