const modeImageBtn = document.getElementById('mode-image');
const modeCameraBtn = document.getElementById('mode-camera');
const uploadSection = document.getElementById('upload-section');

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const workspace = document.getElementById('workspace');
const canvas = document.getElementById('image-canvas');
const ctx = canvas.getContext('2d');
const video = document.getElementById('webcam-video');
const loadingOverlay = document.getElementById('loading-overlay');
const resultsList = document.getElementById('results-list');
const thresholdInput = document.getElementById('threshold');
const thresholdVal = document.getElementById('threshold-val');
const modelSelect = document.getElementById('model-select');

let currentMode = 'image';
let currentImage = null;
let currentDetections = [];
let webcamStream = null;
let detectionInterval = null;
let renderInterval = null;
let isDetecting = false;

// UI Mode Toggle
modeImageBtn.addEventListener('click', () => setMode('image'));
modeCameraBtn.addEventListener('click', () => setMode('camera'));

async function setMode(mode) {
    currentMode = mode;
    if (mode === 'image') {
        modeImageBtn.classList.add('active');
        modeCameraBtn.classList.remove('active');
        uploadSection.classList.remove('hidden');
        stopWebcam();
        if (!currentImage) workspace.classList.add('hidden');
        else renderStaticImage();
    } else {
        modeImageBtn.classList.remove('active');
        modeCameraBtn.classList.add('active');
        uploadSection.classList.add('hidden');
        workspace.classList.remove('hidden');
        await startWebcam();
    }
}

// Webcam Logic
async function startWebcam() {
    try {
        currentDetections = [];
        updateResultsList();
        
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
        video.srcObject = webcamStream;
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            video.play();
            
            // Start render loop
            renderVideoFrame();
            
            // Start detection polling (roughly 2 FPS to save CPU)
            detectionInterval = setInterval(pollWebcamDetection, 500);
        };
    } catch (err) {
        alert('Could not access webcam. Please ensure permissions are granted.');
        console.error(err);
        setMode('image');
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    if (detectionInterval) clearInterval(detectionInterval);
    if (renderInterval) cancelAnimationFrame(renderInterval);
}

function renderVideoFrame() {
    if (currentMode !== 'camera') return;
    
    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Draw bounding boxes on top
    drawDetectionsOverlay();
    
    renderInterval = requestAnimationFrame(renderVideoFrame);
}

async function pollWebcamDetection() {
    if (isDetecting || currentMode !== 'camera') return;
    isDetecting = true;
    
    // Create a temporary hidden canvas to grab the frame safely without overlay
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    tempCanvas.getContext('2d').drawImage(video, 0, 0);
    
    tempCanvas.toBlob(async (blob) => {
        if (!blob) {
            isDetecting = false;
            return;
        }
        
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        formData.append('model', modelSelect.value);
        
        try {
            const response = await fetch('/detect', { method: 'POST', body: formData });
            const data = await response.json();
            if (data.status === 'success') {
                currentDetections = data.results;
                updateResultsList();
            }
        } catch (e) {
            console.error("Detection error", e);
        } finally {
            isDetecting = false;
        }
    }, 'image/jpeg', 0.8);
}


// Image Drag and Drop logic
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drop-zone--over');
});

['dragleave', 'dragend'].forEach(type => {
    dropZone.addEventListener(type, () => {
        dropZone.classList.remove('drop-zone--over');
    });
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drop-zone--over');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFile(fileInput.files[0]);
    }
});

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

// Threshold logic
thresholdInput.addEventListener('input', (e) => {
    thresholdVal.textContent = `${e.target.value}%`;
    if (currentMode === 'image') renderStaticImage();
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            workspace.classList.remove('hidden');
            
            canvas.width = img.width;
            canvas.height = img.height;
            renderStaticImage();
            
            detectObjects(file);
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

async function detectObjects(file) {
    loadingOverlay.classList.remove('hidden');
    currentDetections = [];
    updateResultsList();
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', modelSelect.value);
    
    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            currentDetections = data.results;
            renderStaticImage();
            updateResultsList();
        } else {
            alert('Error during detection: ' + data.message);
        }
    } catch (error) {
        console.error(error);
        alert('Network error while connecting to the backend.');
    } finally {
        loadingOverlay.classList.add('hidden');
    }
}

function renderStaticImage() {
    if (!currentImage) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImage, 0, 0);
    drawDetectionsOverlay();
}

function drawDetectionsOverlay() {
    const threshold = parseInt(thresholdInput.value) / 100;
    
    const colors = [
        '#10b981', '#3b82f6', '#f59e0b', '#ef4444', 
        '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'
    ];
    
    currentDetections.forEach((det) => {
        if (det.score >= threshold) {
            let hash = 0;
            for(let i=0; i<det.label.length; i++) hash = (hash << 5) - hash + det.label.charCodeAt(i);
            const color = colors[Math.abs(hash) % colors.length];
            
            let box = det.box;
            let xmin, ymin, xmax, ymax;
            
            if (box.xmin !== undefined) {
                xmin = box.xmin; ymin = box.ymin;
                xmax = box.xmax; ymax = box.ymax;
            } else if (Array.isArray(box)) {
                [xmin, ymin, xmax, ymax] = box;
            }
            
            ctx.strokeStyle = color;
            ctx.lineWidth = Math.max(3, canvas.width / 400); 
            ctx.beginPath();
            ctx.rect(xmin, ymin, xmax - xmin, ymax - ymin);
            ctx.stroke();
            
            const labelText = `${det.label} ${(det.score * 100).toFixed(0)}%`;
            const fontSize = Math.max(16, canvas.width / 50);
            ctx.font = `${fontSize}px sans-serif`;
            const textMetrics = ctx.measureText(labelText);
            
            ctx.fillStyle = color;
            ctx.fillRect(xmin, ymin - fontSize - 4, textMetrics.width + 8, fontSize + 4);
            
            ctx.fillStyle = '#ffffff';
            ctx.fillText(labelText, xmin + 4, ymin - 4);
        }
    });
}

function updateResultsList() {
    resultsList.innerHTML = '';
    const threshold = parseInt(thresholdInput.value) / 100;
    
    const colors = [
        '#10b981', '#3b82f6', '#f59e0b', '#ef4444', 
        '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'
    ];
    
    let count = 0;
    currentDetections.forEach((det) => {
        if (det.score >= threshold) {
            count++;
            let hash = 0;
            for(let i=0; i<det.label.length; i++) hash = (hash << 5) - hash + det.label.charCodeAt(i);
            const color = colors[Math.abs(hash) % colors.length];
            
            const li = document.createElement('li');
            li.className = 'result-item';
            li.style.borderLeftColor = color;
            li.innerHTML = `
                <span class="result-label">${det.label}</span>
                <span class="result-score">${(det.score * 100).toFixed(1)}%</span>
            `;
            resultsList.appendChild(li);
        }
    });
    
    if (count === 0) {
        resultsList.innerHTML = '<li class="result-item" style="border-left-color: transparent; text-align: center; color: #94a3b8; display: block;">No detections above threshold</li>';
    }
}
