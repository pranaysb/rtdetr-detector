from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
from PIL import Image
import io
from transformers import pipeline

app = FastAPI(title="RT-DETR v2 Object Detection API")

# Initialize pipeline cache
pipelines = {}

def get_pipeline(model_name: str):
    # Default to r50vd if an invalid name is passed
    model_id = "r101vd" if model_name == "r101vd" else "r50vd"
    hf_id = f"PekingU/rtdetr_v2_{model_id}"
    
    if hf_id not in pipelines:
        print(f"Loading {hf_id} model...")
        pipelines[hf_id] = pipeline("object-detection", model=hf_id, device="cpu")
        print(f"Model {hf_id} loaded successfully!")
        
    return pipelines[hf_id]

# Pre-load r50vd to keep startup fast
get_pipeline("r50vd")

# Serve the static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), model: str = Form("r50vd")):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get requested pipeline
        pipe = get_pipeline(model)
        
        # Run inference
        results = pipe(image)
        
        return JSONResponse(content={"status": "success", "results": results})
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/")
def read_root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")
