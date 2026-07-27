import io
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from transformers import pipeline

from src.detector import resolve_device
from src.intrusion import DEFAULT_DEBOUNCE_FRAMES, IntrusionTracker
from src.pipeline import IntrusionPipeline
from src.zone import ZoneRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rtdetr-detector")

app = FastAPI(title="RT-DETR v2 Object Detection API")

# Initialize pipeline cache
pipelines = {}


def get_pipeline(model_name: str):
    # Default to r50vd if an invalid name is passed
    model_id = "r101vd" if model_name == "r101vd" else "r50vd"
    hf_id = f"PekingU/rtdetr_v2_{model_id}"

    if hf_id not in pipelines:
        # Hybrid CPU/GPU selection: auto-detects CUDA, falls back to CPU
        # (never auto-selects MPS — see resolve_device()'s docstring),
        # and can be forced via the DETECTION_DEVICE env var.
        device = resolve_device(os.getenv("DETECTION_DEVICE"))
        print(f"Loading {hf_id} model on device '{device}'...")
        pipelines[hf_id] = pipeline("object-detection", model=hf_id, device=device)
        print(f"Model {hf_id} loaded successfully on '{device}'!")

    return pipelines[hf_id]


# Pre-load r50vd to keep startup fast
get_pipeline("r50vd")

# Serve the static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), model: str = Form("r50vd")):
    # Unchanged from before this session's work: raw, unfiltered
    # multi-class detection, exactly the contract static/script.js's
    # live-webcam and image-upload flows already depend on. Nothing
    # about the person-in-zone module below alters this endpoint.
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


# ---------------------------------------------------------------------------
# Person-in-zone intrusion module — see README.md / LOGS.md for the design.
# ---------------------------------------------------------------------------

_zone_registry = ZoneRegistry(persist_path=Path(__file__).parent / "zones.json")

# One IntrusionPipeline (and therefore one debounce tracker) per
# (camera_id, model) pair — debounce streak state must persist across
# repeated polling calls for the *same* camera, but must never be shared
# across two different camera feeds, which is exactly what keying by
# camera_id gives us.
_intrusion_pipelines: Dict[str, IntrusionPipeline] = {}


def _pipeline_for_camera(camera_id: str, model: str) -> IntrusionPipeline:
    key = f"{camera_id}:{model}"
    if key not in _intrusion_pipelines:
        debounce_frames = int(os.getenv("INTRUSION_DEBOUNCE_FRAMES", str(DEFAULT_DEBOUNCE_FRAMES)))
        pipe = get_pipeline(model)
        _intrusion_pipelines[key] = IntrusionPipeline(
            detect_fn=pipe,
            tracker=IntrusionTracker(required_frames=debounce_frames),
        )
    return _intrusion_pipelines[key]


class ZoneCreateRequest(BaseModel):
    name: str
    points: List[Tuple[float, float]]
    camera_id: Optional[str] = None
    site_id: Optional[str] = None


@app.post("/zones")
def create_zone(req: ZoneCreateRequest):
    try:
        zone = _zone_registry.create(name=req.name, points=req.points, camera_id=req.camera_id, site_id=req.site_id)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})
    return {"status": "success", "zone": zone.to_dict()}


@app.get("/zones")
def list_zones(camera_id: Optional[str] = None):
    return {"status": "success", "zones": [z.to_dict() for z in _zone_registry.list(camera_id=camera_id)]}


@app.delete("/zones/{zone_id}")
def delete_zone(zone_id: str):
    existed = _zone_registry.delete(zone_id)
    if not existed:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Zone not found"})
    return {"status": "success"}


@app.post("/detect/intrusion")
async def detect_intrusion(
    file: UploadFile = File(...),
    model: str = Form("r50vd"),
    camera_id: str = Form("default"),
    threshold: float = Form(0.5),
):
    """Runs one frame through the person-in-zone module for `camera_id`:
    filters detections to `person`, tests every person against every
    zone configured for this camera (bottom-center point, see
    src/geometry.py), and advances that camera's per-zone debounce
    trackers. `events` is only ever non-empty on the exact frame a
    zone's debounce streak first reaches the configured threshold.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        zones = _zone_registry.list(camera_id=camera_id)
        intrusion_pipeline = _pipeline_for_camera(camera_id, model)
        result = intrusion_pipeline.process_frame(image, zones, score_threshold=threshold)

        for event in result.events:
            logger.warning("INTRUSION: camera=%s zone=%s (%s)", camera_id, event.zone_name, event.zone_id)

        return JSONResponse(
            content={
                "status": "success",
                "persons": result.persons,
                "zones": [asdict(z) for z in result.zones],
                "events": [asdict(e) for e in result.events],
            }
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
