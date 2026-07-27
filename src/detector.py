"""Person-class filtering + hybrid CPU/GPU device selection.

Pure/stateless by design — no model loading happens in this module at
all, so it's fully unit-testable against synthetic fake detection output
with zero model weights downloaded (see tests/test_detector.py). The
real RT-DETR v2 pipeline object is constructed in backend.py's existing
`get_pipeline()`, which calls `resolve_device()` below instead of the
previous hardcoded `device="cpu"`.
"""

import logging
from typing import Dict, List, Optional

import torch

logger = logging.getLogger("rtdetr-detector")

PERSON_LABEL = "person"


def resolve_device(explicit: Optional[str] = None) -> str:
    """Pick an inference device: CUDA if available, else CPU.

    MPS is never auto-selected. Mirrors the sibling anpr-pipeline
    repo's device-selection convention (auto-detect, log the choice,
    allow an explicit override) with one deliberate deviation: this
    repo's own README already documents a real Apple-Silicon MPS
    incompatibility with Hugging Face `transformers` (a float64 op
    support gap) — auto-detecting into "mps" would silently reintroduce
    a bug this repo was already built to avoid. An explicit
    `device="mps"` override still works if a caller wants to experiment
    with it, but it is never the *inferred* default.
    """
    if explicit is not None:
        if explicit == "mps":
            logger.warning(
                "Device explicitly set to 'mps' — this repo's README documents a known "
                "float64/MPS compatibility gap in Hugging Face transformers on Apple "
                "Silicon; expect possible failures."
            )
        logger.info("Detection device explicitly set to '%s'", explicit)
        return explicit

    if torch.cuda.is_available():
        logger.info("CUDA available — using GPU for detection inference")
        return "cuda"

    logger.info("No CUDA device found — using CPU for detection inference (MPS deliberately not auto-selected, see resolve_device docstring)")
    return "cpu"


def filter_persons(results: List[Dict], score_threshold: float = 0.0) -> List[Dict]:
    """Keep only `person`-labeled detections at/above `score_threshold`.

    No fine-tuning, no model swap — this is the entire "bring it to a
    person-in-zone module" filtering step, applied to RT-DETR v2's
    unmodified multi-class output. Every other class the base model can
    detect (car, chair, dog, ...) is discarded here, at the one place
    every downstream stage (zone containment, debounce) reads from.
    """
    return [r for r in results if r.get("label") == PERSON_LABEL and r.get("score", 0.0) >= score_threshold]
