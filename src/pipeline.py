"""Orchestrates one frame through detection -> person filter -> zone
containment -> debounce, for a single camera stream.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from PIL import Image

from .detector import filter_persons
from .geometry import bottom_center
from .intrusion import IntrusionEvent, IntrusionTracker
from .zone import Zone


@dataclass
class ZoneOccupancy:
    zone_id: str
    zone_name: str
    occupied: bool
    person_count: int
    streak: int
    intrusion_active: bool


@dataclass
class FrameResult:
    persons: List[Dict]
    zones: List[ZoneOccupancy]
    events: List[IntrusionEvent]


class IntrusionPipeline:
    """`detect_fn` is any callable(image) -> List[Dict] shaped like the
    Hugging Face object-detection pipeline's raw output (unfiltered,
    multi-class). In production this is backend.py's existing
    `get_pipeline(model)` result; in tests it's a fake returning canned
    detections — this class needs no model weights to test.
    """

    def __init__(self, detect_fn: Callable[[Image.Image], List[Dict]], tracker: Optional[IntrusionTracker] = None) -> None:
        self.detect_fn = detect_fn
        self.tracker = tracker or IntrusionTracker()

    def process_frame(self, image: Image.Image, zones: List[Zone], score_threshold: float = 0.5) -> FrameResult:
        raw_results = self.detect_fn(image)
        persons = filter_persons(raw_results, score_threshold=score_threshold)
        points = [bottom_center(_as_box_tuple(p["box"])) for p in persons]

        occupancy: List[ZoneOccupancy] = []
        events: List[IntrusionEvent] = []

        # Every (person, zone) pair is checked here, not just the first
        # of each: the inner list comprehension below tests every one of
        # `points` against the current zone, and the outer loop repeats
        # that for every zone in `zones` — an O(persons x zones)
        # evaluation each frame, by design.
        for zone in zones:
            contained = [pt for pt in points if zone.contains(pt)]
            occupied = len(contained) > 0
            event = self.tracker.update(zone.id, zone.name, occupied)
            if event:
                events.append(event)
            occupancy.append(
                ZoneOccupancy(
                    zone_id=zone.id,
                    zone_name=zone.name,
                    occupied=occupied,
                    person_count=len(contained),
                    streak=self.tracker.get_streak(zone.id),
                    intrusion_active=self.tracker.is_active(zone.id),
                )
            )

        return FrameResult(persons=persons, zones=occupancy, events=events)


def _as_box_tuple(box) -> tuple:
    if isinstance(box, dict):
        return (box["xmin"], box["ymin"], box["xmax"], box["ymax"])
    return tuple(box)
