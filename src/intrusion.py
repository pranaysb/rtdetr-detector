"""Debounced intrusion tracking.

Requires N consecutive occupied frames before firing an event for a
given zone, and resets immediately (not a gradual decay) the first
frame the zone reads empty again.

Scope note, stated plainly rather than silently assumed: RT-DETR here is
a stateless per-frame detector with no cross-frame object identity (no
tracking-by-detection component such as ByteTrack/DeepSORT is wired in —
adding one would mean "swap models," explicitly out of scope per this
module's brief). So debounce is tracked per ZONE ("has this zone had at
least one person in it for N consecutive frames"), not per individual
person ("has this specific person been in the zone for N frames").
Multi-person/multi-zone correctness is still real and exercised: every
detected person is tested against every zone each frame (see
IntrusionPipeline in pipeline.py), so two people in the same zone at
once don't create two counters or double-fire, and the same frame is
independently evaluated against every configured zone.
"""

from dataclasses import dataclass
from typing import Dict, Optional

DEFAULT_DEBOUNCE_FRAMES = 3
# Justification for the default: this repo's own webcam polling loop
# (static/script.js) runs at ~2 FPS (a 500ms setInterval, chosen there to
# save CPU). At that real, measured cadence, 3 consecutive occupied
# frames is ~1.5 seconds of continuous presence before an intrusion
# fires — long enough to reject a single-frame detector flicker (RT-DETR
# occasionally drops or regains a box for one frame on a
# partially-occluded or motion-blurred person, especially at the
# ResNet-50 "fast" setting) without introducing multi-second alerting
# latency, which would defeat the point of a real-time intrusion module.
# Configurable per deployment: raise it for a noisier camera or a lower
# effective frame rate; the practical floor is 2 (1 frame provides zero
# debounce at all).


@dataclass
class IntrusionEvent:
    zone_id: str
    zone_name: str
    frame_count_at_fire: int


@dataclass
class _ZoneState:
    streak: int = 0
    fired: bool = False


class IntrusionTracker:
    """Per-zone debounce state machine.

    One instance per camera stream — don't share a tracker across
    independent camera feeds; streak state has no meaning mixed between
    two physically different video sources.
    """

    def __init__(self, required_frames: int = DEFAULT_DEBOUNCE_FRAMES) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be >= 1")
        self.required_frames = required_frames
        self._state: Dict[str, _ZoneState] = {}

    def update(self, zone_id: str, zone_name: str, occupied: bool) -> Optional[IntrusionEvent]:
        """Advance this zone's debounce state by one frame.

        Returns an `IntrusionEvent` exactly once per continuous
        occupied streak — the frame the streak first reaches
        `required_frames` — never on every subsequent frame while the
        zone stays occupied (that would spam an alert once per poll
        instead of firing a single, actionable event). Returns `None`
        on every other call, including the reset call.
        """
        state = self._state.setdefault(zone_id, _ZoneState())

        if not occupied:
            state.streak = 0
            state.fired = False
            return None

        state.streak += 1

        if state.streak >= self.required_frames and not state.fired:
            state.fired = True
            return IntrusionEvent(zone_id=zone_id, zone_name=zone_name, frame_count_at_fire=state.streak)

        return None

    def get_streak(self, zone_id: str) -> int:
        state = self._state.get(zone_id)
        return state.streak if state else 0

    def is_active(self, zone_id: str) -> bool:
        """True once an intrusion has fired and the zone hasn't emptied since."""
        state = self._state.get(zone_id)
        return bool(state and state.fired)

    def reset(self, zone_id: Optional[str] = None) -> None:
        """Clear debounce state — all zones, or just one."""
        if zone_id is None:
            self._state.clear()
        else:
            self._state.pop(zone_id, None)
