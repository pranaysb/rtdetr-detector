# RT-DETR Person-in-Zone Intrusion Detector
(parent: Optisense )
A FastAPI service that turns RT-DETR v2's raw, multi-class object
detection into a real person-in-zone intrusion module: named polygon
zones, person-only filtering at inference time (no fine-tuning, no
model swap), bottom-center containment, and debounced intrusion events.
The original live-webcam / image-upload multi-class visualizer this repo
started as is unchanged and still works — see "What's still exactly as
it was" below.

## License chain (verified against each package's own metadata — see LOGS.md)

| Layer | Package | License |
|---|---|---|
| Detector | `PekingU/rtdetr_v2_r50vd` / `PekingU/rtdetr_v2_r101vd` | Apache-2.0 (same lineage already verified elsewhere in this project — see the sibling `anpr-pipeline` repo's README) |
| Detection framework | Hugging Face `transformers` | Apache-2.0 |
| Zone geometry | `shapely` | BSD-3-Clause — confirmed directly against the installed package's own metadata (`importlib.metadata`), not assumed from a description: `License: BSD 3-Clause`, classifier `License :: OSI Approved :: BSD License` |
| Web framework | FastAPI / Uvicorn | MIT |

## Pipeline stages

```
frame (webcam poll or uploaded image)
  -> RT-DETR v2 object detection        (unmodified model, unmodified weights)
  -> filter to `person` class only      (src/detector.py — inference-time filter, not a fine-tune)
  -> bottom-center point per person     (src/geometry.py)
  -> point-in-polygon vs. every zone    (src/zone.py, Shapely-backed)
  -> per-zone N-consecutive-frame       (src/intrusion.py)
     debounce, reset on exit
  -> IntrusionEvent, at most once per
     continuous occupied streak
```

Orchestrated end to end by `src/pipeline.py`'s `IntrusionPipeline`, wired
into the FastAPI app in `backend.py`.

## What's new: the person-in-zone module

- **Person-class filtering at inference time** (`src/detector.py`'s
  `filter_persons()`) — the base RT-DETR v2 model and its weights are
  completely unmodified; every other class it can detect (car, chair,
  dog, ...) is simply dropped from the output before anything downstream
  ever sees it.
- **Named polygon zones with camera/site metadata** (`src/zone.py`) —
  `Zone`/`ZoneRegistry`, backed by Shapely for point-in-polygon
  containment (`Polygon.covers()`, not hand-rolled ray-casting). Zones
  persist to a local `zones.json` so a dev-server restart doesn't lose
  them. `POST /zones`, `GET /zones?camera_id=`, `DELETE /zones/{id}`.
- **Bottom-center containment, not box-center** (`src/geometry.py`) —
  `bottom_center()` approximates a person's feet position, which stays
  meaningful under an angled/elevated camera in a way the box's
  geometric centroid (roughly torso height) does not.
- **Debounced intrusion events** (`src/intrusion.py`) — a zone must read
  "occupied" for `N` consecutive frames before an event fires, and the
  streak resets to zero (not a gradual decay) the instant the zone reads
  empty again. Default `N=3`, justified and configurable — see the
  constant's docstring in `src/intrusion.py`, or override per-process
  via the `INTRUSION_DEBOUNCE_FRAMES` env var.
- **Multi-person, multi-zone, every frame** (`src/pipeline.py`) — every
  detected person is tested against every configured zone, every frame;
  multiple people in the same zone increment that zone's `person_count`
  without creating duplicate counters or double-firing.
- **Hybrid CPU/GPU device selection** (`src/detector.py`'s
  `resolve_device()`) — auto-detects CUDA, falls back to CPU, logs which
  was picked, and can be forced via `DETECTION_DEVICE=cuda|cpu|mps`.
  MPS is deliberately never auto-selected (see "A scope decision worth
  knowing about" below) even though it can still be requested explicitly.
- **New endpoint**, additive only: `POST /detect/intrusion` (multipart:
  `file`, `model`, `camera_id`, `threshold`) — returns the filtered
  person detections, every zone's live occupancy/streak/active state for
  that camera, and any `events` that fired on this exact frame.

## What's still exactly as it was

`POST /detect` (raw, unfiltered, multi-class — the endpoint
`static/script.js`'s live-webcam polling loop and image-upload flow both
call) is **untouched in behavior**. The only change anywhere near it is
that `get_pipeline()` now resolves its inference device through the new
hybrid CPU/GPU selector instead of a hardcoded `device="cpu"` — on a
non-CUDA dev machine (like the one this was built and verified on) that
still resolves to `"cpu"`, so nothing observable changes there either.
The static frontend (`static/index.html`/`script.js`/`styles.css`) was
not touched at all.

## A scope decision worth knowing about: no cross-frame person tracking

RT-DETR here is a **stateless per-frame detector** — there is no
tracking-by-detection component (ByteTrack, DeepSORT, etc.) wired in,
and adding one would mean swapping/extending the model stack, explicitly
out of scope for this module ("don't fine-tune, don't swap models — just
filter output"). That means true per-*individual* debounce ("has this
specific person been in the zone for N frames") isn't achievable without
a tracker. What's implemented instead, and what "N consecutive frames"
actually means here, is **per-ZONE occupancy debounce**: has *any*
person been detected in this zone for N consecutive frames. This is
still correctly multi-person/multi-zone-aware — every person is checked
against every zone every frame (see `test_every_person_zone_pair_is_checked_in_one_frame`
in `tests/test_pipeline.py`), and multiple simultaneous occupants of one
zone don't create multiple counters or fire multiple events — but it
does mean the debounce streak survives one person leaving a zone as long
as another person is still in it, rather than being tied to a specific
individual's dwell time. Stated here plainly rather than silently
assumed; revisit if a future requirement genuinely needs per-individual
loitering detection (that would need a tracker, a real scope change).

## Setup

### 1. Install dependencies
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the server
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```
> First run downloads the RT-DETR v2 ResNet-50 weights from Hugging Face
> (~150MB) — unchanged from before this session's work.

### 3. Run the test suite (no model weights required — uses fakes)
```bash
python -m pytest tests/ -v
```
34/34 pass: bounding-box geometry, zone containment (including a point
sitting exactly on a zone edge — a straddling-the-boundary case, not
just clearly-in/clearly-out), debounce state machine (fires exactly on
the Nth consecutive occupied frame, never re-fires while still occupied,
resets to zero on a single empty frame, fires again after a full
exit-then-re-entry, independent per zone), person-class filtering, CUDA/CPU
device resolution, and end-to-end pipeline integration with fake
detections (multi-person/multi-zone in one frame, debounce across
multiple simulated frames).

### 4. Access the web app
[http://localhost:8000](http://localhost:8000) — unchanged image-upload /
live-camera visualizer.

## API

| Endpoint | Method | Notes |
|---|---|---|
| `/detect` | POST | Unchanged — raw multi-class detection, `{file, model}` |
| `/zones` | POST | Create a zone: `{name, points: [[x,y],...], camera_id?, site_id?}` — `400` for fewer than 3 points or a self-intersecting polygon |
| `/zones` | GET | List zones, optional `?camera_id=` filter |
| `/zones/{id}` | DELETE | `404` if the zone doesn't exist |
| `/detect/intrusion` | POST | `{file, model, camera_id, threshold}` — filtered person detections + per-zone occupancy/streak/active state + any events fired this frame |

## Real detection numbers from live verification (not synthetic)

Run against real image input through the real, unmodified RT-DETR v2
ResNet-50 model on this dev machine (CPU — no CUDA device present, auto-detected
and logged correctly): a real photograph with a distant pedestrian
produced a genuine `person` detection at **0.6320** confidence
(`box: {xmin:697, ymin:343, xmax:712, ymax:386}`). Posting that same
image to `/detect/intrusion` 4 times in a row against a zone covering
that position, with the default `N=3` debounce, produced: no event on
frames 1–2 (`streak` 1, 2), an `IntrusionEvent` on frame 3
(`frame_count_at_fire: 3`), and correctly no re-fire on frame 4
(`streak: 4`, `intrusion_active: true`, `events: []`). A second,
non-overlapping zone on the same frames correctly never fired. A
follow-up real frame containing no person correctly reset the zone's
streak to 0 and `intrusion_active` to `false`; the next real occupied
frame correctly restarted the streak at 1, not continuing from where it
left off. Full trace in `LOGS.md`.

## Known limitations to validate against your own footage before trusting in production

1. **No cross-frame person identity/tracking** — see the dedicated
   section above. Debounce is per-zone occupancy, not per-individual
   dwell time.
2. **This repo's own webcam UI polls at ~2 FPS** (`static/script.js`,
   a 500ms interval, chosen there to save CPU) — the default debounce
   `N=3` is tuned against that real, measured cadence (~1.5s to fire).
   A different frame-rate source should probably use a different `N`;
   it is a constructor/env-var parameter specifically so that's not a
   code change.
3. **Zone drawing has no UI yet** — zones are created via the `POST
   /zones` JSON API only; there's no canvas-based polygon-drawing tool
   in `static/`. Out of scope for this phase (a dashboard-integration
   concern, not a detection-module concern) — see the separate
   dashboard-integration proposal.
4. **RT-DETR v2's own published eval numbers apply here unchanged**,
   since the model itself is unmodified — small/distant persons are a
   comparatively weaker case than large/close ones, same caveat the
   sibling `anpr-pipeline` repo documents for its own detector.
5. **In-memory + flat-JSON zone storage, no database.** Fine for a
   single-process dev/demo deployment; a real multi-instance or
   multi-tenant deployment would need a real datastore — explicitly a
   dashboard-integration-proposal question, not resolved here.

## Explicitly not done here (by design)

- No model training or fine-tuning anywhere in this repo.
- No cross-frame object tracking / re-identification (see above).
- No zone-drawing UI.
- No integration into the main Optisense backend/dashboard yet — see
  the separate integration proposal delivered alongside this module,
  which must be approved before any of that code is written.
