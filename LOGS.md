# LOGS — session-by-session history

## 2026-07-23 — Person-in-zone intrusion module (Phase 1)

**Starting state:** This repo was a raw multi-class RT-DETR v2 object
detector — a FastAPI `/detect` endpoint (`{file, model}` → unfiltered,
multi-class results) plus a vanilla HTML/CSS/JS webcam+upload visualizer
in `static/`. No zones, no person filtering, no persistence, no tests.
Relocated this session from `Documents/Trials/rtdetr-app` to
`Documents/rtdetr-detector` (a sibling of the main Optisense repo, as
the brief for this work required) — a plain `mv`, git history intact,
confirmed via `git remote -v` before moving that this was genuinely the
`rtdetr-detector` GitHub repo, not a differently-named clone.

**Goal this session:** Bring this to a real person-in-zone intrusion
module without fine-tuning or swapping the detection model, per an
explicit 7-point brief: person-class filtering, Shapely-backed named
polygon zones, bottom-center containment (not box-center), N-consecutive-frame
debounce with a justified default, every (person, zone) pair checked
(not just the first of each), hybrid CPU/GPU device selection, and zero
regression to the existing webcam/upload functionality. Read this repo's
own README.md/backend.py, the main Optisense repo's PLAN.md/backend
CLAUDE.md/frontend CLAUDE.md/last-5-LOGS-entries of both LOGS.md files,
and the sibling `anpr-pipeline` repo's README.md/LOGS.md for its
established license-verification and testing conventions before writing
any code — matched that standard here rather than reinventing one.

**Done:**
- **License verification, not assumed from the brief's own description.**
  The brief stated Shapely is BSD-3 and "already permissively licensed"
  — checked anyway, against the installed package's own metadata
  (`importlib.metadata.metadata('shapely')`), not just trusted:
  `License: BSD 3-Clause`, classifier `License :: OSI Approved :: BSD
  License`. RT-DETR v2's Apache-2.0 status was pre-confirmed by the
  brief itself (same `PekingU/rtdetr_v2` lineage already verified
  elsewhere in this project, per the sibling `anpr-pipeline` repo) — no
  new check needed there, per explicit instruction.
- **`src/geometry.py`** — `bottom_center(box)`, pure, with a docstring
  explaining why bottom-center (feet-position approximation under an
  angled camera) beats box-centroid (torso height) for zone-containment
  purposes.
- **`src/zone.py`** — `Zone` (Shapely `Polygon`-backed, `contains()`
  deliberately uses `covers()` not `contains()` so a point exactly on
  the drawn boundary line still counts as inside — safety-conservative,
  documented inline) + `ZoneRegistry` (in-memory, optional flat-JSON
  persistence to `zones.json` so a dev-server restart doesn't lose zones
  drawn during testing — no database, since this repo is a standalone
  FastAPI PoC and the AI/detection layer is explicitly out of the main
  Optisense backend's scope). Rejects fewer than 3 points and
  self-intersecting polygons with a clear `ValueError` (surfaced as a
  `400` at the API layer).
- **`src/intrusion.py`** — `IntrusionTracker`, a per-zone debounce state
  machine: `update(zone_id, zone_name, occupied)` returns an
  `IntrusionEvent` exactly once per continuous occupied streak (on the
  exact frame the streak first reaches `required_frames`), never
  re-fires while still occupied, and resets the streak to zero
  immediately (not a gradual decay) the first frame the zone reads
  empty. Default `N=3`, justified against this repo's own real, measured
  webcam-polling cadence (`static/script.js`'s 500ms `setInterval`, ~2
  FPS) — 3 consecutive frames is ~1.5s of continuous presence, long
  enough to reject a single-frame detector flicker without introducing
  multi-second alerting latency. Configurable via the constructor or the
  `INTRUSION_DEBOUNCE_FRAMES` env var.
- **`src/detector.py`** — kept deliberately pure/stateless (no model
  loading in this module at all), so it needs zero model weights to
  test: `resolve_device(explicit=None)` (CUDA if available, else CPU;
  MPS is never auto-selected — this repo's own pre-existing README
  already documented a real Apple-Silicon MPS/float64 incompatibility
  with Hugging Face `transformers`, so auto-detecting into `"mps"` would
  have silently reintroduced a bug this repo was already built to avoid;
  an explicit `device="mps"` override still works, with a logged
  warning, for anyone who wants to experiment with it) and
  `filter_persons(results, score_threshold)` (the actual "don't
  fine-tune, just filter output" requirement, applied to RT-DETR v2's
  unmodified multi-class output).
- **`src/pipeline.py`** — `IntrusionPipeline.process_frame()` ties
  detection → person filter → bottom-center → zone containment →
  debounce together for one camera. Explicitly checks **every**
  (person, zone) pair each frame (nested loop: every zone tested against
  every detected person's point), not just the first of each — this is
  what makes multi-person/multi-zone correctness real rather than
  claimed. `detect_fn` is an injected callable, not a hardcoded model
  call, specifically so this class needs zero model weights to test.
- **`backend.py` updated, `/detect` left behaviorally unchanged.** The
  only touch anywhere near the original endpoint is that `get_pipeline()`
  now resolves its device via `resolve_device()` instead of a hardcoded
  `device="cpu"` — on this non-CUDA dev machine that still resolves to
  `"cpu"`, so nothing observable changed there either; confirmed live,
  not just by inspection (see verification below). New, purely additive
  endpoints: `POST /zones`, `GET /zones`, `DELETE /zones/{id}`, `POST
  /detect/intrusion`. `static/` was not touched at all.
- **34 unit tests, all synthetic fakes, zero model weights required**
  (`tests/test_geometry.py`, `test_zone.py`, `test_intrusion.py`,
  `test_detector.py`, `test_pipeline.py`) — `python -m pytest tests/ -v`:
  **34/34 passed.** Explicit coverage of every case the brief called
  out: a point exactly on a zone edge (`test_point_exactly_on_boundary_edge_counts_as_contained`)
  and exactly on a vertex; debounce firing exactly on the Nth frame, not
  before, not repeatedly after; reset-to-zero on a single empty frame,
  not a gradual decay; firing again after a full exit-then-re-entry;
  zones tracked fully independently of each other; every-(person,zone)-pair
  checked in one frame via a 4-detection/2-zone fixture (including a
  non-person detection sitting inside a zone, confirmed ignored);
  multiple people in one zone incrementing `person_count` without
  double-firing; CUDA-available vs. CUDA-unavailable device resolution,
  mocked via `unittest.mock.patch` (no real GPU needed to test the
  branch).
- **Live verification against the real model and real image input, not
  just fakes** — no webcam device is available in this environment, so
  the identical HTTP contract the webcam polling loop uses
  (`multipart/form-data`, same `/detect`-family endpoints) was exercised
  via real static images instead, which is the same code path minus the
  browser-side polling loop:
  - Booted the real `uvicorn` server (port 8099, scratch verification
    port). Boot log confirmed device auto-detection working exactly as
    designed: `"No CUDA device found — using CPU for detection
    inference (MPS deliberately not auto-selected...)"`.
  - Scanned several real street-scene photographs (already present
    locally) through the **unmodified** `/detect` endpoint to find real
    pedestrian detections — found two: `0.5179` and `0.6320` confidence.
  - Created two real zones via `POST /zones`: one covering the real
    detected person's position, one deliberately empty.
  - Posted the same real image to `/detect/intrusion` **4 times** in a
    row (camera_id="verify-cam", default N=3): frame 1 → `streak: 1,
    events: []`; frame 2 → `streak: 2, events: []`; frame 3 → `streak:
    3, intrusion_active: true, events: [{frame_count_at_fire: 3}]`;
    frame 4 → `streak: 4, intrusion_active: true, events: []` (no
    re-fire). The empty zone stayed `occupied: false, streak: 0` across
    all 4 real frames.
  - Posted a real frame with **no detected person** — confirmed the
    previously-active zone's `streak` reset to `0` and `intrusion_active`
    to `false`. Posted the original occupied real image again — confirmed
    the streak restarted at `1`, not continuing from where it left off.
  - Confirmed `POST /detect` still returns unfiltered multi-class
    results (`car` and `person` both present, 3 total detections) on the
    same real image — the person-filtering module has zero effect on
    the original endpoint.
  - Confirmed both new error paths live: `POST /zones` with 2 points →
    real `400` with a clear message; `DELETE` on a nonexistent zone id →
    real `404`.
  - Confirmed `static/index.html` and `static/script.js` still serve
    `200` — the original visualizer is unaffected.
  - Cleaned up the two verification zones and the resulting `zones.json`
    afterward, so a fresh clone/run starts with no zones configured.
- Updated this repo's `README.md` (license table, pipeline-stage diagram,
  new-vs-unchanged sections, the explicit no-cross-frame-tracking scope
  decision, setup/API docs, the real verification numbers above, a
  known-limitations section and an explicitly-not-done section — same
  structure the sibling `anpr-pipeline` repo's README already
  established) and added this `LOGS.md`.

**Not done / blocked:**
- No dashboard/backend integration — per the brief's explicit
  instruction, stopping here for a written proposal, not writing that
  code without approval first.
- No zone-drawing UI (documented as an explicit scope decision in
  README.md, not an oversight).
- No cross-frame person tracking/re-identification (documented as an
  explicit, reasoned scope decision in README.md — adding one would be
  "swap models," out of scope for this module).
- No real webcam-device verification (no camera available in this
  environment) — substituted with the identical HTTP contract via real
  static images, stated plainly as a substitution, not silently implied
  to be full webcam coverage.

**Decisions made:**
- Kept `src/detector.py` fully pure/stateless (no model loading) purely
  so `filter_persons()`/`resolve_device()` are unit-testable without any
  model weights — the actual HF pipeline object stays constructed in
  `backend.py`'s existing `get_pipeline()`, now just parameterized by
  device instead of hardcoding it.
- Debounce is per-zone occupancy, not per-individual person, and this is
  documented as a deliberate scope decision (no tracker was added) — not
  silently glossed over as if per-person debounce had been built.
- `/detect` was left completely behaviorally unchanged rather than
  folding person-filtering into it — a new, additive `/detect/intrusion`
  endpoint was added instead, specifically so the existing webcam/upload
  UI has zero regression risk.

**Notes for next session:**
- Read this file and README.md's "scope decision" section before
  touching `src/intrusion.py` or `src/pipeline.py` — the per-zone (not
  per-person) debounce semantics are load-bearing for how multi-person
  scenarios behave and are easy to accidentally "fix" into something
  that actually needs a tracker.
- Dashboard-integration proposal delivered separately alongside this
  session's work, per the brief's explicit "stop before touching the
  main dashboard/backend, propose first" instruction — do not start
  that code without it being approved first.
- This repo now lives at `Documents/rtdetr-detector`, a sibling of the
  main `Optisense AI` repo, not its old `Documents/Trials/rtdetr-app`
  path — update any saved absolute paths/aliases accordingly.
