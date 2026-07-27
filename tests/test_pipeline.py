from src.intrusion import IntrusionTracker
from src.pipeline import IntrusionPipeline
from src.zone import Zone

ZONE_A = Zone.create(name="Zone A", points=[(0, 0), (10, 0), (10, 10), (0, 10)])
ZONE_B = Zone.create(name="Zone B", points=[(100, 100), (110, 100), (110, 110), (100, 110)])


def _detection(xmin, ymin, xmax, ymax, label="person", score=0.9):
    return {"score": score, "label": label, "box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}}


def test_every_person_zone_pair_is_checked_in_one_frame():
    # Person 1 -> bottom-center (5, 10) -> inside Zone A only.
    # Person 2 -> bottom-center (105, 110) -> inside Zone B only.
    # Person 3 -> bottom-center (50, 50) -> inside neither.
    # A non-person detection sitting inside Zone A must be ignored entirely.
    detections = [
        _detection(0, 0, 10, 10),  # person 1 -> Zone A
        _detection(100, 100, 110, 110),  # person 2 -> Zone B
        _detection(45, 45, 55, 55),  # person 3 -> neither zone
        _detection(0, 0, 10, 10, label="car", score=0.99),  # ignored: not a person
    ]

    def fake_detect(_image):
        return detections

    pipeline = IntrusionPipeline(detect_fn=fake_detect, tracker=IntrusionTracker(required_frames=1))
    result = pipeline.process_frame(image=None, zones=[ZONE_A, ZONE_B], score_threshold=0.5)

    assert len(result.persons) == 3  # the car was filtered out before zone-checking even runs

    by_zone = {z.zone_id: z for z in result.zones}
    assert by_zone[ZONE_A.id].occupied is True
    assert by_zone[ZONE_A.id].person_count == 1
    assert by_zone[ZONE_B.id].occupied is True
    assert by_zone[ZONE_B.id].person_count == 1

    # required_frames=1, so both zones should have already fired this frame.
    fired_zone_ids = {e.zone_id for e in result.events}
    assert fired_zone_ids == {ZONE_A.id, ZONE_B.id}


def test_multiple_people_in_the_same_zone_count_but_do_not_double_fire():
    detections = [_detection(1, 1, 2, 2), _detection(3, 3, 4, 4), _detection(5, 5, 6, 6)]

    def fake_detect(_image):
        return detections

    pipeline = IntrusionPipeline(detect_fn=fake_detect, tracker=IntrusionTracker(required_frames=1))
    result = pipeline.process_frame(image=None, zones=[ZONE_A], score_threshold=0.5)

    assert result.zones[0].person_count == 3
    assert len(result.events) == 1  # one event for the zone, not one per person


def test_debounce_end_to_end_across_multiple_frames_no_event_until_nth():
    occupied_frame = [_detection(1, 1, 2, 2)]
    empty_frame = []

    calls = {"frame": 0}

    def fake_detect(_image):
        return occupied_frame

    tracker = IntrusionTracker(required_frames=3)
    pipeline = IntrusionPipeline(detect_fn=fake_detect, tracker=tracker)

    r1 = pipeline.process_frame(image=None, zones=[ZONE_A], score_threshold=0.5)
    assert r1.events == []
    r2 = pipeline.process_frame(image=None, zones=[ZONE_A], score_threshold=0.5)
    assert r2.events == []
    r3 = pipeline.process_frame(image=None, zones=[ZONE_A], score_threshold=0.5)
    assert len(r3.events) == 1
    assert r3.events[0].zone_id == ZONE_A.id


def test_debounce_end_to_end_resets_on_exit_and_can_fire_again():
    tracker = IntrusionTracker(required_frames=2)

    def occupied(_image):
        return [_detection(1, 1, 2, 2)]

    def empty(_image):
        return []

    pipeline = IntrusionPipeline(detect_fn=occupied, tracker=tracker)
    assert pipeline.process_frame(image=None, zones=[ZONE_A], score_threshold=0.5).events == []
    fired = pipeline.process_frame(image=None, zones=[ZONE_A], score_threshold=0.5).events
    assert len(fired) == 1

    pipeline.detect_fn = empty
    exited = pipeline.process_frame(image=None, zones=[ZONE_A], score_threshold=0.5)
    assert exited.zones[0].occupied is False
    assert exited.zones[0].intrusion_active is False

    pipeline.detect_fn = occupied
    assert pipeline.process_frame(image=None, zones=[ZONE_A], score_threshold=0.5).events == []
    re_fired = pipeline.process_frame(image=None, zones=[ZONE_A], score_threshold=0.5).events
    assert len(re_fired) == 1
