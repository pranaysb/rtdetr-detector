from unittest.mock import patch

from src.detector import filter_persons, resolve_device

FAKE_RESULTS = [
    {"score": 0.95, "label": "person", "box": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 20}},
    {"score": 0.88, "label": "car", "box": {"xmin": 20, "ymin": 20, "xmax": 40, "ymax": 50}},
    {"score": 0.40, "label": "person", "box": {"xmin": 5, "ymin": 5, "xmax": 15, "ymax": 25}},
    {"score": 0.99, "label": "chair", "box": {"xmin": 1, "ymin": 1, "xmax": 2, "ymax": 2}},
]


def test_filter_persons_drops_non_person_classes():
    filtered = filter_persons(FAKE_RESULTS, score_threshold=0.0)
    labels = {r["label"] for r in filtered}
    assert labels == {"person"}
    assert len(filtered) == 2


def test_filter_persons_respects_score_threshold():
    filtered = filter_persons(FAKE_RESULTS, score_threshold=0.5)
    assert len(filtered) == 1
    assert filtered[0]["score"] == 0.95


def test_filter_persons_on_empty_input():
    assert filter_persons([], score_threshold=0.0) == []


def test_filter_persons_with_no_person_detections():
    only_cars = [r for r in FAKE_RESULTS if r["label"] != "person"]
    assert filter_persons(only_cars) == []


def test_resolve_device_explicit_override_is_returned_as_is():
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"  # allowed explicitly, just not auto-selected


@patch("src.detector.torch.cuda.is_available", return_value=True)
def test_resolve_device_auto_selects_cuda_when_available(_mock):
    assert resolve_device(None) == "cuda"


@patch("src.detector.torch.cuda.is_available", return_value=False)
def test_resolve_device_falls_back_to_cpu_never_mps(_mock):
    assert resolve_device(None) == "cpu"
