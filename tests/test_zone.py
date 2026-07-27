import pytest

from src.zone import Zone, ZoneRegistry

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_point_clearly_inside_zone():
    zone = Zone.create(name="Restricted", points=SQUARE)
    assert zone.contains((5, 5)) is True


def test_point_clearly_outside_zone():
    zone = Zone.create(name="Restricted", points=SQUARE)
    assert zone.contains((50, 50)) is False


def test_point_exactly_on_boundary_edge_counts_as_contained():
    # Straddling a zone edge — the exact case called out in the brief.
    # A point sitting precisely on the polygon's edge should be treated
    # as inside (safety-conservative: covers(), not contains() — see
    # Zone.contains()'s docstring).
    zone = Zone.create(name="Restricted", points=SQUARE)
    assert zone.contains((10, 5)) is True  # on the right edge
    assert zone.contains((5, 0)) is True  # on the bottom edge


def test_point_exactly_on_a_vertex_counts_as_contained():
    zone = Zone.create(name="Restricted", points=SQUARE)
    assert zone.contains((0, 0)) is True


def test_point_just_outside_the_boundary_is_not_contained():
    zone = Zone.create(name="Restricted", points=SQUARE)
    assert zone.contains((10.01, 5)) is False


def test_zone_requires_at_least_three_points():
    with pytest.raises(ValueError):
        Zone.create(name="Line", points=[(0, 0), (1, 1)])


def test_zone_rejects_self_intersecting_polygon():
    # A classic bowtie/self-intersecting shape.
    bowtie = [(0, 0), (10, 10), (10, 0), (0, 10)]
    with pytest.raises(ValueError):
        Zone.create(name="Bowtie", points=bowtie)


def test_zone_to_dict_round_trips_points_as_lists():
    zone = Zone.create(name="Restricted", points=SQUARE, camera_id="cam-1", site_id="site-1")
    d = zone.to_dict()
    assert d["name"] == "Restricted"
    assert d["camera_id"] == "cam-1"
    assert d["site_id"] == "site-1"
    assert d["points"] == [list(p) for p in SQUARE]


def test_registry_create_list_get_delete():
    registry = ZoneRegistry()
    zone = registry.create(name="Loading Bay", points=SQUARE, camera_id="cam-1")

    assert registry.get(zone.id) is zone
    assert registry.list() == [zone]
    assert registry.list(camera_id="cam-1") == [zone]
    assert registry.list(camera_id="cam-2") == []

    assert registry.delete(zone.id) is True
    assert registry.get(zone.id) is None
    assert registry.delete(zone.id) is False  # already gone


def test_registry_persists_and_reloads_from_disk(tmp_path):
    persist_path = tmp_path / "zones.json"
    registry = ZoneRegistry(persist_path=persist_path)
    zone = registry.create(name="Dock A", points=SQUARE, camera_id="cam-9")

    assert persist_path.exists()

    reloaded = ZoneRegistry(persist_path=persist_path)
    reloaded_zone = reloaded.get(zone.id)
    assert reloaded_zone is not None
    assert reloaded_zone.name == "Dock A"
    assert reloaded_zone.camera_id == "cam-9"
    assert reloaded_zone.contains((5, 5)) is True
