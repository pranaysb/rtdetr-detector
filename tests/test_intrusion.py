from src.intrusion import IntrusionTracker


def test_never_fires_below_debounce_threshold():
    tracker = IntrusionTracker(required_frames=3)
    assert tracker.update("z1", "Zone 1", occupied=True) is None
    assert tracker.update("z1", "Zone 1", occupied=True) is None
    # Only 2 of the required 3 consecutive frames — no event yet.
    assert tracker.is_active("z1") is False


def test_fires_exactly_on_the_nth_consecutive_frame():
    tracker = IntrusionTracker(required_frames=3)
    assert tracker.update("z1", "Zone 1", occupied=True) is None
    assert tracker.update("z1", "Zone 1", occupied=True) is None
    event = tracker.update("z1", "Zone 1", occupied=True)

    assert event is not None
    assert event.zone_id == "z1"
    assert event.frame_count_at_fire == 3
    assert tracker.is_active("z1") is True


def test_does_not_refire_every_frame_while_still_occupied():
    tracker = IntrusionTracker(required_frames=2)
    assert tracker.update("z1", "Zone 1", occupied=True) is None
    first_fire = tracker.update("z1", "Zone 1", occupied=True)
    assert first_fire is not None

    # Zone stays occupied for several more frames — must not spam a new
    # event on every single one of them.
    for _ in range(5):
        assert tracker.update("z1", "Zone 1", occupied=True) is None
    assert tracker.is_active("z1") is True


def test_resets_immediately_on_a_single_empty_frame():
    tracker = IntrusionTracker(required_frames=3)
    tracker.update("z1", "Zone 1", occupied=True)
    tracker.update("z1", "Zone 1", occupied=True)
    # One frame with nobody in the zone before reaching the threshold —
    # the streak must reset to zero, not just pause.
    assert tracker.update("z1", "Zone 1", occupied=False) is None
    assert tracker.get_streak("z1") == 0

    # Re-entering now requires a full fresh streak of 3, not "1 more".
    assert tracker.update("z1", "Zone 1", occupied=True) is None
    assert tracker.update("z1", "Zone 1", occupied=True) is None
    assert tracker.update("z1", "Zone 1", occupied=True) is not None


def test_fires_again_after_a_full_exit_and_re_entry():
    tracker = IntrusionTracker(required_frames=2)
    assert tracker.update("z1", "Zone 1", occupied=True) is None
    assert tracker.update("z1", "Zone 1", occupied=True) is not None
    assert tracker.is_active("z1") is True

    # Person leaves the zone entirely.
    assert tracker.update("z1", "Zone 1", occupied=False) is None
    assert tracker.is_active("z1") is False

    # A new intrusion (same or a different person) must fire again, not
    # be silently suppressed because "this zone already fired once".
    assert tracker.update("z1", "Zone 1", occupied=True) is None
    second_fire = tracker.update("z1", "Zone 1", occupied=True)
    assert second_fire is not None


def test_zones_are_tracked_independently():
    tracker = IntrusionTracker(required_frames=2)
    tracker.update("z1", "Zone 1", occupied=True)
    # Zone 2 has never seen anyone — its streak must stay at 0
    # regardless of Zone 1's state.
    assert tracker.get_streak("z1") == 1
    assert tracker.get_streak("z2") == 0

    fire_z2 = tracker.update("z2", "Zone 2", occupied=True)
    assert fire_z2 is None  # only 1 frame so far for z2
    fire_z2 = tracker.update("z2", "Zone 2", occupied=True)
    assert fire_z2 is not None
    # Zone 1 is unaffected by Zone 2's frames.
    assert tracker.is_active("z1") is False


def test_required_frames_of_one_fires_on_first_occupied_frame():
    tracker = IntrusionTracker(required_frames=1)
    event = tracker.update("z1", "Zone 1", occupied=True)
    assert event is not None
    assert event.frame_count_at_fire == 1


def test_rejects_invalid_required_frames():
    import pytest

    with pytest.raises(ValueError):
        IntrusionTracker(required_frames=0)


def test_reset_clears_a_single_zone_only():
    tracker = IntrusionTracker(required_frames=3)
    tracker.update("z1", "Zone 1", occupied=True)
    tracker.update("z2", "Zone 2", occupied=True)

    tracker.reset("z1")
    assert tracker.get_streak("z1") == 0
    assert tracker.get_streak("z2") == 1


def test_reset_with_no_zone_id_clears_everything():
    tracker = IntrusionTracker(required_frames=3)
    tracker.update("z1", "Zone 1", occupied=True)
    tracker.update("z2", "Zone 2", occupied=True)

    tracker.reset()
    assert tracker.get_streak("z1") == 0
    assert tracker.get_streak("z2") == 0
