from src.geometry import bottom_center


def test_bottom_center_of_square_box():
    assert bottom_center((0, 0, 10, 20)) == (5.0, 20.0)


def test_bottom_center_of_offset_box():
    # xmin=100, ymin=50, xmax=140, ymax=250 -> center-x=120, bottom-y=250
    assert bottom_center((100, 50, 140, 250)) == (120.0, 250.0)


def test_bottom_center_uses_ymax_not_box_centroid():
    # A tall, narrow box (person standing far from an angled camera) —
    # the centroid would be at y=(ymin+ymax)/2=125, well above the feet.
    box = (10, 0, 30, 250)
    x, y = bottom_center(box)
    assert y == 250.0  # bottom edge, not the vertical midpoint (125.0)
    assert x == 20.0
