"""Geometric helpers for the person-in-zone intrusion module."""

from typing import Tuple

BBox = Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax
Point = Tuple[float, float]


def bottom_center(box: BBox) -> Point:
    """Bottom-center point of a bounding box.

    Used as the position tested against a zone polygon instead of the
    box centroid: for a person standing under an angled/elevated camera,
    the bottom-center of the detection box approximates where their feet
    touch the ground far better than the box's geometric center does
    (the center sits roughly at torso/waist height, which drifts further
    from the true ground position the more oblique the camera angle is
    — exactly the case a real gate/perimeter camera usually is).
    """
    xmin, ymin, xmax, ymax = box
    return ((xmin + xmax) / 2.0, ymax)
