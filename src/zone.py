"""Named polygon zones with camera/site metadata.

Point-in-polygon containment is delegated entirely to Shapely (BSD-3
Clause, confirmed against the installed package's own metadata — see
LOGS.md) rather than hand-rolled ray-casting/winding-number code.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

Coordinate = Tuple[float, float]


@dataclass
class Zone:
    id: str
    name: str
    points: List[Coordinate]
    camera_id: Optional[str] = None
    site_id: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(f"Zone '{self.name}' needs at least 3 points, got {len(self.points)}")
        polygon = Polygon(self.points)
        if not polygon.is_valid:
            raise ValueError(f"Zone '{self.name}' polygon is self-intersecting or otherwise invalid")
        # Not a dataclass field on purpose — a derived/cached value, not
        # part of the zone's own identity or serialized shape.
        self._polygon = polygon

    def contains(self, point: Coordinate) -> bool:
        """True if `point` is inside the zone, boundary included.

        `covers()` is used deliberately, not `contains()`: Shapely's
        `contains()` treats a point exactly on the polygon's edge as
        outside (strict topological interior only). For an intrusion
        zone, a person standing exactly on the drawn boundary line
        should still count as "in the zone" — a safety-conservative
        choice, since a missed intrusion is a worse failure than an
        edge-case false positive.
        """
        return self._polygon.covers(ShapelyPoint(point))

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "points": [list(p) for p in self.points],
            "camera_id": self.camera_id,
            "site_id": self.site_id,
        }

    @classmethod
    def create(
        cls,
        name: str,
        points: Iterable[Coordinate],
        camera_id: Optional[str] = None,
        site_id: Optional[str] = None,
    ) -> "Zone":
        return cls(id=str(uuid.uuid4()), name=name, points=list(points), camera_id=camera_id, site_id=site_id)


class ZoneRegistry:
    """In-memory zone store with optional JSON-file persistence.

    No database here by design — this repo is a standalone FastAPI PoC,
    and the AI/detection layer is explicitly out of the main Optisense
    backend's scope (see PLAN.md §0/§6 in the main repo). A flat JSON
    file is enough to survive a dev-server restart without losing zones
    drawn during testing; a real deployment would replace this with
    whatever the Phase 2 dashboard-integration proposal recommends.
    """

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        self._zones: Dict[str, Zone] = {}
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load()

    def create(
        self,
        name: str,
        points: List[Coordinate],
        camera_id: Optional[str] = None,
        site_id: Optional[str] = None,
    ) -> Zone:
        zone = Zone.create(name=name, points=points, camera_id=camera_id, site_id=site_id)
        self._zones[zone.id] = zone
        self._save()
        return zone

    def list(self, camera_id: Optional[str] = None) -> List[Zone]:
        zones = list(self._zones.values())
        if camera_id is not None:
            zones = [z for z in zones if z.camera_id == camera_id]
        return zones

    def get(self, zone_id: str) -> Optional[Zone]:
        return self._zones.get(zone_id)

    def delete(self, zone_id: str) -> bool:
        existed = self._zones.pop(zone_id, None) is not None
        if existed:
            self._save()
        return existed

    def _save(self) -> None:
        if not self._persist_path:
            return
        data = [z.to_dict() for z in self._zones.values()]
        self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        data = json.loads(self._persist_path.read_text())
        for row in data:
            zone = Zone(
                id=row["id"],
                name=row["name"],
                points=[tuple(p) for p in row["points"]],
                camera_id=row.get("camera_id"),
                site_id=row.get("site_id"),
            )
            self._zones[zone.id] = zone
