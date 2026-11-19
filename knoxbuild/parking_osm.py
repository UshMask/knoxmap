"""OSM parking policy and access geometry in map-local tiles."""
from __future__ import annotations

from collections import Counter
from math import isfinite

from shapely import LineString, Point, STRtree, box
from shapely.geometry import mapping

from generator.osm import classify, load_project_cache
from generator.rendering.crossings import _at_ground
from generator.rendering.geometry import _feature_coords_px, _is_polygon
from generator.rendering.projection import shape_px
from generator.rendering.widths import _way_width_m

from .reference import compare_count

ROAD_CLASSES = {"road_minor", "road_medium", "road_major", "road_service"}
GATES = {"gate", "lift_gate", "swing_gate", "bollard", "chain"}
PARKING_BANS = {"no", "no_parking", "no_standing", "no_stopping", "loading_only", "charging_only"}
OFF_ROAD_PARKING = {"separate", "street_side", "on_kerb", "half_on_kerb", "shoulder", "layby"}
NON_SURFACE_LOTS = {"underground", "rooftop", "multi-storey", "garage", "garage_boxes"}
RESTRICTED_ACCESS = {"no", "private", "permit", "delivery", "emergency"}


def lot_policy(tags):
    """Return the supported row orientation, or a reason to leave the area alone."""
    if tags.get("landuse") == "garages" or tags.get("amenity") == "bus_station":
        return None, "different_use"
    if (tags.get("parking") in NON_SURFACE_LOTS or tags.get("location") in {"roof", "rooftop"}
            or not _at_ground(tags)):
        return None, "not_surface"
    if (tags.get("restriction") in PARKING_BANS or
            (tags.get("motorcar") or tags.get("motor_vehicle") or tags.get("access"))
            in RESTRICTED_ACCESS):
        return None, "restricted_access"
    orientation = tags.get("orientation") or tags.get("parking:orientation")
    if orientation and orientation not in {"parallel", "perpendicular"}:
        return None, "unsupported_orientation"
    return orientation or ("parallel" if tags.get("parking") in OFF_ROAD_PARKING else "perpendicular"), None


def side_policy(tags, side):
    """Read modern side tags first, with the older lane scheme as a fallback."""
    def value(suffix):
        return tags.get(f"parking:{side}{suffix}", tags.get(f"parking:both{suffix}"))

    lane = tags.get(f"parking:lane:{side}", tags.get("parking:lane:both"))
    condition = tags.get(f"parking:condition:{side}", tags.get("parking:condition:both"))
    placement = value("") or lane
    if (placement in PARKING_BANS or (value(":restriction") or condition) in PARKING_BANS
            or (value(":motorcar") or value(":motor_vehicle") or value(":access"))
            in RESTRICTED_ACCESS):
        return None, "osm_parking_restriction"
    if placement in OFF_ROAD_PARKING:
        return None, "off_carriageway_parking"
    orientation = value(":orientation") or (lane if lane in {"parallel", "diagonal", "perpendicular"} else None)
    if orientation and orientation not in {"parallel", "perpendicular"}:
        return None, "unsupported_orientation"
    return orientation or "parallel", None


def compare_lot_capacity(props, parts, parking_osm):
    osm_id = props.get("osm_id")
    raw = props.get("capacity")
    if "capacity" not in props and "osm_kind" not in props and parking_osm is not None:
        raw = parking_osm.lot_capacity(props)
    comparison = compare_count(
        "parking_spaces", sum(part["capacity"] for part in parts),
        source="map_sketch" if isinstance(osm_id, int) and osm_id < 0 else "OSM",
        field="capacity", raw=raw,
        comparable=all(part["within_map"] for part in parts),
    )
    reasons = Counter()
    for part in parts:
        reasons.update(part["layout_rejections"])
        if part["reason"] and part["reason"] != "no_space":
            reasons[part["reason"]] += 1
        if not part["within_map"]:
            reasons["scope_outside_map"] += 1
    if not reasons and comparison["delta"]:
        reasons["unexplained_layout_difference"] = 1
    return {"osm_id": osm_id, "osm_kind": props.get("osm_kind"),
            "name": props.get("name"), "part_bounds": [part["bounds"] for part in parts],
            **comparison, "reasons": reasons,
            **{key: sum(part[key] for part in parts) for key in ("eligible", "target", "exported")}}


class ParkingOSM:
    @classmethod
    def load(cls, out_dir, map_name, info, proj):
        from .street_context import _drawn_features

        _, features = load_project_cache(out_dir, map_name, info)
        features += _drawn_features(out_dir, map_name)
        if features and info.get("straight_roads"):
            from generator.octilinear import replay

            replay(features, proj, classify, _is_polygon, info, out_dir, map_name)
        policy = cls(features, proj)
        selection = shape_px(info.get("shape"), proj)
        if selection is not None:
            policy.extent = policy.extent.intersection(selection)
        return policy

    def __init__(self, features, proj):
        self.clearances = []
        self.roads = []
        road_lines = []
        crossings = []
        mpt = max(0.05, proj.meters_per_tile)
        frame = box(0, 0, proj.width, proj.height)
        self.extent = frame
        tagged_roads = False
        self.excluded_osm_lots = Counter()
        self.lot_tags = {}
        for feat in features:
            tags = feat.tags
            if tags.get("amenity") == "parking":
                self.lot_tags.setdefault(feat.osm_id, {})[feat.kind] = tags
                _, reason = lot_policy(tags)
                if reason:
                    self.excluded_osm_lots[reason] += 1
            if not _at_ground(tags) or not feat.geometry or feat.kind not in {"way", "node"}:
                continue
            cat = classify(tags)
            if feat.kind == "way":
                points = _feature_coords_px(feat, proj)[0]
                if len(set(points)) < 2:
                    continue
                line = LineString(points)
                if not frame.intersects(line):
                    continue
                if cat in ROAD_CLASSES and not _is_polygon(feat):
                    half_width = _way_width_m(feat, cat) / (2 * mpt)
                    policies = tuple(side_policy(tags, side) for side in ("left", "right"))
                    tagged_roads |= policies != (("parallel", None),) * 2
                    road_lines.append(line)
                    self.roads.append((feat.osm_id, half_width, policies))
                    if cat == "road_service":
                        self.clearances.append(("service_access", feat.osm_id, line.buffer(
                            half_width + 1 / mpt, cap_style="square", join_style="bevel", quad_segs=2)))
                if tags.get("footway") == "crossing":
                    self.clearances.append(("crossing", feat.osm_id, line.buffer(
                        2 / mpt, cap_style="square", join_style="bevel", quad_segs=2)))
            else:
                point = Point(proj.to_px(*feat.geometry[0]))
                if not frame.covers(point):
                    continue
                if tags.get("highway") == "crossing":
                    crossings.append((feat.osm_id, point))
                if tags.get("barrier") in GATES or tags.get("amenity") == "parking_entrance":
                    width = 0.0
                    try:
                        width = float(tags.get("width", "").removesuffix(" m").replace(",", "."))
                    except ValueError:
                        pass
                    radius = max(3.0, min(8.0, width / 2 + 1)) if isfinite(width) else 3.0
                    self.clearances.append(("gate", feat.osm_id, point.buffer(radius / mpt, quad_segs=4)))
        self.road_tree = STRtree(road_lines)
        self.tagged_roads = tagged_roads
        self.side_tolerance = 2 / mpt
        for osm_id, point in crossings:
            near = self.road_tree.nearest(point)
            reach = self.roads[near][1] if near is not None else 3.5 / mpt
            self.clearances.append(("crossing", osm_id, point.buffer(reach + 2 / mpt, quad_segs=4)))
        self.clear_tree = STRtree([shape for _, _, shape in self.clearances])

    def lot_capacity(self, props):
        """Read the snapshot for older area files that lack the capacity tag."""
        by_kind = self.lot_tags.get(props.get("osm_id"), {})
        kind = props.get("osm_kind")
        if kind is None and len(by_kind) == 1:
            kind = next(iter(by_kind))
        return by_kind.get(kind, {}).get("capacity")

    def restriction(self, x, y, width, height):
        nearby = self.clear_tree.query(box(x, y, x + width, y + height), predicate="intersects")
        if nearby.size:
            return self.clearances[min(nearby)][0]
        return None

    def kerb_policy(self, rect):
        if not self.tagged_roads:
            return "parallel", None
        x, y, w, h = rect
        point = Point(x + (w - 1) / 2, y + (h - 1) / 2)
        nearest = min(self.road_tree.query_nearest(point), key=lambda i: self.roads[i][0])
        road = self.road_tree.geometries[nearest]
        _, half_width, policies = self.roads[nearest]
        if road.distance(point) > half_width + self.side_tolerance:
            return "parallel", None
        along = road.project(point)
        a, b = road.interpolate(along - 0.01 if along > 0.01 else 0), road.interpolate(along + 0.01)
        # North-up game tiles have y growing south, so left has negative cross product.
        cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)
        if abs(cross) < 1e-8:
            return "parallel", None
        return policies[0 if cross < 0 else 1]

    def report(self):
        return [{"kind": kind, "osm_id": osm_id, "geometry": mapping(shape)}
                for kind, osm_id, shape in self.clearances]
