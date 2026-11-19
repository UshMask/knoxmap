"""Mapped crosswalks and inferred crossings and stop lines at junctions.

Build 42 ships kerbs and lane lines but no zebra crossing, so a crossing here
is laid out of the white lane-line tiles the road rules already place
(street_trafficlines_01). A line along the street on the near edge of every
tile across the carriageway, two tiles deep, reads from above as a row of
stripes - which is what a painted crossing looks like on a map.

Where one goes is decided from the streets themselves, not from the tarmac. A
junction is a point where three or more street ends meet, or where one street
ends on the length of another. Two street ends in line are one street carrying
on - OpenStreetMap splits a way at every change of lanes or name - and get
nothing; counting the ways near a point would paint a crossing into the middle
of every long road.

Where OSM has a crossing, it takes precedence over the guessed crossing on
that approach. Otherwise a crossing sits just outside the junction box with
a stop line a little further back on the approach side.
Diagonal streets are left alone: their stripes cannot be laid on the tile grid
without a zigzag, and the lane lines skip them for the same reason.
"""
from __future__ import annotations

import logging
import math
from collections import Counter

from .. import pz_colors as C
from .geometry import _feature_coords_px, _is_polygon
from .widths import _way_width_m

CROSSWALK_TILES = 2        # how deep along the street the stripes run
CROSSWALK_MIN_TILES = 4.5  # a lane narrower than this is not painted
JUNCTION_REACH = 1.5       # tiles: how near another street counts as meeting
STOP_LINE_GAP = 2          # tiles between the crossing and the stop line
MAX_JUNCTIONS = 4000       # a guard on pathological data
CELL = 4.0                 # tiles: the grid the segment index is kept on
MAPPED_REACH = 8.0         # tiles: look for a mapped crossing on the same approach

_log = logging.getLogger("knoxmap.renderer.crossings")

# Tiles the crossing is allowed to paint over: the lane lines run down the
# middle of the street a crossing crosses, and a line through the stripes is a
# line nobody paints in a real one.
_LINE_TILES = (C.LINE_WHITE_N, C.LINE_WHITE_W, C.LINE_YELLOW_N, C.LINE_YELLOW_W,
               C.EDGE_LINE_N, C.EDGE_LINE_S, C.EDGE_LINE_W, C.EDGE_LINE_E)
# Straight kerbs, the ones a crossing can cut through. A dropped kerb is just
# no kerb: the pavement under it shows through.
_KERB_TILES = (C.KERB_N, C.KERB_S, C.KERB_E, C.KERB_W)


def _at_ground(tags) -> bool:
    return (all(str(tags.get(key, "")).strip().lower() in ("", "no", "false", "0")
                for key in ("bridge", "tunnel", "covered"))
            and all(str(tags.get(key, "0")).strip() in ("", "0")
                    for key in ("layer", "level")))


def _street_ways(buckets, proj, categories=("road_major", "road_medium", "road_minor")
                 ) -> list[tuple[list[tuple[float, float]], float]]:
    """(ring in pixels, width in tiles) for every street worth a crossing."""
    ways = []
    for cat in categories:
        for feat in buckets.get(cat, []):
            if _is_polygon(feat):
                continue
            if not _at_ground(feat.tags):
                continue
            width = _way_width_m(feat, cat) / proj.meters_per_tile
            if width < (3 if cat == "road_service" else CROSSWALK_MIN_TILES):
                continue
            for ring in _feature_coords_px(feat, proj):
                if len(ring) >= 2:
                    ways.append(([(float(x), float(y)) for x, y in ring], width))
    return ways


def _segments(ways) -> list[list[tuple[float, float, float, float]]]:
    """(x0, y0, x1, y1) for each piece of each street, street by street."""
    return [[(a[0], a[1], b[0], b[1]) for a, b in zip(ring, ring[1:]) if a != b]
            for ring, _width in ways]


def _near_point(px, py, segment) -> tuple[float, float, float]:
    """Closest point on a segment to (px, py), and its squared distance."""
    x0, y0, x1, y1 = segment
    dx, dy = x1 - x0, y1 - y0
    length = dx * dx + dy * dy
    t = 0.0 if not length else max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / length))
    nx, ny = x0 + dx * t, y0 + dy * t
    return nx, ny, (px - nx) ** 2 + (py - ny) ** 2


class _SegmentIndex:
    """The streets, bucketed by where they run, so a point asks only its own
    neighbourhood - a shapely tree over a county's worth of ways spent most of
    its time building a geometry per query."""

    def __init__(self, segments):
        self.segments = segments
        self.cells: dict[tuple[int, int], list[int]] = {}
        for index, (x0, y0, x1, y1) in enumerate(segments):
            steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / (CELL / 2)) + 1)
            last = None
            for step in range(steps + 1):
                t = step / steps
                key = (int(math.floor((x0 + (x1 - x0) * t) / CELL)),
                       int(math.floor((y0 + (y1 - y0) * t) / CELL)))
                if key != last:
                    self.cells.setdefault(key, []).append(index)
                    last = key

    def near(self, x: float, y: float, radius: int = 1):
        """The segments in the cells around a point, nearest first is not
        needed: the callers measure the distance themselves."""
        cx, cy = int(math.floor(x / CELL)), int(math.floor(y / CELL))
        seen = set()
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for index in self.cells.get((cx + dx, cy + dy), ()):
                    if index in seen:
                        continue
                    seen.add(index)
                    yield index


def _unique(points, cell: float = 3.0) -> list[tuple[float, float]]:
    seen, out = set(), []
    for x, y in points:
        key = (int(math.floor(x / cell)), int(math.floor(y / cell)))
        if key in seen:
            continue
        seen.add(key)
        out.append((x, y))
    return out


def _tile(value: float) -> int:
    """The tile a coordinate falls in, rounding a half up rather than to even.

    round() takes 26.5 to 26 and 13.5 to 14, so a crossing whose offset landed
    on a half tile put both of its stripe rows on the same square and came out
    one tile deep.
    """
    return int(math.floor(value + 0.5))


def _stop_tile(junction: float, offset: float, direction: int, mapped: float | None) -> int:
    stop = _tile(junction + direction * (offset + CROSSWALK_TILES + STOP_LINE_GAP))
    if mapped is None:
        return stop
    start = _tile(mapped - (CROSSWALK_TILES - 1) / 2)
    if direction > 0:
        return max(stop, start + CROSSWALK_TILES + STOP_LINE_GAP)
    return min(stop, start - STOP_LINE_GAP - 1)


def junction_points(ways) -> list[tuple[float, float]]:
    """Tile points where three or more street arms meet."""
    flat = [segment for own in _segments(ways) for segment in own]
    if not flat:
        return []
    index = _SegmentIndex(flat)
    # Two street ends meeting at a shared node are one street carrying on;
    # three or more is a junction.
    ends: dict[tuple[int, int], int] = {}
    for x0, y0, x1, y1 in flat:
        for key in ((round(x0), round(y0)), (round(x1), round(y1))):
            ends[key] = ends.get(key, 0) + 1
    points = [(float(key[0]), float(key[1])) for key, count in ends.items()
              if count >= 3]
    # A side street ending on the length of another has only its own end at
    # that point, so the count above never sees it.
    reach2 = JUNCTION_REACH ** 2
    for position, (x0, y0, x1, y1) in enumerate(flat):
        for x, y in ((x0, y0), (x1, y1)):
            joined = False
            for other in index.near(x, y):
                if other == position:
                    continue
                ox0, oy0, ox1, oy1 = flat[other]
                _nx, _ny, distance2 = _near_point(x, y, flat[other])
                if distance2 > reach2:
                    continue
                if min(math.hypot(x - ox0, y - oy0),
                       math.hypot(x - ox1, y - oy1)) <= JUNCTION_REACH:
                    continue        # a street carrying on, not joining
                joined = True
                break
            if joined:
                points.append((x, y))
    return _unique(points)[:MAX_JUNCTIONS]


def _mapped_crossings(buckets, proj, segments, widths):
    """Match mapped footway segments to street centre lines they cross."""
    features = [f for cat in ("paved_path", "dirt_path") for f in buckets.get(cat, ())
                if f.kind == "way" and f.tags.get("footway") == "crossing"]
    if not features:
        return {}, []

    from shapely.geometry import LineString
    from shapely.strtree import STRtree

    # A mapper can mark a crossing over a service road. Guessed crossings
    # still leave those roads alone.
    service = _street_ways(buckets, proj, ("road_service",))
    segments = list(segments)
    for parts, (_, width) in zip(_segments(service), service):
        segments.extend(parts)
        widths.extend([width] * len(parts))
    lines = [LineString(((x0, y0), (x1, y1))) for x0, y0, x1, y1 in segments]
    streets = STRtree(lines)
    matched, candidates = {}, []
    for feat in features:
        marked = (feat.tags.get("crossing") != "unmarked"
                  and feat.tags.get("crossing:markings") != "no")
        status = "unsupported_level" if not _at_ground(feat.tags) else "unmatched_road"
        if status == "unmatched_road":
            points = _feature_coords_px(feat, proj)[0]
            for (x0, y0), (x1, y1) in zip(points, points[1:]):
                dx, dy = x1 - x0, y1 - y0
                if not (dx or dy):
                    continue
                path = LineString(((x0, y0), (x1, y1)))
                for i in sorted(streets.query(path, predicate="intersects")):
                    rx, ry, ex, ey = segments[i]
                    ux, uy = ex - rx, ey - ry
                    # A path that follows a street, even if it touches its
                    # centre line at an end, is not a crossing of that street.
                    if abs(dx * uy - dy * ux) < 0.65 * path.length * lines[i].length:
                        continue
                    if min(abs(ux), abs(uy)) > 0.2 * max(abs(ux), abs(uy)):
                        if status == "unmatched_road":
                            status = "unsupported_direction"
                        continue
                    point = path.intersection(lines[i])
                    if point.geom_type != "Point" or point.is_empty:
                        continue
                    along_x = abs(ux) >= abs(uy)
                    key = (_tile(point.x), _tile(point.y), along_x)
                    status = "duplicate"
                    if key not in matched or (matched[key][-1] and not marked):
                        matched[key] = (feat.osm_id, point.x, point.y, widths[i], marked)
        candidates.append({"osm_id": feat.osm_id, "status": status, "marked": marked})
    matched_ids = {item[0] for item in matched.values()}
    return matched, [row for row in candidates if row["osm_id"] not in matched_ids]


def paint_crossings(veg, landscape, buckets, proj) -> dict:
    """Paint mapped crossings, then fill junction approaches with no mapped one."""
    import numpy as np

    counts = {"crosswalk": 0, "stop_line": 0, "kerb_cut": 0}
    ways = _street_ways(buckets, proj)
    flat, owner = [], []
    for index, own in enumerate(_segments(ways)):
        for segment in own:
            flat.append(segment)
            owner.append(index)
    mapped, report = _mapped_crossings(buckets, proj, flat, [ways[i][1] for i in owner])
    points = junction_points(ways)
    if not points and not mapped:
        if report:
            counts["mapped"] = report
            _log.info("OSM crossings: %s", dict(Counter(r["status"] for r in report)))
        return counts
    index = _SegmentIndex(flat)

    ground = np.asarray(landscape.convert("RGB"))
    height, width = ground.shape[:2]
    asphalt = np.zeros((height, width), dtype=bool)
    for colour in (C.MEDIUM_ASPHALT, C.DARK_ASPHALT, C.DARKEST_ASPHALT,
                   C.DARK_POTHOLE, C.LIGHT_POTHOLE):
        asphalt |= np.all(ground == colour, axis=2)
    veg_px = veg.load()

    def road(x, y) -> bool:
        return 0 <= x < width and 0 <= y < height and bool(asphalt[y, x])

    def put(x, y, colour, along_x) -> bool:
        """One line tile, where the street is under it and nothing else is."""
        if not (0 <= x < width and 0 <= y < height):
            return False
        existing = veg_px[x, y]
        if existing not in (C.VEG_NOTHING, *_LINE_TILES):
            return False
        if not road(x, y):
            return False
        # The line sits on the edge between two tiles, so the square behind it
        # has to be street too or the paint hangs over the kerb.
        behind = road(x, y - 1) if along_x else road(x - 1, y)
        if not behind:
            return False
        if existing == colour:
            return False        # this stripe is already painted
        veg_px[x, y] = colour
        return True

    def cut(x, y, dx, dy) -> bool:
        """Drop the kerb where a crossing runs out of road.

        Walk out from the last stripe square until a straight kerb turns up, or
        something that is not street-side emptiness does. Taking the kerb tile
        away leaves the pavement showing, which from above is what a dropped
        kerb at a crossing looks like.
        """
        for k in range(0, 4):
            cx, cy = x + dx * k, y + dy * k
            if not (0 <= cx < width and 0 <= cy < height):
                return False
            existing = veg_px[cx, cy]
            if existing in _KERB_TILES:
                veg_px[cx, cy] = C.VEG_NOTHING
                return True
            if existing != C.VEG_NOTHING and existing not in _LINE_TILES:
                return False
        return False

    anchors = []
    for (_, _, along_x), (osm_id, x, y, road_width, marked) in mapped.items():
        status = "no_asphalt"
        if road(_tile(x), _tile(y)):
            anchors.append((x, y, along_x))
            axis, across = (x, y) if along_x else (y, x)
            start = _tile(axis - (CROSSWALK_TILES - 1) / 2)
            low = math.floor(across - road_width / 2)
            high = math.ceil(across + road_width / 2)
            colour = C.LINE_WHITE_N if along_x else C.LINE_WHITE_W
            laid = False
            for tile in range(start, start + CROSSWALK_TILES):
                if marked:
                    for side in range(low, high + 1):
                        px, py = (tile, side) if along_x else (side, tile)
                        laid |= put(px, py, colour, along_x)
                for end, outward in ((low, -1), (high, 1)):
                    px, py = (tile, end) if along_x else (end, tile)
                    dx, dy = (0, outward) if along_x else (outward, 0)
                    counts["kerb_cut"] += cut(px, py, dx, dy)
            counts["crosswalk"] += laid
            status = "painted" if laid else "no_stripe_tiles" if marked else "unmarked"
        report.append({"osm_id": osm_id, "status": status, "marked": marked,
                       "tile": [_tile(x), _tile(y)]})

    if report:
        counts["mapped"] = report
        _log.info("OSM crossings: %s", dict(Counter(r["status"] for r in report)))
    mapped_index = _SegmentIndex([(x, y, x, y) for x, y, _along_x in anchors])

    def mapped_on_approach(x, y, nx, ny, along_x):
        junction, guessed = (nx, x) if along_x else (ny, y)
        outermost = None
        for i in mapped_index.near(x, y, math.ceil(MAPPED_REACH / CELL)):
            mx, my, axis = anchors[i]
            if axis != along_x:
                continue
            candidate = mx if along_x else my
            offset = (guessed - junction) * (candidate - junction)
            lateral = abs(my - y) if along_x else abs(mx - x)
            if offset >= 0 and lateral <= 2.5 and (x - mx) ** 2 + (y - my) ** 2 <= MAPPED_REACH ** 2:
                if outermost is None or abs(candidate - junction) > abs(outermost - junction):
                    outermost = candidate
        return outermost

    for jx, jy in points:
        near = []
        seen = set()
        for position in index.near(jx, jy):
            way = owner[position]
            if way in seen:
                continue
            nx, ny, distance2 = _near_point(jx, jy, flat[position])
            if distance2 > 2.5 ** 2:
                continue
            seen.add(way)
            near.append((nx, ny, ways[way][1], flat[position]))
        if len(near) < 2:
            continue
        widest = max(road_width for _x, _y, road_width, _seg in near)
        for nx, ny, road_width, segment in near:
            # The street's own direction where it passes this junction.
            x0, y0, x1, y1 = segment
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if not length:
                continue
            ux, uy = dx / length, dy / length
            # The junction box is as deep as half the widest street through it.
            offset = widest / 2 + 1
            across = road_width / 2
            if abs(uy) <= 0.2 * abs(ux):            # a street running east-west
                step = 1 if ux > 0 else -1
                for side in (1, -1):
                    direction = step * side
                    first = _tile(nx + direction * offset)
                    mapped_axis = mapped_on_approach(first, ny, nx, ny, True) if anchors else None
                    if mapped_axis is None:
                        laid = False
                        y_lo = int(math.floor(ny - across))
                        y_hi = int(math.ceil(ny + across))
                        for k in range(CROSSWALK_TILES):
                            x = first + direction * k
                            for y in range(y_lo, y_hi + 1):
                                laid |= put(x, y, C.LINE_WHITE_N, True)
                            counts["kerb_cut"] += cut(x, y_lo, 0, -1)
                            counts["kerb_cut"] += cut(x, y_hi, 0, 1)
                        counts["crosswalk"] += laid
                    stop = _stop_tile(nx, offset, direction, mapped_axis)
                    # Right-hand traffic: approaching from the west the near
                    # lanes are the southern half, from the east the northern.
                    low = ny - across if step * side > 0 else ny
                    high = ny if step * side > 0 else ny + across
                    laid = False
                    for y in range(int(math.floor(low)), int(math.ceil(high)) + 1):
                        laid |= put(stop, y, C.LINE_WHITE_W, False)
                    counts["stop_line"] += laid
            elif abs(ux) <= 0.2 * abs(uy):          # a street running north-south
                step = 1 if uy > 0 else -1
                for side in (1, -1):
                    direction = step * side
                    first = _tile(ny + direction * offset)
                    mapped_axis = mapped_on_approach(nx, first, nx, ny, False) if anchors else None
                    if mapped_axis is None:
                        laid = False
                        x_lo = int(math.floor(nx - across))
                        x_hi = int(math.ceil(nx + across))
                        for k in range(CROSSWALK_TILES):
                            y = first + direction * k
                            for x in range(x_lo, x_hi + 1):
                                laid |= put(x, y, C.LINE_WHITE_W, False)
                            counts["kerb_cut"] += cut(x_lo, y, -1, 0)
                            counts["kerb_cut"] += cut(x_hi, y, 1, 0)
                        counts["crosswalk"] += laid
                    stop = _stop_tile(ny, offset, direction, mapped_axis)
                    # Approaching from the north the near lanes are the
                    # western half, from the south the eastern.
                    low = nx - across if step * side > 0 else nx
                    high = nx if step * side > 0 else nx + across
                    laid = False
                    for x in range(int(math.floor(low)), int(math.ceil(high)) + 1):
                        laid |= put(x, stop, C.LINE_WHITE_N, True)
                    counts["stop_line"] += laid
    return counts
