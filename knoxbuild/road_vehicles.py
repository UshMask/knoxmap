"""Place roadside parking and traffic jams on the rendered carriageway."""
from __future__ import annotations

from collections import defaultdict, deque
from hashlib import blake2s
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from shapely import LineString, Point, STRtree

from generator.osm import ROAD_HIERARCHY_CODES

Rect = tuple[int, int, int, int]
STALL_W, STALL_H = 3, 5
JAM_WIDTH, JAM_LENGTH, JAM_SPACING = 3, 24, 90
JAM_AREA = 600             # tiles: share jam demand among nearby corridors
SERVICE = ROAD_HIERARCHY_CODES["service"]
SECONDARY = ROAD_HIERARCHY_CODES["secondary"]
PRIMARY = ROAD_HIERARCHY_CODES["primary"]
MOTORWAY = ROAD_HIERARCHY_CODES["motorway"]
KERB_RATES = {2: 0.18, 3: 0.55, 4: 0.38, 5: 0.28, 6: 0.16, 7: 0.08}


def zone_chunks(rect: Rect) -> set[tuple[int, int]]:
    x, y, w, h = rect
    return {(cx, cy) for cy in range(y // 8, (y + h - 1) // 8 + 1)
            for cx in range(x // 8, (x + w - 1) // 8 + 1)}


def _straight_edges(mask, length, step, minimum_width):
    """Run a vehicle-sized window along straight edges with room for a live lane."""
    for horizontal, surface in ((True, mask), (False, mask.T)):
        cross, along = surface.shape
        for start in range(2, along - length - 1, step):
            # Extra space at either end keeps bays away from corners and junctions.
            strip = surface[:, start - 2:start + length + 2]
            edges = np.flatnonzero(np.diff(np.pad(strip.all(axis=1), 1))).tolist()
            for lo, hi in zip(edges[::2], edges[1::2]):
                width = hi - lo
                if width < minimum_width:
                    continue
                choices = []
                for low_side, edge, side in ((True, lo - 1, lo), (False, hi, hi - STALL_W)):
                    if 0 <= edge < cross and not strip[edge].any():
                        rect = ((start, side, length, STALL_W) if horizontal else
                                (side, start, STALL_W, length))
                        choices.append((rect, low_side))
                if choices:
                    yield width, choices


def _center(rect: Rect):
    x, y, w, h = rect
    return x + w // 2, y + h // 2


def _score(seed: int, purpose: str, key: str | tuple) -> float:
    """A repeatable roll for one candidate, independent of its neighbours."""
    data = repr((seed, purpose, key)).encode("utf-8")
    return int.from_bytes(blake2s(data, digest_size=8).digest(), "big") / (1 << 64)


def roadside_bays(asphalt, hierarchy, density, seed, unavailable, taken, story_chunks,
                  story_centers, report, kerb_policy=None):
    if density <= 0:
        return
    # Old projects have only the ground bitmap. Limit inferred streets to
    # narrow strips, so a paved square is still handled as a possible car park.
    road = asphalt if hierarchy is None else asphalt & (hierarchy >= SERVICE)
    for width, choices in _straight_edges(road, STALL_H, 14, 2 * STALL_W):
        if hierarchy is None and width > 14:
            continue
        occupied_width = 0
        for rect, low_side in sorted(choices, key=lambda side: _score(seed, "kerb side", side[0])):
            orientation, reason = kerb_policy(rect) if kerb_policy else ("parallel", None)
            if reason:
                report["roadside_skipped"][reason] += 1
                continue
            x, y, w, h = rect
            depth = STALL_W
            if orientation == "perpendicular":
                width_delta = 0 if low_side else STALL_H - STALL_W
                rect = ((x, y - width_delta, STALL_W, STALL_H) if w > h else
                        (x - width_delta, y, STALL_H, STALL_W))
                x, y, w, h = rect
                depth = STALL_H
            if width - occupied_width - depth < STALL_W:
                continue
            xc, yc = _center(rect)
            tier = int(hierarchy[yc, xc]) if hierarchy is not None else None
            if tier is not None and (tier not in KERB_RATES
                                     or (hierarchy[y:y+h, x:x+w] == MOTORWAY).any()):
                continue
            chunks = zone_chunks(rect)
            if not chunks.isdisjoint(story_centers):
                report["roadside_reserved_for_stories"] += 1
                continue
            if not chunks.isdisjoint(taken) or unavailable(*rect):
                continue
            report["roadside_candidates"] += 1
            rate = 0.45 if tier is None else KERB_RATES[tier]
            if not chunks.isdisjoint(story_chunks):
                rate *= 0.3  # keep the rest of the road available for some parking
            if _score(seed, "kerb occupancy", rect) >= rate * density:
                continue
            taken.update(chunks)
            occupied_width += depth
            yield rect


def street_lookup(path: str | None, origin: tuple[int, int] = (0, 0)):
    """Index the paper map's named centre lines in project-local tiles."""
    if not path or not Path(path).is_file():
        return lambda x, y: ""
    ox, oy = origin
    lines, names, widths = [], [], []
    for street in ElementTree.parse(path).getroot().iterfind("street"):
        coords = [(float(p.attrib["x"]) - ox, float(p.attrib["y"]) - oy)
                  for p in street.iterfind("points/point")]
        if len(coords) >= 2:
            lines.append(LineString(coords))
            names.append(street.attrib["name"])
            widths.append(float(street.attrib["width"]))
    index = STRtree(lines)
    if not lines:
        return lambda x, y: ""

    def name_at(x, y):
        point = Point(x, y)
        nearest = min(index.query_nearest(point), key=lambda i: (-widths[i], names[i]))
        return names[nearest] if lines[nearest].distance(point) <= widths[nearest] / 2 + 2 else ""

    return name_at


def traffic_jams(asphalt, hierarchy, density, seed, unavailable, taken, street_name, in_town, report):
    if hierarchy is None or density <= 0:
        return
    by_area = defaultdict(lambda: defaultdict(list))
    expected = defaultdict(float)
    for _, sides in _straight_edges(asphalt & (hierarchy >= SECONDARY), JAM_LENGTH,
                                    JAM_LENGTH, 2 * JAM_WIDTH):
        sides = [rect for rect, _ in sides if zone_chunks(rect).isdisjoint(taken)
                 and not unavailable(*rect)]
        if not sides:
            continue
        rect = min(sides, key=lambda rect: _score(seed, "jam side", rect))
        x, y = _center(rect)
        name = street_name(x, y)
        # Maps without street names still spread jams by area and direction.
        key = name or (x // 300, y // 300, rect[2] > rect[3])
        area = x // JAM_AREA, y // JAM_AREA
        by_area[area][key].append((rect, name))
        report["jam_candidates"] += 1
        chance = 0.16 if hierarchy[y, x] >= PRIMARY else 0.07
        if in_town(x, y):
            chance *= 0.25
        expected[area] += min(1.0, chance * density)

    # Each area has a local quota. Take turns across its corridors so the
    # busiest road does not take every zone and edits do not reroll the city.
    centers = []
    street_centers = defaultdict(list)
    for area in sorted(by_area, key=lambda area: _score(seed, "jam area", area)):
        budget = int(expected[area] + _score(seed, "jam quota", area))
        if not budget:
            continue
        corridors = by_area[area]
        queue = deque(corridors[key] for key in sorted(
            corridors, key=lambda key: _score(seed, "jam corridor", (area, key))))
        for group in queue:
            group.sort(key=lambda entry: _score(seed, "jam location", entry[0]), reverse=True)
        while queue and budget:
            group = queue.popleft()
            while group:
                rect, name = group.pop()
                x, y = _center(rect)
                if any((x - a) ** 2 + (y - b) ** 2 < JAM_SPACING ** 2 for a, b in centers):
                    continue
                if name and any((x - a) ** 2 + (y - b) ** 2 < (2 * JAM_SPACING) ** 2
                                for a, b in street_centers[name]):
                    continue
                chunks = zone_chunks(rect)
                if not chunks.isdisjoint(taken):
                    continue
                taken.update(chunks)
                centers.append((x, y))
                if name:
                    street_centers[name].append((x, y))
                budget -= 1
                yield rect, name
                if group:
                    queue.append(group)
                break
