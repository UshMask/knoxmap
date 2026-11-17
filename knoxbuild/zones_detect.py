"""Zone detection (ParkingStall and TownZone) for Project KnoxMap."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .road_vehicles import STALL_H, STALL_W, roadside_bays, street_lookup, traffic_jams, zone_chunks
from .settings import Settings

if TYPE_CHECKING:
    from .world import Zone

CAR_PARK_LANE = 17            # two rows of 5, and a 7-tile lane


def car_park_stalls(bounds: tuple[int, int, int, int], w: int, h: int):
    """Where a stall could go in a car park with these bounds, as (x, y, sw, sh),
    in the order they are tried - only the ones that lie on the map.

    This used to walk the whole bounding box of the car park and throw away what
    was off the map afterwards. A car park a few hundred metres past the edge
    costs nothing; a polygon of the whole country mapped as one (it happens) was
    hundreds of millions of steps. Starting each run at the first position that
    can be on the map visits the same stalls in the same order."""
    x0, y0, x1, y1 = bounds
    across = (x1 - x0) >= (y1 - y0)          # rows run along the long side
    sw, sh = (STALL_W, STALL_H) if across else (STALL_H, STALL_W)
    # b runs along the rows, a across them; a + row is the offset of a stall's row.
    b_origin, b_len, b_map = (x0, x1 - x0, w) if across else (y0, y1 - y0, h)
    a_origin, a_len, a_map = (y0, y1 - y0, h) if across else (x0, x1 - x0, w)
    lane = CAR_PARK_LANE
    a_lo = max(0, -a_origin - STALL_H)
    a_start = -(-a_lo // lane) * lane
    a_stop = min(a_len, a_map - STALL_H - a_origin + 1)
    b_lo = max(0, -b_origin)
    b_start = -(-b_lo // STALL_W) * STALL_W
    b_stop = min(b_len, b_map - STALL_W - b_origin + 1)
    for a in range(a_start, a_stop, lane):
        for row in (0, STALL_H):
            for b in range(b_start, b_stop, STALL_W):
                x, y = (x0 + b, y0 + a + row) if across else (x0 + a + row, y0 + b)
                if 0 <= x and x + sw <= w and 0 <= y and y + sh <= h:
                    yield x, y, sw, sh


def _detect_zones(landscape_path: str, placements, rng_seed: int = 7,
                  settings: Settings | None = None, areas=None, drives=(),
                  keep_clear=(), road_hierarchy: np.ndarray | None = None,
                  veg_path: str | None = None, streets_path: str | None = None,
                  vehicle_report: dict | None = None):
    """Parking stalls along the roads, town zones over built-up ground.

    ParkingStall zones support parked cars and pileups, while narrow Nav
    zones support random road stories. TownZone marks built-up ground.

    The optional report describes the authored zones in local map tiles.
    """
    import random

    from PIL import Image
    from shapely import STRtree, box

    from generator import pz_colors as C
    from generator.osm import ROAD_HIERARCHY_CODES

    from .world import Zone

    ASPHALT = {C.MEDIUM_ASPHALT, C.DARK_ASPHALT, C.DARKEST_ASPHALT,
               C.DARK_POTHOLE, C.LIGHT_POTHOLE}

    settings = settings or Settings()
    rng = random.Random(rng_seed)
    zones: list[Zone] = []
    report = vehicle_report if vehicle_report is not None else {}
    counts = {key: 0 for key in ("mapped_parking", "yards", "unmapped_parking", "roadside",
                                "traffic_jams", "road_stories")}
    report.update(counts=counts, roadside_candidates=0, roadside_reserved_for_stories=0,
                  jam_candidates=0, roadside=[], jams=[])
    lot_density = settings.parking_density
    yard_density = getattr(settings, "vehicle_yard_density", None)
    kerb_density = getattr(settings, "vehicle_kerb_density", None)
    jam_density = getattr(settings, "vehicle_jam_density", None)
    yard_density = lot_density if yard_density is None else yard_density
    kerb_density = lot_density if kerb_density is None else kerb_density
    jam_density = lot_density if jam_density is None else jam_density
    drives = tuple(drives)
    lots = [shape for shape, props in areas._items if props.get("category") == "parking"] if areas else []
    reserved = STRtree([*lots, *(box(x, y, x + sw, y + sh) for x, y, sw, sh in drives)])
    clear_boxes = [(x, y, x + sw, y + sh) for x, y, sw, sh in keep_clear]
    buildings = [box(p.tile_x, p.tile_y, p.tile_x + p.width, p.tile_y + p.height)
                 for p in placements]
    built = STRtree(buildings)

    def is_reserved(x, y, sw, sh):
        return len(reserved) > 0 and reserved.query(
            box(x, y, x + sw, y + sh), predicate="intersects").size > 0

    def unavailable(x, y, sw, sh):
        return (_in_boxes(x, y, sw, sh, clear_boxes) or is_reserved(x, y, sw, sh)
                or built.query(box(x, y, x + sw, y + sh), predicate="intersects").size > 0)

    with Image.open(landscape_path) as source:
        land_arr = np.asarray(source.convert("RGB"))
    h, w = land_arr.shape[:2]
    land_key = _pack_rgb(land_arr)
    del land_arr
    asphalt_mask = np.isin(land_key, [_code(c) for c in ASPHALT])
    hard_mask = asphalt_mask | np.isin(land_key, [_code(C.PALE_CONCRETE), _code(C.PAVING)])
    if road_hierarchy is not None:
        road_hierarchy = road_hierarchy[:h, :w]

    def is_asphalt(x: int, y: int) -> bool:
        """Carriageway or car park, as opposed to the pavement beside it."""
        if not (0 <= x < w and 0 <= y < h):
            return False
        # The three street shades plus the two pothole shades that weather
        # them. Kerbs and paving slabs are deliberately out: parking a car on
        # the pavement is where stalls ended up once roads grew kerbs, because
        # any grey pixel in a wide range counted as road.
        return bool(asphalt_mask[y, x])

    taken: set[tuple[int, int]] = set()

    # A car on most drives, as in Knox County.
    for x, y, sw, sh in drives:
        if rng.random() < min(1.0, 0.6 * yard_density):
            taken.update(zone_chunks((x, y, sw, sh)))
            zones.append(Zone("ParkingStall", x, y, sw, sh))
            counts["yards"] += 1

    # Car parks OpenStreetMap maps as such: rows of stalls across the whole of
    # each, two rows back to back and then a lane, the way a real one is laid
    # out. Sampling the tarmac every 14 tiles found at most a stall or two in
    # anything smaller than a supermarket's, so most car parks had no cars.
    painted_stalls: list[tuple[int, int, int, int, bool, bool]] = []
    island_boxes: list[tuple[int, int, int, int, bool]] = []
    extra_lamps: list[tuple[int, int, tuple[int, int, int]]] = []
    if lots:
        from shapely.prepared import prep
        for lot in lots:
            lot_prep = prep(lot)
            bounds = tuple(int(v) for v in lot.bounds)
            across = (bounds[2] - bounds[0]) >= (bounds[3] - bounds[1])
            a_origin = bounds[1] if across else bounds[0]
            b_origin = bounds[0] if across else bounds[1]
            lot_stalls: list[tuple[int, int, int, int, bool, int, int]] = []
            for x, y, sw, sh in car_park_stalls(bounds, w, h):
                stall = box(x, y, x + sw, y + sh)
                if not lot_prep.contains(stall):
                    continue
                if not (hard_mask[y, x] and hard_mask[y, x + sw - 1]
                        and hard_mask[y + sh - 1, x] and hard_mask[y + sh - 1, x + sw - 1]):
                    continue
                if built.query(stall, predicate="intersects").size:
                    continue
                a_coord = y if across else x
                b_coord = x if across else y
                is_first_row = ((a_coord - a_origin) % CAR_PARK_LANE) == 0
                band = (a_coord - a_origin) // CAR_PARK_LANE
                slot = (b_coord - b_origin) // STALL_W
                lot_stalls.append((x, y, sw, sh, is_first_row, band, slot))

            if not lot_stalls:
                continue

            # Reserve a stall-pair column every 6 stalls along long rows as a
            # curbed divider island with a light pole and greenery.
            band_slots: dict[int, set[int]] = {}
            for _x, _y, _sw, _sh, _first, band, slot in lot_stalls:
                band_slots.setdefault(band, set()).add(slot)
            island_keys: set[tuple[int, int]] = set()
            for band, slots in band_slots.items():
                ordered = sorted(slots)
                run: list[int] = []
                for s in ordered:
                    if not run or s == run[-1] + 1:
                        run.append(s)
                    else:
                        if len(run) >= 6:
                            for idx in range(5, len(run) - 1, 6):
                                island_keys.add((band, run[idx]))
                        run = [s]
                if len(run) >= 6:
                    for idx in range(5, len(run) - 1, 6):
                        island_keys.add((band, run[idx]))

            grouped_islands: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
            for x, y, sw, sh, is_first_row, band, slot in lot_stalls:
                if (band, slot) in island_keys:
                    grouped_islands.setdefault((band, slot), []).append((x, y, sw, sh))
                    continue
                painted_stalls.append((x, y, sw, sh, is_first_row, across))
                if rng.random() >= min(1.0, 0.55 * lot_density):
                    continue
                taken.update(zone_chunks((x, y, sw, sh)))
                zones.append(Zone("ParkingStall", x, y, sw, sh))
                counts["mapped_parking"] += 1

            for parts in grouped_islands.values():
                ix0 = min(p[0] for p in parts)
                iy0 = min(p[1] for p in parts)
                ix1 = max(p[0] + p[2] for p in parts)
                iy1 = max(p[1] + p[3] for p in parts)
                island_boxes.append((ix0, iy0, ix1, iy1, across))

            # Medium lots without divider islands still get edge light poles.
            if not grouped_islands and len(lot_stalls) >= 4:
                x0, y0, sw0, sh0, _, _, _ = lot_stalls[0]
                lx, ly = (x0, max(0, y0 - 1)) if across else (max(0, x0 - 1), y0)
                if 0 <= lx < w and 0 <= ly < h and hard_mask[ly, lx]:
                    lamp = C.LAMP_S if across else C.LAMP_E
                    extra_lamps.append((lx, ly, lamp))

    # One town zone per cell that holds buildings, covering their extent with a
    # margin so gardens and the street frontage count as town too.
    by_cell: dict[tuple[int, int], list] = {}
    for p in placements:
        by_cell.setdefault((p.cell_x, p.cell_y), []).append(p)
    for (cx, cy), group in by_cell.items():
        x0 = max(0, min(p.tile_x for p in group) - 8)
        y0 = max(0, min(p.tile_y for p in group) - 8)
        x1 = min(w, max(p.tile_x + p.width for p in group) + 8)
        y1 = min(h, max(p.tile_y + p.height for p in group) + 8)
        # Clamp into the owning cell so the rectangle stays where it is written.
        x0 = max(x0, cx * 300)
        y0 = max(y0, cy * 300)
        x1 = min(x1, (cx + 1) * 300)
        y1 = min(y1, (cy + 1) * 300)
        if x1 - x0 > 4 and y1 - y0 > 4:
            zones.append(Zone("TownZone", x0, y0, x1 - x0, y1 - y0))

    veg_arr = None
    if veg_path:
        with Image.open(veg_path) as v_src:
            veg_arr = np.asarray(v_src.convert("RGB"))
    town_boxes = [(z.tile_x, z.tile_y, z.tile_x + z.width, z.tile_y + z.height)
                  for z in zones if z.kind == "TownZone"]

    def in_town(x, y):
        return _in_boxes(x, y, 1, 1, town_boxes)

    street_name = street_lookup(streets_path)
    jams = list(traffic_jams(asphalt_mask, road_hierarchy, jam_density,
                             random.Random(rng_seed + 1), unavailable, taken, street_name, in_town,
                             report))
    for (x, y, sw, sh), name in jams:
        zones.append(Zone("ParkingStall", x, y, sw, sh,
                          "trafficjame" if sw > sh else "trafficjamn"))
        report["jams"].append({"street": name, "x": x, "y": y, "width": sw, "height": sh})
    counts["traffic_jams"] = len(jams)
    story_chunks = set()
    story_centers = set()
    if road_hierarchy is not None:
        road_mask = asphalt_mask & (road_hierarchy >= ROAD_HIERARCHY_CODES["residential"])
        for (x, y, sw, sh), _ in jams:
            road_mask[y:y + sh, x:x + sw] = False
        for zone in _road_story_zones(road_mask, clear_boxes, is_reserved):
            zones.append(zone)
            counts["road_stories"] += 1
            story_chunks.update(zone_chunks((zone.tile_x, zone.tile_y, zone.width, zone.height)))
            story_centers.add(((zone.tile_x + zone.width // 2) // 8,
                               (zone.tile_y + zone.height // 2) // 8))

    for rect in roadside_bays(asphalt_mask, road_hierarchy, kerb_density,
                              random.Random(rng_seed + 2), unavailable, taken, story_chunks,
                              story_centers, report):
        zones.append(Zone("ParkingStall", *rect))
        x, y, sw, sh = rect
        report["roadside"].append({"street": street_name(x + sw // 2, y + sh // 2),
                                   "x": x, "y": y, "width": sw, "height": sh})
        counts["roadside"] += 1

    # Broad asphalt with no road class can still be an unmapped parking area.
    if lot_density > 0:
        for y in range(4, h - STALL_H - 4, 14):
            for x in range(4, w - STALL_W - 4, 14):
                if not is_asphalt(x, y):
                    continue
                if road_hierarchy is not None and road_hierarchy[y:y + STALL_H, x:x + STALL_W].any():
                    continue
                if not all(is_asphalt(a, b) for a, b in ((x - 8, y), (x + 8, y),
                                                        (x, y - 8), (x, y + 8))):
                    continue
                rect = (x, y, STALL_W, STALL_H)
                chunks = zone_chunks(rect)
                if (not asphalt_mask[y:y + STALL_H, x:x + STALL_W].all()
                        or not chunks.isdisjoint(taken) or not chunks.isdisjoint(story_chunks)
                        or unavailable(*rect)
                        or rng.random() >= 0.75 * lot_density):
                    continue
                taken.update(chunks)
                zones.append(Zone("ParkingStall", *rect))
                counts["unmapped_parking"] += 1
    zones.extend(_forage_and_nav_zones(None, veg_arr, w, h, town_boxes,
                                       land_key=land_key))

    if veg_path and veg_arr is not None and (painted_stalls or island_boxes or extra_lamps):
        veg_out = _paint_parking_lots(
            veg_arr, painted_stalls, island_boxes, extra_lamps,
            lamp_density=(getattr(settings, "street_lamp_density", 1.0)
                          if getattr(settings, "street_lamp_mode", "context") == "context" else 0.0),
        )
        if veg_out is not None:
            import knoxfiles
            knoxfiles.save(Image.fromarray(veg_out), veg_path, format="BMP")

    return zones


def _paint_parking_lots(
    veg_arr: np.ndarray,
    stalls: list[tuple[int, int, int, int, bool, bool]],
    islands: list[tuple[int, int, int, int, bool]] = (),
    extra_lamps: list[tuple[int, int, tuple[int, int, int]]] = (),
    *, lamp_density: float = 1.0,
) -> np.ndarray | None:
    """Paint bay lines, wheel stops, divider islands, and light poles onto _veg.bmp."""
    from generator import pz_colors as C
    from generator.rendering.furniture import sample_fixture
    from knoxbuild.yards import _rule_ready

    paint_blocks = _rule_ready("Parking block")
    h, w = veg_arr.shape[:2]
    out = veg_arr.copy()
    empty = np.all(out == C.VEG_NOTHING, axis=2)
    stall_origins = {(x, y) for x, y, _sw, _sh, _first, _across in stalls}
    changed = False

    def put(tx: int, ty: int, colour: tuple[int, int, int]) -> None:
        nonlocal changed
        if 0 <= tx < w and 0 <= ty < h and empty[ty, tx]:
            out[ty, tx] = colour
            empty[ty, tx] = False
            changed = True

    for x, y, sw, sh, is_first_row, across in stalls:
        if across:
            if paint_blocks:
                head_y = y + sh - 1 if is_first_row else y
                for dx, colour in enumerate(C.PARKING_BLOCK_X):
                    put(x + dx, head_y, colour)
            dy_range = range(0, sh - 1) if is_first_row else range(1, sh)
            right_end = (x + sw, y) not in stall_origins
            for dy in dy_range:
                put(x, y + dy, C.EDGE_LINE_W)
                if right_end:
                    put(x + sw - 1, y + dy, C.EDGE_LINE_E)
        else:
            if paint_blocks:
                head_x = x + sw - 1 if is_first_row else x
                for dy, colour in enumerate(C.PARKING_BLOCK_Y):
                    put(head_x, y + dy, colour)
            dx_range = range(0, sw - 1) if is_first_row else range(1, sw)
            bottom_end = (x, y + sh) not in stall_origins
            for dx in dx_range:
                put(x + dx, y, C.EDGE_LINE_N)
                if bottom_end:
                    put(x + dx, y + sh - 1, C.EDGE_LINE_S)

    for ix0, iy0, ix1, iy1, across in islands:
        if ix1 - ix0 < 3 or iy1 - iy0 < 3:
            continue
        # Perimeter curbstones around the divider island.
        put(ix0, iy0, C.KERB_NW)
        put(ix1 - 1, iy0, C.KERB_NE)
        put(ix0, iy1 - 1, C.KERB_SW)
        put(ix1 - 1, iy1 - 1, C.KERB_SE)
        for tx in range(ix0 + 1, ix1 - 1):
            put(tx, iy0, C.KERB_N)
            put(tx, iy1 - 1, C.KERB_S)
        for ty in range(iy0 + 1, iy1 - 1):
            put(ix0, ty, C.KERB_W)
            put(ix1 - 1, ty, C.KERB_E)

        # Interior: light pole in the middle, greenery/tree along the spine.
        cx, cy = (ix0 + ix1) // 2, (iy0 + iy1) // 2
        if sample_fixture(lamp_density, cx + cy * w):
            put(cx, cy, C.LAMP_S if across else C.LAMP_E)
        if across:
            for ty in range(iy0 + 1, iy1 - 1):
                if ty == cy:
                    continue
                put(cx, ty, C.TREE_SMALL if ty in (iy0 + 2, iy1 - 3) else C.BUSHES)
        else:
            for tx in range(ix0 + 1, ix1 - 1):
                if tx == cx:
                    continue
                put(tx, cy, C.TREE_SMALL if tx in (ix0 + 2, ix1 - 3) else C.BUSHES)

    for lx, ly, lamp in extra_lamps:
        if sample_fixture(lamp_density, lx + ly * w):
            put(lx, ly, lamp)

    return out if changed else None


def _in_boxes(x: int, y: int, sw: int, sh: int,
              boxes: list[tuple[int, int, int, int]]) -> bool:
    return any(x < x1 and x0 < x + sw and y < y1 and y0 < y + sh
               for x0, y0, x1, y1 in boxes)


FORAGE_BLOCK = 20


def _pack_rgb(arr: np.ndarray) -> np.ndarray:
    return ((arr[:, :, 0].astype(np.uint32) << 16)
            | (arr[:, :, 1].astype(np.uint32) << 8)
            | arr[:, :, 2].astype(np.uint32))


def _code(c: tuple[int, int, int]) -> int:
    return (c[0] << 16) | (c[1] << 8) | c[2]


def _road_story_zones(road_mask: np.ndarray, clear_boxes, is_reserved):
    """Find straight carriageways wide enough for the game's random road stories."""
    from .world import CELL_SIZE, Zone

    # B42 accepts rectangular Nav zones longer than 30 tiles and narrower than 15.
    # The game's stories also need room across the road for at least one vehicle.
    length, min_width, max_width = 40, 6, 12
    height, width = road_mask.shape
    for y in range(0, height, CELL_SIZE):
        for x in range(0, width, CELL_SIZE):
            cell = road_mask[y:y + CELL_SIZE, x:x + CELL_SIZE]
            used = np.zeros_like(cell)
            for horizontal in (True, False):
                road = cell if horizontal else cell.T
                occupied = used if horizontal else used.T
                for start in range(0, road.shape[1] - length + 1, length // 2):
                    strip = slice(start, start + length)
                    cross_section = np.all(road[:, strip], axis=1)
                    edges = np.flatnonzero(np.diff(np.pad(cross_section, 1))).tolist()
                    for lo, hi in zip(edges[::2], edges[1::2]):
                        span = hi - lo
                        if span < min_width:
                            continue
                        breadth = min(span, max_width)
                        side = lo + (span - breadth) // 2
                        footprint = occupied[side:side + breadth, strip]
                        if footprint.any():
                            continue
                        zx, zy, zw, zh = (x + start, y + side, length, breadth) if horizontal else (
                            x + side, y + start, breadth, length)
                        if _in_boxes(zx, zy, zw, zh, clear_boxes) or is_reserved(zx, zy, zw, zh):
                            continue
                        footprint[:] = True
                        yield Zone("Nav", zx, zy, zw, zh)


def _forage_and_nav_zones(land_arr: np.ndarray | None, veg_arr: np.ndarray | None,
                          w: int, h: int,
                          town_boxes: list[tuple[int, int, int, int]],
                          land_key: np.ndarray | None = None):
    """Emit per-cell Forest, DeepForest, Vegitation, Farm, and Nav zones."""
    from generator import pz_colors as C
    from .world import CELL_SIZE, Zone

    if land_key is None:
        land_key = _pack_rgb(land_arr)
    veg_key = _pack_rgb(veg_arr) if veg_arr is not None else None

    nav_mask = np.isin(land_key, [_code(C.MEDIUM_ASPHALT), _code(C.DARKEST_ASPHALT),
                                  _code(C.DARK_ASPHALT), _code(C.DARK_POTHOLE),
                                  _code(C.LIGHT_POTHOLE)])
    farm_mask = np.isin(land_key, [_code(C.DIRT_ROWS_H), _code(C.DIRT_ROWS_V)])
    grass_mask = np.isin(land_key, [_code(C.DARK_GRASS), _code(C.MEDIUM_GRASS),
                                    _code(C.LIGHT_GRASS)])
    if veg_key is not None:
        farm_mask |= np.isin(veg_key, [_code(C.CORN_1), _code(C.CORN_2)])
        deep_mask = np.isin(veg_key, [_code(C.TREES), _code(C.TREE_BIG),
                                      _code(C.TREE_HUGE), _code(C.TREE_GIANT)])
        forest_mask = deep_mask | np.isin(veg_key, [
            _code(C.TREES_DARK_GRASS), _code(C.SPARSE_TREES),
            _code(C.BUSHES_TREES_DARK_GRASS), _code(C.TREE_SMALL),
        ])
        veg_open_mask = grass_mask | np.isin(veg_key, [
            _code(C.BUSHES), _code(C.DENSE_BUSHES_GRASS),
            _code(C.SHORT_GRASS), _code(C.GRASS_ON_DARK), _code(C.FLOWERS),
        ])
    else:
        deep_mask = np.zeros((h, w), dtype=bool)
        forest_mask = np.zeros((h, w), dtype=bool)
        veg_open_mask = grass_mask

    town_mask = np.zeros((h, w), dtype=bool)
    for tx0, ty0, tx1, ty1 in town_boxes:
        town_mask[max(0, ty0):min(h, ty1), max(0, tx0):min(w, tx1)] = True

    out: list[Zone] = []
    bs = FORAGE_BLOCK
    for cy in range((h + CELL_SIZE - 1) // CELL_SIZE):
        cy0, cy1 = cy * CELL_SIZE, min(h, (cy + 1) * CELL_SIZE)
        for cx in range((w + CELL_SIZE - 1) // CELL_SIZE):
            cx0, cx1 = cx * CELL_SIZE, min(w, (cx + 1) * CELL_SIZE)
            for y0 in range(cy0, cy1, bs):
                y1 = min(cy1, y0 + bs)
                run_kind: str | None = None
                run_x0 = cx0
                run_x1 = cx0
                for x0 in range(cx0, cx1, bs):
                    x1 = min(cx1, x0 + bs)
                    area = (x1 - x0) * (y1 - y0)
                    if area <= 0:
                        continue
                    kind: str | None = None
                    if int(nav_mask[y0:y1, x0:x1].sum()) * 100 >= area * 35:
                        kind = "Nav"
                    elif int(deep_mask[y0:y1, x0:x1].sum()) * 100 >= area * 60:
                        kind = "DeepForest"
                    elif int(forest_mask[y0:y1, x0:x1].sum()) * 100 >= area * 25:
                        kind = "Forest"
                    elif int(farm_mask[y0:y1, x0:x1].sum()) * 100 >= area * 35:
                        kind = "Farm"
                    elif (not town_mask[y0:y1, x0:x1].any()
                          and int(veg_open_mask[y0:y1, x0:x1].sum()) * 100 >= area * 50):
                        kind = "Vegitation"

                    if kind == run_kind:
                        run_x1 = x1
                    else:
                        if run_kind is not None:
                            out.append(Zone(run_kind, run_x0, y0, run_x1 - run_x0, y1 - y0))
                        run_kind = kind
                        run_x0 = x0
                        run_x1 = x1
                if run_kind is not None:
                    out.append(Zone(run_kind, run_x0, y0, run_x1 - run_x0, y1 - y0))
    return out
