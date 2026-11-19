"""Street furniture and road wear: lamps, signs, hydrants, drains, cracks and litter."""
from __future__ import annotations

import os
import random
from collections import defaultdict

from PIL import Image, ImageFilter

from knoxbuild.settings import DEFAULT_STREET_SETTINGS, StreetSettings

from .. import pz_colors as C
from ..osm import ROAD_HIERARCHY_BY_CODE
from .geometry import _draw_roads, _is_polygon, _pixel_width
from .widths import _way_width_m


# How often the vanilla streets have each, measured on their squares
# (tools/building_stats.py's neighbourhoods): grime on 9% of the tarmac, cracks
# on under 0.5%, a lamp per ~290 squares of street, and so on. Spacings are in
# tiles along the kerb.
# The grime was smooth dark patches on the road that read as shadows across it,
# so it is off; set above 0 to bring it back.
GRIME_SHARE = 0.0
# Roads at least this wide get a faded edge line down each side.
EDGE_LINE_MIN_WIDTH = 5
# Straight edge runs shorter than this are staircase steps on a diagonal road;
# placing axis-aligned white edge lines on them leaves broken jagged dashes.
EDGE_LINE_MIN_RUN = 6
CRACK_SHARE = 0.012
LITTER_SHARE = 0.002
LAMP_EVERY = 18
# Hydrants and benches only stand where there is a building to serve.
NEAR_BUILDING_TILES = 12
# A speed limit sign every so many tiles along a main street's kerb (Erika's
# Tiles only). The available signs display miles per hour.
SPEED_SIGN_EVERY = 80
SPEED_LIMITS = {f"{limit} mph": limit for limit, _ in C.SPEED_SIGNS}
LIT, UNLIT = 64, 128
ROAD_LIGHTING = {"yes": LIT, "no": UNLIT}
HYDRANT_EVERY = 38
DRAIN_EVERY = 24
# A bench is two squares long and needs both free.
BENCH_EVERY = 110
# A manhole cover on the pavement, one square in a few hundred.
MANHOLE_SHARE = 0.003
# Which way a prop on each kerb faces. The tools' orient is the direction an
# object points - the kerbs say so, since roads.py puts KERB_W on the strip
# whose road is to its west - so a prop stands with its back to the pavement
# and its face across the carriageway.
FACES_THE_ROAD = {C.EDGE_LINE_W: "E", C.EDGE_LINE_E: "W",
                  C.EDGE_LINE_N: "S", C.EDGE_LINE_S: "N"}
# A bench is backless, so it has no facing: the catalogue draws it in two axes
# (orient W along y, orient N along x) and a bench lies along its kerb.
BENCH_RUNS = {C.EDGE_LINE_W: (C.BENCH_Y, True), C.EDGE_LINE_E: (C.BENCH_Y, True),
              C.EDGE_LINE_N: (C.BENCH_X, False), C.EDGE_LINE_S: (C.BENCH_X, False)}
# A utility pole every so many tiles of kerb. Trunk roads and motorways carry
# their cables on pylons, which are structures of their own.
UTILITY_POLE_EVERY = {
    "motorway": 1 << 30, "trunk": 1 << 30, "primary": 60,
    "secondary": 50, "tertiary": 45, "residential": 40,
    "service": 90, "footway": 1 << 30,
}
ROAD_FURNITURE_INTERVALS = {
    "motorway": {"lamp": 55, "hydrant": 1 << 30, "speed": 55, "drain": 40},
    "trunk": {"lamp": 36, "hydrant": 80, "speed": 65, "drain": 32},
    "primary": {"lamp": 20, "hydrant": 42, "speed": 75, "drain": 24},
    "secondary": {"lamp": 18, "hydrant": 38, "speed": 85, "drain": 24},
    "tertiary": {"lamp": 16, "hydrant": 35, "speed": 100, "drain": 24},
    "residential": {"lamp": 14, "hydrant": 32, "speed": 120, "drain": 24},
    "service": {"lamp": 30, "hydrant": 75, "speed": 1 << 30, "drain": 34},
    "footway": {"lamp": 22, "hydrant": 1 << 30, "speed": 1 << 30, "drain": 1 << 30},
}

SNAP_OFFSETS = sorted(
    ((dx, dy) for dx in range(-3, 4) for dy in range(-3, 4) if dx * dx + dy * dy <= 9),
    key=lambda offset: offset[0] ** 2 + offset[1] ** 2,
)
ROAD_OFFSETS = sorted(
    ((dx, dy) for dx in range(-6, 7) for dy in range(-6, 7) if 0 < dx * dx + dy * dy <= 36),
    key=lambda offset: offset[0] ** 2 + offset[1] ** 2,
)
FURNITURE_CLEARANCE = 12
NEIGHBOURS = ((-1, 0), (0, -1), (1, 0), (0, 1))
MAPPED_PROPS = {
    ("highway", "street_lamp"): "lamp",
    ("amenity", "waste_basket"): "bin",
    ("amenity", "post_box"): "postbox",
    ("shop", "kiosk"): "kiosk",
}


def sample_fixture(density: float, key: int) -> bool:
    """Thin contextual candidates without shifting another pass's random stream."""
    return density >= 1.0 or (density > 0.0 and random.Random(key).random() < density)


def _scaled_interval(interval: int, density: float) -> int:
    if not density or interval >= (1 << 29):
        return 1 << 30
    return max(1, round(interval / density))


def _road_furniture_mask(buckets, proj, ground_rings, *, crisp=False):
    """Pack a way's explicit lighting and supported speed into its road tiles."""
    mask = None
    for category in ("road_service", "road_minor", "road_medium", "road_major"):
        strokes = defaultdict(list)
        for feat in buckets.get(category, ()):
            if _is_polygon(feat):
                continue
            tags = feat.tags
            speed = SPEED_LIMITS.get(tags.get("maxspeed", "").strip().lower(), 0)
            if any(key.startswith("maxspeed:") and key not in {"maxspeed:type", "maxspeed:source"}
                   for key in tags):
                speed = 0
            code = ROAD_LIGHTING.get(tags.get("lit"), 0) | speed
            if code:
                width = _pixel_width(_way_width_m(feat, category), proj.meters_per_tile)
                strokes[code].append((width, ground_rings(feat)))
        if strokes and mask is None:
            mask = Image.new("L", (proj.width, proj.height))
        for code, lines in sorted(strokes.items()):
            _draw_roads(mask, lines, code, crisp=crisp)
    return mask


def _osm_furniture(features):
    """Read mapped props and public places that can use a bin."""
    for feat in features:
        if feat.kind != "node" or not feat.geometry:
            continue
        tags = feat.tags
        if tags.get("location") == "underground" or tags.get("indoor") == "yes":
            continue
        for (key, value), kind in MAPPED_PROPS.items():
            if tags.get(key) == value:
                yield kind, feat
        if tags.get("highway") == "bus_stop" or (
                tags.get("public_transport") == "platform" and tags.get("bus") == "yes"):
            yield "stop bin", feat


def _stop_access_tiles(features, proj, width, height):
    """Leave a tile of clearance around mapped doors, gates and crossings."""
    reserved = set()
    for feat in features:
        tags = feat.tags
        points = feat.geometry
        if feat.kind == "node" and (tags.get("entrance") not in (None, "no")
                                    or tags.get("barrier") in ("gate", "lift_gate", "swing_gate")
                                    or tags.get("highway") == "crossing"):
            ends = points[:1]
        elif feat.kind == "way" and tags.get("footway") == "crossing":
            ends = points[:1] + points[-1:]
        else:
            continue
        for lat, lon in ends:
            x, y = map(round, proj.to_px(lat, lon))
            reserved.update((px, py)
                            for px in range(max(0, x - 1), min(width, x + 2))
                            for py in range(max(0, y - 1), min(height, y + 2)))
    return reserved


def _stop_bin_tile(x, y, pavement, roadside, free, reserved, near_bin):
    """Reach a verge from the waiting tile, or use a platform edge with a bypass."""
    h, w = pavement.shape
    candidates = [(x + dx, y + dy) for dx, dy in SNAP_OFFSETS
                  if 0 <= x + dx < w and 0 <= y + dy < h and free(x + dx, y + dy)]
    walkable = {(px, py) for px, py in candidates if pavement[py, px]}
    waiting = next((point for point in candidates if point in walkable), None)
    if waiting is None:
        return None, "no_free_waiting_surface"
    def reachable_without(blocked=None):
        reached = {waiting}
        pending = [waiting]
        while pending:
            px, py = pending.pop()
            for dx, dy in NEIGHBOURS:
                step = (px + dx, py + dy)
                if step != blocked and step in walkable and step not in reached:
                    reached.add(step)
                    pending.append(step)
        return reached

    connected = reachable_without()

    def placement(px, py):
        if (px, py) in (waiting, (x, y)):
            return None
        sides = [(px + dx, py + dy) for dx, dy in NEIGHBOURS]
        if roadside[py, px] and any(point in connected for point in sides):
            return "verge"
        if (px, py) not in connected or all(
                0 <= sx < w and 0 <= sy < h and pavement[sy, sx] for sx, sy in sides):
            return None
        # The other three tiles of an inward square remain open.
        if any({(px + dx, py), (px, py + dy), (px + dx, py + dy)} <= connected
               for dx in (-1, 1) for dy in (-1, 1)) and (
                   len(reachable_without((px, py))) == len(connected) - 1):
            return "platform_edge"
        return None

    choices = []
    reason = "no_clear_side_tile"
    for px, py in candidates:
        surface = placement(px, py)
        if not surface:
            continue
        if reserved(px, py):
            reason = "protected_access"
        elif near_bin(px, py):
            reason = "near_existing_bin"
        else:
            choices.append((surface, px, py))
    if choices:
        surface, px, py = min(choices, key=lambda choice: choice[0] != "verge")
        return (px, py), surface
    return None, reason


def _paint_osm_furniture(out, empty, roadside, pavement, asphalt,
                         buildings, features, proj, clip, counts, settings, bin_report, passage):
    """Place OSM nodes on nearby free ground and index the bins and lamps."""
    h, w = empty.shape
    placed = {"lamp": defaultdict(list), "bin": defaultdict(list)}
    distance = FURNITURE_CLEARANCE

    def near(kind, x, y):
        cells = placed[kind]
        return bool(cells) and any(
            (x - px) ** 2 + (y - py) ** 2 < distance ** 2
            for cy in range(y // distance - 1, y // distance + 2)
            for cx in range(x // distance - 1, x // distance + 2)
            for px, py in cells.get((cx, cy), ()))

    if proj is None:
        return near
    if clip is not None:
        from shapely.geometry import Point

    def free(x, y):
        return (empty[y, x] and (buildings is None or not buildings[y, x])
                and (clip is None or clip.covers(Point(x, y))))

    def near_bin(x, y):
        return near("bin", x, y)

    def put(kind, x, y, colour):
        out[y, x] = colour
        empty[y, x] = False
        counts[kind] = counts.get(kind, 0) + 1
        if kind in placed:
            placed[kind][x // distance, y // distance].append((x, y))

    # Explicit objects take priority when an inferred stop bin needs a tile.
    features = tuple(features)
    props = sorted(_osm_furniture(features),
                   key=lambda item: (item[0] == "stop bin", item[1].osm_id))
    access_tiles = _stop_access_tiles(features, proj, w, h) if props else set()

    def reserved(x, y):
        return (x, y) in access_tiles or passage(x, y)
    colours = {}
    if settings.street_lamp_mode != "off":
        colours["lamp"] = {f: getattr(C, f"LAMP_{f}") for f in FACES_THE_ROAD.values()}
    for kind, label, enabled in (("kiosk", "Kiosk N", settings.street_kiosks),
                                 ("postbox", "Postbox N", settings.street_postboxes)):
        if enabled and any(prop == kind for prop, _ in props) and _rule_ready(label):
            colours[kind] = {f: getattr(C, f"{kind.upper()}_{f}") for f in FACES_THE_ROAD.values()}

    for what, feat in props:
        ax, ay = proj.to_px(*feat.geometry[0])
        x, y = round(ax), round(ay)
        if not (0 <= x < w and 0 <= y < h):
            continue
        entry = None
        if bin_report is not None and what in ("bin", "stop bin"):
            entry = {
                "osm_id": feat.osm_id, "anchor": [x, y],
                "source": "osm_waste_basket" if what == "bin" else "osm_bus_stop_inferred",
                "status": "skipped", "reason": "settings",
            }
            bin_report.append(entry)
        if what == "stop bin":
            if settings.street_bin_mode != "context":
                continue
            if not sample_fixture(settings.street_bin_density, feat.osm_id):
                if entry is not None:
                    entry["reason"] = "density"
                continue
        elif what == "bin":
            if settings.street_bin_mode == "off":
                continue
        elif what not in colours:
            continue
        if buildings is not None and buildings[y, x]:
            if entry is not None:
                entry["reason"] = "inside_building"
            continue
        if clip is not None and not clip.covers(Point(ax, ay)):
            if entry is not None:
                entry["reason"] = "outside_shape"
            continue
        if what == "stop bin":
            tile, reason = ((None, "near_existing_bin") if near_bin(x, y) else
                            _stop_bin_tile(x, y, pavement, roadside, free, reserved, near_bin))
            if tile is not None:
                put("bin", *tile, C.BIN)
                if entry is not None:
                    entry.update(status="placed", reason=reason, tile=list(tile))
            elif entry is not None:
                entry["reason"] = reason
            continue
        skipped = "no_free_surface"
        for dx, dy in SNAP_OFFSETS:
            px, py = x + dx, y + dy
            if not (0 <= px < w and 0 <= py < h and free(px, py)):
                continue
            if not (roadside[py, px] or pavement[py, px]):
                continue
            if reserved(px, py):
                skipped = "protected_access"
                continue
            colour = C.BIN
            if what != "bin":
                facing = "S"
                for rx, ry in ROAD_OFFSETS:
                    nx, ny = px + rx, py + ry
                    if 0 <= nx < w and 0 <= ny < h and asphalt[ny, nx]:
                        facing = (("W" if rx < 0 else "E") if abs(rx) >= abs(ry)
                                  else ("N" if ry < 0 else "S"))
                        break
                colour = colours[what][facing]
            put(what, px, py, colour)
            if entry is not None:
                entry.update(status="placed", reason="mapped", tile=[px, py])
            break
        else:
            if entry is not None:
                entry["reason"] = skipped
    return near


def _rule_ready(label: str) -> bool:
    """Whether the map tools' Rules.txt carries one of our rules yet.

    A colour with no rule behind it is dropped with a warning, so this is what
    keeps a pole off the verge on an install that Setup has not been run on
    since the poles were added.
    """
    if os.environ.get("KNOXMAP_NO_POLES") == "1":
        return False
    try:
        import knoxpaths
        rules = knoxpaths.mapping_tools_dir() / "config" / "Rules.txt"
        return f"KnoxMap road {label}" in rules.read_text(encoding="utf-8", errors="replace")
    except Exception:      # noqa: BLE001 - no tools, no pole
        return False


def _mod_tiles_ready() -> bool:
    """Whether maps may use Erika's Tiles: installed and set up, and not
    turned off with KNOXMAP_NO_MOD_TILES=1."""
    if os.environ.get("KNOXMAP_NO_MOD_TILES") == "1":
        return False
    try:
        import knoxpaths
        if not knoxpaths.erikas_tiles_ready():
            return False
        # Tools set up before the signs existed have no rule for them, and a
        # colour without a rule is dropped with a warning: run Setup again.
        rules = knoxpaths.mapping_tools_dir() / "config" / "Rules.txt"
        return "KnoxMap road Speed limit 25 S" in rules.read_text(encoding="utf-8", errors="replace")
    except Exception:      # noqa: BLE001 - no tools, no mod tiles
        return False


def _long_runs(mask, min_len: int, axis: int):
    """Keep only contiguous True runs of length >= min_len along `axis`."""
    import numpy as np

    if min_len <= 1:
        return mask
    h, w = mask.shape
    if (axis == 1 and w < min_len) or (axis == 0 and h < min_len):
        return np.zeros_like(mask)
    starts = mask.copy()
    for offset in range(1, min_len):
        if axis == 1:
            starts[:, :-offset] &= mask[:, offset:]
            starts[:, -offset:] = False
        else:
            starts[:-offset, :] &= mask[offset:, :]
            starts[-offset:, :] = False
    out = starts.copy()
    for offset in range(1, min_len):
        if axis == 1:
            out[:, offset:] |= starts[:, :-offset]
        else:
            out[offset:, :] |= starts[:-offset, :]
    return out


def _paint_street_furniture(veg: Image.Image, landscape: Image.Image,
                            road_hierarchy: Image.Image | None = None,
                            buildings: Image.Image | None = None, *,
                            road_furniture: Image.Image | None = None,
                            features=(), proj=None, clip=None,
                            settings: StreetSettings = DEFAULT_STREET_SETTINGS,
                            bin_report: list[dict] | None = None,
                            passage_mask=None) -> dict:
    """Lamps, hydrants and drains along the kerbs; grime, cracks and litter.

    A generated street was clean tarmac, kerb and pavement and nothing else;
    Knox County's have all of these, and a road without them reads as a
    diagram. They go by the kerbs already painted: a lamp stands on the
    pavement one tile back from a straight kerb with its arm over the road, a
    hydrant likewise, a drain in the gutter in front of it.
    """
    import numpy as np

    rng = np.random.default_rng(4321)
    ground = np.asarray(landscape.convert("RGB"))
    vp = np.array(veg.convert("RGB"))
    h, w = ground.shape[:2]
    road_classes = (np.asarray(road_hierarchy.convert("L"))
                    if road_hierarchy is not None else None)
    road_tags = np.asarray(road_furniture) if road_furniture is not None else None

    def is_colour(arr, colours):
        mask = np.zeros(arr.shape[:2], dtype=bool)
        for c in colours:
            mask |= np.all(arr == c, axis=2)
        return mask

    empty = np.all(vp == 0, axis=2)
    asphalt = is_colour(ground, (C.MEDIUM_ASPHALT, C.DARKEST_ASPHALT))
    driveable = is_colour(ground, (C.MEDIUM_ASPHALT, C.DARK_ASPHALT,
                                  C.DARKEST_ASPHALT))
    pavement = is_colour(ground, (C.PALE_CONCRETE,))
    roadside = is_colour(
        ground, (C.DARK_GRASS, C.MEDIUM_GRASS, C.LIGHT_GRASS, C.PAVING,
                 C.DIRT_ROWS_H, C.DIRT_ROWS_V))
    # Hydrants and benches stand on the grass or paving beside the sidewalk.
    verge = is_colour(ground, (C.DARK_GRASS, C.MEDIUM_GRASS, C.LIGHT_GRASS,
                               C.PAVING))
    near_buildings = None
    building_tiles = None
    if buildings is not None:
        from . import native_pixels

        mask = buildings if buildings.mode == "L" else buildings.convert("L")
        building_tiles = np.asarray(mask)
        mw, mh = mask.size
        raw = native_pixels.box_any(mask.tobytes(), mw, mh, NEAR_BUILDING_TILES)
        if raw is not None:
            near_buildings = np.frombuffer(raw, dtype=bool).reshape(mh, mw).copy()
        else:
            reach = ImageFilter.MaxFilter(2 * NEAR_BUILDING_TILES + 1)
            near_buildings = np.asarray(mask.filter(reach)) > 0
    counts = {}

    # Edge lines: tarmac with the road's edge on exactly one side and enough
    # road across from it to be a carriageway, not a car park aisle's end.
    # (Dark tarmac, 100, is car parks and yards, and is left out.)
    k = EDGE_LINE_MIN_WIDTH
    edge_lines = 0
    diag_kerbs = {}
    # What counts as "no road on the far side": a car park's dark tarmac is
    # road too, and a kerb drawn against it stood in the middle of a paved
    # yard in game - a white strip with nothing to kerb.
    beside = driveable
    for colour, (dx, dy) in ((C.EDGE_LINE_W, (-1, 0)), (C.EDGE_LINE_E, (1, 0)),
                             (C.EDGE_LINE_N, (0, -1)), (C.EDGE_LINE_S, (0, 1))):
        outside = np.zeros_like(asphalt)
        across = np.ones_like(asphalt)
        if dx:
            if dx < 0:
                outside[:, 1:] = ~beside[:, :-1]
                for i in range(1, k):
                    across[:, :-i] &= asphalt[:, i:]
                    across[:, -i:] = False
            else:
                outside[:, :-1] = ~beside[:, 1:]
                for i in range(1, k):
                    across[:, i:] &= asphalt[:, :-i]
                    across[:, :i] = False
            side_a = np.zeros_like(asphalt); side_b = np.zeros_like(asphalt)
            side_a[1:, :] = asphalt[:-1, :]; side_b[:-1, :] = asphalt[1:, :]
        else:
            if dy < 0:
                outside[1:, :] = ~beside[:-1, :]
                for i in range(1, k):
                    across[:-i, :] &= asphalt[i:, :]
                    across[-i:, :] = False
            else:
                outside[:-1, :] = ~beside[1:, :]
                for i in range(1, k):
                    across[i:, :] &= asphalt[:-i, :]
                    across[:i, :] = False
            side_a = np.zeros_like(asphalt); side_b = np.zeros_like(asphalt)
            side_a[:, 1:] = asphalt[:, :-1]; side_b[:, :-1] = asphalt[:, 1:]
        # Straight edges only: the road continues along the line both ways, and
        # the run is long enough not to be a staircase step on a diagonal road.
        candidate = asphalt & outside & across
        long_edge = _long_runs(candidate, EDGE_LINE_MIN_RUN, 0 if dx else 1)
        pick = long_edge & side_a & side_b & empty
        diag_kerbs[colour] = candidate & (side_a | side_b) & ~pick
        vp[pick] = colour
        empty &= ~pick
        edge_lines += int(pick.sum())
    counts["edge lines"] = edge_lines

    # Along each straight edge of the carriageway, road on one side: the lamp
    # stands a tile or two off the tarmac (past the kerb or on the verge) with
    # its arm reaching back over the road, the drain sits in the gutter.
    # (edge colour just painted, step away from the road, lamp colour)
    straight = [(C.EDGE_LINE_W, (-1, 0), C.LAMP_E), (C.EDGE_LINE_E, (1, 0), C.LAMP_W),
                (C.EDGE_LINE_N, (0, -1), C.LAMP_S), (C.EDGE_LINE_S, (0, 1), C.LAMP_N)]
    taken: dict[str, set] = {
        "lamp": set(), "hydrant": set(), "drain": set(), "speed": set(),
        "pole": set(), "bench": set(),
    }
    pole = C.UTILITY_POLE if _rule_ready("Utility pole") else None
    out = vp.copy()

    def passage(x, y):
        return passage_mask is not None and passage_mask[y, x]

    near_osm_prop = _paint_osm_furniture(
        out, empty, roadside, pavement, asphalt, building_tiles, features, proj, clip,
        counts, settings, bin_report, passage)
    # Speed signs stand where drivers keeping right see their faces: on the
    # east kerb of a north-south road facing south, the north kerb of an
    # east-west one facing east. Erika's signs only face those two ways.
    speed_side = ({C.EDGE_LINE_E: "S", C.EDGE_LINE_N: "E"}
                  if road_tags is not None and settings.street_sign_density and _mod_tiles_ready() else {})
    lamp_density = settings.street_lamp_density if settings.street_lamp_mode == "context" else 0.0
    intervals = {}
    for hierarchy in (None, *ROAD_FURNITURE_INTERVALS):
        spacing = ROAD_FURNITURE_INTERVALS.get(hierarchy, {})
        lamp_every = spacing.get("lamp", LAMP_EVERY)
        intervals[hierarchy] = (
            _scaled_interval(lamp_every, lamp_density),
            _scaled_interval(2 * lamp_every, lamp_density),
            _scaled_interval(spacing.get("speed", SPEED_SIGN_EVERY), settings.street_sign_density),
            _scaled_interval(spacing.get("hydrant", HYDRANT_EVERY), settings.street_hydrant_density),
            _scaled_interval(UTILITY_POLE_EVERY.get(hierarchy, 55), settings.street_pole_density),
            spacing.get("drain", DRAIN_EVERY),
        )
    verge_buildings = near_buildings if settings.street_verge_scope == "built_up" else None

    def hierarchy_at(gx, gy, sx, sy):
        if road_classes is None:
            return None
        for n in range(24):
            x, y = gx - n * sx, gy - n * sy
            if not (0 <= x < w and 0 <= y < h):
                break
            if asphalt[y, x] and road_classes[y, x]:
                return ROAD_HIERARCHY_BY_CODE.get(int(road_classes[y, x]))
        return None

    def beside_the_pavement(gx, gy, sx, sy):
        """The first free grass or paving tile that touches the sidewalk."""
        for n in range(2, 6):
            x, y = gx + n * sx, gy + n * sy
            if not (0 <= x < w and 0 <= y < h):
                return None
            if not (empty[y, x] and not driveable[y, x] and verge[y, x]) or passage(x, y):
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and pavement[ny, nx]:
                    return x, y
        return None

    def off_the_pavement(gx, gy, sx, sy):
        """First free non-sidewalk roadside tile off the kerb."""
        x1, y1 = gx + sx, gy + sy
        x2, y2 = gx + 2 * sx, gy + 2 * sy
        if (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h
                and pavement[y2, x2] and empty[y1, x1]
                and not driveable[y1, x1] and roadside[y1, x1] and not passage(x1, y1)):
            return x1, y1
        for n in range(2, 6):
            x, y = gx + n * sx, gy + n * sy
            if not (0 <= x < w and 0 <= y < h) or driveable[y, x]:
                return None
            if empty[y, x] and roadside[y, x] and not passage(x, y):
                return x, y
        return None

    for edge, (sx, sy), lamp in straight:
        kerb_mask = np.all(vp == edge, axis=2) | diag_kerbs[edge]
        ys, xs = np.nonzero(kerb_mask)
        order = rng.permutation(len(xs))
        for i in order.tolist():
            gx, gy = int(xs[i]), int(ys[i])     # the gutter square itself
            bx, by = gx + 2 * sx, gy + 2 * sy   # past the kerb or the verge
            if not (0 <= bx < w and 0 <= by < h):
                continue
            hierarchy = hierarchy_at(gx, gy, sx, sy)
            lit_every, fallback_every, speed_every, hydrant_every, pole_every, drain_every = intervals[hierarchy]
            facing = speed_side.get(edge)
            tags = int(road_tags[gy, gx]) if road_tags is not None else 0
            speed = C.SPEED_SIGNS.get((tags & (LIT - 1), facing))
            for what, every, colour, at in (
                    ("speed", speed_every, speed, (bx, by)),
                    ("lamp", lit_every if tags & LIT else fallback_every,
                     None if tags & UNLIT else lamp, (bx, by)),
                    ("hydrant", hydrant_every, C.HYDRANT, (bx, by)),
                    # Last of the verge props, so a lamp or a hydrant that is
                    # due on the same square keeps it: the poles fall in the
                    # gaps between them, as they do along a real street.
                    ("pole", pole_every, pole, (bx, by)),
                    ("drain", drain_every, C.STORM_DRAIN, (gx, gy))):
                if colour is None or every >= (1 << 29):
                    continue
                px, py = at
                if what == "hydrant":
                    spot = beside_the_pavement(gx, gy, sx, sy)
                    if spot is None:
                        continue
                    px, py = spot
                    if verge_buildings is not None and not verge_buildings[py, px]:
                        continue
                elif what in ("speed", "lamp", "pole"):
                    spot = off_the_pavement(gx, gy, sx, sy)
                    if spot is None:
                        continue
                    px, py = spot
                if not (0 <= px < w and 0 <= py < h):
                    continue
                # Speed signs keep a spacing per facing: the north-south roads,
                # met first, took every cell and left east-west ones none.
                cell = (px // every, py // every, facing if what == "speed" else None)
                if cell in taken[what]:
                    continue
                if what == "lamp":
                    if (not tags & LIT and near_buildings is not None
                            and not near_buildings[py, px]):
                        continue
                    if near_osm_prop("lamp", px, py):
                        continue
                # A drain takes the gutter square from its edge line; other
                # props need an empty sidewalk or verge tile, not a road or lot.
                if what == "drain" and not np.array_equal(vp[py, px], edge):
                    continue
                if what != "drain" and (not empty[py, px] or driveable[py, px]
                                         or not roadside[py, px]):
                    continue
                taken[what].add(cell)
                out[py, px] = colour
                empty[py, px] = False
                counts[what] = counts.get(what, 0) + 1
                break

    # Wear comes in patches, as it does on a real road and on the vanilla
    # map; picked square by square it came out as a chequerboard. Smooth noise
    # - coarse random values, blurred up to full size - gives the patches.
    coarse = Image.fromarray((rng.random((max(1, h // 5), max(1, w // 5))) * 255).astype(np.uint8))
    noise = np.asarray(coarse.resize((w, h), Image.BICUBIC)
                       .filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    level = (np.quantile(noise[asphalt], 1 - GRIME_SHARE)
             if GRIME_SHARE > 0 and asphalt.any() else 256)
    pick = asphalt & empty & (noise >= level)
    out[pick] = C.ASPHALT_GRIME
    empty &= ~pick
    counts["grime"] = int(pick.sum())
    # Scattered things: each square rolls once, and each kind has its own band
    # of the roll, so no square gets two.
    roll = rng.random((h, w))
    scattered = [(asphalt, CRACK_SHARE, C.ASPHALT_CRACKS, "cracks"),
                 (pavement, LITTER_SHARE * settings.street_litter_density, C.LITTER, "litter")]
    if _rule_ready("Manhole cover"):
        from . import native_pixels

        asphalt_img = Image.fromarray(asphalt.astype(np.uint8) * 255, mode="L")
        raw_road = native_pixels.box_any(asphalt_img.tobytes(), w, h, 4)
        if raw_road is not None:
            near_road = np.frombuffer(raw_road, dtype=bool).reshape(h, w).copy()
        else:
            near_road = np.asarray(asphalt_img.filter(ImageFilter.MaxFilter(9))) > 0
        scattered.append((pavement & near_road, MANHOLE_SHARE, C.MANHOLE, "manholes"))
    lo = 0.0
    for mask, share, colour, name in scattered:
        pick = mask & empty & (roll >= lo) & (roll < lo + share)
        lo += share
        out[pick] = colour
        empty &= ~pick
        counts[name] = int(pick.sum())
    # Benches: three squares along the kerb with the seat facing the road. They
    # need the whole run free, so they take a pass of their own rather than a
    # square in the props loop.
    if settings.street_bench_density and _rule_ready("Bench Y0"):
        bench_every = _scaled_interval(BENCH_EVERY, settings.street_bench_density)
        for edge, (sx, sy), _lamp in straight:
            tiles, along_y = BENCH_RUNS[edge]
            dx, dy = (0, 1) if along_y else (1, 0)
            kerb_mask = np.all(vp == edge, axis=2) | diag_kerbs[edge]
            ys, xs = np.nonzero(kerb_mask)
            for i in rng.permutation(len(xs)).tolist():
                gx, gy = int(xs[i]), int(ys[i])
                # A bench leans on a straight kerb. On a road at an angle the
                # kerb steps a square at a time, so the squares either side of
                # this one along its own axis are road, not edge - and a seat
                # laid along that step hangs off the pavement.
                if along_y:
                    if not (0 < gy < h - 1 and kerb_mask[gy - 1, gx]
                            and kerb_mask[gy + 1, gx]):
                        continue
                elif not (0 < gx < w - 1 and kerb_mask[gy, gx - 1]
                          and kerb_mask[gy, gx + 1]):
                    continue
                spot = beside_the_pavement(gx, gy, sx, sy)
                if spot is None:
                    continue
                px, py = spot
                if verge_buildings is not None and not verge_buildings[py, px]:
                    continue
                cell = (px // bench_every, py // bench_every)
                if cell in taken["bench"]:
                    continue
                run = [(px + k * dx, py + k * dy) for k in range(len(tiles))]
                if not all(0 <= x < w and 0 <= y < h and empty[y, x]
                           and not driveable[y, x] and roadside[y, x] and not passage(x, y)
                           for x, y in run):
                    continue
                for (x, y), colour in zip(run, tiles):
                    out[y, x] = colour
                    empty[y, x] = False
                taken["bench"].add(cell)
                counts["bench"] = counts.get("bench", 0) + 1
    veg.paste(Image.fromarray(out), (0, 0))
    return counts
