"""Independent stairwells in a long residential block."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .footprint import Footprint, split_at
from .layout.core import MIN_ROOM, STAIR_RUN
from .layout.furnishing import CORE_WIDE

# Initial estimate for elongated blocks with no mapped residential entrances.
SECTION_FRONTAGE_M = 18
TURNING_HALL_WIDTH = 7


@dataclass(frozen=True)
class Section:
    footprint: Footprint
    core: tuple[int, int, int, int]
    door: tuple[int, int, str]


def _fit_section(fp: Footprint, front: str, target: float | None) -> Section | None:
    """Fit a hall from the facade inward, leaving room for flats on both sides."""
    return (_fit_hall(fp, front, target, TURNING_HALL_WIDTH)
            or _fit_hall(fp, front, target, CORE_WIDE))


def _fit_hall(fp: Footprint, front: str, target: float | None, hall_width: int) -> Section | None:
    along_x = front in ("N", "S")
    mask = fp.mask if along_x else fp.mask.T
    reverse = front in ("S", "E")
    depth, width = mask.shape
    preferred = target if target is not None else width / 2
    candidates = []
    for start in range(MIN_ROOM, width - MIN_ROOM - hall_width + 1):
        center = start + hall_width // 2
        if target is not None and abs(center + 0.5 - target) > 1.5:
            continue
        strip = mask[:, start:start + hall_width].all(axis=1)
        full = np.flatnonzero(strip)
        if not len(full):
            continue
        edge = int(full[-1] if reverse else full[0])
        step = -1 if reverse else 1
        end = edge
        while 0 <= end + step < depth and strip[end + step]:
            end += step
        if abs(end - edge) + 1 < STAIR_RUN + 3:
            continue
        if 0 <= edge - step < depth and mask[edge - step, center]:
            continue
        lo, hi = sorted((edge, end))
        wall = edge + int(reverse)
        if along_x:
            core, door = (start, lo, start + hall_width - 1, hi), (center, wall, "N")
        else:
            core, door = (lo, start, hi, start + hall_width - 1), (wall, center, "W")
        candidates.append((abs(center + 0.5 - preferred), -abs(end - edge), core, door))
    if not candidates:
        return None
    _, _, core, door = min(candidates)
    return Section(fp, core, door)


def turning_flights(core: tuple[int, int, int, int]):
    """Two native PZ flights with separate openings and a shared walking area."""
    x, y, right, bottom = core
    if min(right - x + 1, bottom - y + 1) < TURNING_HALL_WIDTH:
        return None
    if bottom - y >= right - x:
        return (x + 4, y + 1, "N"), (x + 1, y + 6, "W")
    return (x + 1, y + 4, "W"), (x + 6, y + 1, "N")


def plan_sections(fp: Footprint, entrances: list, street: str,
                  metres_per_tile: float, *, estimate: bool = False) -> tuple[list[Section], str]:
    """Plan adjacent sections for a straight slab, or keep its original plan."""
    width, depth = sorted((fp.width, fp.height), reverse=True)
    if (width < 2.5 * depth or fp.angle > 8 or fp.tiles < 0.82 * width * depth
            or width * metres_per_tile < 1.8 * SECTION_FRONTAGE_M):
        return [], ""
    along_x = fp.width >= fp.height
    fronts = ("N", "S") if along_x else ("W", "E")
    start, near = (fp.x0, fp.y0) if along_x else (fp.y0, fp.x0)
    known = {}
    for x, y, tags in entrances:
        if tags.get("entrance") != "staircase" or tags.get("access") == "no":
            continue
        if str(tags.get("level", "0")) not in ("0", "ground"):
            continue
        along, across = (x, y) if along_x else (y, x)
        front = fronts[int(across >= near + depth / 2)]
        if abs(across - (near + (depth if front == fronts[1] else 0))) > 3:
            return [], ""
        key = tags.get("ref") or tags.get("addr:flats") or (round(x), round(y))
        if key in known:
            return [], ""  # two mapped doors might serve the same staircase
        known[key] = front, along - start
    if known:
        targets = sorted(along for _, along in known.values())
        if len(targets) < 2 or len({front for front, _ in known.values()}) != 1:
            return [], ""
        front = next(iter(known.values()))[0]
        edges = [0] + [round((a + b) / 2) for a, b in zip(targets, targets[1:])] + [width]
        source = "mapped"
    else:
        if not estimate or entrances:
            return [], ""  # preserve buildings with no reliable entrance count
        count = round(width * metres_per_tile / SECTION_FRONTAGE_M)
        edges = [round(width * k / count) for k in range(count + 1)]
        targets = [None] * count
        front = street if street in fronts else fronts[1]
        source = "estimated"
    parts = split_at(fp, edges)
    if len(parts) != len(targets) or sum(part.tiles for part in parts) != fp.tiles:
        return [], ""
    sections = []
    for part, target in zip(parts, targets):
        offset = (part.x0 if along_x else part.y0) - start
        section = _fit_section(part, front, None if target is None else target - offset)
        if section is None:
            return [], ""
        sections.append(section)
    return sections, source
