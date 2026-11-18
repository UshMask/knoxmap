"""Which cells of a compiled map are out of date, so a recompile does only those.

Compiling a big town is most of an hour, and a change to one street used to cost
all of it: clear_stale (compile_map.py) can only compare file times, and a
build rewrites every building's file, so everything looked newer than the lots.

This keeps, after a compile that finished cleanly, a fingerprint of everything
each 300-tile cell of the world was compiled from:

  terrain  the ground and vegetation pixels. These become the cell's .tmx.
  content  the cell's entry in the .pzw (its buildings and zones), the .tbx
           files those buildings are, and the zombie spawn map over the cell.
           These, with the terrain, become the lot files.

and one fingerprint of what applies to the whole world (the rules the bitmaps
are converted by, the compiler, the world's size and origin). Next time, cells
whose fingerprints are unchanged keep their lots; the rest, and their
neighbours (a lot file is 256 tiles, a cell 300, and blends and buildings
reach across an edge), are deleted so WorldEd makes them again. When anything
global changed, or there is no record, nothing is trusted and the older
file-time rule decides.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

CELL = 300          # tiles in one cell of the world, as the .pzw counts them
LOT = 256           # tiles in one lot file
SPAWN_SCALE = 10    # the spawn map has one pixel to ten tiles
STATE_FILE = "compile_state.json"
VERSION = 1

_CELL = re.compile(r'<cell x="(\d+)" y="(\d+)" map="[^"]*">(.*?)</cell>', re.S)


def _sha(*parts: bytes) -> str:
    h = hashlib.blake2b(digest_size=12)
    for part in parts:
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)
    return h.hexdigest()


def _key(x: int, y: int) -> str:
    return f"{x},{y}"


def _unkey(key: str) -> tuple[int, int]:
    x, y = key.split(",")
    return int(x), int(y)


def _pzw_facts(text: str) -> dict:
    size = re.search(r'<world version="[^"]*" width="(\d+)" height="(\d+)"', text)
    origin = re.search(r'<worldOrigin origin="(-?\d+),(-?\d+)"', text)
    bmp = re.search(r'<bmp path="([^"]+)"', text)
    return {"width": int(size.group(1)) if size else 0,
            "height": int(size.group(2)) if size else 0,
            "origin": (int(origin.group(1)), int(origin.group(2))) if origin else (0, 0),
            "base": Path(bmp.group(1)).stem if bmp else ""}


def _global_fingerprint(project: Path, pzw_text: str, facts: dict) -> str:
    """What changes every cell at once. A folder that has been moved changes
    it too, because the project file names it: that costs one full compile and
    is never wrong."""
    import knoxpaths

    parts = [json.dumps([VERSION, facts["width"], facts["height"], facts["origin"],
                         facts["base"]]).encode()]
    head = pzw_text.split("<bmp ", 1)[0].replace(project.as_posix(), "<project>")
    parts.append(head.encode("utf-8", "replace"))
    tools = knoxpaths.mapping_tools_dir()
    if tools:
        for rel in (("config", "Rules.txt"), ("config", "Blends.txt"),
                    ("settings", "PZTools.ini")):
            p = tools.joinpath(*rel)
            parts.append(p.read_bytes() if p.exists() else b"-")
    cli = knoxpaths.worlded_cli()
    if cli and Path(cli).exists():
        st = Path(cli).stat()
        parts.append(f"{st.st_size}:{int(st.st_mtime)}".encode())
    return _sha(*parts)


def fingerprints(project: Path) -> dict:
    """What every cell of `project` would be compiled from right now."""
    from knoxbuild.bitmaps import cell_hashes

    pzw = project / f"{project.name}.pzw"
    text = pzw.read_text(encoding="utf-8", errors="replace")
    facts = _pzw_facts(text)
    base = facts["base"] or project.name

    ground = cell_hashes(str(project / f"{base}.bmp"), CELL)
    veg_path = project / f"{base}_veg.bmp"
    veg = cell_hashes(str(veg_path), CELL) if veg_path.exists() else {}
    spawn_path = project / f"{base}_ZombieSpawnMap.bmp"
    spawn = (cell_hashes(str(spawn_path), CELL // SPAWN_SCALE)
             if spawn_path.exists() else {})

    tbx_hash: dict[str, str] = {}

    def tbx(rel: str) -> str:
        if rel not in tbx_hash:
            p = project / rel
            tbx_hash[rel] = (_sha(p.read_bytes()) if p.is_file() else "missing")
        return tbx_hash[rel]

    terrain: dict[str, str] = {}
    content: dict[str, str] = {}
    bodies = {(int(x), int(y)): body for x, y, body in _CELL.findall(text)}
    for y in range(facts["height"]):
        for x in range(facts["width"]):
            k = _key(x, y)
            terrain[k] = _sha(ground.get((x, y), "-").encode(),
                              veg.get((x, y), "-").encode())
            body = bodies.get((x, y), "")
            used = sorted(set(re.findall(r'<lot [^>]*map="([^"]+)"', body)))
            content[k] = _sha(body.encode("utf-8", "replace"),
                              spawn.get((x, y), "-").encode(),
                              "".join(f"{u}={tbx(u)};" for u in used).encode())
    return {"version": VERSION, "global": _global_fingerprint(project, text, facts),
            "origin": list(facts["origin"]), "size": [facts["width"], facts["height"]],
            "terrain": terrain, "content": content}


def load(project: Path) -> dict | None:
    try:
        with open(project / STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("version") == VERSION else None


def save(project: Path, fp: dict) -> None:
    tmp = project / (STATE_FILE + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(fp, f)
        os.replace(tmp, project / STATE_FILE)
    except OSError:
        pass          # no record means the next compile uses the older rule


def forget(project: Path) -> None:
    try:
        (project / STATE_FILE).unlink()
    except OSError:
        pass


def _grow(cells: set[tuple[int, int]], size: tuple[int, int]) -> set[tuple[int, int]]:
    w, h = size
    return {(x + dx, y + dy) for x, y in cells for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if 0 <= x + dx < w and 0 <= y + dy < h}


def lot_files_of(cells, origin: tuple[int, int]) -> set[tuple[int, int]]:
    """The (x, y) of every lot file that overlaps any of these world cells."""
    ox, oy = origin
    out: set[tuple[int, int]] = set()
    for x, y in cells:
        x0, y0 = (ox + x) * CELL, (oy + y) * CELL
        for lx in range(x0 // LOT, (x0 + CELL - 1) // LOT + 1):
            for ly in range(y0 // LOT, (y0 + CELL - 1) // LOT + 1):
                out.add((lx, ly))
    return out


def worlded_export(project: Path, paths: list[str], marker: Path) -> tuple:
    """Require every lot to postdate the last build and its WorldEd inputs.

    Stock WorldEd does not write a KnoxMap completion record. A full export
    replaces all three files for every lot coordinate. A cancelled or selected
    export can leave older triples mixed in with the new ones.
    """
    import knoxpaths
    from tools.compile_cells import lot_names

    def local(name: str) -> Path:
        # Wine's default Z: drive is the host filesystem.
        if os.name != "nt" and name[:3].lower() == "z:/":
            name = name[2:]
        return project / name

    def snapshot(path: Path) -> tuple:
        st = path.stat()
        return path, st.st_mtime_ns, st.st_size, st.st_ino

    pzw = project / f"{project.name}.pzw"
    before = snapshot(pzw)
    world = ET.parse(pzw).getroot()
    origin = world.find("./GenerateLots/worldOrigin")
    bitmap = world.find("bmp")
    if world.tag != "world" or origin is None or bitmap is None:
        raise ValueError("The WorldEd project is missing its origin or bitmap.")
    w, h = int(world.attrib["width"]), int(world.attrib["height"])
    ox, oy = map(int, origin.attrib["origin"].split(","))
    ground = local(bitmap.attrib["path"])
    vegetation = ground.with_name(f"{ground.stem}_veg.bmp")
    terrain = [snapshot(ground), snapshot(vegetation)]
    tools = knoxpaths.mapping_tools_dir()
    if tools:
        terrain += [snapshot(p) for p in (tools / "config" / "Rules.txt",
                                          tools / "config" / "Blends.txt") if p.is_file()]
    inputs = [before, snapshot(marker), *terrain]
    spawn = world.find("./GenerateLots/ZombieSpawnMap")
    if spawn is not None and spawn.get("path"):
        inputs.append(snapshot(local(spawn.attrib["path"])))
    inputs.extend(snapshot(local(name)) for name in sorted({
        lot.attrib["map"] for lot in world.findall("./cell/lot")
    }))
    maps = {(int(cell.attrib["x"]), int(cell.attrib["y"])): cell.get("map")
            for cell in world.findall("cell")}
    coords = [(x, y) for y in range(h) for x in range(w)]
    if not coords or {Path(path).name for path in paths} != lot_names(lot_files_of(coords, (ox, oy))):
        raise ValueError("WorldEd has not exported the entire map.")
    terrain_time = max(p[1] for p in terrain)
    converted = [snapshot(local(maps[x, y]) if maps.get((x, y)) else
                          project / "tmx" / f"{ground.stem}_{ox + x}_{oy + y}.tmx")
                 for x, y in coords]
    if any(tmx[1] < terrain_time for tmx in converted):
        raise ValueError("The converted maps are out of date.")
    inputs.extend(converted)
    if tools:
        settings = tools / "settings" / "PZTools.ini"
        if settings.is_file():
            inputs.append(snapshot(settings))

    outputs = [snapshot(Path(path)) for path in paths]
    if min(p[1] for p in outputs) <= max(p[1] for p in inputs):
        raise ValueError("Some of the lot files are out of date.")
    return inputs, outputs


def worlded_lot_hashes(folder: Path) -> dict:
    """Validate the staged Build 42 files before recording their hashes."""
    from tools.compile_cells import lot_paths

    hashes = {}
    for header in folder.glob("*.lotheader"):
        xy = header.stem
        data = header.read_bytes()
        if data[:8] != b"LOTH\x01\0\0\0":
            raise ValueError(f"Invalid lot header: {header.name}")
        tiles, = struct.unpack_from("<I", data, 8)
        pos = 12
        for _ in range(tiles):
            pos = data.index(b"\n", pos) + 1
        cw, ch, lo, hi, rooms = struct.unpack_from("<IIiiI", data, pos)
        if (cw, ch) != (8, 8) or not -32 <= lo <= hi < 32:
            raise ValueError(f"Invalid lot dimensions: {header.name}")
        pos += 20
        for _ in range(rooms):
            pos = data.index(b"\n", pos) + 1
            _, rects = struct.unpack_from("<iI", data, pos)
            pos += 8 + 16 * rects
            objects, = struct.unpack_from("<I", data, pos)
            pos += 4 + 12 * objects
        buildings, = struct.unpack_from("<I", data, pos)
        pos += 4
        for _ in range(buildings):
            count, = struct.unpack_from("<I", data, pos)
            pos += 4
            if count > (len(data) - pos) // 4:
                raise ValueError(f"Invalid building: {header.name}")
            if any(room >= rooms for room in struct.unpack_from(f"<{count}I", data, pos)):
                raise ValueError(f"Invalid room reference: {header.name}")
            pos += 4 * count
        if len(data) - pos != 1024:
            raise ValueError(f"Incomplete lot header: {header.name}")
        hashes[header.name] = hashlib.sha256(data).hexdigest()

        pack = folder / f"world_{xy}.lotpack"
        data = pack.read_bytes()
        if data[:12] != struct.pack("<4sII", b"LOTP", 1, 1024):
            raise ValueError(f"Invalid lot pack: {pack.name}")
        offsets = (*struct.unpack_from("<1024Q", data, 12), len(data))
        if offsets[0] != 12 + 1024 * 8:
            raise ValueError(f"Invalid lot offsets: {pack.name}")
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            if start >= end or end > len(data):
                raise ValueError(f"Invalid lot offsets: {pack.name}")
            pos, squares = start, 0
            while pos < end:
                count, value = struct.unpack_from("<ii", data, pos)
                if count == -1 and value > 0:
                    squares += value
                    pos += 8
                elif 1 <= count <= (end - pos - 4) // 4 and -1 <= value < rooms:
                    ids = struct.unpack_from(f"<{count - 1}I", data, pos + 8)
                    if any(tile >= tiles for tile in ids):
                        raise ValueError(f"Invalid tile reference: {pack.name}")
                    squares += 1
                    pos += 4 + 4 * count
                else:
                    raise ValueError(f"Invalid lot squares: {pack.name}")
            if pos != end or squares != 64 * (hi - lo + 1):
                raise ValueError(f"Incomplete lot chunk: {pack.name}")
        hashes[pack.name] = hashlib.sha256(data).hexdigest()

        chunkdata = folder / f"chunkdata_{xy}.bin"
        data = chunkdata.read_bytes()
        if data[:2] != b"\0\x01":
            raise ValueError(f"Invalid chunk data: {chunkdata.name}")
        pos = 2
        for _ in range(1024):
            kind = data[pos]
            if kind > 5:
                raise ValueError(f"Invalid chunk data: {chunkdata.name}")
            pos += 65 if kind == 2 else 1
        if pos != len(data):
            raise ValueError(f"Incomplete chunk data: {chunkdata.name}")
        hashes[chunkdata.name] = hashlib.sha256(data).hexdigest()
    if not hashes or hashes.keys() != {p.name for p in lot_paths(folder)}:
        raise ValueError("The staged lot files are incomplete.")
    return hashes


def plan(old: dict | None, new: dict, lots_present: bool) -> dict | None:
    """What to throw away, or None when the old record cannot be trusted.

    Returns {"terrain": cells whose .tmx must be made again, "lots": cells
    whose lot files must, "changed": how many cells changed themselves}."""
    if not old or not lots_present:
        return None
    if old.get("global") != new["global"] or old.get("size") != new["size"] \
            or old.get("origin") != new["origin"]:
        return None
    size = (new["size"][0], new["size"][1])
    changed_terrain, changed_any = set(), set()
    for k, h in new["terrain"].items():
        if old.get("terrain", {}).get(k) != h:
            changed_terrain.add(_unkey(k))
            changed_any.add(_unkey(k))
    for k, h in new["content"].items():
        if old.get("content", {}).get(k) != h:
            changed_any.add(_unkey(k))
    return {"terrain": _grow(changed_terrain, size), "lots": _grow(changed_any, size),
            "changed": len(changed_any)}


def discard(project: Path, plan_: dict, origin: tuple[int, int]) -> dict:
    """Delete the lot files the plan says are out of date and, when any ground
    changed, the .tmx maps (all of them: see below), and forget the .tmx in the
    project so WorldEd does not stop on a missing one. Returns how many went."""
    base = _pzw_facts((project / f"{project.name}.pzw").read_text(
        encoding="utf-8", errors="replace"))["base"]
    ox, oy = origin
    lots = project / "lots"
    removed_lots = 0
    for lx, ly in lot_files_of(plan_["lots"], origin):
        for name in (f"{lx}_{ly}.lotheader", f"chunkdata_{lx}_{ly}.bin",
                     f"world_{lx}_{ly}.lotpack"):
            try:
                (lots / name).unlink()
                removed_lots += 1
            except OSError:
                pass
    removed_tmx = 0
    if plan_["terrain"]:
        # All of them, not just the changed cells'. WorldEd's BMP to TMX step
        # does not make up a partial set: with some maps present it leaves the
        # missing ones missing (its log says "source TMX cells 375" for 384),
        # and the lots over those cells are never written. Converting the whole
        # bitmap again is the first batch's job and takes a few minutes; the
        # lots of the cells that did not change are still kept.
        for p in (project / "tmx").glob(f"{base}_*.tmx"):
            try:
                p.unlink()
                removed_tmx += 1
            except OSError:
                pass
        pzw = project / f"{project.name}.pzw"
        text = pzw.read_text(encoding="utf-8", errors="replace")
        new = re.sub(r'(<cell x="\d+" y="\d+" )map="[^"]*\.tmx"', r'\1map=""', text)
        if new != text:
            pzw.write_text(new, encoding="utf-8")
    return {"lots": removed_lots, "tmx": removed_tmx}


def batch_needs_work(project: Path, box: tuple[int, int, int, int],
                     origin: tuple[int, int], size: tuple[int, int]) -> bool:
    """Whether any lot file this batch of cells would write is missing.

    The compiler itself skips a batch whose cells all have a lot header, but
    only after starting, and starting means loading the whole tile catalogue:
    a minute or more for nothing. A batch writes exactly the lot files that
    overlap its own cells (WorldEd's log says "output cells 30" for a 4x4
    batch), so those are the ones looked for.
    """
    x0, y0, x1, y1 = box
    cells = {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)
             if 0 <= x < size[0] and 0 <= y < size[1]}
    lots = project / "lots"
    return any(not (lots / f"{lx}_{ly}.lotheader").exists()
               for lx, ly in lot_files_of(cells, origin))


def origin_of(pzw: Path) -> tuple[int, int]:
    """The world origin the project's lot files are numbered from."""
    return _pzw_facts(pzw.read_text(encoding="utf-8", errors="replace"))["origin"]


def wipe(project: Path) -> None:
    """Throw away everything compiled and the record of it, so the next
    compile starts from nothing."""
    forget(project)
    for folder in ("lots", "tmx"):
        for p in (project / folder).glob("*"):
            if p.is_file():
                try:
                    p.unlink()
                except OSError:
                    pass
    pzw = project / f"{project.name}.pzw"
    if pzw.exists():
        text = pzw.read_text(encoding="utf-8", errors="replace")
        new = re.sub(r'(<cell x="\d+" y="\d+" )map="[^"]*\.tmx"', r'\1map=""', text)
        if new != text:
            pzw.write_text(new, encoding="utf-8")
