import random
import unittest

import numpy as np

from knoxbuild.apartment_sections import plan_sections
from knoxbuild.audit import audit
from knoxbuild.basements import _run, plan_basement, reserve
from knoxbuild.buildheader import placed_rooms
from knoxbuild.buildtiles import parse_tbx_rooms
from knoxbuild.footprint import place
from knoxbuild.layout import build_building
from knoxbuild.layout.openings import _room_at
from knoxbuild.settings import Settings
from knoxbuild.street_context import _party_walls
from knoxbuild.tbx import render_tbx


def block(points=((10, 10), (83, 10), (83, 24), (10, 24))):
    fp, reason = place(list(points), np.zeros((120, 120), dtype=bool), alignment="real")
    assert fp is not None, reason
    return fp


def hall_distance(floor, start, goal, blocked):
    hall = {(x, y) for y, row in enumerate(floor.grid) for x, r in enumerate(row)
            if r and floor.rooms[r - 1].is_core} - blocked
    frontier, seen, steps = {start} & hall, {start}, 0
    while frontier:
        if goal in frontier:
            return steps
        frontier = {(x + dx, y + dy) for x, y in frontier
                    for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0))} & (hall - seen)
        seen.update(frontier)
        steps += 1
    return None


class ApartmentSections(unittest.TestCase):
    def test_estimated_sections_have_independent_public_stairs(self):
        fp = block()
        self.assertEqual(plan_sections(fp, [], "N", 1.0), ([], ""))
        sections, source = plan_sections(fp, [], "N", 1.0, estimate=True)
        self.assertEqual((len(sections), source), (4, "estimated"))
        self.assertEqual(sum(s.footprint.tiles for s in sections), fp.tiles)
        owner = np.full((120, 120), -1)
        for i, section in enumerate(sections):
            part = section.footprint
            owner[part.y0:part.y0 + part.height,
                  part.x0:part.x0 + part.width][part.mask] = i
        lots = []
        for i, section in enumerate(sections):
            part = section.footprint
            party = _party_walls(owner, i, part.x0, part.y0, part.mask, [5] * 4)
            building = build_building(part.width, part.height, 5, kind="apartment",
                                      mask=part.mask_list(), party=party, seed=558 + i,
                                      section=(section.core, section.door))
            self.assertEqual(len(building.stairs), 4)
            self.assertEqual([d for _, _, d in building.stairs], ["N", "W", "N", "W"])
            for z, (below, above) in enumerate(zip(building.stairs, building.stairs[1:]), 1):
                incoming, outgoing = _run(*below), _run(*above)
                self.assertTrue(set(incoming).isdisjoint(outgoing))
                distance = hall_distance(building.storeys[z], incoming[0], outgoing[-1],
                                         set(incoming[1:-1] + outgoing[1:-1]))
                self.assertIn(distance, range(1, 7))
            for level, floor in enumerate(building.storeys):
                self.assertFalse(set(floor.doors) & party.keys())
                outside = {}
                for x, y, direction in floor.doors:
                    a = _room_at(floor, x - (direction == "W"), y - (direction == "N"))
                    b = _room_at(floor, x, y)
                    if 0 in (a, b):
                        outside[x, y, direction] = max(a, b)
                self.assertEqual(list(outside), [section.door] if level == 0 else [])
                if outside:
                    inside = outside[section.door]
                    self.assertTrue(floor.rooms[inside - 1].is_core)
            problems = audit(building, "apartment")
            for problem in ("room with no way in", "flat with no front door",
                            "flat with several front doors", "no way in from outside"):
                self.assertNotIn(problem, problems)
            lots.append((parse_tbx_rooms(render_tbx(building, "Apartment section")),
                         part.x0, part.y0, 0))
        self.assertEqual(len(placed_rooms(lots)), 1)

    def test_basement_has_one_door_from_the_hall_and_leaves_the_upstairs_route(self):
        section = plan_sections(block(), [], "N", 1.0, estimate=True)[0][0]
        fp = section.footprint
        settings = Settings(basement_share=1)
        spec = reserve([dict(kind="apartment", x0=fp.x0, y0=fp.y0,
                             w=fp.width, h=fp.height, mask=fp.mask)], settings,
                       None, None, random.Random(1), (120, 120))[0]
        building = build_building(fp.width, fp.height, 6, kind="apartment",
                                  settings=settings, section=(section.core, section.door))
        ground = building.storeys[0]
        before = set(ground.doors)
        basement = plan_basement(building, spec, 0, settings, public_only=True)
        self.assertIsNotNone(basement)
        (x, y, direction), = set(ground.doors) - before
        a, b = _room_at(ground, x - (direction == "W"), y - (direction == "N")), _room_at(ground, x, y)
        self.assertTrue(a and b and a != b)
        self.assertTrue(ground.rooms[a - 1].is_core != ground.rooms[b - 1].is_core)
        self.assertEqual((ground.rooms[a - 1].unit, ground.rooms[b - 1].unit), (0, 0))
        outgoing = _run(*building.stairs[0])
        self.assertTrue(ground.holes.isdisjoint(outgoing))
        self.assertIsNotNone(hall_distance(ground, section.door[:2], outgoing[-1],
                                          ground.holes | set(outgoing[1:-1])))

    def test_mapped_entrances_control_boundaries_and_door_positions(self):
        fp = block()
        doors = [(20, 10, {"entrance": "staircase", "ref": "1"}),
                 (44, 10, {"entrance": "staircase", "ref": "2"}),
                 (69, 10, {"entrance": "staircase", "ref": "3"}),
                 (30, 10, {"entrance": "service"})]
        sections, source = plan_sections(fp, doors, "S", 1.0)
        self.assertEqual(source, "mapped")
        self.assertEqual(len(sections), 3)
        for section, (x, y, _) in zip(sections, doors):
            door_x, door_y, direction = section.door
            self.assertEqual(direction, "N")
            self.assertLessEqual(abs(door_x + section.footprint.x0 + 0.5 - x), 1)
            self.assertEqual(door_y + section.footprint.y0, y)

    def test_square_tower_and_ambiguous_entrance_keep_existing_layout(self):
        tower = block(((10, 10), (45, 10), (45, 45), (10, 45)))
        self.assertEqual(plan_sections(tower, [], "N", 1.0, estimate=True), ([], ""))
        ambiguous = [(20, 10, {"entrance": "yes"})]
        self.assertEqual(plan_sections(block(), ambiguous, "N", 1.0, estimate=True), ([], ""))
