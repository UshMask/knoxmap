"""Tests for foraging and navigation zones in objects.lua / .pzw."""
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from generator import pz_colors as C
from generator.osm import ROAD_HIERARCHY_CODES
from knoxbuild.settings import Settings
from knoxbuild.world import render_pzw
from knoxbuild.zones_detect import _detect_zones
from tools.make_map_mod import write_objects


class ForageAndNavZoneTests(unittest.TestCase):
    def test_road_story_zones_follow_straight_carriageways(self):
        land = np.full((140, 120, 3), C.MEDIUM_GRASS, dtype=np.uint8)
        ranks = np.zeros(land.shape[:2], dtype=np.uint8)
        for region, tier in (
            (np.s_[10:19, :120], "residential"),
            (np.s_[50:130, 10:18], "secondary"),
            (np.s_[70:78, 30:110], "footway"),
        ):
            land[region] = C.MEDIUM_ASPHALT
            ranks[region] = ROAD_HIERARCHY_CODES[tier]

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "land.bmp")
            Image.fromarray(land, "RGB").save(path)
            zones = _detect_zones(path, [], road_hierarchy=ranks,
                                  keep_clear=[(45, 8, 30, 12)],
                                  settings=Settings(vehicle_jam_density=0))

        stories = [(z.tile_x, z.tile_y, z.width, z.height) for z in zones
                   if z.kind == "Nav" and min(z.width, z.height) < 15
                   and max(z.width, z.height) > 30]
        self.assertEqual(stories, [(0, 10, 40, 9), (80, 10, 40, 9), (10, 60, 8, 40)])

    def test_detects_all_forage_and_nav_zone_kinds(self):
        w, h = 100, 100
        land = np.full((h, w, 3), C.MEDIUM_GRASS, dtype=np.uint8)
        veg = np.full((h, w, 3), C.VEG_NOTHING, dtype=np.uint8)

        # Row 0..20: asphalt road -> Nav
        land[0:20, 0:60] = C.MEDIUM_ASPHALT
        # Row 20..40: dense trees -> DeepForest
        veg[20:40, 0:60] = C.TREES
        # Row 40..60: sparse/medium trees -> Forest
        veg[40:60, 0:40] = C.SPARSE_TREES
        # Row 60..80: farmland -> Farm
        land[60:80, 0:60] = C.DIRT_ROWS_H
        # Row 80..100: open grass -> Vegitation

        with tempfile.TemporaryDirectory() as td:
            lp = os.path.join(td, "land.bmp")
            vp = os.path.join(td, "veg.bmp")
            Image.fromarray(land, "RGB").save(lp)
            Image.fromarray(veg, "RGB").save(vp)
            zones = _detect_zones(lp, [], veg_path=vp)

        kinds = {z.kind for z in zones}
        self.assertTrue({"Nav", "DeepForest", "Forest", "Farm", "Vegitation"}.issubset(kinds), kinds)
        # Adjacent 20x20 Nav blocks (0..60) merge into one 60x20 zone.
        nav = [z for z in zones if z.kind == "Nav" and z.tile_y == 0]
        self.assertEqual(len(nav), 1)
        self.assertEqual((nav[0].tile_x, nav[0].width, nav[0].height), (0, 60, 20))

    def test_zones_stay_within_cell_and_export_to_objects_lua(self):
        w, h = 340, 40
        land = np.full((h, w, 3), C.MEDIUM_ASPHALT, dtype=np.uint8)
        veg = np.full((h, w, 3), C.VEG_NOTHING, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            lp = os.path.join(td, "land.bmp")
            vp = os.path.join(td, "veg.bmp")
            Image.fromarray(land, "RGB").save(lp)
            Image.fromarray(veg, "RGB").save(vp)
            zones = _detect_zones(lp, [], veg_path=vp)
            nav_zones = [z for z in zones if z.kind == "Nav"]
            for z in nav_zones:
                self.assertEqual(z.tile_x // 300, (z.tile_x + z.width - 1) // 300)

            pzw_path = os.path.join(td, "test.pzw")
            with open(pzw_path, "w", encoding="utf-8") as f:
                f.write(render_pzw(2, 1, "land.bmp", [], zones=zones))
            counts = write_objects(td, td)
            self.assertGreater(counts.get("Nav", 0), 0)
            with open(os.path.join(td, "objects.lua"), encoding="utf-8") as f:
                lua = f.read()
            self.assertIn('type = "Nav"', lua)
