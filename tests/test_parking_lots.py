import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
from shapely.geometry import box

from generator import pz_colors as C
from generator import renderer
from generator.osm import OSMFeature, ROAD_HIERARCHY_CODES, classify
from generator.rendering.geometry import _is_polygon
from generator.rendering.outputs import _areas_geojson
from generator.rendering.parking import square_parking_lots
from generator.rendering.projection import Projector
from knoxbuild.areas import AreaIndex
from knoxbuild.parking_osm import ParkingOSM, lot_policy
from knoxbuild.road_vehicles import roadside_bays
from knoxbuild.settings import Settings
from knoxbuild.zones_detect import _detect_zones


class ParkingLotDressingTest(unittest.TestCase):
    def test_capacity_comparison_preserves_source_and_counts_a_multipart_site_once(self):
        def xy_ring(x0, y0, x1, y1):
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]

        rings = [xy_ring(8, 8, 17, 13), xy_ring(25, 8, 34, 13)]
        narrow = xy_ring(43, 10, 47, 42)
        clipped = xy_ring(57, 50, 71, 63)
        features = [
            OSMFeature(61, "relation", {"amenity": "parking", "capacity": "10"}, [],
                       [("outer", ring) for ring in rings]),
            OSMFeature(61, "way", {"amenity": "parking", "capacity": "4"}, narrow),
            OSMFeature(62, "way", {"amenity": "parking", "capacity": "9"}, clipped),
        ]
        proj = SimpleNamespace(width=64, height=80, meters_per_tile=1, to_px=lambda x, y: (x, y))
        report = {}
        with tempfile.TemporaryDirectory() as td:
            geo = _areas_geojson({"parking": features})
            with open(os.path.join(td, "map_areas.geojson"), "w", encoding="utf-8") as out:
                json.dump(geo, out)
            path = os.path.join(td, "map.bmp")
            Image.new("RGB", (64, 80), C.DARK_ASPHALT).save(path)
            _detect_zones(path, [], areas=AreaIndex.load(td, "map", proj), vehicle_report=report,
                          settings=Settings(parking_density=4, vehicle_parking_occupancy=0))

        whole, thin, partial = report["capacity_comparisons"]
        self.assertEqual((whole["osm_kind"], thin["osm_kind"]), ("relation", "way"))
        self.assertEqual(len(whole["part_bounds"]), 2)
        self.assertEqual(whole["reference"], {"source": "OSM", "field": "capacity", "raw": "10", "value": 10})
        self.assertEqual((whole["generated"], whole["delta"], whole["exported"]), (6, -4, 0))
        self.assertEqual((thin["delta"], thin["status"]), (-4, "below_reference"))
        self.assertIn("grid_too_narrow", thin["reasons"])
        self.assertEqual((partial["delta"], partial["status"]), (None, "not_comparable"))
        self.assertIn("scope_outside_map", partial["reasons"])

    def test_legacy_area_reads_capacity_from_the_saved_osm_snapshot(self):
        tags = {"amenity": "parking"}
        proj = SimpleNamespace(width=60, height=60, meters_per_tile=1, to_px=lambda x, y: (x, y))
        policy = ParkingOSM([
            OSMFeature(1, "way", {**tags, "capacity": "0"}, []),
            OSMFeature(2, "way", tags, []),
            OSMFeature(2, "relation", {**tags, "capacity": "25"}, []),
        ], proj)
        areas = SimpleNamespace(_items=[
            (box(8, 8, 13, 16), {"category": "parking", "osm_id": 1}),
            (box(22, 8, 27, 16), {"category": "parking", "osm_id": 2}),
            (box(38, 8, 43, 16), {"category": "parking", "capacity": "6;8"}),
            (box(46, 8, 51, 16), {"category": "parking", "osm_id": 1, "osm_kind": "way"}),
        ])
        report = {}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "map.bmp")
            Image.new("RGB", (60, 60), C.DARK_ASPHALT).save(path)
            _detect_zones(path, [], areas=areas, parking_osm=policy, vehicle_report=report)
        zero, ambiguous, invalid, current = report["capacity_comparisons"]
        self.assertEqual(zero["reference"]["value"], 0)
        self.assertEqual(zero["status"], "above_reference")
        self.assertEqual((ambiguous["status"], ambiguous["delta"]), ("missing_reference", None))
        self.assertEqual((invalid["status"], invalid["reference"]["raw"]), ("invalid_reference", "6;8"))
        self.assertEqual(current["status"], "missing_reference")

    def test_osm_side_tags_follow_way_order_and_keep_restrictions(self):
        proj = SimpleNamespace(width=90, height=50, meters_per_tile=1, to_px=lambda x, y: (x, y))
        tags = {"highway": "residential", "width": "10", "parking:both": "lane",
                "parking:left:restriction": "no_stopping",
                "parking:right:orientation": "perpendicular"}
        for points, forbidden, allowed in (([(0, 25), (89, 25)], 20, 27),
                                            ([(89, 25), (0, 25)], 27, 20)):
            rules = ParkingOSM([OSMFeature(1, "way", tags, points)], proj)
            self.assertEqual(rules.kerb_policy((16, forbidden, 5, 3)), (None, "osm_parking_restriction"))
            self.assertEqual(rules.kerb_policy((16, allowed, 5, 3)), ("perpendicular", None))

    def test_surface_lots_keep_crossing_gate_and_service_lane_clear(self):
        proj = SimpleNamespace(width=100, height=80, meters_per_tile=1, to_px=lambda x, y: (x, y))
        rules = ParkingOSM([
            OSMFeature(1, "way", {"highway": "footway", "footway": "crossing"}, [(38, 5), (38, 70)]),
            OSMFeature(2, "node", {"barrier": "lift_gate"}, [(53, 23)]),
            OSMFeature(3, "way", {"highway": "service", "service": "parking_aisle", "width": "4"},
                       [(62, 5), (62, 70)]),
        ], proj)
        areas = SimpleNamespace(_items=[
            (box(8, 8, 82, 52), {"category": "parking", "orientation": "parallel", "osm_id": 99}),
            (box(8, 54, 78, 75), {"category": "parking", "parking": "multi-storey", "osm_id": 100}),
        ])
        image = Image.new("RGB", (100, 80), C.MEDIUM_GRASS)
        image.paste(C.DARK_ASPHALT, (8, 8, 82, 75))
        report = {}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "map.bmp")
            image.save(path)
            _detect_zones(path, [], areas=areas, parking_osm=rules, vehicle_report=report,
                          settings=Settings(parking_density=2, vehicle_kerb_density=0,
                                            vehicle_yard_density=0, vehicle_jam_density=0))
        self.assertEqual(set(report["clearance_rejections"]), {"crossing", "gate", "service_access"})
        self.assertTrue({"crossing", "gate", "service_access"} <= report["capacity_comparisons"][0]["reasons"].keys())
        self.assertEqual(report["lots"][1]["reason"], "not_surface")
        self.assertTrue(report["parking"])
        self.assertTrue(all((row["width"], row["height"]) == (5, 3) and row["y"] < 52
                            for row in report["parking"]))
        self.assertTrue(all(not rules.restriction(*(row[k] for k in ("x", "y", "width", "height")))
                            for row in report["parking"]))
        self.assertEqual(lot_policy({"access": "private"}), (None, "restricted_access"))
        self.assertEqual(lot_policy({"parking": "street_side"}), ("parallel", None))

    def test_perpendicular_kerb_bays_require_a_remaining_lane(self):
        for width, allowed in ((7, False), (8, True)):
            asphalt = np.zeros((36, 80), dtype=bool)
            asphalt[10:10 + width, 4:76] = True
            report = {"roadside_candidates": 0, "roadside_reserved_for_stories": 0}
            found = list(roadside_bays(asphalt, None, 10, 5, lambda *_: False,
                                      set(), set(), set(), report,
                                      lambda _: ("perpendicular", None)))
            self.assertEqual(bool(found), allowed)
            self.assertTrue(all((w, h) == (3, 5) and y in (10, 13) for _, y, w, h in found))

    def test_each_vehicle_setting_controls_its_own_locations(self):
        lot_box = (8, 8, 62, 50)
        yard_box = (76, 4, 93, 16)
        street_box = (4, 60, 350, 68)
        highway_box = (4, 101, 350, 126)
        land = Image.new("RGB", (360, 144), C.MEDIUM_GRASS)
        for bounds in (lot_box, yard_box, street_box, highway_box):
            land.paste(C.DARK_ASPHALT, bounds)
        hierarchy = np.zeros((144, 360), dtype=np.uint8)
        hierarchy[60:68, 4:350] = ROAD_HIERARCHY_CODES["residential"]
        hierarchy[101:126, 4:350] = ROAD_HIERARCHY_CODES["motorway"]
        areas = SimpleNamespace(_items=[(box(*lot_box), {"category": "parking"})])
        knobs = {"parking_density": lot_box, "vehicle_yard_density": yard_box,
                 "vehicle_kerb_density": street_box, "vehicle_jam_density": highway_box}

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "map.bmp")
            land.save(path)
            for enabled, (x0, y0, x1, y1) in knobs.items():
                with self.subTest(only=enabled):
                    settings = Settings.from_dict({key: 4 if key == enabled else 0 for key in knobs})
                    zones = _detect_zones(path, [], settings=settings, areas=areas,
                                          drives=[(76, 4, 17, 12)], road_hierarchy=hierarchy)
                    stalls = [z for z in zones if z.kind == "ParkingStall"]
                    self.assertTrue(stalls)
                    self.assertTrue(all(x0 <= z.tile_x < x1 and y0 <= z.tile_y < y1 for z in stalls))

    def test_old_density_applies_to_each_new_control_unless_overridden(self):
        settings = Settings.from_dict({"preset": "suburb", "parking_density": 0.45,
                                       "vehicle_jam_density": 0})
        values = settings.to_dict()
        self.assertEqual(values["vehicle_yard_density"], 0.45)
        self.assertEqual(values["vehicle_kerb_density"], 0.45)
        self.assertEqual(values["vehicle_jam_density"], 0)
        self.assertEqual(values["vehicle_parking_occupancy"], 100)
        self.assertEqual(Settings.from_dict(values).to_dict(), values)

    def test_target_occupancy_filters_export_without_moving_or_erasing_stalls(self):
        lot = (8, 8, 17, 13)  # Three marked bays, so a 50% target rounds up to two.
        street = (4, 24, 160, 32)
        driveway = (8, 40, 13, 43)
        areas = SimpleNamespace(_items=[(box(*lot), {"category": "parking", "osm_id": 5})])
        land = Image.new("RGB", (170, 50), C.MEDIUM_GRASS)
        land.paste(C.DARK_ASPHALT, lot)
        land.paste(C.DARK_ASPHALT, street)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "map.bmp")
            veg = os.path.join(td, "veg.bmp")
            land.save(path)

            def build(percent, seed=1):
                Image.new("RGB", land.size, C.VEG_NOTHING).save(veg)
                report = {}
                settings = Settings.from_dict({"parking_density": 4, "vehicle_yard_density": 4,
                                               "vehicle_parking_occupancy": percent, "seed": seed})
                zones = _detect_zones(path, [], areas=areas, settings=settings, drives=[(8, 40, 5, 3)],
                                      veg_path=veg, vehicle_report=report)
                with Image.open(veg) as image:
                    markings = image.tobytes()
                return report, zones, markings

            full, _, full_markings = build(100)
            empty, empty_zones, empty_markings = build(0)
            half, half_zones, half_markings = build(50)
            repeat, repeat_zones, _ = build(50)
            other_seed, _, _ = build(50, seed=2)

        self.assertEqual(half, repeat)
        self.assertEqual(half_zones, repeat_zones)
        self.assertNotEqual(half["roadside"], other_seed["roadside"])
        self.assertEqual(half_markings, empty_markings)
        self.assertEqual(full_markings, empty_markings)
        self.assertNotEqual(full_markings, bytes(C.VEG_NOTHING) * (170 * 50))
        self.assertEqual([(z.tile_x, z.tile_y, z.tile_x + z.width, z.tile_y + z.height)
                          for z in empty_zones if z.kind == "ParkingStall"], [driveway])
        self.assertEqual(half["lots"][0]["eligible"], 3)
        self.assertEqual(half["lots"][0]["target"], 2)
        self.assertEqual(half["counts"]["mapped_parking"], 2)
        self.assertTrue(half["roadside"])
        self.assertTrue(all(row in full["roadside"] for row in half["roadside"]))
        self.assertEqual(half["occupancy"]["roadside"]["eligible"], len(full["roadside"]))
        self.assertIsNone(half["occupancy"]["actual"])
        self.assertFalse(empty["roadside"])

    def test_osm_parking_lot_gets_stall_lines_and_wheel_stops(self):
        w, h = 60, 60
        with tempfile.TemporaryDirectory() as td:
            lp = os.path.join(td, "map.bmp")
            vp = os.path.join(td, "map_veg.bmp")
            land = Image.new("RGB", (w, h), C.DARK_ASPHALT)
            veg = Image.new("RGB", (w, h), C.VEG_NOTHING)
            # Pre-existing tree pixel inside the lot should be preserved.
            veg.putpixel((12, 12), C.TREES)
            land.save(lp, format="BMP")
            veg.save(vp, format="BMP")

            lot = box(6, 6, 42, 30)
            areas = SimpleNamespace(_items=[(lot, {"category": "parking"})])

            with mock.patch("knoxbuild.yards._rule_ready", return_value=True):
                zones = _detect_zones(lp, [], areas=areas, veg_path=vp)
            stalls = [z for z in zones if z.kind == "ParkingStall"]
            self.assertGreater(len(stalls), 0)

            with Image.open(vp) as out_img:
                out = np.asarray(out_img.convert("RGB"))

            self.assertEqual(tuple(out[12, 12]), C.TREES)

            has_block = np.any(
                np.all(out == C.PARKING_BLOCK_X[0], axis=2)
                | np.all(out == C.PARKING_BLOCK_Y[0], axis=2)
            )
            has_line = np.any(
                np.all(out == C.EDGE_LINE_W, axis=2)
                | np.all(out == C.EDGE_LINE_N, axis=2)
            )
            has_island_kerb = np.any(np.all(out == C.KERB_NW, axis=2))
            has_lamp = np.any(
                np.all(out == C.LAMP_S, axis=2)
                | np.all(out == C.LAMP_E, axis=2)
            )
            has_greenery = np.any(
                np.all(out == C.BUSHES, axis=2)
                | np.all(out == C.TREE_SMALL, axis=2)
            )
            self.assertTrue(has_block)
            self.assertTrue(has_line)
            self.assertTrue(has_island_kerb)
            self.assertTrue(has_lamp)
            self.assertTrue(has_greenery)

            # No ParkingStall zone should overlap a divider island's curb or lamp.
            for z in stalls:
                patch = out[z.tile_y:z.tile_y + z.height, z.tile_x:z.tile_x + z.width]
                self.assertFalse(np.any(np.all(patch == C.KERB_NW, axis=2)))
                self.assertFalse(np.any(np.all(patch == C.LAMP_S, axis=2)))
                self.assertFalse(np.any(np.all(patch == C.LAMP_E, axis=2)))

    def test_medium_parking_lot_gets_edge_light_pole(self):
        w, h = 40, 40
        with tempfile.TemporaryDirectory() as td:
            lp = os.path.join(td, "map.bmp")
            vp = os.path.join(td, "map_veg.bmp")
            Image.new("RGB", (w, h), C.DARK_ASPHALT).save(lp, format="BMP")
            Image.new("RGB", (w, h), C.VEG_NOTHING).save(vp, format="BMP")

            # 4 stalls wide -> no divider island, should still get an edge light pole.
            lot = box(6, 6, 20, 18)
            areas = SimpleNamespace(_items=[(lot, {"category": "parking"})])

            _detect_zones(lp, [], areas=areas, veg_path=vp)
            with Image.open(vp) as out_img:
                out = np.asarray(out_img.convert("RGB"))

            has_lamp = np.any(
                np.all(out == C.LAMP_S, axis=2)
                | np.all(out == C.LAMP_E, axis=2)
            )
            self.assertTrue(has_lamp)


class ParkingLotSquaringTest(unittest.TestCase):
    SOUTH, WEST, NORTH, EAST = 40.0, -85.0, 40.0027, -84.9965

    @staticmethod
    def _ring_latlon(proj: Projector, pts_px: list[tuple[float, float]]):
        ring = [proj.to_latlon(x, y) for x, y in pts_px]
        ring.append(ring[0])
        return ring

    def test_angled_splotch_parking_renders_as_orthogonal_rectangle(self):
        proj = Projector.build(self.SOUTH, self.WEST, self.NORTH, self.EAST, 1.0, 0.0)
        # 28-degree rotated blobby 36x22 parking lot around (120, 120).
        cx, cy = 120.0, 120.0
        angle = math.radians(28)
        ca, sa = math.cos(angle), math.sin(angle)
        local_blob = [
            (-18, -10), (-4, -13), (12, -11), (20, -6),
            (17, 10), (3, 12), (-14, 9), (-21, 1),
        ]
        pts_px = [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in local_blob]
        lot = OSMFeature(10, "way", {"amenity": "parking"}, self._ring_latlon(proj, pts_px))
        road = OSMFeature(11, "way", {"highway": "residential"},
                          [proj.to_latlon(165, 20), proj.to_latlon(165, 220)])
        bldg_pts = [(105, 160), (145, 160), (145, 185), (105, 185)]
        bldg = OSMFeature(12, "way", {"building": "retail"}, self._ring_latlon(proj, bldg_pts))

        with tempfile.TemporaryDirectory() as out:
            res = renderer.render([lot, road, bldg],
                                  self.SOUTH, self.WEST, self.NORTH, self.EAST,
                                  meters_per_tile=1.0, output_dir=out, map_name="lot")
            with open(os.path.join(out, "lot_areas.geojson"), encoding="utf-8") as f:
                areas = json.load(f)
            parking_feats = [ft for ft in areas["features"]
                             if ft["properties"].get("category") == "parking"]
            self.assertEqual(len(parking_feats), 1)
            coords = parking_feats[0]["geometry"]["coordinates"][0][0]
            px_coords = [proj.to_px(lat, lon) for lon, lat in coords]
            self.assertEqual(len(px_coords), 5)
            for (x0, y0), (x1, y1) in zip(px_coords, px_coords[1:]):
                self.assertTrue(abs(x1 - x0) < 1e-3 or abs(y1 - y0) < 1e-3,
                                f"non-orthogonal parking edge {(x0, y0)} -> {(x1, y1)}")

            land = np.asarray(Image.open(res.landscape_path).convert("RGB"))
            # Interior is painted DARK_ASPHALT while sidewalk along the road stays intact.
            self.assertEqual(tuple(land[120, 120]), C.DARK_ASPHALT)
            self.assertIn(C.PALE_CONCRETE, [tuple(land[120, x]) for x in range(158, 165)])

    def test_l_shaped_parking_keeps_orthogonal_arms_without_swallowing_building(self):
        proj = Projector.build(self.SOUTH, self.WEST, self.NORTH, self.EAST, 1.0, 0.0)
        # Slightly ragged L-shape hugging north and west sides of an inner courtyard.
        l_pts = [
            (40, 40), (95, 42), (96, 63), (62, 62),
            (61, 98), (39, 97), (40, 62),
        ]
        lot = OSMFeature(20, "way", {"amenity": "parking"}, self._ring_latlon(proj, l_pts))
        courtyard_bldg = OSMFeature(
            21, "way", {"building": "yes"},
            self._ring_latlon(proj, [(65, 65), (95, 65), (95, 95), (65, 95)]),
        )
        feats = [lot, courtyard_bldg]
        changed = square_parking_lots(feats, proj, classify, _is_polygon)
        self.assertEqual(changed, 1)
        px_rings = [[proj.to_px(la, lo) for la, lo in ring]
                    for role, ring in (feats[0].role_geoms or [("outer", feats[0].geometry)])
                    if role == "outer"]
        self.assertGreater(len(px_rings[0]), 5)
        for ring in px_rings:
            for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
                self.assertTrue(abs(x1 - x0) < 1e-3 or abs(y1 - y0) < 1e-3)
        from shapely.geometry import Point, Polygon
        poly = Polygon(px_rings[0])
        self.assertFalse(poly.contains(Point(80, 80)))

    def test_squaring_preserves_public_sidewalk_under_overlapping_lot(self):
        proj = Projector.build(self.SOUTH, self.WEST, self.NORTH, self.EAST, 1.0, 0.0)
        lot_pts = [(80, 80), (140, 82), (138, 120), (78, 118)]
        lot = OSMFeature(30, "way", {"amenity": "parking"}, self._ring_latlon(proj, lot_pts))
        road = OSMFeature(31, "way", {"highway": "secondary"},
                          [proj.to_latlon(135, 60), proj.to_latlon(135, 140)])
        with tempfile.TemporaryDirectory() as out:
            res = renderer.render([lot, road],
                                  self.SOUTH, self.WEST, self.NORTH, self.EAST,
                                  meters_per_tile=1.0, output_dir=out, map_name="sw")
            land = np.asarray(Image.open(res.landscape_path).convert("RGB"))
            # Road is centered at x=135; its west sidewalk should not be eaten by parking asphalt.
            west_row = [tuple(land[100, x]) for x in range(125, 136)]
            self.assertIn(C.PALE_CONCRETE, west_row)


if __name__ == "__main__":
    unittest.main()
