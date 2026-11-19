import unittest
from unittest import mock

from PIL import Image

from generator import pz_colors as C
from generator import renderer
from generator.osm import OSMFeature
from generator.rendering import furniture
from knoxbuild.settings import StreetSettings


class TileProjector:
    width = height = 40
    meters_per_tile = 1.0

    @staticmethod
    def to_px(x, y):
        return x, y


def osm_node(osm_id, x, y, **tags):
    return OSMFeature(osm_id, "node", tags, [(x, y)])


class RoadFurniturePlacement(unittest.TestCase):
    def test_mapped_bin_snaps_off_the_pedestrian_line(self):
        import numpy as np

        vegetation, ground = self.make_road(C.DARK_GRASS)
        passage = np.zeros((40, 40), dtype=bool)
        passage[:, 13] = True
        report = []
        renderer._paint_street_furniture(
            vegetation, ground, proj=TileProjector(), passage_mask=passage, bin_report=report,
            features=[osm_node(1, 13, 9, amenity="waste_basket")])
        self.assertEqual(report[0]["tile"], [14, 9])
        self.assertNotEqual(vegetation.getpixel((13, 9)), C.BIN)
        self.assertEqual(vegetation.getpixel((14, 9)), C.BIN)

    def make_road(self, roadside):
        ground = Image.new("RGB", (40, 40), C.DARK_GRASS)
        pixels = ground.load()
        for y in range(40):
            for x in range(8, 13):
                pixels[x, y] = C.MEDIUM_ASPHALT
            pixels[13, y] = C.PALE_CONCRETE
            pixels[14, y] = roadside
        vegetation = Image.new("RGB", ground.size, C.VEG_NOTHING)
        vegetation.putpixel((12, 20), C.EDGE_LINE_E)
        return vegetation, ground

    def test_signs_and_lamps_skip_asphalt_destination(self):
        for mod_tiles, prop in ((True, "speed"), (False, "lamp")):
            with self.subTest(prop=prop):
                vegetation, ground = self.make_road(C.DARK_ASPHALT)
                road_tags = Image.new("L", ground.size, 25) if mod_tiles else None
                with mock.patch.object(furniture, "_mod_tiles_ready", return_value=mod_tiles):
                    renderer._paint_street_furniture(vegetation, ground, road_furniture=road_tags)
                blocked_color = (C.SPEED_SIGNS[(25, "S")] if prop == "speed"
                                 else C.LAMP_W)
                self.assertNotEqual(vegetation.getpixel((14, 20)), blocked_color)

    def test_lamp_can_use_a_grass_verge(self):
        vegetation, ground = self.make_road(C.DARK_GRASS)
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=False):
            counts = renderer._paint_street_furniture(vegetation, ground)
        self.assertGreater(counts.get("lamp", 0), 0)

    def road_tags(self, **tags):
        way = OSMFeature(1, "way", {"highway": "residential", "width": "5", **tags},
                         [(10, 0), (10, 39)])
        return furniture._road_furniture_mask(
            {"road_minor": [way]}, TileProjector(), lambda feat: [feat.geometry])

    def test_speed_signs_use_supported_osm_values(self):
        for tags, expected in (({"maxspeed": "35 mph"}, 35),
                               ({"maxspeed": "35"}, None),
                               ({"maxspeed": "RU:urban"}, None),
                               ({"maxspeed": "30 mph"}, None),
                               ({"maxspeed": "35 mph", "maxspeed:forward": "25 mph"}, None),
                               ({}, None)):
            with self.subTest(tags=tags):
                vegetation, ground = self.make_road(C.DARK_GRASS)
                with mock.patch.object(furniture, "_mod_tiles_ready", return_value=True):
                    renderer._paint_street_furniture(
                        vegetation, ground, road_furniture=self.road_tags(**tags))
                signs = set(vegetation.get_flattened_data()) & set(C.SPEED_SIGNS.values())
                self.assertEqual(signs, {C.SPEED_SIGNS[(expected, "S")]} if expected else set())

    def test_unlit_road_keeps_only_the_mapped_lamp(self):
        vegetation, ground = self.make_road(C.DARK_GRASS)
        counts = renderer._paint_street_furniture(
            vegetation, ground, road_furniture=self.road_tags(lit="no"),
            features=[osm_node(2, 13, 20, highway="street_lamp")], proj=TileProjector())
        self.assertEqual(counts.get("lamp"), 1)
        self.assertEqual(vegetation.getpixel((13, 20)), C.LAMP_W)

    def test_lighting_is_inferred_near_buildings_or_from_lit_yes(self):
        no_buildings = Image.new("L", (40, 40))
        for tags, expected in (({}, False), ({"lit": "yes"}, True)):
            with self.subTest(tags=tags):
                vegetation, ground = self.make_road(C.DARK_GRASS)
                counts = renderer._paint_street_furniture(
                    vegetation, ground, buildings=no_buildings, road_furniture=self.road_tags(**tags))
                self.assertEqual(bool(counts.get("lamp", 0)), expected)

    def test_generated_lamps_leave_space_around_mapped_lamps(self):
        vegetation, ground = self.make_road(C.DARK_GRASS)
        renderer._paint_street_furniture(
            vegetation, ground, features=[osm_node(2, 13, 20, highway="street_lamp")],
            proj=TileProjector())
        lamps = {C.LAMP_W, C.LAMP_E, C.LAMP_N, C.LAMP_S}
        nearby = {(x, y) for x in range(40) for y in range(40)
                  if vegetation.getpixel((x, y)) in lamps
                  and (x - 13) ** 2 + (y - 20) ** 2 < furniture.FURNITURE_CLEARANCE ** 2}
        self.assertEqual(nearby, {(13, 20)})

    def test_kiosks_and_postboxes_require_mapped_nodes_and_enabled_settings(self):
        nodes = [
            osm_node(1, 13, 8, shop="kiosk"),
            osm_node(2, 13, 30, amenity="post_box"),
        ]
        for features, enabled, expected in (
                ([], 1, 0), (nodes, 1, 1), (nodes, 0, 0)):
            with self.subTest(mapped=bool(features), enabled=enabled):
                vegetation, ground = self.make_road(C.DARK_GRASS)
                with mock.patch.object(furniture, "_rule_ready", return_value=True):
                    counts = renderer._paint_street_furniture(
                        vegetation, ground, features=features, proj=TileProjector(),
                        settings=StreetSettings(street_kiosks=enabled, street_postboxes=enabled))
                self.assertEqual(counts.get("kiosk", 0), expected)
                self.assertEqual(counts.get("postbox", 0), expected)
                if expected:
                    self.assertEqual(vegetation.getpixel((13, 8)), C.KIOSK_W)
                    self.assertEqual(vegetation.getpixel((13, 30)), C.POSTBOX_W)

    def test_lamp_modes_and_zero_density_keep_mapped_points_when_enabled(self):
        for mode, density, count in (("osm", 3.0, 1), ("context", 0.0, 1), ("off", 3.0, 0)):
            with self.subTest(mode=mode, density=density):
                vegetation, ground = self.make_road(C.DARK_GRASS)
                counts = renderer._paint_street_furniture(
                    vegetation, ground, proj=TileProjector(), road_furniture=self.road_tags(lit="yes"),
                    features=[osm_node(2, 13, 20, highway="street_lamp")],
                    settings=StreetSettings(street_lamp_mode=mode, street_lamp_density=density))
                self.assertEqual(counts.get("lamp", 0), count)


class RoadsideProps(unittest.TestCase):
    """Procedural roadside props and objects placed from mapped points."""

    def long_road(self, length=180):
        ground = Image.new("RGB", (length, 24), C.DARK_GRASS)
        pixels = ground.load()
        for y in range(24):
            for x in range(4, 12):
                pixels[x, y] = C.MEDIUM_ASPHALT
            pixels[12, y] = C.PALE_CONCRETE
            pixels[13, y] = C.DARK_GRASS
        vegetation = Image.new("RGB", ground.size, C.VEG_NOTHING)
        for y in range(24):
            vegetation.putpixel((11, y), C.EDGE_LINE_E)
        return vegetation, ground

    def buildings_near(self, size, box):
        mask = Image.new("L", size, 0)
        mask.paste(255, box)
        return mask

    def test_utility_poles_line_a_street_once_the_rules_carry_them(self):
        vegetation, ground = self.long_road()
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=False), \
                mock.patch.object(furniture, "_rule_ready", return_value=True):
            counts = renderer._paint_street_furniture(vegetation, ground)
        self.assertGreater(counts.get("pole", 0), 0)
        colours = {vegetation.getpixel((x, y)) for x in range(ground.width)
                   for y in range(ground.height)}
        self.assertIn(C.UTILITY_POLE, colours)

    def test_without_the_pole_rule_no_poles_are_painted(self):
        # A colour with no rule behind it is dropped with a warning, so an
        # install set up before the poles must not get the colour at all.
        vegetation, ground = self.long_road()
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=False), \
                mock.patch.object(furniture, "_rule_ready", return_value=False):
            counts = renderer._paint_street_furniture(vegetation, ground)
        self.assertEqual(counts.get("pole", 0), 0)
        colours = {vegetation.getpixel((x, y)) for x in range(ground.width)
                   for y in range(ground.height)}
        self.assertNotIn(C.UTILITY_POLE, colours)

    def test_no_patio_tables_on_the_roadside(self):
        vegetation, ground = self.long_road()
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=False):
            renderer._paint_street_furniture(
                vegetation, ground, None,
                self.buildings_near(ground.size, (10, 8, 60, 16)))
        colours = {vegetation.getpixel((x, y)) for x in range(ground.width)
                   for y in range(ground.height)}
        for tile in (C.TABLE_X0, C.TABLE_X1, C.TABLE_Y0, C.TABLE_Y1):
            self.assertNotIn(tile, colours, "a bench is still being laid")

    def test_benches_stand_along_the_kerb_once_the_rules_carry_them(self):
        vegetation, ground = self.long_road()
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=False), \
                mock.patch.object(furniture, "_rule_ready", return_value=True):
            counts = renderer._paint_street_furniture(
                vegetation, ground, None,
                self.buildings_near(ground.size, (10, 0, 60, 24)))
        self.assertGreater(counts.get("bench", 0), 0)
        # The road runs north-south, so the bench lies along y: two squares,
        # the catalogue's orient W pair (0,0 then 0,1).
        heads = [(x, y) for x in range(ground.width) for y in range(ground.height - 1)
                 if vegetation.getpixel((x, y)) == C.BENCH_Y[0]]
        self.assertTrue(heads, "no bench was laid")
        for x, y in heads:
            with self.subTest(at=(x, y)):
                self.assertEqual(vegetation.getpixel((x, y + 1)), C.BENCH_Y[1])

    def test_a_bench_never_stands_on_a_road_at_an_angle(self):
        """An angled road is a staircase of kerbs; a seat laid along one step
        hangs off the pavement, so benches are for straight kerbs only."""
        length, height = 180, 120
        ground = Image.new("RGB", (length, height), C.DARK_GRASS)
        pixels = ground.load()
        for y in range(height):
            shift = y                      # one square across per square down
            for x in range(4 + shift, 12 + shift):
                if x < length:
                    pixels[x, y] = C.MEDIUM_ASPHALT
            if 12 + shift < length:
                pixels[12 + shift, y] = C.PALE_CONCRETE
        vegetation = Image.new("RGB", ground.size, C.VEG_NOTHING)
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=False), \
                mock.patch.object(furniture, "_rule_ready", return_value=True):
            counts = renderer._paint_street_furniture(
                vegetation, ground, None,
                self.buildings_near(ground.size, (10, 0, 120, 120)))
        self.assertEqual(counts.get("bench", 0), 0,
                         "a bench was laid along an angled road")
        self.assertFalse([1 for x in range(ground.width) for y in range(ground.height)
                          if vegetation.getpixel((x, y)) in {*C.BENCH_Y, *C.BENCH_X}])

    def test_the_kiosk_and_collection_box_follow_osm_points(self):
        vegetation, ground = self.long_road()
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=False), \
                mock.patch.object(furniture, "_rule_ready", return_value=True):
            counts = renderer._paint_street_furniture(
                vegetation, ground, proj=TileProjector(), features=[
                    osm_node(1, 13, 10, shop="kiosk"),
                    osm_node(2, 13, 18, amenity="post_box"),
                    osm_node(3, 100, 18, amenity="post_office"),
                ])
        self.assertEqual(counts.get("kiosk"), 1)
        self.assertEqual(counts.get("postbox"), 1)
        self.assertEqual(vegetation.getpixel((13, 10)), C.KIOSK_W)
        self.assertEqual(vegetation.getpixel((13, 18)), C.POSTBOX_W)

    def test_mapped_props_skip_buildings_and_snap_off_the_road(self):
        vegetation, ground = self.long_road()
        buildings = self.buildings_near(ground.size, (20, 10, 23, 14))
        with mock.patch.object(furniture, "_rule_ready", return_value=True):
            counts = renderer._paint_street_furniture(
                vegetation, ground, buildings=buildings, proj=TileProjector(), features=[
                    osm_node(1, 21, 11, shop="kiosk"),
                    osm_node(2, 10, 18, amenity="post_box"),
                ])
        self.assertEqual(counts.get("kiosk", 0), 0)
        self.assertEqual(vegetation.getpixel((12, 18)), C.POSTBOX_W)

    def test_bins_follow_explicit_points_and_stops_with_pavement(self):
        vegetation, ground = self.long_road(length=80)
        # A separate platform well away from the street.
        ground.paste(C.PALE_CONCRETE, (40, 5, 45, 12))
        report = []
        counts = renderer._paint_street_furniture(
            vegetation, ground, proj=TileProjector(), bin_report=report, features=[
                osm_node(1, 13, 4, amenity="waste_basket"),
                osm_node(2, 12, 6, highway="bus_stop"),
                osm_node(3, 41, 6, public_transport="platform", bus="yes"),
                osm_node(4, 42, 6, highway="bus_stop"),
                osm_node(5, 70, 8, highway="bus_stop"),
            ])
        self.assertEqual(counts.get("bin"), 2)
        self.assertEqual(vegetation.getpixel((13, 4)), C.BIN)
        self.assertNotEqual(vegetation.getpixel((41, 6)), C.BIN)
        self.assertEqual(vegetation.getpixel((39, 6)), C.BIN)
        self.assertEqual([(e["osm_id"], e["source"], e["reason"]) for e in report], [
            (1, "osm_waste_basket", "mapped"),
            (2, "osm_bus_stop_inferred", "near_existing_bin"),
            (3, "osm_bus_stop_inferred", "verge"),
            (4, "osm_bus_stop_inferred", "near_existing_bin"),
            (5, "osm_bus_stop_inferred", "no_free_waiting_surface"),
        ])

    def test_stop_bin_leaves_a_narrow_sidewalk_clear(self):
        vegetation, ground = self.long_road()
        counts = renderer._paint_street_furniture(
            vegetation, ground, proj=TileProjector(),
            features=[osm_node(1, 12, 6, highway="bus_stop")])
        self.assertEqual(counts.get("bin"), 1)
        self.assertEqual(vegetation.getpixel((13, 6)), C.BIN)

    def test_stop_leaves_the_only_verge_tile_free_when_gate_uses_it(self):
        vegetation, ground = self.long_road()
        vegetation.paste(C.TREES, (13, 0, 14, 24))
        vegetation.putpixel((13, 6), C.VEG_NOTHING)
        report = []
        counts = renderer._paint_street_furniture(
            vegetation, ground, proj=TileProjector(), bin_report=report,
            features=[osm_node(1, 12, 6, highway="bus_stop"),
                      osm_node(2, 13, 6, barrier="gate")])
        self.assertEqual(counts.get("bin", 0), 0)
        self.assertEqual(report[0]["reason"], "protected_access")
        self.assertNotIn("tile", report[0])

    def test_stop_bin_uses_platform_edge_only_with_a_paved_bypass(self):
        ground = Image.new("RGB", (32, 32), C.DARK_ASPHALT)
        for width, expected in ((1, "no_clear_side_tile"), (3, "platform_edge")):
            with self.subTest(width=width):
                ground.paste(C.PALE_CONCRETE, (10, 10, 10 + width, 20))
                vegetation = Image.new("RGB", ground.size, C.VEG_NOTHING)
                report = []
                renderer._paint_street_furniture(
                    vegetation, ground, proj=TileProjector(), bin_report=report,
                    features=[osm_node(1, 10, 15, highway="bus_stop")])
                self.assertEqual(report[0]["reason"], expected)
                self.assertNotEqual(vegetation.getpixel((10, 15)), C.BIN)

    def test_zero_stop_bin_density_keeps_the_mapped_bin(self):
        vegetation, ground = self.long_road()
        counts = renderer._paint_street_furniture(
            vegetation, ground, proj=TileProjector(), features=[
                osm_node(1, 13, 4, amenity="waste_basket"),
                osm_node(2, 12, 22, highway="bus_stop"),
            ], settings=StreetSettings(street_bin_density=0))
        self.assertEqual(counts.get("bin"), 1)
        self.assertEqual(vegetation.getpixel((13, 4)), C.BIN)

    def test_no_fabricated_street_mailboxes_postboxes_kiosks_or_bins(self):
        vegetation, ground = self.long_road()
        with mock.patch.object(furniture, "_rule_ready", return_value=True):
            renderer._paint_street_furniture(
                vegetation, ground, buildings=self.buildings_near(ground.size, (15, 0, 60, 24)))
        removed = {C.MAILBOX, C.BIN}
        removed.update(getattr(C, f"{kind}_{facing}")
                       for kind in ("KIOSK", "POSTBOX") for facing in "NESW")
        self.assertFalse(set(vegetation.get_flattened_data()) & removed)

    def test_without_the_rules_none_of_the_new_props_are_painted(self):
        vegetation, ground = self.long_road()
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=False), \
                mock.patch.object(furniture, "_rule_ready", return_value=False):
            counts = renderer._paint_street_furniture(
                vegetation, ground, proj=TileProjector(), features=[
                    osm_node(1, 13, 10, shop="kiosk"),
                    osm_node(2, 13, 18, amenity="post_box"),
                ])
        for what in ("bench", "kiosk", "postbox"):
            self.assertEqual(counts.get(what, 0), 0)

    def test_a_bench_never_stands_half_on_the_kerb(self):
        vegetation, ground = self.long_road()
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=False), \
                mock.patch.object(furniture, "_rule_ready", return_value=True):
            renderer._paint_street_furniture(
                vegetation, ground, None,
                self.buildings_near(ground.size, (10, 0, 60, 24)))
        bench = {*C.BENCH_Y, *C.BENCH_X}
        for x in range(ground.width):
            for y in range(ground.height - 1):
                if vegetation.getpixel((x, y)) in bench:
                    with self.subTest(at=(x, y)):
                        self.assertNotEqual(ground.getpixel((x, y)), C.MEDIUM_ASPHALT)
                        self.assertNotEqual(ground.getpixel((x, y + 1)), C.MEDIUM_ASPHALT)

    def test_manhole_covers_lie_on_the_pavement(self):
        vegetation, ground = self.long_road()
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=False), \
                mock.patch.object(furniture, "_rule_ready", return_value=True), \
                mock.patch.object(furniture, "MANHOLE_SHARE", 0.2):
            renderer._paint_street_furniture(vegetation, ground)
        covers = [(x, y) for x in range(ground.width) for y in range(ground.height)
                  if vegetation.getpixel((x, y)) == C.MANHOLE]
        self.assertTrue(covers, "no manhole cover was laid")
        for x, y in covers:
            with self.subTest(at=(x, y)):
                self.assertEqual(ground.getpixel((x, y)), C.PALE_CONCRETE)

    def test_props_only_stand_where_a_building_is(self):
        vegetation, ground = self.long_road()
        mask = self.buildings_near(ground.size, (10, 8, 60, 16))
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=False):
            counts = renderer._paint_street_furniture(vegetation, ground, None, mask)
        props = [C.HYDRANT]
        xs = [x for x in range(ground.width) for y in range(ground.height)
              if vegetation.getpixel((x, y)) in props]
        self.assertGreater(counts.get("hydrant", 0), 0)
        self.assertTrue(xs)
        self.assertLessEqual(max(xs) - min(xs), 60 + 2 * furniture.NEAR_BUILDING_TILES,
                             f"props out on open road at {sorted(set(xs))}")

    def test_props_stand_beside_the_sidewalk_and_not_on_it(self):
        vegetation, ground = self.long_road()
        mask = self.buildings_near(ground.size, (10, 0, 60, 24))
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=False):
            renderer._paint_street_furniture(vegetation, ground, None, mask)
        props = [C.HYDRANT]
        for x in range(ground.width):
            for y in range(ground.height):
                if vegetation.getpixel((x, y)) not in props:
                    continue
                with self.subTest(at=(x, y)):
                    self.assertNotEqual(ground.getpixel((x, y)), C.PALE_CONCRETE,
                                        "a prop is standing on the sidewalk")
                    neighbours = [ground.getpixel((x + dx, y + dy))
                                  for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                                  if 0 <= x + dx < ground.width and 0 <= y + dy < ground.height]
                    self.assertIn(C.PALE_CONCRETE, neighbours,
                                  "a prop is not beside a sidewalk")

    def test_no_broken_edge_lines_on_diagonal_staircase(self):
        """A diagonal road with 3-tile staircase steps must not get broken
        white edge-line dashes on the steps."""
        w, h = 80, 40
        ground = Image.new("RGB", (w, h), C.DARK_GRASS)
        pixels = ground.load()
        for x in range(w):
            y0 = 10 + x // 3
            for y in range(y0, min(h, y0 + 8)):
                pixels[x, y] = C.MEDIUM_ASPHALT
        vegetation = Image.new("RGB", ground.size, C.VEG_NOTHING)
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=False):
            counts = renderer._paint_street_furniture(vegetation, ground)
        self.assertEqual(counts.get("edge lines", 0), 0)
        edge_colors = {C.EDGE_LINE_N, C.EDGE_LINE_S, C.EDGE_LINE_W, C.EDGE_LINE_E}
        placed = [(x, y) for x in range(w) for y in range(h)
                  if vegetation.getpixel((x, y)) in edge_colors]
        self.assertEqual(placed, [])

    def test_lamps_signs_and_poles_never_stand_on_wide_sidewalk(self):
        ground = Image.new("RGB", (180, 24), C.DARK_GRASS)
        pixels = ground.load()
        for y in range(24):
            for x in range(4, 12):
                pixels[x, y] = C.MEDIUM_ASPHALT
            # 2-tile sidewalk at x=12, 13; grass begins at x=14.
            pixels[12, y] = C.PALE_CONCRETE
            pixels[13, y] = C.PALE_CONCRETE
        vegetation = Image.new("RGB", ground.size, C.VEG_NOTHING)
        for y in range(24):
            vegetation.putpixel((11, y), C.EDGE_LINE_E)
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=True), \
                mock.patch.object(furniture, "_rule_ready", return_value=True):
            counts = renderer._paint_street_furniture(vegetation, ground)
        self.assertGreater(counts.get("lamp", 0), 0)
        for x in (12, 13):
            for y in range(24):
                self.assertIn(
                    vegetation.getpixel((x, y)),
                    (C.VEG_NOTHING, C.LITTER, C.MANHOLE),
                    f"fixture placed on sidewalk at {(x, y)}",
                )

    def test_manholes_skip_distant_concrete_aprons(self):
        ground = Image.new("RGB", (80, 80), C.MEDIUM_GRASS)
        pixels = ground.load()
        for y in range(80):
            for x in range(4, 12):
                pixels[x, y] = C.MEDIUM_ASPHALT
            for x in range(12, 14):
                pixels[x, y] = C.PALE_CONCRETE
        for y in range(10, 70):
            for x in range(30, 75):
                pixels[x, y] = C.PALE_CONCRETE
        vegetation = Image.new("RGB", ground.size, C.VEG_NOTHING)
        with mock.patch.object(furniture, "_mod_tiles_ready", return_value=False):
            renderer._paint_street_furniture(vegetation, ground)
        distant_manholes = [
            (x, y) for x in range(20, 80) for y in range(80)
            if vegetation.getpixel((x, y)) == C.MANHOLE
        ]
        self.assertEqual(distant_manholes, [])


if __name__ == "__main__":
    unittest.main()
