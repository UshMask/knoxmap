import unittest
import random
from types import SimpleNamespace

from generator.osm import OSMFeature
from generator.rendering.buildings import _buildings_geojson, _houses_from_addresses
from knoxbuild import build, layout
from knoxbuild.profile import BuildingProfile
from knoxbuild.settings import Settings


class BuildingProfileTests(unittest.TestCase):
    def test_exported_way_and_relation_dates_reach_the_profile(self):
        ring = [(1, 2), (1, 3), (2, 3), (1, 2)]
        way = OSMFeature(1, "way", {"building": "house", "start_date": "1922-06"}, ring)
        relation = OSMFeature(2, "relation", {"building": "house", "start_date": "1976"},
                              [], role_geoms=[("outer", ring)])
        undated = OSMFeature(3, "way", {"building": "house"}, ring)
        modern = OSMFeature(4, "way", {"building": "house", "start_date": "2014"}, ring)
        tags = [feat["properties"] for feat in
                _buildings_geojson([way, relation, undated, modern])["features"]]
        self.assertEqual([tag.get("start_date") for tag in tags], ["1922-06", "1976", None, "2014"])
        profiles = [BuildingProfile.infer(tag, "house", 0.1, 100, 2, 4) for tag in tags]
        self.assertEqual([p.year_built for p in profiles[:2]], [1922, 1976])
        self.assertTrue(all(p.dated for p in profiles[:2]))
        self.assertFalse(profiles[2].dated)
        self.assertEqual(profiles[2], profiles[3])

    def test_inferred_footprint_does_not_inherit_a_business_start_date(self):
        point = OSMFeature(71, "node", {"addr:housenumber": "5", "shop": "bakery",
                                        "start_date": "1992"}, [(20, 20)])
        proj = SimpleNamespace(width=60, height=60, meters_per_tile=1,
                               to_px=lambda lat, lon: (lon, lat),
                               to_latlon=lambda x, y: (y, x))
        footprint = _houses_from_addresses([point], [], proj)
        properties = _buildings_geojson(footprint)["features"][0]["properties"]
        self.assertEqual(properties["shop"], "bakery")
        self.assertNotIn("start_date", properties)

    def test_inference_is_reproducible_and_uses_explicit_tags(self):
        tags = {"start_date": "1922", "building:condition": "poor",
                "class": "luxury", "building": "mansion"}
        profile = BuildingProfile.infer(tags, "house", 0.12, 700, 2, 41)
        self.assertEqual(profile, BuildingProfile.infer(tags, "house", 0.12, 700, 2, 41))
        self.assertEqual(profile.year_built, 1922)
        self.assertEqual(profile.era, "prewar")
        self.assertEqual(profile.setting, "suburban")
        self.assertEqual(profile.wear, 0.76)
        self.assertGreater(profile.wealth, 0.7)

    def test_neighborhood_setting_uses_density_and_building_type(self):
        rural = BuildingProfile.infer({}, "house", 0.02, 100, 1, 4)
        suburban = BuildingProfile.infer({}, "house", 0.14, 100, 1, 4)
        urban = BuildingProfile.infer({}, "apartment", 0.4, 100, 4, 4)
        barn = BuildingProfile.infer({"building": "barn"}, None, 0.2, 100, 1, 4)
        self.assertEqual((rural.setting, suburban.setting, urban.setting, barn.setting),
                         ("rural", "suburban", "urban", "rural"))

    def test_profile_weights_shell_furniture_and_floor_families(self):
        old_worn = BuildingProfile.infer(
            {"start_date": "1900", "condition": "poor"}, "house", 0.1, 100, 1, 2)
        new_clean = BuildingProfile.infer(
            {"start_date": "1985", "condition": "excellent", "class": "luxury"},
            "house", 0.35, 700, 2, 3)
        self.assertGreater(old_worn.style_weight("brick"), old_worn.style_weight("render"))
        self.assertGreater(new_clean.furniture_weight("dresser_black"),
                           new_clean.furniture_weight("dresser_pale"))
        self.assertGreater(old_worn.floor_weight("civic_worn"),
                           old_worn.floor_weight("wood_pale"))
        self.assertLess(new_clean.clutter, old_worn.clutter)

    def test_garage_tags_generate_garage_rooms_and_industrial_shells(self):
        for value in ("garage", "garages"):
            with self.subTest(building=value):
                self.assertEqual(build.classify_building({"building": value}),
                                 "garage")

        plan = layout.build_plan(12, 10, kind="garage", commercial=True, seed=8)
        self.assertTrue(plan.rooms)
        self.assertEqual({room.kind for room in plan.rooms}, {"garage"})
        shell = build.pick_style(build.STYLE_AS["garage"], 20, 30,
                                 random.Random(8), Settings(), density=0.2)
        self.assertTrue(shell["name"].startswith("industrial"))


if __name__ == "__main__":
    unittest.main()
