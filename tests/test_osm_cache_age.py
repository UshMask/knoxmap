"""An older download is still the map's roads.

The download kept beside a map is stamped with the filter list it was fetched
with, and a map drawn before that list grew has an older stamp. Fetching is
right to refuse it - the new filters are there to bring back things the old
ones never asked for - but the steps after the map is drawn only ever read
this file. Turning it away there did not get the newer tags: the paper map
came out with the buildings on it and not one road, river or wood, and every
shop and surgery mapped as a point was lost with it.
"""
import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path

from generator import osm


def write_cache(path, bbox, filters, features=()):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"filters": filters, "bbox": list(bbox),
                   "features": list(features)}, f)


ROAD = {"osm_id": 1, "kind": "way", "tags": {"highway": "residential"},
        "geometry": [[1.0, 1.0], [1.0, 2.0]], "role_geoms": []}
BOX = (1.0, 2.0, 3.0, 4.0)


class AnOlderDownload(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        self.path = os.path.join(folder, "m_osm.json.gz")

    def test_fetching_still_refuses_one_made_with_an_older_filter_list(self):
        write_cache(self.path, BOX, osm.FILTERS_VERSION - 1, [ROAD])
        self.assertIsNone(osm.load_cache(self.path, BOX))

    def test_reading_takes_whatever_is_there(self):
        write_cache(self.path, BOX, osm.FILTERS_VERSION - 4, [ROAD])
        got = osm.load_cache(self.path, BOX, filters=None)
        self.assertEqual(len(got or []), 1)
        self.assertEqual(got[0].tags, {"highway": "residential"})

    def test_a_download_of_the_current_list_is_taken_either_way(self):
        write_cache(self.path, BOX, osm.FILTERS_VERSION, [ROAD])
        self.assertEqual(len(osm.load_cache(self.path, BOX) or []), 1)
        self.assertEqual(len(osm.load_cache(self.path, BOX, filters=None) or []), 1)

    def test_a_download_of_somewhere_else_is_refused_however_it_is_read(self):
        write_cache(self.path, (9.0, 9.0, 9.5, 9.5), osm.FILTERS_VERSION, [ROAD])
        self.assertIsNone(osm.load_cache(self.path, BOX))
        self.assertIsNone(osm.load_cache(self.path, BOX, filters=None))

    def test_old_project_without_cache_metadata_uses_the_default_snapshot(self):
        write_cache(self.path, BOX, osm.FILTERS_VERSION - 1, [ROAD])
        bbox = dict(zip(("south", "west", "north", "east"), BOX))
        path, features = osm.load_project_cache(os.path.dirname(self.path), "m", {"bbox": bbox})
        self.assertEqual(path, self.path)
        self.assertEqual([feature.osm_id for feature in features], [1])

    def test_copied_project_prefers_its_snapshot_to_the_old_absolute_path(self):
        original_dir = self.enterContext(tempfile.TemporaryDirectory())
        original = os.path.join(original_dir, Path(self.path).name)
        write_cache(original, BOX, osm.FILTERS_VERSION, [{**ROAD, "osm_id": 2}])
        info = {"osm_cache": original, "osm_bbox": BOX}
        project = os.path.dirname(self.path)
        path, features = osm.load_project_cache(project, "m", info)
        self.assertEqual(path, original)
        self.assertEqual([feature.osm_id for feature in features], [2])
        write_cache(self.path, BOX, osm.FILTERS_VERSION - 1, [ROAD])
        path, features = osm.load_project_cache(project, "m", info)
        self.assertEqual(path, self.path)
        self.assertEqual([feature.osm_id for feature in features], [1])
        write_cache(self.path, (9.0, 9.0, 9.5, 9.5), osm.FILTERS_VERSION, [ROAD])
        path, features = osm.load_project_cache(project, "m", info)
        self.assertEqual(path, original)
        self.assertEqual([feature.osm_id for feature in features], [2])
        Path(self.path).write_bytes(b"not a gzip snapshot")
        self.assertEqual(osm.load_project_cache(project, "m", info)[0], original)
        write_cache(self.path, BOX, osm.FILTERS_VERSION)
        self.assertEqual(osm.load_project_cache(project, "m", info), (self.path, []))

    def test_rotated_snapshot_path_stays_relative_to_the_project(self):
        project = os.path.dirname(self.path)
        rotated = osm.cache_path(project, "m_turned")
        write_cache(rotated, BOX, osm.FILTERS_VERSION, [ROAD])
        reference = osm.project_cache_reference(project, rotated)
        self.assertEqual(reference, "m_turned_osm.json.gz")
        path, features = osm.load_project_cache(project, "m",
                                               {"osm_cache": reference, "osm_bbox": BOX})
        self.assertEqual(path, rotated)
        self.assertEqual([feature.osm_id for feature in features], [1])


if __name__ == "__main__":
    unittest.main()
