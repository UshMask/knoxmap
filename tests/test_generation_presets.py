"""Reusable settings must stay independent of the map that supplied them."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import app
from web import presets


class GenerationPresets(unittest.TestCase):
    url = "/api/generation-presets"

    def setUp(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = folder / "generation_presets.json"
        self.areas = folder / "presets.json"
        self.areas.write_text('{"Existing area":{}}', encoding="utf-8")
        self.enterContext(patch.object(presets, "GENERATION_PRESETS_FILE", self.path))
        self.enterContext(patch.object(presets, "PRESETS_FILE", self.areas))
        self.client = app.test_client()

    def test_create_apply_rename_update_and_delete(self):
        self.assertEqual(self.client.get(self.url).json["presets"], [])
        data = {"name": "Киров", "south": 10, "metersPerTile": 6,
                "settings": {"preset": "city", "tree_density": 0.83, "south": 10}}
        result = self.client.post(self.url, json=data)
        self.assertEqual(result.status_code, 200)
        entry, = result.json["presets"]
        preset_url = f"{self.url}/{entry['id']}"
        self.assertEqual(set(entry), {"id", "name", "settings"})
        self.assertEqual(entry["settings"]["tree_density"], 0.83)
        self.assertEqual(entry["settings"]["apartment_chance"], 0.85)
        self.assertNotIn("south", entry["settings"])
        self.assertEqual(app.test_client().get(self.url).json["presets"], [entry])

        result = self.client.patch(preset_url, json={"name": "Новая область"})
        self.assertEqual(result.json["presets"][0]["settings"], entry["settings"])
        result = self.client.patch(preset_url, json={"settings": {"tree_density": 0.48}})
        self.assertEqual(result.json["presets"][0]["name"], "Новая область")
        self.assertEqual(result.json["presets"][0]["settings"]["tree_density"], 0.48)
        self.assertEqual(self.client.delete(preset_url).json["presets"], [])
        self.assertEqual(self.areas.read_text(encoding="utf-8"), '{"Existing area":{}}')

    def test_invalid_and_duplicate_requests_leave_the_catalog_intact(self):
        self.assertEqual(self.client.post(self.url, json={"name": "Broken"}).status_code, 400)
        self.assertFalse(self.path.exists())
        self.client.post(self.url, json={"name": "First", "settings": {}})
        before = self.path.read_bytes()
        duplicate = self.client.post(self.url, json={"name": " first ", "settings": {}})
        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_library_is_never_overwritten(self):
        self.path.write_text('{"broken": 1}', encoding="utf-8")
        self.assertEqual(self.client.get(self.url).status_code, 500)
        self.assertEqual(self.client.post(self.url, json={"name": "New", "settings": {}}).status_code, 500)
        self.assertEqual(self.path.read_text(encoding="utf-8"), '{"broken": 1}')
