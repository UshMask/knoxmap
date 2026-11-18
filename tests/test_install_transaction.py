"""Installing a map keeps the previous mod until its replacement is ready."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tools import compile_cells, compile_map, compile_state, make_map_mod


class InstallTransaction(unittest.TestCase):
    def setUp(self):
        work = self.enterContext(tempfile.TemporaryDirectory())
        self.project = Path(work, "town")
        self.lots = self.project / "lots"
        self.lots.mkdir(parents=True)
        for name in ("0_0.lotheader", "world_0_0.lotpack", "chunkdata_0_0.bin"):
            (self.lots / name).write_bytes(b"new")
        self.mods = Path(work, "mods")
        self.destination = self.mods / "town"
        self.installed = self.destination / "common/media/maps" / make_map_mod.folder_name("Town", "town")
        self.installed.mkdir(parents=True)
        (self.installed / "0_0.lotheader").write_bytes(b"old")
        (self.destination / "extra.txt").write_bytes(b"keep")

    def install(self):
        return make_map_mod.package(str(self.project), "Town", "town", mods_dir=str(self.mods))

    def check_previous(self):
        self.assertEqual((self.installed / "0_0.lotheader").read_bytes(), b"old")
        self.assertEqual((self.destination / "extra.txt").read_bytes(), b"keep")
        self.assertEqual(list(self.mods.glob(".knoxmap-install-*")), [])

    def test_copy_failure_keeps_the_previous_mod(self):
        copy2 = make_map_mod.shutil.copy2

        def fail_new_cell(source, target):
            if Path(source).parent == self.lots:
                raise OSError("copy failed")
            return copy2(source, target)

        with mock.patch.object(make_map_mod.shutil, "copy2", side_effect=fail_new_cell):
            with self.assertRaisesRegex(OSError, "copy failed"):
                self.install()
        self.check_previous()

    def test_failed_swap_restores_the_previous_mod(self):
        rename = Path.rename

        def fail_publish(path, target):
            if path.name == "new":
                raise OSError("rename failed")
            return rename(path, target)

        with mock.patch.object(Path, "rename", fail_publish):
            with self.assertRaisesRegex(OSError, "rename failed"):
                self.install()
        self.check_previous()

    def test_running_or_interrupted_compile_refuses_install(self):
        with compile_map._only_one(self.project, "compile"):
            with self.assertRaises(compile_map.ProjectBusy):
                self.install()
        (self.project / compile_map.INCOMPLETE_LOTS).touch()
        with self.assertRaises(make_map_mod.InvalidLots):
            self.install()
        self.check_previous()

    def test_lock_also_blocks_a_second_process(self):
        with compile_map._only_one(self.project, "compile"):
            called = subprocess.run([
                sys.executable, "-c",
                "import sys; from tools.compile_map import claim; claim(sys.argv[1], 'other')",
                str(self.project),
            ], cwd=compile_map.BASE_DIR, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(called.returncode, 0)
        self.assertIn("ProjectBusy", called.stderr)
        self.check_previous()

    def test_incomplete_cell_refuses_install(self):
        (self.lots / "world_0_0.lotpack").unlink()
        with self.assertRaises(make_map_mod.InvalidLots):
            self.install()
        self.check_previous()

    def test_native_record_detects_a_missing_whole_cell(self):
        expected = compile_cells.lot_hashes(self.lots)
        expected.update({name: "digest" for name in compile_cells.lot_names({(0, 1)})})
        (self.project / compile_cells.CACHE).write_text(json.dumps({"outputs": expected}),
                                                      encoding="utf-8")
        with self.assertRaises(make_map_mod.InvalidLots):
            self.install()
        self.check_previous()

    def worlded_project(self):
        from knoxbuild.world import render_pzw

        self.enterContext(mock.patch("knoxpaths.mapping_tools_dir", return_value=None))
        source_time = time.time_ns() - 60_000_000_000
        (self.project / "tmx").mkdir()
        pzw = self.project / "town.pzw"
        with mock.patch("knoxbuild.world.origin", return_value=(0, 0)):
            pzw.write_text(render_pzw(1, 1, "town.bmp", [], map_name="town",
                                      project_dir=str(self.project)), encoding="utf-8")
        tmx = self.project / "tmx/town_0_0.tmx"
        for source in ("town.bmp", "town_veg.bmp", "town_ZombieSpawnMap.bmp", "tmx/town_0_0.tmx"):
            (self.project / source).write_bytes(b"source")
        for path in (pzw, tmx, *self.project.glob("*.bmp")):
            os.utime(path, ns=(source_time, source_time))
        marker = self.project / compile_map.INCOMPLETE_LOTS
        compile_map.mark_lots_stale(self.project)
        marked = source_time + 2_000_000_000
        os.utime(marker, ns=(marked, marked))
        return marker

    def write_worlded_lots(self):
        # One 300-tile project cell covers four 256-tile output cells.
        header = struct.pack("<4sIIIIiiII", b"LOTH", 1, 0, 8, 8, 0, 0, 0, 0) + bytes(1024)
        pack = (struct.pack("<4sII1024Q", b"LOTP", 1, 1024, *range(8204, 16396, 8))
                + struct.pack("<ii", -1, 64) * 1024)
        for x, y in ((0, 0), (0, 1), (1, 0), (1, 1)):
            for name, data in ((f"{x}_{y}.lotheader", header),
                               (f"world_{x}_{y}.lotpack", pack),
                               (f"chunkdata_{x}_{y}.bin", b"\0\x01" + bytes(1024))):
                (self.lots / name).write_bytes(data)

    def worlded_export(self):
        marker = self.worlded_project()
        self.write_worlded_lots()
        return marker

    def test_full_manual_export_reconciles_stale_lots_after_a_native_compile(self):
        cache = self.project / compile_cells.CACHE
        cache.write_text(json.dumps({"outputs": compile_cells.lot_hashes(self.lots)}),
                         encoding="utf-8")
        (self.project / compile_state.STATE_FILE).touch()
        marker = self.worlded_export()
        _, cells, _ = self.install()
        self.assertEqual(cells, 4)
        self.assertFalse(marker.exists())
        self.assertFalse((self.project / compile_state.STATE_FILE).exists())
        expected = compile_cells.lot_hashes(self.lots)
        self.assertEqual(compile_cells.lot_hashes(self.installed), expected)
        self.assertEqual(json.loads(cache.read_text(encoding="utf-8")), {"outputs": expected})
        self.install()
        (self.lots / "1_1.lotheader").write_bytes(b"bad")
        with self.assertRaisesRegex(make_map_mod.InvalidLots, "cell files changed"):
            self.install()
        self.assertEqual(compile_cells.lot_hashes(self.installed), expected)

    def test_manual_worlded_fallback_installs_through_the_app(self):
        from app import app
        from web import common, compiling, generation

        marker = self.worlded_project()
        client = app.test_client()
        request = {"mapName": "town"}
        with (mock.patch.object(common, "OUTPUT_DIR", self.project.parent),
              mock.patch.object(compiling, "_worlded_exe", return_value=None),
              mock.patch.object(generation, "_worlded_exe", return_value=Path("WorldEd.exe")),
              mock.patch("knoxpaths.command_for", return_value=["WorldEd.exe"]),
              mock.patch("subprocess.Popen") as launch,
              mock.patch.object(make_map_mod, "default_mods_dir", return_value=str(self.mods))):
            fallback = client.post("/api/compile", json={**request, "backend": "worlded"})
            self.assertEqual(fallback.status_code, 400)
            self.assertIn("Open in WorldEd", fallback.get_json()["error"])
            opened = client.post("/api/worlded", json=request)
            self.assertEqual(opened.status_code, 200, opened.get_json())
            launch.assert_called_once_with(["WorldEd.exe", str(self.project.resolve() / "town.pzw")])
            self.write_worlded_lots()
            installed = client.post("/api/install", json={**request, "title": "Town"})
        self.assertEqual(installed.status_code, 200, installed.get_json())
        self.assertEqual(installed.get_json()["cells"], 4)
        self.assertFalse(marker.exists())

    def test_a_later_build_invalidates_an_unreconciled_export(self):
        marker = self.worlded_export()
        old = marker.stat().st_mtime_ns + 1_000_000_000
        for cell in self.lots.iterdir():
            os.utime(cell, ns=(old, old))
        compile_map.mark_lots_stale(self.project)
        with self.assertRaisesRegex(make_map_mod.InvalidLots, "lot files are out of date"):
            self.install()
        self.assertTrue(marker.exists())
        self.check_previous()

    def test_manual_export_missing_a_whole_cell_is_refused(self):
        marker = self.worlded_export()
        for path in compile_cells.lot_names({(1, 1)}):
            (self.lots / path).unlink()
        with self.assertRaisesRegex(make_map_mod.InvalidLots, "entire map"):
            self.install()
        self.assertTrue(marker.exists())
        self.check_previous()

    def test_manual_export_with_an_old_file_is_refused(self):
        marker = self.worlded_export()
        old = marker.stat().st_mtime_ns - 1_000_000_000
        os.utime(self.lots / "world_1_1.lotpack", ns=(old, old))
        with self.assertRaisesRegex(make_map_mod.InvalidLots, "lot files are out of date"):
            self.install()
        self.assertTrue(marker.exists())
        self.check_previous()

    def test_manual_export_from_old_tmx_is_refused(self):
        marker = self.worlded_export()
        old = (self.project / "town.bmp").stat().st_mtime_ns - 1_000_000_000
        os.utime(self.project / "tmx/town_0_0.tmx", ns=(old, old))
        with self.assertRaisesRegex(make_map_mod.InvalidLots, "converted maps are out of date"):
            self.install()
        self.assertTrue(marker.exists())
        self.check_previous()

    def test_manual_export_with_any_truncated_file_is_refused(self):
        marker = self.worlded_export()
        for name in ("1_1.lotheader", "world_1_1.lotpack", "chunkdata_1_1.bin"):
            with self.subTest(name=name):
                path = self.lots / name
                data = path.read_bytes()
                path.write_bytes(data[:-1])
                with self.assertRaisesRegex(make_map_mod.InvalidLots, "damaged or incomplete"):
                    self.install()
                path.write_bytes(data)
                self.assertTrue(marker.exists())
                self.check_previous()

    def test_valid_looking_export_cannot_clear_a_publication_marker(self):
        marker = self.worlded_export()
        marker.write_text("publishing", encoding="utf-8")
        compile_map.mark_lots_stale(self.project)
        with self.assertRaisesRegex(make_map_mod.InvalidLots, "out of date or incomplete"):
            self.install()
        self.assertEqual(marker.read_text(encoding="utf-8"), "publishing")
        self.check_previous()

    def test_manual_export_changed_while_copying_is_not_published(self):
        marker = self.worlded_export()
        cell = self.lots / "1_1.lotheader"
        data = cell.read_bytes()

        def on_progress(done, total):
            if done == total:
                cell.write_bytes(data[:-1] + b"\xff")

        with self.assertRaisesRegex(make_map_mod.InvalidLots, "changed during installation"):
            make_map_mod.package(str(self.project), "Town", "town", mods_dir=str(self.mods),
                                 on_progress=on_progress)
        self.assertTrue(marker.exists())
        self.check_previous()

    def test_success_replaces_map_data_and_keeps_other_files(self):
        (self.installed / "stale.lotheader").write_bytes(b"old")
        root, cells, _ = self.install()
        self.assertEqual((Path(root), cells), (self.destination, 1))
        self.assertEqual((self.installed / "0_0.lotheader").read_bytes(), b"new")
        self.assertFalse((self.installed / "stale.lotheader").exists())
        self.assertEqual((self.destination / "extra.txt").read_bytes(), b"keep")
