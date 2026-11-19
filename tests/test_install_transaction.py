"""Installing a map keeps the previous mod until its replacement is ready."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import compile_cells, compile_map, make_map_mod


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

    def test_success_replaces_map_data_and_keeps_other_files(self):
        (self.installed / "stale.lotheader").write_bytes(b"old")
        root, cells, _ = self.install()
        self.assertEqual((Path(root), cells), (self.destination, 1))
        self.assertEqual((self.installed / "0_0.lotheader").read_bytes(), b"new")
        self.assertFalse((self.installed / "stale.lotheader").exists())
        self.assertEqual((self.destination / "extra.txt").read_bytes(), b"keep")
