"""Platform integration checks without loading models or starting a desktop app."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api_server
from seedvr_studio.paths import ROOT, VENV_PYTHON


@unittest.skipIf(os.name == "nt", "Linux integration checks")
class LinuxSupportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_server.app)

    def test_render_interpreter_uses_linux_venv(self) -> None:
        self.assertEqual(VENV_PYTHON, ROOT / ".venv/bin/python")
        self.assertTrue(VENV_PYTHON.is_file())

    def test_updates_do_not_launch_windows_updater(self) -> None:
        with patch.object(api_server, "launch_updater") as launch:
            self.assertFalse(self.client.get("/api/update/check").json()["supported"])
            self.assertEqual(self.client.post("/api/update/apply").status_code, 409)
            launch.assert_not_called()

    def test_output_folder_opens_with_xdg_and_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory) / "outputs"
            outputs.mkdir()
            video = outputs / "example.mp4"
            video.touch()
            (Path(directory) / "outside.mp4").touch()
            with patch.object(api_server, "OUTPUTS", outputs), patch.object(api_server.subprocess, "Popen") as popen:
                response = self.client.post("/api/open-folder", data={"output_path": "/media/example.mp4"})
                self.assertEqual(response.status_code, 200)
                popen.assert_called_once_with(["xdg-open", str(outputs)], close_fds=True)
                popen.reset_mock()
                response = self.client.post("/api/open-folder", data={"output_path": "../outside.mp4"})
                self.assertEqual(response.status_code, 404)
                popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
