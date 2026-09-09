"""Job creation from a local source path and delivery beside the source."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api_server
from seedvr_studio import dialogs
from seedvr_studio.media import VideoInfo


class _InertThread:
    """Stand-in for threading.Thread so job creation never starts a render."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def start(self) -> None:
        pass


class SourcePathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_server.app)
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.video = self.root / "clips" / "holiday.mov"
        self.video.parent.mkdir()
        self.video.write_bytes(b"not really a video")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _create(self, **data: str):
        with patch.object(api_server, "OUTPUTS", self.outputs), patch.object(api_server, "ensure_workspace"), patch.object(api_server, "Thread", _InertThread):
            api_server.JOBS.clear()
            return self.client.post("/api/jobs", data={"job_type": "full", **data})

    def test_file_path_is_used_without_upload(self) -> None:
        response = self._create(source_path=str(self.video))
        self.assertEqual(response.status_code, 200, response.text)
        job = api_server.JOBS[response.json()["id"]]
        values = job["_values"]
        self.assertEqual(values["original_stem"], "holiday")
        self.assertEqual(values["delivery_dir"], str(self.video.parent))
        self.assertEqual(Path(job["_source"]).name, "source.mov")
        self.assertEqual(Path(job["_source"]).read_bytes(), b"not really a video")

    def test_folder_path_requires_an_upload_and_delivers_there(self) -> None:
        self.assertEqual(self._create(source_path=str(self.video.parent)).status_code, 400)
        with patch.object(api_server, "OUTPUTS", self.outputs), patch.object(api_server, "ensure_workspace"), patch.object(api_server, "Thread", _InertThread):
            response = self.client.post("/api/jobs", data={"job_type": "full", "source_path": str(self.video.parent)}, files={"file": ("party.mp4", b"bytes", "video/mp4")})
        self.assertEqual(response.status_code, 200, response.text)
        values = api_server.JOBS[response.json()["id"]]["_values"]
        self.assertEqual(values["original_stem"], "party")
        self.assertEqual(values["delivery_dir"], str(self.video.parent))

    def test_upload_wins_over_a_stale_file_path(self) -> None:
        with patch.object(api_server, "OUTPUTS", self.outputs), patch.object(api_server, "ensure_workspace"), patch.object(api_server, "Thread", _InertThread):
            response = self.client.post("/api/jobs", data={"job_type": "preview", "source_path": str(self.video)}, files={"file": ("second.mp4", b"second", "video/mp4")})
        self.assertEqual(response.status_code, 200, response.text)
        job = api_server.JOBS[response.json()["id"]]
        self.assertEqual(job["_values"]["original_stem"], "second")
        self.assertEqual(Path(job["_source"]).read_bytes(), b"second")
        self.assertEqual(job["_values"]["delivery_dir"], str(self.video.parent))

    def test_missing_path_and_missing_upload_are_rejected(self) -> None:
        self.assertEqual(self._create(source_path=str(self.root / "nope.mp4")).status_code, 400)
        self.assertEqual(self._create().status_code, 400)

    def test_full_render_is_delivered_beside_the_source(self) -> None:
        response = self._create(source_path=str(self.video))
        job = api_server.JOBS[response.json()["id"]]
        job_dir = Path(job["_job_dir"])

        def fake_render(backend_name, source, output, settings, progress_callback=None):
            output.write_bytes(b"restored")
            return output

        info = VideoInfo(duration=2.0, width=640, height=360, fps=25.0, frames=50)
        with patch.object(api_server, "OUTPUTS", self.outputs), patch.object(api_server, "probe", return_value=info), patch.object(api_server, "render", fake_render):
            api_server._render_job(job["id"], Path(job["_source"]), job_dir, dict(job["_values"]))
        self.assertEqual(job["status"], job.get("error") or "complete")
        self.assertTrue((job_dir / "holiday-seed.mp4").is_file())
        delivered = self.video.parent / "holiday-seed.mp4"
        self.assertEqual(job["saved_path"], str(delivered))
        self.assertEqual(delivered.read_bytes(), b"restored")

    def test_local_video_endpoints_preview_a_path(self) -> None:
        info = VideoInfo(duration=2.0, width=640, height=360, fps=25.0, frames=50)
        with patch.object(api_server, "probe", return_value=info):
            response = self.client.get("/api/local-video/info", params={"path": str(self.video)})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["name"], "holiday.mov")
        self.assertEqual(response.json()["folder"], str(self.video.parent))
        self.assertEqual(response.json()["fps"], 25.0)
        streamed = self.client.get("/api/local-video", params={"path": str(self.video)})
        self.assertEqual(streamed.status_code, 200)
        self.assertEqual(streamed.content, b"not really a video")
        self.assertEqual(self.client.get("/api/local-video/info", params={"path": str(self.root / "missing.mp4")}).status_code, 404)
        (self.root / "notes.txt").touch()
        self.assertEqual(self.client.get("/api/local-video", params={"path": str(self.root / "notes.txt")}).status_code, 415)

    def test_cleanup_keeps_running_jobs_and_undelivered_results(self) -> None:
        preview = self.outputs / "js-preview-aaaa"; preview.mkdir(); (preview / "restored.mp4").touch()
        delivered = self.outputs / "js-full-bbbb"; delivered.mkdir(); (delivered / "clip-seed.mp4").touch(); (delivered / "delivered.json").write_text("{}")
        undelivered = self.outputs / "js-full-cccc"; undelivered.mkdir(); (undelivered / "clip-seed.mp4").touch()
        running = self.outputs / "js-full-dddd"; running.mkdir()
        with patch.object(api_server, "OUTPUTS", self.outputs):
            api_server.JOBS.clear()
            api_server.JOBS["dddd"] = {"status": "running", "_job_dir": str(running)}
            result = self.client.post("/api/workspace/clear").json()
        self.assertEqual(sorted(result["removed"]), ["js-full-bbbb", "js-preview-aaaa"])
        self.assertEqual(sorted(result["kept"]), ["js-full-cccc", "js-full-dddd"])
        self.assertFalse(preview.exists()); self.assertFalse(delivered.exists())
        self.assertTrue(undelivered.exists()); self.assertTrue(running.exists())

    def test_output_presets_scale_from_source_or_fix_size(self) -> None:
        self.assertEqual(api_server.resolve_output_size("1.5x source", 1280, 720), (1080, 1920))
        self.assertEqual(api_server.resolve_output_size("2x source", 720, 1280), (1440, 2560))
        self.assertEqual(api_server.resolve_output_size("Original / enhancement only", 853, 480), (480, 852))
        self.assertEqual(api_server.resolve_output_size("Original enhancement only", 640, 360), (360, 640))
        self.assertEqual(api_server.resolve_output_size("4K / 2160p", 640, 360), (2160, 3840))
        self.assertEqual(api_server.resolve_output_size("unknown preset", 640, 360), (1080, 1920))
        config = self.client.get("/api/config").json()["output_presets"]
        self.assertIn("2x source", config)
        self.assertEqual(config["2x source"]["scale"], 2.0)

    def test_source_divisor_accepts_scale_or_legacy_flag(self) -> None:
        self.assertEqual(api_server._source_divisor({"source_scale": "3"}), 3.0)
        self.assertEqual(api_server._source_divisor({"source_scale": 4}), 4.0)
        self.assertEqual(api_server._source_divisor({"source_scale": "1", "half_source": True}), 1.0)
        self.assertEqual(api_server._source_divisor({"half_source": "true"}), 2.0)
        self.assertEqual(api_server._source_divisor({"source_scale": "7"}), 1.0)
        self.assertEqual(api_server._source_divisor({}), 1.0)

    def test_half_source_keeps_output_size_from_original(self) -> None:
        response = self._create(source_path=str(self.video), output_preset="2x source", source_scale="3")
        job = api_server.JOBS[response.json()["id"]]
        job_dir = Path(job["_job_dir"])
        seen: dict[str, object] = {}

        def fake_render(backend_name, source, output, settings, progress_callback=None):
            seen["source"] = source; seen["settings"] = settings
            output.write_bytes(b"restored")
            return output

        def fake_downscale(source, target, divisor):
            seen["divisor"] = divisor; target.write_bytes(b"small"); return target

        infos = {"source": VideoInfo(2.0, 1280, 720, 25.0, 50), "downscaled": VideoInfo(2.0, 426, 240, 25.0, 50)}
        with patch.object(api_server, "OUTPUTS", self.outputs), patch.object(api_server, "probe", side_effect=lambda path: infos["downscaled" if Path(path).stem == "downscaled" else "source"]), patch.object(api_server, "make_downscaled", fake_downscale), patch.object(api_server, "render", fake_render):
            api_server._render_job(job["id"], Path(job["_source"]), job_dir, dict(job["_values"]))
        self.assertEqual(job["status"], job.get("error") or "complete")
        self.assertEqual(Path(seen["source"]).name, "downscaled.mov")
        self.assertEqual(seen["divisor"], 3.0)
        self.assertEqual((seen["settings"].resolution, seen["settings"].max_resolution), (1440, 2560))

    def test_pick_file_endpoint_uses_the_first_working_dialog(self) -> None:
        import subprocess
        chosen = subprocess.CompletedProcess([], 0, stdout=f"{self.video}\n", stderr="")
        with patch.object(dialogs, "available_pickers", return_value=["kdialog", "zenity"]), patch.object(dialogs.subprocess, "run", side_effect=[FileNotFoundError("kdialog"), chosen]) as run:
            response = self.client.post("/api/pick-file", data={"initial": str(self.video.parent)})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"path": str(self.video), "cancelled": False})
        self.assertEqual(run.call_args_list[1].args[0][0], "zenity")
        cancelled = subprocess.CompletedProcess([], 1, stdout="", stderr="")
        with patch.object(dialogs, "available_pickers", return_value=["kdialog"]), patch.object(dialogs.subprocess, "run", return_value=cancelled):
            self.assertEqual(self.client.post("/api/pick-file").json(), {"path": None, "cancelled": True})
        with patch.object(dialogs, "available_pickers", return_value=[]):
            self.assertEqual(self.client.post("/api/pick-file").status_code, 503)

    def test_preview_job_reports_its_slot_on_the_original_timeline(self) -> None:
        with patch.object(api_server, "OUTPUTS", self.outputs), patch.object(api_server, "ensure_workspace"), patch.object(api_server, "Thread", _InertThread):
            api_server.JOBS.clear()
            response = self.client.post("/api/jobs", data={"job_type": "preview", "source_path": str(self.video), "preview_start": "2", "preview_seconds": "3"})
        job = api_server.JOBS[response.json()["id"]]
        job_dir = Path(job["_job_dir"])

        def fake_clip(source, target, start, length, fps=0.0):
            seen_clip.update(start=start, length=length, fps=fps)
            target.write_bytes(b"clip"); return target

        def fake_render(backend_name, source, output, settings, progress_callback=None):
            output.write_bytes(b"restored"); return output

        seen_clip: dict[str, float] = {}
        info = VideoInfo(duration=4.0, width=640, height=360, fps=25.0, frames=100)
        with patch.object(api_server, "OUTPUTS", self.outputs), patch.object(api_server, "probe", return_value=info), patch.object(api_server, "make_clip", fake_clip), patch.object(api_server, "render", fake_render):
            api_server._render_job(job["id"], Path(job["_source"]), job_dir, dict(job["_values"]))
        self.assertEqual(job["status"], job.get("error") or "complete")
        self.assertEqual(job["preview_start"], 2.0)
        self.assertEqual(job["preview_seconds"], 2.0)  # clamped to the 4 s source
        self.assertEqual(seen_clip, {"start": 2.0, "length": 2.0, "fps": 25.0})
        self.assertEqual(job["source_duration"], 4.0)
        self.assertNotIn("saved_path", job)
        # A post-only reprocess of that preview must report the same slot and clip.
        (job_dir / "tensorrt_decoded").mkdir(); (job_dir / "tensorrt_decoded" / "decoded_001.pt").touch()

        def fake_reprocess(job_dir_, source, output, **kwargs):
            seen["source"] = source; output.write_bytes(b"post"); return output

        seen: dict[str, object] = {}
        api_server.JOBS["rp1"] = {"id": "rp1", "status": "queued", "job_type": "reprocess"}
        with patch.object(api_server, "OUTPUTS", self.outputs), patch.object(api_server, "probe", return_value=info), patch.object(api_server, "reprocess_tensorrt", fake_reprocess):
            api_server._reprocess_job("rp1", job_dir / "restored.mp4", {})
        reprocessed = api_server.JOBS["rp1"]
        self.assertEqual(reprocessed["status"], reprocessed.get("error") or "complete")
        self.assertEqual(Path(seen["source"]).name, "preview-source.mp4")
        self.assertEqual(reprocessed["preview_start"], 2.0)
        self.assertEqual(reprocessed["source_duration"], 4.0)
        self.assertTrue(reprocessed["original_url"].endswith("/preview-source.mp4"))

    def test_delivery_target_never_overwrites(self) -> None:
        first = api_server._delivery_target(self.video.parent, "holiday")
        self.assertEqual(first, self.video.parent / "holiday-seed.mp4")
        first.touch()
        self.assertEqual(api_server._delivery_target(self.video.parent, "holiday"), self.video.parent / "holiday-seed-2.mp4")


if __name__ == "__main__":
    unittest.main()
