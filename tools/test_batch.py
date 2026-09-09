"""Batch queue behaviour: expansion, skipping, ordering, persistence, and the worker loop."""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api_server
from seedvr_studio.batch import BatchQueue, expand_paths, expected_output


def _wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class BatchQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.clips = self.root / "clips"
        self.clips.mkdir()
        for name in ("b.mp4", "a.mov", "notes.txt", "c-seed.mp4"):
            (self.clips / name).write_bytes(b"x")
        self.store = self.root / "batch-queue.json"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_expand_paths_lists_videos_in_a_folder_sorted_without_previous_results(self) -> None:
        files, missing = expand_paths([str(self.clips), str(self.clips / "a.mov"), str(self.root / "nope.mp4")])
        self.assertEqual([path.name for path in files], ["a.mov", "b.mp4"])
        self.assertEqual(missing, [str(self.root / "nope.mp4")])

    def test_output_goes_to_vrupscale_subfolder(self) -> None:
        self.assertEqual(expected_output(self.clips / "a.mov"), self.clips / "VRUpscale" / "a-seed.mp4")

    def test_add_skips_existing_results_and_duplicates_and_persists(self) -> None:
        (self.clips / "VRUpscale").mkdir()
        (self.clips / "VRUpscale" / "b-seed.mp4").touch()
        queue = BatchQueue(self.store, lambda item, settings, progress: {"status": "done"})
        result = queue.add([str(self.clips)], {"model_label": "3B"})
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["skipped"], ["b.mp4"])
        self.assertEqual(queue.add([str(self.clips / "a.mov")], {})["duplicates"], ["a.mov"])
        statuses = {item["name"]: item["status"] for item in queue.snapshot()["items"]}
        self.assertEqual(statuses, {"a.mov": "queued", "b.mp4": "skipped"})
        saved = json.loads(self.store.read_text())
        self.assertEqual([item["name"] for item in saved["items"]], ["a.mov", "b.mp4"])
        self.assertNotIn("settings", queue.snapshot()["items"][0])
        # A running item interrupted by a restart comes back as queued.
        saved["items"][0]["status"] = "running"
        self.store.write_text(json.dumps(saved))
        reloaded = BatchQueue(self.store, lambda item, settings, progress: {"status": "done"})
        self.assertEqual(reloaded.snapshot()["items"][0]["status"], "queued")

    def test_worker_runs_items_in_order_and_survives_failures(self) -> None:
        order: list[str] = []
        settings_seen: list[dict] = []

        def runner(item, settings, progress):
            order.append(item["name"])
            settings_seen.append(settings)
            progress(0.5, "half way")
            if item["name"] == "a.mov":
                raise RuntimeError("boom")
            return {"status": "done", "output": "/out/b-seed.mp4", "job_id": "j1"}

        queue = BatchQueue(self.store, runner)
        queue.add([str(self.clips)], {"seed": "7"})
        self.assertTrue(queue.start())
        self.assertTrue(_wait(lambda: queue.state == "idle" and all(item["status"] != "queued" for item in queue.items)))
        self.assertEqual(order, ["a.mov", "b.mp4"])
        self.assertEqual(settings_seen[0], {"seed": "7"})
        by_name = {item["name"]: item for item in queue.snapshot()["items"]}
        self.assertEqual(by_name["a.mov"]["status"], "failed")
        self.assertEqual(by_name["a.mov"]["message"], "boom")
        self.assertEqual(by_name["b.mp4"]["status"], "done")
        self.assertEqual(by_name["b.mp4"]["output"], "/out/b-seed.mp4")
        self.assertFalse(queue.start())  # nothing left to do
        self.assertTrue(queue.retry(by_name["a.mov"]["id"]))
        self.assertEqual(queue.clear_finished(), 1)
        self.assertEqual([item["name"] for item in queue.items], ["a.mov"])

    def test_pause_finishes_current_item_then_stops(self) -> None:
        release = threading.Event()
        started = threading.Event()

        def runner(item, settings, progress):
            started.set()
            release.wait(5)
            return {"status": "done"}

        queue = BatchQueue(self.store, runner)
        queue.add([str(self.clips)], {})
        queue.start()
        self.assertTrue(started.wait(5))
        queue.pause()
        self.assertEqual(queue.state, "pausing")
        release.set()
        self.assertTrue(_wait(lambda: queue.state == "idle"))
        statuses = [item["status"] for item in queue.items]
        self.assertEqual(statuses, ["done", "queued"])

    def test_stop_cancels_current_item(self) -> None:
        cancelled = threading.Event()

        def runner(item, settings, progress):
            cancelled.wait(5)
            return {"status": "cancelled", "message": "Render stopped"}

        queue = BatchQueue(self.store, runner, cancel=cancelled.set)
        queue.add([str(self.clips)], {})
        queue.start()
        self.assertTrue(_wait(lambda: queue.current is not None))
        queue.stop()
        self.assertTrue(_wait(lambda: queue.state == "idle"))
        self.assertEqual([item["status"] for item in queue.items], ["cancelled", "queued"])


class BatchEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api_server.app)
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.clips = self.root / "clips"
        self.clips.mkdir()
        (self.clips / "one.mp4").write_bytes(b"one")
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()
        self.queue = BatchQueue(self.root / "queue.json", api_server._batch_runner, lambda: None)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_add_and_run_a_batch_through_the_api(self) -> None:
        def fake_render_job(job_id, source, job_dir, values):
            target = Path(str(values["delivery_dir"])) / f"{values['original_stem']}-seed.mp4"
            target.write_bytes(b"restored")
            api_server._update(job_id, status="complete", saved_path=str(target), output=str(target))

        with patch.object(api_server, "BATCH", self.queue), patch.object(api_server, "OUTPUTS", self.outputs), patch.object(api_server, "ensure_workspace"), patch.object(api_server, "_render_job", fake_render_job):
            api_server.JOBS.clear()
            response = self.client.post("/api/batch/add", data={"paths": json.dumps([str(self.clips)]), "settings": json.dumps({"seed": "5", "output_preset": "2x source", "job_type": "preview"})})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["added"], 1)
            self.assertEqual(self.client.post("/api/batch/start").status_code, 200)
            self.assertTrue(_wait(lambda: self.queue.state == "idle"))
            item = self.client.get("/api/batch").json()["items"][0]
            self.assertEqual(item["status"], "done", item)
            self.assertEqual(item["output"], str(self.clips / "VRUpscale" / "one-seed.mp4"))
            self.assertTrue((self.clips / "VRUpscale" / "one-seed.mp4").is_file())
            job = api_server.JOBS[item["job_id"]]
            self.assertEqual(job["_values"]["job_type"], "full")
            self.assertEqual(job["_values"]["seed"], 5)
            self.assertEqual(job["_values"]["sharpen_strength"], 0.25)  # missing fields get server defaults
            self.assertEqual(job["_values"]["model_label"], "3B FP16 — best 3B quality")
            self.assertEqual(Path(job["_source"]).read_bytes(), b"one")
            # Second add of the same folder skips the finished file.
            response = self.client.post("/api/batch/add", data={"paths": json.dumps([str(self.clips)]), "settings": "{}"})
            self.assertEqual(response.json()["skipped"], ["one.mp4"])

    def test_single_renders_are_refused_while_a_batch_runs(self) -> None:
        with patch.object(api_server, "BATCH", self.queue), patch.object(api_server, "OUTPUTS", self.outputs), patch.object(api_server, "ensure_workspace"):
            self.queue.state = "running"
            try:
                response = self.client.post("/api/jobs", data={"job_type": "preview", "source_path": str(self.clips / "one.mp4")})
            finally:
                self.queue.state = "idle"
        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
