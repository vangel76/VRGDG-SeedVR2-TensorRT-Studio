"""Browser regression tests; start ./run.sh and install Playwright first."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


@unittest.skipIf(sync_playwright is None, "Playwright is required for browser tests")
class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.video = Path(cls.directory.name) / "test.webm"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc2=size=128x128:rate=30", "-t", "3", "-c:v", "libvpx",
            str(cls.video),
        ], check=True)
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(
            executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE"),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.browser.close()
        cls.playwright.stop()
        cls.directory.cleanup()

    def setUp(self) -> None:
        self.page = self.browser.new_page()
        self.page.set_default_timeout(5000)
        self.page.route("**/viewer-test.webm", lambda route: route.fulfill(
            body=self.video.read_bytes(), content_type="video/webm"))
        self.page.route("**/api/outputs", lambda route: route.fulfill(
            body="[]", content_type="application/json"))
        self.page.goto(os.environ.get("SEEDVR_TEST_URL", "http://127.0.0.1:7870"))

    def tearDown(self) -> None:
        self.page.close()

    def check_seek(self, ids: list[str]) -> None:
        self.page.wait_for_function("Number(document.querySelector('#seek').max) > 2.9")
        slider = self.page.locator("#seek")
        slider.scroll_into_view_if_needed()
        box = slider.bounding_box()
        slider.click(position={"x": box["width"] * 0.75, "y": box["height"] / 2})
        for video in ids:
            self.page.wait_for_function("id => document.getElementById(id).currentTime > 2", arg=video)
        self.page.mouse.move(box["x"] + box["width"] * 0.75, box["y"] + box["height"] / 2)
        self.page.mouse.down()
        self.page.mouse.move(box["x"] + box["width"] * 0.25, box["y"] + box["height"] / 2, steps=8)
        self.page.mouse.up()
        for video in ids:
            self.page.wait_for_function("id => { const t = document.getElementById(id).currentTime; return t > 0.4 && t < 1.2; }", arg=video)

    def load_media(self, video: str) -> None:
        self.page.locator(f"#{video}").evaluate("async video => { const response = await fetch('/viewer-test.webm'); video.src = URL.createObjectURL(await response.blob()); video.load(); }")
        self.page.wait_for_function("id => document.getElementById(id).readyState >= 2", arg=video)

    def test_original_only_timeline_and_playback(self) -> None:
        self.page.locator("#source-file").set_input_files(self.video)
        self.page.locator('[data-mode="original"]').click()
        self.check_seek(["before"])
        self.page.locator("#play").click()
        self.page.wait_for_function("!document.querySelector('#before').paused")

    def test_restored_only_timeline_and_frame_step(self) -> None:
        self.load_media("after")
        self.page.locator('[data-mode="restored"]').click()
        self.check_seek(["after"])
        previous = self.page.locator("#after").evaluate("video => video.currentTime")
        self.page.locator("#next").click()
        self.assertGreater(self.page.locator("#after").evaluate("video => video.currentTime"), previous)

    def test_comparison_timeline_seeks_both_videos(self) -> None:
        self.load_media("before")
        self.load_media("after")
        self.check_seek(["before", "after"])


if __name__ == "__main__":
    unittest.main()
