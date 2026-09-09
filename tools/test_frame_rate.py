from __future__ import annotations

import math
import unittest

from seedvr_studio.media import FrameRateRequiredError, align_clip_to_frames, resolve_frame_rate


class ResolveFrameRateTests(unittest.TestCase):
    def test_detected_frame_rates_are_preserved(self) -> None:
        self.assertEqual(resolve_frame_rate(48.0), 48.0)
        self.assertEqual(resolve_frame_rate(29.97), 29.97)

    def test_override_is_used_when_detection_fails(self) -> None:
        self.assertEqual(resolve_frame_rate(0.0, 23.976), 23.976)
        self.assertEqual(resolve_frame_rate(math.nan, 60), 60.0)

    def test_invalid_frame_rates_require_user_input(self) -> None:
        for detected, override in ((0, 0), (math.nan, 0), (241, -1), (None, "bad")):
            with self.subTest(detected=detected, override=override):
                with self.assertRaises(FrameRateRequiredError):
                    resolve_frame_rate(detected, override)


class AlignClipToFramesTests(unittest.TestCase):
    def test_start_snaps_down_to_the_frame_shown_at_that_time(self) -> None:
        start, length, first = align_clip_to_frames(2.397, 0.5, 24.0)
        self.assertEqual(first, 57)
        self.assertAlmostEqual(start, 57 / 24)
        self.assertAlmostEqual(length, 12 / 24)

    def test_exact_boundaries_and_fractional_rates(self) -> None:
        self.assertEqual(align_clip_to_frames(3.0, 3.0, 24.0)[2], 72)
        start, length, first = align_clip_to_frames(3.0, 1.0, 29.97)
        self.assertEqual(first, 89)
        self.assertAlmostEqual(length, 30 / 29.97)
        self.assertEqual(align_clip_to_frames(0.0, 0.01, 24.0), (0.0, 1 / 24, 0))

    def test_unknown_fps_leaves_values_alone(self) -> None:
        self.assertEqual(align_clip_to_frames(2.397, 0.5, 0.0), (2.397, 0.5, 0))


if __name__ == "__main__":
    unittest.main()
