"""Face tracking and blend maths without loading any model."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_restore import FACE_TEMPLATE, FaceTracker, detail_preserving_merge, iou
from run_rtx_vsr import plan_passes


def _det(x, y, size):
    bbox = np.array([x, y, x + size, y + size], dtype=np.float32)
    landmarks = FACE_TEMPLATE / 512.0 * size + np.array([x, y], dtype=np.float32)
    return bbox, landmarks


class TrackerTests(unittest.TestCase):
    def test_iou(self) -> None:
        self.assertAlmostEqual(iou(np.array([0, 0, 10, 10]), np.array([5, 0, 15, 10])), 1 / 3, places=5)
        self.assertEqual(iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])), 0.0)

    def test_tracks_persist_and_landmarks_smooth(self) -> None:
        tracker = FaceTracker(history=5)
        first = tracker.update([_det(100, 100, 200)])
        self.assertEqual([r[0] for r in first], [1])
        # Same face moved by 4 px: same id, smoothed landmarks between old and new.
        second = tracker.update([_det(104, 100, 200)])
        self.assertEqual(second[0][0], 1)
        self.assertEqual(second[0][3], 2)
        raw = _det(104, 100, 200)[1]
        self.assertTrue(np.all(second[0][2][:, 0] < raw[:, 0]))
        self.assertTrue(np.all(second[0][2][:, 0] > _det(100, 100, 200)[1][:, 0]))
        # A second, separate face gets a new id.
        third = tracker.update([_det(108, 100, 200), _det(900, 50, 120)])
        self.assertEqual(sorted(r[0] for r in third), [1, 2])

    def test_missing_faces_expire_after_max_missed(self) -> None:
        tracker = FaceTracker(max_missed=1)
        tracker.update([_det(0, 0, 100)])
        tracker.update([])
        self.assertIn(1, tracker.tracks)
        tracker.update([])
        self.assertNotIn(1, tracker.tracks)
        # Re-appearing face becomes a new track.
        self.assertEqual(tracker.update([_det(0, 0, 100)])[0][0], 2)

    def test_state_round_trip(self) -> None:
        tracker = FaceTracker()
        tracker.update([_det(10, 10, 50)])
        tracker.update([_det(12, 10, 50)])
        restored = FaceTracker.from_state(tracker.to_state())
        self.assertEqual(restored.next_id, 2)
        self.assertEqual(len(restored.tracks[1]["landmarks"]), 2)
        self.assertEqual(restored.update([_det(14, 10, 50)])[0][0], 1)


class MergeTests(unittest.TestCase):
    def test_keep_detail_zero_returns_restored_and_one_keeps_original_texture(self) -> None:
        rng = np.random.default_rng(0)
        original = rng.random((64, 64, 3), dtype=np.float32)
        restored = np.full((64, 64, 3), 0.5, dtype=np.float32)
        np.testing.assert_array_equal(detail_preserving_merge(restored, original, 0.0, 1.0), restored)
        merged = detail_preserving_merge(restored, original, 1.0, 1.0)
        # Mean stays with the restorer, fine variation comes from the original.
        self.assertAlmostEqual(float(merged.mean()), 0.5, places=2)
        self.assertGreater(float(merged.std()), 0.2)


class RtxPlanTests(unittest.TestCase):
    def test_up_to_4x_is_one_pass_and_more_chains(self) -> None:
        self.assertEqual(plan_passes(608, 352, 1216, 704), [(1216, 704)])
        self.assertEqual(plan_passes(608, 352, 2432, 1408), [(2432, 1408)])
        self.assertEqual(plan_passes(608, 352, 608, 352), [(608, 352)])
        passes = plan_passes(480, 270, 3840, 2160)  # 8x
        self.assertEqual(passes[-1], (3840, 2160))
        self.assertEqual(len(passes), 2)
        self.assertEqual(passes[0], (1920, 1080))


if __name__ == "__main__":
    unittest.main()
