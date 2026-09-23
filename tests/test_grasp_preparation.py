"""Common hardware/simulation preparation guards without loading models."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from airbot_yolo import Detection
from grasp_preparation import prepare_color_grasp


class PreparationTest(unittest.TestCase):
    def setUp(self):
        self.snapshot = SimpleNamespace(color=np.zeros((40, 40, 3), np.uint8),
                                        depth=np.ones((40, 40)), timestamp=123.,
                                        state={'trans': [0, 0, 0], 'orient': [0, 0, 0, 1]})
        self.detection = Detection((10, 10, 30, 30), 'block', .9, 'blue', 1.)
        self.mask = np.zeros((40, 40), dtype=bool)
        self.mask[10:30, 10:30] = True
        self.segment = Mock()
        self.segment.inference_bbox.return_value = self.mask
        self.camera = Mock(has_hardware_depth=True, depth_factor=1.,
                           intrinsic=[[100, 0, 20], [0, 100, 20], [0, 0, 1]])
        self.camera.create_point_cloud.return_value = np.ones((40, 40, 3))
        self.grasp = Mock(length_minor=4., angle=0.)
        self.grasp.inference.return_value = (np.array([.3, -.1, .03]), np.array([0, 0, 0, 1]),
                                             np.array([[.3, -.1, .04]]))

    def prepare(self):
        return prepare_color_grasp(self.snapshot, self.detection, [self.detection],
                                   self.segment, self.camera, self.grasp, {}, save_cloud=False)

    def test_shared_pipeline_keeps_snapshot_pose_and_width_units(self):
        info, preview = self.prepare()
        self.assertEqual(info['snapshot_timestamp'], 123.)
        self.assertAlmostEqual(info['o_width'], .04)
        self.assertEqual(info['o_bbox'], [10, 10, 30, 30])
        self.assertEqual(self.grasp.inference.call_args.kwargs['end_pose'], [[0, 0, 0], [0, 0, 0, 1]])
        self.assertEqual(preview.shape, self.snapshot.color.shape)
        self.assertTrue(np.any(preview[self.mask]))

    def test_missing_small_and_outside_masks_stop_before_prediction(self):
        for mask in (None, np.zeros((40, 40), bool), np.eye(40, dtype=bool),
                     np.ones((40, 40), bool)):
            self.segment.inference_bbox.return_value = mask
            with self.assertRaises(ValueError):
                self.prepare()
        self.grasp.inference.assert_not_called()

    def test_missing_depth_stops_before_prediction(self):
        self.snapshot.depth[:] = 0
        with self.assertRaisesRegex(ValueError, '有效深度'):
            self.prepare()
        self.grasp.inference.assert_not_called()

    def test_prediction_failure_is_not_a_grasp(self):
        self.grasp.inference.return_value = None, None, None
        with self.assertRaisesRegex(ValueError, '位姿计算失败'):
            self.prepare()
