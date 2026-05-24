import os
import sys
import unittest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.training.models.inflation import inflate_conv2d_to_3d, map_state_dict_2d_to_3d


class TestInflation(unittest.TestCase):
    def test_inflate_conv2d_to_3d_center(self):
        weight_2d = torch.randn(2, 3, 3, 3)
        weight_3d = inflate_conv2d_to_3d(weight_2d, kernel_depth=3)
        self.assertEqual(weight_3d.shape, (2, 3, 3, 3, 3))
        self.assertTrue(torch.allclose(weight_3d[:, :, 1, :, :], weight_2d))
        self.assertTrue(torch.allclose(weight_3d[:, :, 0, :, :], torch.zeros_like(weight_2d)))
        self.assertTrue(torch.allclose(weight_3d[:, :, 2, :, :], torch.zeros_like(weight_2d)))

    def test_map_state_dict_2d_to_3d(self):
        state_2d = {
            "conv.weight": torch.randn(2, 3, 3, 3),
            "bias": torch.randn(2),
        }
        state_3d = {
            "conv.weight": torch.zeros(2, 3, 3, 3, 3),
            "bias": torch.zeros(2),
        }
        mapped, missing = map_state_dict_2d_to_3d(state_2d, state_3d)
        self.assertIn("conv.weight", mapped)
        self.assertIn("bias", mapped)
        self.assertEqual(len(missing), 0)


if __name__ == "__main__":
    unittest.main()
