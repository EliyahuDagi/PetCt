
import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.segmentation_utils import SegmentationPredictor

class TestSegmentation(unittest.TestCase):
    def test_predictor_initialization(self):
        # We assume the bundle is downloaded by the script or mock it.
        # Here we just check import and instantation logic if we mock bundle path
        pass

    def test_dummy_transforms(self):
        # Test if transforms chain can run on random data
        from monai.transforms import Compose, EnsureChannelFirstd, ToTensord
        trans = Compose([
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            ToTensord(keys=["image"])
        ])
        data = {"image": np.zeros((10, 10, 10))}
        out = trans(data)
        self.assertEqual(out["image"].shape, (1, 10, 10, 10))

if __name__ == '__main__':
    unittest.main()
