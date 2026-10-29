import os
import unittest
from unittest.mock import patch

import torch

from cosyvoice.utils.device import get_inference_device


class InferenceDeviceTests(unittest.TestCase):
    @patch.dict(os.environ, {'COSYVOICE_DEVICE': 'auto'})
    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.cuda.is_available', return_value=False)
    def test_auto_uses_mps(self, _cuda_available, _mps_available):
        self.assertEqual(get_inference_device(), torch.device('mps'))

    @patch.dict(os.environ, {'COSYVOICE_DEVICE': 'auto'})
    @patch('torch.backends.mps.is_available', return_value=False)
    @patch('torch.cuda.is_available', return_value=False)
    def test_auto_falls_back_to_cpu(self, _cuda_available, _mps_available):
        self.assertEqual(get_inference_device(), torch.device('cpu'))

    @patch.dict(os.environ, {'COSYVOICE_DEVICE': 'mps'})
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_forced_mps_fails_when_unavailable(self, _mps_available):
        with self.assertRaisesRegex(RuntimeError, 'MPS is unavailable'):
            get_inference_device()


if __name__ == '__main__':
    unittest.main()
