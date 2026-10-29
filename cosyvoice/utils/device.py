import os

import torch


def get_inference_device():
    requested = os.environ.get('COSYVOICE_DEVICE', 'auto').lower()
    if requested == 'auto':
        if torch.cuda.is_available():
            requested = 'cuda'
        elif torch.backends.mps.is_available():
            requested = 'mps'
        else:
            requested = 'cpu'
    elif requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('COSYVOICE_DEVICE=cuda requested, but CUDA is unavailable')
    elif requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('COSYVOICE_DEVICE=mps requested, but MPS is unavailable')
    elif requested not in ('cpu', 'cuda', 'mps'):
        raise ValueError('COSYVOICE_DEVICE must be auto, cpu, cuda, or mps')
    return torch.device(requested)
