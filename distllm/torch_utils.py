"""Torch utilities with optional Intel XPU support.

This module provides a centralized way to handle device selection
with support for CUDA, Intel XPU (via intel_extension_for_pytorch),
and CPU fallback.
"""

from __future__ import annotations

from typing import Literal

import torch

# Try to import Intel Extension for PyTorch for XPU support
try:
    import intel_extension_for_pytorch as ipex  # noqa: F401

    _HAS_IPEX = True
except ImportError:
    _HAS_IPEX = False


DeviceType = Literal['cuda', 'xpu', 'cpu', 'auto']


def has_xpu() -> bool:
    """Check if XPU is available.

    Returns
    -------
    bool
        True if Intel XPU is available, False otherwise.
    """
    return _HAS_IPEX and hasattr(torch, 'xpu') and torch.xpu.is_available()


def has_cuda() -> bool:
    """Check if CUDA is available.

    Returns
    -------
    bool
        True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()


def get_device(preferred: DeviceType | None = None) -> torch.device:
    """Get the best available device for computation.

    Parameters
    ----------
    preferred : DeviceType | None, optional
        The preferred device type. If None or 'auto', the best available
        device will be selected automatically. Options are:
        - 'cuda': Use NVIDIA GPU
        - 'xpu': Use Intel GPU (requires intel_extension_for_pytorch)
        - 'cpu': Use CPU
        - 'auto' or None: Auto-detect best available device

    Returns
    -------
    torch.device
        The selected device.

    Raises
    ------
    RuntimeError
        If the preferred device is not available.
    """
    # Handle explicit device selection
    if preferred is not None and preferred != 'auto':
        if preferred == 'cuda':
            if not has_cuda():
                raise RuntimeError(
                    'CUDA device requested but CUDA is not available.',
                )
            return torch.device('cuda')
        elif preferred == 'xpu':
            if not has_xpu():
                raise RuntimeError(
                    'XPU device requested but XPU is not available. '
                    'Make sure intel_extension_for_pytorch is installed.',
                )
            return torch.device('xpu')
        elif preferred == 'cpu':
            return torch.device('cpu')
        else:
            # Try to use the device string directly
            return torch.device(preferred)

    # Auto-detect: prefer CUDA > XPU > CPU
    if has_cuda():
        return torch.device('cuda')
    if has_xpu():
        return torch.device('xpu')
    return torch.device('cpu')


def get_device_info() -> dict[str, bool | str]:
    """Get information about available devices.

    Returns
    -------
    dict[str, bool | str]
        A dictionary containing:
        - 'cuda_available': bool
        - 'xpu_available': bool
        - 'ipex_installed': bool
        - 'selected_device': str (the device that would be auto-selected)
    """
    selected = get_device()
    return {
        'cuda_available': has_cuda(),
        'xpu_available': has_xpu(),
        'ipex_installed': _HAS_IPEX,
        'selected_device': str(selected),
    }

