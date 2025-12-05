#!/usr/bin/env python
"""Test script for XPU support in distllm.

This script tests Intel XPU support by:
1. Checking available devices (CUDA, XPU, CPU)
2. Loading a small embedding model
3. Running inference on the detected device

Requirements:
- For XPU support: pip install intel_extension_for_pytorch
- For testing: pip install transformers torch

Usage:
    python examples/test_xpu.py
    python examples/test_xpu.py --device xpu  # Force XPU
    python examples/test_xpu.py --device cuda  # Force CUDA
    python examples/test_xpu.py --device cpu   # Force CPU
"""

from __future__ import annotations

import argparse
import sys
import time


def print_section(title: str) -> None:
    """Print a section header."""
    print(f'\n{"=" * 60}')
    print(f' {title}')
    print('=' * 60)


def test_device_detection() -> None:
    """Test device detection utilities."""
    print_section('Device Detection')

    from distllm.torch_utils import get_device_info
    from distllm.torch_utils import has_cuda
    from distllm.torch_utils import has_xpu

    info = get_device_info()

    print(f'CUDA available:      {has_cuda()}')
    print(f'XPU available:       {has_xpu()}')
    print(f'IPEX installed:      {info["ipex_installed"]}')
    print(f'Auto-selected device: {info["selected_device"]}')


def test_encoder(device: str | None = None) -> None:
    """Test encoder with specified device."""
    print_section(f'Encoder Test (device={device or "auto"})')

    from distllm.embed.encoders.auto import AutoEncoder
    from distllm.embed.encoders.auto import AutoEncoderConfig
    from distllm.torch_utils import get_device

    # Use a small, fast model for testing
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    print(f'Loading model: {model_name}')
    print(f'Requested device: {device or "auto"}')

    # Create config with device setting
    config = AutoEncoderConfig(
        pretrained_model_name_or_path=model_name,
        quantization=False,  # Disable quantization for XPU compatibility
        half_precision=False,
        eval_mode=True,
        device=device,
    )

    # Time the model loading
    start = time.perf_counter()
    encoder = AutoEncoder(config)
    load_time = time.perf_counter() - start

    print(f'Model loaded in {load_time:.2f}s')
    print(f'Model device: {encoder.device}')
    print(f'Model dtype: {encoder.dtype}')
    print(f'Embedding size: {encoder.embedding_size}')

    # Test inference
    print('\nRunning inference test...')
    test_texts = [
        'This is a test sentence for XPU support.',
        'Intel Extension for PyTorch enables XPU acceleration.',
        'distllm supports distributed inference at scale.',
    ]

    # Tokenize
    batch_encoding = encoder.tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )

    # Move to device
    batch_encoding = batch_encoding.to(encoder.device)

    # Time inference
    start = time.perf_counter()
    with __import__('torch').no_grad():
        embeddings = encoder.encode(batch_encoding)
    inference_time = time.perf_counter() - start

    print(f'Inference completed in {inference_time * 1000:.2f}ms')
    print(f'Output shape: {embeddings.shape}')
    print(f'Output device: {embeddings.device}')

    # Verify output
    assert embeddings.shape[0] == len(test_texts), 'Batch size mismatch'
    assert embeddings.shape[2] == encoder.embedding_size, 'Embedding size mismatch'

    print('\n✓ Encoder test passed!')


def main() -> int:
    """Run XPU support tests."""
    parser = argparse.ArgumentParser(description='Test XPU support in distllm')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'xpu', 'cpu', 'auto'],
        help='Device to use for testing (default: auto-detect)',
    )
    args = parser.parse_args()

    print('distllm XPU Support Test')
    print('========================')

    try:
        test_device_detection()
        test_encoder(args.device)

        print_section('All Tests Passed!')
        return 0

    except Exception as e:
        print(f'\n✗ Test failed with error: {e}')
        import traceback

        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

