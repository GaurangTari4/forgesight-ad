"""Pixel-equivalence coverage for allocation-reducing visualization changes."""
import numpy as np

import demo_app


def test_heatmap_overlay_matches_reference_exactly():
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    heatmap = rng.random((64, 64), dtype=np.float32)
    reference = demo_app._build_heatmap_overlay_reference(image, heatmap)
    optimized = demo_app.build_heatmap_overlay(image, heatmap)
    assert optimized.shape == reference.shape == image.shape
    assert np.array_equal(optimized, reference)


def test_mask_overlay_retains_image_alignment_and_finite_values():
    image = np.full((32, 48, 3), 100, dtype=np.uint8)
    mask = np.zeros((32, 48), dtype=np.float32)
    mask[10:20, 15:30] = 1.0
    overlay = demo_app.build_mask_overlay(image, mask)
    assert overlay.shape == image.shape
    assert np.isfinite(overlay).all()
