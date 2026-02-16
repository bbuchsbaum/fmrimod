#!/usr/bin/env python3.9
"""Integration test for SBHM pipeline."""

import numpy as np
from fmrimod.single.sbhm import (
    build_sbhm_library,
    sbhm_prepass,
    sbhm_match,
    sbhm_amplitude,
    sbhm_single_trial,
)
from fmrimod.single import SbhmConfig

def test_sbhm_library():
    """Test library construction."""
    print("Testing library construction...")
    library_H = np.random.randn(20, 50)
    library = build_sbhm_library(library_H, r=3, normalize=True)

    assert library.B.shape == (20, 3), f"Expected B shape (20, 3), got {library.B.shape}"
    assert library.S.shape == (3,), f"Expected S shape (3,), got {library.S.shape}"
    assert library.A.shape == (50, 3), f"Expected A shape (50, 3), got {library.A.shape}"
    print("✓ Library construction passed")

def test_sbhm_prepass():
    """Test prepass regression."""
    print("Testing prepass...")
    Y = np.random.randn(100, 500)
    A_agg = np.random.randn(100, 3)
    confounds = np.random.randn(100, 5)

    beta_bar, G = sbhm_prepass(Y, A_agg, confounds=confounds)

    assert beta_bar.shape == (3, 500), f"Expected beta_bar shape (3, 500), got {beta_bar.shape}"
    assert G.shape == (3, 3), f"Expected G shape (3, 3), got {G.shape}"
    print("✓ Prepass passed")

def test_sbhm_match():
    """Test HRF matching."""
    print("Testing matching...")
    beta_bar = np.random.randn(3, 500)
    S = np.array([10.0, 5.0, 2.0])
    A = np.random.randn(50, 3)

    # Test top_k=1
    result = sbhm_match(beta_bar, S, A, shrink=True, top_k=1)
    assert result["matched_idx"].shape == (500,), f"Expected matched_idx shape (500,), got {result['matched_idx'].shape}"
    assert result["margin"].shape == (500,), f"Expected margin shape (500,), got {result['margin'].shape}"
    assert result["alpha_coords"].shape == (3, 500), f"Expected alpha_coords shape (3, 500), got {result['alpha_coords'].shape}"

    # Test top_k=3
    result = sbhm_match(beta_bar, S, A, shrink=False, top_k=3)
    assert result["matched_idx"].shape == (500, 3), f"Expected matched_idx shape (500, 3), got {result['matched_idx'].shape}"
    assert result["weights"].shape == (500, 3), f"Expected weights shape (500, 3), got {result['weights'].shape}"
    print("✓ Matching passed")

def test_sbhm_amplitude():
    """Test amplitude estimation."""
    print("Testing amplitude estimation...")
    Y = np.random.randn(100, 500)
    X_trials = np.random.randn(100, 60)  # 20 trials x 3 basis
    alpha_coords = np.random.randn(3, 500)
    confounds = np.random.randn(100, 5)

    # Test global_ls
    betas = sbhm_amplitude(Y, X_trials, alpha_coords, confounds=confounds, method="global_ls", K=3)
    assert betas.shape == (20, 500), f"Expected betas shape (20, 500), got {betas.shape}"

    # Test lss1 (smaller for speed)
    Y_small = Y[:, :10]
    alpha_small = alpha_coords[:, :10]
    betas = sbhm_amplitude(Y_small, X_trials, alpha_small, confounds=confounds, method="lss1", K=3)
    assert betas.shape == (20, 10), f"Expected betas shape (20, 10), got {betas.shape}"

    # Test oasis_voxel (smaller for speed)
    betas = sbhm_amplitude(Y_small, X_trials, alpha_small, confounds=confounds, method="oasis_voxel", K=3)
    assert betas.shape == (20, 10), f"Expected betas shape (20, 10), got {betas.shape}"

    print("✓ Amplitude estimation passed")

def test_sbhm_pipeline():
    """Test full SBHM pipeline."""
    print("Testing full pipeline...")

    # Build library
    library_H = np.random.randn(20, 50)
    library = build_sbhm_library(library_H, r=3)

    # Generate synthetic data
    Y = np.random.randn(100, 50)  # Smaller for speed
    X = np.random.randn(100, 60)  # 20 trials x 3 basis
    confounds = np.random.randn(100, 5)
    trial_labels = [f"trial_{i}" for i in range(20)]

    # Test with default config
    config = SbhmConfig(r=3, amplitude_method="oasis_voxel")
    result = sbhm_single_trial(Y, X, confounds=confounds, config=config, trial_labels=trial_labels, library=library)

    assert result.betas.shape == (20, 50), f"Expected betas shape (20, 50), got {result.betas.shape}"
    assert result.method == "sbhm", f"Expected method 'sbhm', got {result.method}"
    assert result.trial_labels == trial_labels, "Trial labels mismatch"
    assert "matched_idx" in result.extra, "Missing matched_idx in extra"
    assert "margin" in result.extra, "Missing margin in extra"
    assert "alpha_coords" in result.extra, "Missing alpha_coords in extra"

    print("✓ Full pipeline passed")

if __name__ == "__main__":
    np.random.seed(42)

    test_sbhm_library()
    test_sbhm_prepass()
    test_sbhm_match()
    test_sbhm_amplitude()
    test_sbhm_pipeline()

    print("\n✅ All SBHM integration tests passed!")
