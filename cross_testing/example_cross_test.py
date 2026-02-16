#!/usr/bin/env python
"""Example of how to use the cross-testing infrastructure for custom comparisons."""

import numpy as np
from cross_testing.utils import REquivalenceTester


def main():
    """Run example cross-test comparisons."""
    # Initialize the tester
    tester = REquivalenceTester()
    
    print("Running example cross-tests...\n")
    
    # Example 1: Compare HRF evaluations
    print("1. Comparing HRF evaluations:")
    t = np.linspace(0, 30, 100)
    
    # Python
    from fmrimod import spm_canonical
    py_hrf = spm_canonical(t)
    
    # R
    tester.r.assign('t', t)
    r_hrf = tester.r('hrf_spmg1(t)')
    
    # Compare
    try:
        tester.compare_arrays(r_hrf, py_hrf)
        print("   ✅ SPM canonical HRF matches perfectly!")
    except AssertionError as e:
        print(f"   ❌ Mismatch: {e}")
    
    # Example 2: Compare regressor output
    print("\n2. Comparing regressor evaluations:")
    
    # Python
    from fmrimod import regressor, SamplingFrame
    py_reg = regressor(onsets=[10, 30, 50], hrf="spmg1", duration=2.0)
    sf = SamplingFrame(blocklens=100, tr=2.0)
    py_result = py_reg.evaluate(sf.samples)
    
    # R
    r_code = """
    reg <- regressor(
        onsets = c(10, 30, 50),
        hrf = HRF_SPMG1,
        duration = 2.0
    )
    sf <- sampling_frame(100, TR = 2.0)
    result <- evaluate(reg, samples(sf, global = TRUE))
    """
    r_vars = tester.run_r_code(r_code)
    
    # Compare
    try:
        tester.compare_arrays(r_vars['result'], py_result)
        print("   ✅ Regressor evaluation matches perfectly!")
    except AssertionError as e:
        print(f"   ❌ Mismatch: {e}")
    
    # Example 3: Compare design matrices
    print("\n3. Comparing design matrices:")
    
    # Python
    from fmrimod import regressor_set
    py_rset = regressor_set(
        onsets=[5, 15, 25, 35],
        fac=['A', 'B', 'A', 'B'],
        hrf="gamma"
    )
    py_design = py_rset.evaluate(sf.samples)
    
    # R
    r_code = """
    rset <- regressor_set(
        onsets = c(5, 15, 25, 35),
        fac = factor(c('A', 'B', 'A', 'B')),
        hrf = HRF_GAMMA
    )
    design <- regressor_matrix(rset, sf)
    """
    r_vars = tester.run_r_code(r_code)
    
    # Compare
    try:
        tester.compare_arrays(r_vars['design'], py_design)
        print("   ✅ Design matrices match perfectly!")
        print(f"      Shape: {py_design.shape}")
        print(f"      Conditions: {py_rset.levels}")
    except AssertionError as e:
        print(f"   ❌ Mismatch: {e}")
    
    # Example 4: Performance comparison
    print("\n4. Performance comparison:")
    import time
    
    n_events = 500
    onsets = np.sort(np.random.uniform(0, 1000, n_events))
    
    # Python timing
    start = time.time()
    py_reg = regressor(onsets=onsets, hrf="spmg1")
    py_result = py_reg.evaluate(np.linspace(0, 1000, 2000))
    py_time = time.time() - start
    
    # R timing
    tester.r.assign('onsets', onsets)
    start = time.time()
    tester.r("""
    reg <- regressor(onsets = onsets, hrf = HRF_SPMG1)
    result <- evaluate(reg, seq(0, 1000, length.out = 2000))
    """)
    r_time = time.time() - start
    
    print(f"   Python: {py_time*1000:.1f}ms")
    print(f"   R: {r_time*1000:.1f}ms")
    print(f"   Speed ratio (R/Python): {r_time/py_time:.2f}x")
    
    # Example 5: Custom tolerance comparison
    print("\n5. Using custom tolerances:")
    
    # Get appropriate tolerance for matrix operations
    tol = tester.get_tolerance('matrix')
    
    # Create a large matrix operation
    from fmrimod import penalty_matrix, get_hrf
    hrf = get_hrf("bspline")
    py_penalty = penalty_matrix(hrf, order=2)
    
    # R equivalent
    r_code = """
    hrf <- get_hrf(HRF_BSPLINE)
    penalty <- penalty_matrix(hrf, order = 2)
    """
    r_vars = tester.run_r_code(r_code)
    
    # Compare with relaxed tolerance
    try:
        tester.compare_arrays(r_vars['penalty'], py_penalty, **tol)
        print("   ✅ Penalty matrices match within tolerance!")
        print(f"      Used rtol={tol['rtol']}, atol={tol['atol']}")
    except AssertionError as e:
        print(f"   ❌ Mismatch: {e}")
    
    print("\nExample cross-tests completed!")


if __name__ == "__main__":
    main()