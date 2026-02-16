"""Compare performance between R and Python implementations."""

import time
import pytest
import numpy as np
from fmrimod import regressor, regressor_set, SamplingFrame


class TestPerformance:
    """Compare performance between implementations."""
    
    @pytest.mark.benchmark
    def test_large_scale_evaluation(self, r_tester):
        """Test performance with large datasets."""
        n_events = 1000
        n_timepoints = 10000
        
        # Generate data
        np.random.seed(42)
        onsets = np.sort(np.random.uniform(0, 5000, n_events))
        times = np.linspace(0, 5000, n_timepoints)
        
        # Python timing
        py_start = time.time()
        py_reg = regressor(onsets=onsets, hrf="spmg1")
        py_result = py_reg.evaluate(times)
        py_time = time.time() - py_start
        
        # R timing
        r_tester.r.assign('onsets', onsets)
        r_tester.r.assign('times', times)
        r_start = time.time()
        r_tester.r("""
        reg <- regressor(onsets = onsets, hrf = HRF_SPMG1)
        result <- evaluate(reg, times)
        """)
        r_time = time.time() - r_start
        
        # Report
        print(f"\nPerformance Comparison (Large Scale):")
        print(f"  Events: {n_events}, Timepoints: {n_timepoints}")
        print(f"  Python: {py_time:.3f}s")
        print(f"  R: {r_time:.3f}s")
        print(f"  Ratio (R/Python): {r_time/py_time:.2f}x")
        
        # Still verify equivalence
        r_result = r_tester.r['result']
        tol = r_tester.get_tolerance('large')
        r_tester.compare_arrays(r_result, py_result, **tol)
    
    @pytest.mark.benchmark
    def test_design_matrix_construction(self, r_tester):
        """Test performance of design matrix construction."""
        # Complex experimental design
        n_conditions = 5
        n_events_per_condition = 50
        n_scans = 500
        
        # Generate events
        np.random.seed(42)
        all_onsets = []
        all_conditions = []
        
        for cond in range(n_conditions):
            onsets = np.sort(np.random.uniform(10, 900, n_events_per_condition))
            all_onsets.extend(onsets)
            all_conditions.extend([cond] * n_events_per_condition)
        
        # Sort by onset time
        sort_idx = np.argsort(all_onsets)
        all_onsets = np.array(all_onsets)[sort_idx]
        all_conditions = np.array(all_conditions)[sort_idx]
        
        # Python timing
        py_start = time.time()
        sf = SamplingFrame(blocklens=n_scans, tr=2.0)
        py_rset = regressor_set(
            onsets=all_onsets,
            fac=all_conditions,
            hrf="spmg1"
        )
        py_design = py_rset.evaluate(sf.samples)
        py_time = time.time() - py_start
        
        # R timing
        r_tester.r.assign('onsets', all_onsets)
        r_tester.r.assign('conditions', all_conditions)
        r_tester.r.assign('n_scans', n_scans)
        
        r_start = time.time()
        r_tester.r("""
        sf <- sampling_frame(n_scans, TR = 2.0)
        rset <- regressor_set(
            onsets = onsets,
            fac = as.factor(conditions),
            hrf = HRF_SPMG1
        )
        design <- regressor_matrix(rset, sf)
        """)
        r_time = time.time() - r_start
        
        # Report
        print(f"\nPerformance Comparison (Design Matrix):")
        print(f"  Conditions: {n_conditions}, Events/condition: {n_events_per_condition}")
        print(f"  Design shape: {py_design.shape}")
        print(f"  Python: {py_time:.3f}s")
        print(f"  R: {r_time:.3f}s")
        print(f"  Ratio (R/Python): {r_time/py_time:.2f}x")
        
        # Verify equivalence
        r_design = r_tester.r['design']
        tol = r_tester.get_tolerance('matrix')
        r_tester.compare_arrays(r_design, py_design, **tol)
    
    @pytest.mark.benchmark
    def test_hrf_evaluation_performance(self, r_tester):
        """Test HRF evaluation performance."""
        # Test different HRF types
        hrf_types = ["spmg1", "gamma", "gaussian"]
        n_points = 50000
        
        print("\nHRF Evaluation Performance:")
        
        for hrf_type in hrf_types:
            t = np.linspace(0, 100, n_points)
            
            # Python
            from fmrimod import get_hrf
            py_hrf = get_hrf(hrf_type)
            
            py_start = time.time()
            py_result = py_hrf(t)
            py_time = time.time() - py_start
            
            # R
            r_hrf_name = f"HRF_{hrf_type.upper()}"
            r_tester.r.assign('t', t)
            r_tester.r.assign('hrf_name', r_hrf_name)
            
            r_start = time.time()
            r_tester.r(f"""
            hrf_obj <- get_hrf({r_hrf_name})
            result <- hrf_obj(t)
            """)
            r_time = time.time() - r_start
            
            print(f"  {hrf_type}:")
            print(f"    Python: {py_time*1000:.1f}ms")
            print(f"    R: {r_time*1000:.1f}ms")
            print(f"    Ratio: {r_time/py_time:.2f}x")
            
            # Verify equivalence
            r_result = r_tester.r['result']
            r_tester.compare_arrays(r_result, py_result, rtol=1e-8)
    
    @pytest.mark.benchmark
    def test_sparse_operations(self, r_tester):
        """Test performance with sparse operations."""
        # Create sparse scenario
        n_events = 20
        n_timepoints = 5000
        
        # Widely spaced events
        onsets = np.linspace(100, 4900, n_events)
        
        # Python timing - sparse
        py_start = time.time()
        py_reg = regressor(onsets=onsets, hrf="spmg1")
        sf = SamplingFrame(blocklens=n_timepoints, tr=1.0)
        py_sparse = py_reg.evaluate(sf.samples, sparse=True)
        py_time_sparse = time.time() - py_start
        
        # Python timing - dense
        py_start = time.time()
        py_dense = py_reg.evaluate(sf.samples, sparse=False)
        py_time_dense = time.time() - py_start
        
        # R timing
        r_tester.r.assign('onsets', onsets)
        r_tester.r.assign('n_timepoints', n_timepoints)
        
        # R sparse
        r_start = time.time()
        r_tester.r("""
        reg <- regressor(onsets = onsets, hrf = HRF_SPMG1)
        sf <- sampling_frame(n_timepoints, TR = 1.0)
        result_sparse <- evaluate(reg, samples(sf, global = TRUE), sparse = TRUE)
        """)
        r_time_sparse = time.time() - r_start
        
        # R dense
        r_start = time.time()
        r_tester.r("""
        result_dense <- evaluate(reg, samples(sf, global = TRUE), sparse = FALSE)
        """)
        r_time_dense = time.time() - r_start
        
        # Report
        print(f"\nSparse Operations Performance:")
        print(f"  Events: {n_events}, Timepoints: {n_timepoints}")
        print(f"  Sparse:")
        print(f"    Python: {py_time_sparse:.3f}s")
        print(f"    R: {r_time_sparse:.3f}s")
        print(f"    Ratio: {r_time_sparse/py_time_sparse:.2f}x")
        print(f"  Dense:")
        print(f"    Python: {py_time_dense:.3f}s")
        print(f"    R: {r_time_dense:.3f}s")
        print(f"    Ratio: {r_time_dense/py_time_dense:.2f}x")
        
        # Calculate sparsity
        if hasattr(py_sparse, 'nnz'):
            sparsity = 1 - (py_sparse.nnz / (py_sparse.shape[0] * py_sparse.shape[1]))
            print(f"  Sparsity: {sparsity:.1%}")