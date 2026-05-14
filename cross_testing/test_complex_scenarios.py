"""Test complex real-world scenarios for R-Python equivalence."""

import numpy as np
import pandas as pd

from fmrimod import SamplingFrame, get_hrf, regressor, regressor_set
from fmrimod.hrf import hrf_library, penalty_matrix, reconstruction_matrix
from fmrimod.hrf.generators import gamma_generator


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_full_glm_pipeline(self, r_tester):
        """Test complete GLM pipeline."""
        # Generate synthetic fMRI data parameters
        n_scans = 200
        tr = 2.0
        
        # Create events
        event_times = np.array([10, 30, 50, 70, 90, 110, 130, 150])
        conditions = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        
        # Python pipeline
        sf = SamplingFrame(blocklens=n_scans, tr=tr)
        py_rset = regressor_set(
            onsets=event_times,
            fac=conditions,
            hrf="spmg1"
        )
        py_design = py_rset.evaluate(sf.samples)
        
        # R pipeline
        r_tester.r.assign('event_times', event_times)
        r_tester.r.assign('conditions', conditions)
        r_tester.r.assign('n_scans', n_scans)
        r_tester.r.assign('tr', tr)
        
        r_vars = r_tester.run_r_code("""
        sf <- sampling_frame(n_scans, TR = tr)
        rset <- regressor_set(
            onsets = event_times,
            fac = as.factor(conditions),
            hrf = HRF_SPMG1
        )
        design <- regressor_matrix(rset, sf)
        """)
        
        # numerical_floor: independent R/Python HRF convolution paths need
        # float64-level tolerance; this pins parity without bitwise identity.
        r_tester.compare_arrays(r_vars['design'], py_design, rtol=1e-10)
    
    def test_mixed_event_types(self, r_tester):
        """Test design with mixed event types and parameters."""
        # Different event types with different properties
        event_data = {
            'onset': [5, 10, 15, 25, 35, 45, 55, 65],
            'condition': ['visual', 'motor', 'visual', 'motor', 
                         'visual', 'motor', 'visual', 'motor'],
            'duration': [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            'amplitude': [1.0, 1.5, 0.8, 1.2, 1.1, 1.3, 0.9, 1.4]
        }
        
        # Python
        sf = SamplingFrame(blocklens=100, tr=2.0)
        py_rset = regressor_set(
            onsets=event_data['onset'],
            fac=event_data['condition'],
            duration=event_data['duration'],
            amplitude=event_data['amplitude'],
            hrf="spmg1"
        )
        py_design = py_rset.evaluate(sf.samples)
        
        # R
        r_tester.r.assign('onsets', event_data['onset'])
        r_tester.r.assign('conditions', event_data['condition'])
        r_tester.r.assign('durations', event_data['duration'])
        r_tester.r.assign('amplitudes', event_data['amplitude'])
        
        r_vars = r_tester.run_r_code("""
        sf <- sampling_frame(100, TR = 2.0)
        rset <- regressor_set(
            onsets = onsets,
            fac = as.factor(conditions),
            duration = durations,
            amplitude = amplitudes,
            hrf = HRF_SPMG1
        )
        design <- regressor_matrix(rset, sf)
        """)
        
        # numerical_floor: independent R/Python event-duration convolution
        # paths need float64-level tolerance.
        r_tester.compare_arrays(r_vars['design'], py_design, rtol=1e-10)
    
    def test_hrf_library_glm(self, r_tester):
        """Test GLM with HRF library (basis functions)."""
        # Create HRF library
        param_grid = pd.DataFrame({
            'shape': [4, 6, 8],
            'rate': [1.0, 1.0, 1.0]
        })
        
        # Python
        py_lib = hrf_library(gamma_generator, param_grid)
        
        # Create regressor with HRF library
        onsets = [10, 30, 50, 70]
        py_reg = regressor(onsets=onsets, hrf=py_lib)
        
        sf = SamplingFrame(blocklens=100, tr=2.0)
        py_result = py_reg.evaluate(sf.samples)
        
        # R
        r_tester.r.assign('onsets', onsets)
        r_vars = r_tester.run_r_code("""
        # Create HRF library
        param_grid <- expand.grid(shape = c(4, 6, 8), rate = 1.0)
        lib <- hrf_library(gamma_generator, param_grid)
        
        # Create regressor
        reg <- regressor(onsets = onsets, hrf = lib)
        sf <- sampling_frame(100, TR = 2.0)
        result <- evaluate(reg, samples(sf, global = TRUE))
        """)
        
        # Compare - should have 3 columns (one per basis function)
        assert py_result.shape[1] == 3
        # numerical_floor: HRF library evaluation crosses independent
        # R/Python gamma-generator and convolution paths.
        r_tester.compare_arrays(r_vars['result'], py_result, rtol=1e-10)
    
    def test_multi_block_experiment(self, r_tester):
        """Test complex multi-block experiment."""
        # Simulate a multi-run experiment
        blocklens = [150, 150, 120]
        TRs = [2.0, 2.0, 1.5]
        start_times = [0.0, 310.0, 630.0]
        
        # Events distributed across runs
        events = pd.DataFrame({
            'onset': [10, 30, 50, 80, 110, 140,  # Run 1
                     320, 340, 370, 400, 430,    # Run 2  
                     640, 660, 680, 700],         # Run 3
            'condition': ['A', 'B', 'A', 'B', 'A', 'B',
                         'A', 'B', 'A', 'B', 'A',
                         'B', 'A', 'B', 'A'],
            'duration': [2.0] * 15
        })
        
        # Python
        sf = SamplingFrame(
            blocklens=blocklens,
            tr=TRs,
            start_time=start_times
        )
        
        py_rset = regressor_set(
            onsets=events['onset'].values,
            fac=events['condition'].values,
            duration=events['duration'].values,
            hrf="spmg1"
        )
        py_design = py_rset.evaluate(sf.samples)
        
        # R
        r_tester.r.assign('blocklens', blocklens)
        r_tester.r.assign('TRs', TRs)
        r_tester.r.assign('start_times', start_times)
        r_tester.r.assign('onsets', events['onset'].values)
        r_tester.r.assign('conditions', events['condition'].values)
        r_tester.r.assign('durations', events['duration'].values)
        
        r_vars = r_tester.run_r_code("""
        sf <- sampling_frame(
            blocklens = blocklens,
            TR = TRs,
            start_time = start_times
        )
        
        rset <- regressor_set(
            onsets = onsets,
            fac = as.factor(conditions),
            duration = durations,
            hrf = HRF_SPMG1
        )
        
        design <- regressor_matrix(rset, sf)
        """)
        
        # numerical_floor: multi-block sampling and convolution cross
        # independent R/Python implementations.
        r_tester.compare_arrays(r_vars['design'], py_design, rtol=1e-10)
    
    def test_regularized_estimation(self, r_tester):
        """Test regularized HRF estimation scenario."""
        # Use B-spline basis for flexible HRF estimation
        hrf = get_hrf("bspline")
        
        # Python - get penalty matrix
        py_penalty = penalty_matrix(hrf, order=2)
        
        # Python - get reconstruction matrix
        sf = SamplingFrame(blocklens=100, tr=2.0)
        py_recon = reconstruction_matrix(hrf, sf)
        
        # R
        r_vars = r_tester.run_r_code("""
        hrf <- get_hrf(HRF_BSPLINE)
        
        # Penalty matrix
        penalty <- penalty_matrix(hrf, order = 2)
        
        # Reconstruction matrix
        sf <- sampling_frame(100, TR = 2.0)
        recon <- reconstruction_matrix(hrf, sf)
        """)
        
        # numerical_floor: spline penalty/reconstruction matrices cross
        # independent R/Python basis construction paths.
        r_tester.compare_arrays(r_vars['penalty'], py_penalty, rtol=1e-10)
        r_tester.compare_arrays(r_vars['recon'], py_recon, rtol=1e-10)
    
    def test_concatenated_runs(self, r_tester):
        """Test concatenating multiple experimental runs."""
        # Create individual runs
        sf1 = SamplingFrame(blocklens=100, tr=2.0)
        sf2 = SamplingFrame(blocklens=120, tr=2.0)
        
        # Python - concatenate
        py_concat = sf1.concatenate(sf2)
        
        # Events spanning both runs
        onsets = [10, 30, 50, 210, 230, 250]
        py_reg = regressor(onsets=onsets, hrf="spmg1")
        py_result = py_reg.evaluate(py_concat.samples)
        
        # R
        r_tester.r.assign('onsets', onsets)
        r_vars = r_tester.run_r_code("""
        sf1 <- sampling_frame(100, TR = 2.0)
        sf2 <- sampling_frame(120, TR = 2.0)
        sf_concat <- concatenate(sf1, sf2)
        
        reg <- regressor(onsets = onsets, hrf = HRF_SPMG1)
        result <- evaluate(reg, samples(sf_concat, global = TRUE))
        """)
        
        # numerical_floor: concatenated-run convolution crosses independent
        # R/Python sampling-frame implementations.
        r_tester.compare_arrays(r_vars['result'], py_result, rtol=1e-10)
