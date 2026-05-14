"""Test regressor equivalence between R and Python."""

import numpy as np

from fmrimod import SamplingFrame, regressor, regressor_set


class TestRegressorEquivalence:
    """Test regressor equivalence."""
    
    def test_basic_regressor(self, r_tester):
        """Test basic regressor creation and evaluation."""
        onsets = [10, 30, 50]
        duration = 2.0
        
        # Python
        py_reg = regressor(
            onsets=onsets,
            hrf="spmg1",
            duration=duration
        )
        sf = SamplingFrame(blocklens=100, tr=2.0)
        py_result = py_reg.evaluate(sf.samples)
        
        # R
        r_tester.r.assign('onsets', onsets)
        r_tester.r.assign('duration', duration)
        r_vars = r_tester.run_r_code("""
        reg <- regressor(
            onsets = onsets,
            hrf = HRF_SPMG1,
            duration = duration
        )
        sf <- sampling_frame(100, TR = 2.0)
        result <- evaluate(reg, samples(sf, global = TRUE))
        """)
        
        # numerical_floor: independent R/Python HRF convolution paths need
        # float64-level tolerance; this pins parity without bitwise identity.
        r_tester.compare_arrays(r_vars['result'], py_result, rtol=1e-10)
    
    def test_regressor_with_amplitudes(self, r_tester):
        """Test regressor with varying amplitudes."""
        onsets = [5, 15, 25, 35]
        amplitudes = [1.0, 0.5, 1.5, 0.8]
        
        # Python
        py_reg = regressor(
            onsets=onsets,
            amplitude=amplitudes,
            hrf="gamma"
        )
        sf = SamplingFrame(blocklens=50, tr=2.0)
        py_result = py_reg.evaluate(sf.samples)
        
        # R
        r_tester.r.assign('onsets', onsets)
        r_tester.r.assign('amplitudes', amplitudes)
        r_vars = r_tester.run_r_code("""
        reg <- regressor(
            onsets = onsets,
            amplitude = amplitudes,
            hrf = HRF_GAMMA
        )
        sf <- sampling_frame(50, TR = 2.0)
        result <- evaluate(reg, samples(sf, global = TRUE))
        """)
        
        # numerical_floor: amplitude-weighted convolution crosses independent
        # R/Python implementations.
        r_tester.compare_arrays(r_vars['result'], py_result, rtol=1e-10)
    
    def test_regressor_with_durations(self, r_tester):
        """Test regressor with varying durations."""
        onsets = [10, 30, 50]
        durations = [1.0, 2.0, 3.0]
        
        # Python
        py_reg = regressor(
            onsets=onsets,
            duration=durations,
            hrf="spmg1"
        )
        times = np.linspace(0, 80, 200)
        py_result = py_reg.evaluate(times)
        
        # R
        r_tester.r.assign('onsets', onsets)
        r_tester.r.assign('durations', durations)
        r_tester.r.assign('times', times)
        r_vars = r_tester.run_r_code("""
        reg <- regressor(
            onsets = onsets,
            duration = durations,
            hrf = HRF_SPMG1
        )
        result <- evaluate(reg, times)
        """)
        
        # numerical_floor: variable-duration convolution crosses independent
        # R/Python implementations.
        r_tester.compare_arrays(r_vars['result'], py_result, rtol=1e-10)
    
    def test_regressor_summation(self, r_tester):
        """Test regressor with summation disabled."""
        onsets = [10, 11, 12]  # Close events
        
        # Python - with summation (default)
        py_reg_sum = regressor(onsets=onsets, hrf="spmg1", summate=True)
        # Python - without summation
        py_reg_nosum = regressor(onsets=onsets, hrf="spmg1", summate=False)
        
        times = np.linspace(0, 50, 100)
        py_result_sum = py_reg_sum.evaluate(times)
        py_result_nosum = py_reg_nosum.evaluate(times)
        
        # R
        r_tester.r.assign('onsets', onsets)
        r_tester.r.assign('times', times)
        r_vars = r_tester.run_r_code("""
        reg_sum <- regressor(onsets = onsets, hrf = HRF_SPMG1, summate = TRUE)
        reg_nosum <- regressor(onsets = onsets, hrf = HRF_SPMG1, summate = FALSE)
        result_sum <- evaluate(reg_sum, times)
        result_nosum <- evaluate(reg_nosum, times)
        """)
        
        # numerical_floor: close-event summation crosses independent R/Python
        # convolution implementations.
        r_tester.compare_arrays(r_vars['result_sum'], py_result_sum, rtol=1e-10)
        r_tester.compare_arrays(r_vars['result_nosum'], py_result_nosum, rtol=1e-10)
    
    def test_regressor_set_basic(self, r_tester):
        """Test basic regressor set functionality."""
        # Create events with conditions
        event_times = np.array([10, 30, 50, 70])
        conditions = np.array([0, 1, 0, 1])
        
        # Python
        sf = SamplingFrame(blocklens=100, tr=2.0)
        py_rset = regressor_set(
            onsets=event_times,
            fac=conditions,
            hrf="spmg1"
        )
        py_design = py_rset.evaluate(sf.samples)
        
        # R
        r_tester.r.assign('event_times', event_times)
        r_tester.r.assign('conditions', conditions)
        r_vars = r_tester.run_r_code("""
        sf <- sampling_frame(100, TR = 2.0)
        rset <- regressor_set(
            onsets = event_times,
            fac = as.factor(conditions),
            hrf = HRF_SPMG1
        )
        design <- regressor_matrix(rset, sf)
        """)
        
        # numerical_floor: regressor-set design construction crosses
        # independent R/Python factor and convolution paths.
        r_tester.compare_arrays(r_vars['design'], py_design, rtol=1e-10)
    
    def test_regressor_set_with_names(self, r_tester):
        """Test regressor set with named conditions."""
        event_times = np.array([5, 15, 25, 35, 45])
        conditions = ['A', 'B', 'A', 'B', 'C']
        
        # Python
        py_rset = regressor_set(
            onsets=event_times,
            fac=conditions,
            hrf="gamma"
        )
        
        # Check levels
        assert py_rset.levels == ['A', 'B', 'C']
        assert len(py_rset.regressors) == 3
        
        # R
        r_tester.r.assign('event_times', event_times)
        r_tester.r.assign('conditions', conditions)
        r_vars = r_tester.run_r_code("""
        rset <- regressor_set(
            onsets = event_times,
            fac = as.factor(conditions),
            hrf = HRF_GAMMA
        )
        levels <- rset$levels
        n_regs <- length(rset$regs)
        """)
        
        # Compare structure
        r_levels = list(r_vars['levels'])
        assert r_levels == py_rset.levels
        assert int(r_vars['n_regs'][0]) == len(py_rset.regressors)
    
    def test_multi_block_regressor(self, r_tester):
        """Test regressor evaluation across multiple blocks."""
        # Multi-block sampling frame
        sf = SamplingFrame(
            blocklens=[50, 50],
            tr=[2.0, 2.0],
            start_time=[0.0, 110.0]
        )
        
        # Events in both blocks
        onsets = [10, 30, 120, 140]
        
        # Python
        py_reg = regressor(onsets=onsets, hrf="spmg1")
        py_result = py_reg.evaluate(sf.samples)
        
        # R
        r_tester.r.assign('onsets', onsets)
        r_vars = r_tester.run_r_code("""
        sf <- sampling_frame(
            blocklens = c(50, 50),
            TR = c(2.0, 2.0),
            start_time = c(0.0, 110.0)
        )
        reg <- regressor(onsets = onsets, hrf = HRF_SPMG1)
        result <- evaluate(reg, samples(sf, global = TRUE))
        """)
        
        # numerical_floor: multi-block regressor evaluation crosses independent
        # R/Python sampling-frame implementations.
        r_tester.compare_arrays(r_vars['result'], py_result, rtol=1e-10)
    
    def test_regressor_sparse_output(self, r_tester):
        """Test sparse matrix output from regressor."""
        # Create regressor with few events
        onsets = [20, 60]
        
        # Python
        py_reg = regressor(onsets=onsets, hrf="spmg1")
        sf = SamplingFrame(blocklens=100, tr=2.0)
        py_sparse = py_reg.evaluate(sf.samples, sparse=True)
        
        # R
        r_tester.r.assign('onsets', onsets)
        r_vars = r_tester.run_r_code("""
        reg <- regressor(onsets = onsets, hrf = HRF_SPMG1)
        sf <- sampling_frame(100, TR = 2.0)
        result_sparse <- evaluate(reg, samples(sf, global = TRUE), sparse = TRUE)
        result_dense <- as.matrix(result_sparse)
        """)
        
        # numerical_floor: sparse-to-dense regressor output crosses independent
        # R/Python sparse materialization paths.
        py_dense = py_sparse.toarray() if hasattr(py_sparse, 'toarray') else py_sparse
        r_tester.compare_arrays(r_vars['result_dense'], py_dense, rtol=1e-10)
