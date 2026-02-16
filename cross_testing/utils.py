"""Core utilities for R-Python cross-testing."""

import numpy as np
import pandas as pd
from typing import Any, List, Union, Dict, Optional
import warnings
import uuid


try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import conversion as ro_conversion

    # Configure global converters (rpy2>=3.6 compatible).
    # This replaces deprecated pandas2ri.activate()/numpy2ri.activate().
    ro_conversion.set_conversion(
        ro.default_converter + numpy2ri.converter + pandas2ri.converter
    )
    
    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False
    warnings.warn(
        "rpy2 not available. Cross-testing with R will not be possible. "
        "Install rpy2 to enable R-Python cross-testing."
    )


class REquivalenceTester:
    """Base class for R-Python equivalence testing."""
    
    def __init__(self):
        """Initialize R environment and load packages."""
        if not RPY2_AVAILABLE:
            raise RuntimeError(
                "rpy2 is not installed. Please install it with: "
                "pip install rpy2"
            )
        
        # Import R packages
        self.base = importr('base')
        try:
            self.fmrihrf = importr('fmrihrf')
        except Exception as e:
            raise RuntimeError(
                "Failed to load R fmrihrf package. Please install it with:\n"
                "R -e 'remotes::install_github(\"bbuchsbaum/fmrihrf\")'"
            ) from e
        
        # Store R global environment
        self.r = _RSessionProxy(ro.r, ro, numpy2ri, pandas2ri, localconverter)

        # Attach package namespace for tests that call bare R function names.
        self.r("suppressPackageStartupMessages(library(fmrihrf))")

        # Compatibility aliases for older/newer API naming used by tests.
        self.r("""
        if (!exists("get_hrf", mode = "function")) {
          get_hrf <- function(x = "spmg1", ...) {
            if (inherits(x, "HRF")) {
              return(x)
            }
            getHRF(x, ...)
          }
        }
        if (!exists("hrf_spmg2", mode = "function")) hrf_spmg2 <- function(t) HRF_SPMG2(t)
        if (!exists("hrf_spmg3", mode = "function")) hrf_spmg3 <- function(t) HRF_SPMG3(t)
        if (!exists("gamma_generator", mode = "function")) {
          gamma_generator <- function(shape = 6, rate = 1) {
            as_hrf(hrf_gamma, name = "gamma", params = list(shape = shape, rate = rate))
          }
        }
        if (!exists("regressor_matrix", mode = "function")) {
          regressor_matrix <- function(rset, sf, ...) evaluate(rset, samples(sf, global = TRUE), ...)
        }
        if (!exists("concatenate", mode = "function")) {
          concatenate <- function(sf1, sf2) {
            s1 <- samples(sf1, global = TRUE)
            if (length(s1) == 0) {
              last_end <- 0
            } else {
              last_end <- max(s1) + sf1$TR[length(sf1$TR)]
            }
            sampling_frame(
              blocklens = c(sf1$blocklens, sf2$blocklens),
              TR = c(sf1$TR, sf2$TR),
              start_time = c(sf1$start_time, sf2$start_time + last_end),
              precision = min(sf1$precision, sf2$precision)
            )
          }
        }
        """)
        
    def compare_arrays(
        self, 
        r_result: Any, 
        py_result: np.ndarray, 
        rtol: float = 1e-10, 
        atol: float = 1e-12,
        check_shape: bool = True
    ) -> None:
        """Compare R and Python arrays.
        
        Args:
            r_result: R object to compare
            py_result: Python array to compare
            rtol: Relative tolerance
            atol: Absolute tolerance
            check_shape: Whether to check shapes match
        """
        # Convert R object to numpy
        if hasattr(r_result, '__array__'):
            r_array = np.array(r_result)
        else:
            # Try to convert R object
            with localconverter(ro.default_converter + numpy2ri.converter):
                r_array = np.array(r_result)
        
        # Handle different shapes (R is column-major, Python is row-major)
        if check_shape and r_array.ndim == 2 and py_result.ndim == 2:
            if (r_array.shape[0] == py_result.shape[1] and 
                r_array.shape[1] == py_result.shape[0]):
                r_array = r_array.T
        
        # Compare
        np.testing.assert_allclose(r_array, py_result, rtol=rtol, atol=atol)
        
    def compare_objects(
        self, 
        r_obj: Any, 
        py_obj: Any, 
        attributes: List[str],
        rtol: float = 1e-10,
        atol: float = 1e-12
    ) -> None:
        """Compare R and Python objects by attributes.
        
        Args:
            r_obj: R object
            py_obj: Python object
            attributes: List of attributes to compare
            rtol: Relative tolerance for numeric comparisons
            atol: Absolute tolerance for numeric comparisons
        """
        for attr in attributes:
            r_val = self.get_r_attribute(r_obj, attr)
            py_val = getattr(py_obj, attr)
            
            # Convert and compare based on type
            if isinstance(py_val, (int, float, np.ndarray)):
                self.compare_arrays(r_val, py_val, rtol=rtol, atol=atol)
            else:
                # String or other comparison
                assert str(r_val) == str(py_val), f"Attribute {attr} mismatch"
    
    def get_r_attribute(self, r_obj: Any, attr: str) -> Any:
        """Extract attribute from R object.
        
        Args:
            r_obj: R object
            attr: Attribute name
            
        Returns:
            Attribute value
        """
        # Try different methods to get attribute
        try:
            # Method 1: Direct attribute access
            return self.r[f"attr({r_obj.r_repr()}, '{attr}')"]
        except:
            try:
                # Method 2: Use $ operator
                return self.r[f"{r_obj.r_repr()}${attr}"]
            except:
                # Method 3: Use slot access for S4 objects
                return self.r[f"{r_obj.r_repr()}@{attr}"]
    
    def run_r_code(self, code: str) -> Dict[str, Any]:
        """Run R code and return variables.
        
        Args:
            code: R code to execute
            
        Returns:
            Dictionary of variables defined in the code
        """
        env_name = f".py_env_{uuid.uuid4().hex}"
        self.r(f"{env_name} <- new.env(parent = .GlobalEnv)")
        self.r.assign(".py_code", code)
        self.r(f"eval(parse(text = .py_code), envir = {env_name})")

        vars_in_env = list(self.r(f"ls(envir = {env_name})"))
        result = {}
        for var in vars_in_env:
            self.r.assign(".py_var_name", str(var))
            result[str(var)] = self.r(f"get(.py_var_name, envir = {env_name})")

        self.r(f"rm({env_name}, envir = .GlobalEnv)")
        self.r("rm(.py_code, .py_var_name, envir = .GlobalEnv)")
        return result
    
    def call_r_function(
        self, 
        func_name: str, 
        *args, 
        package: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Call an R function with arguments.
        
        Args:
            func_name: Name of R function
            *args: Positional arguments
            package: Optional package name
            **kwargs: Keyword arguments
            
        Returns:
            Result of R function call
        """
        if package:
            func = getattr(importr(package), func_name)
        else:
            func = self.r[func_name]
        
        # Convert Python arguments to R
        r_args = []
        for arg in args:
            if isinstance(arg, (list, np.ndarray)):
                with localconverter(ro.default_converter + numpy2ri.converter):
                    r_args.append(ro.conversion.py2rpy(arg))
            else:
                r_args.append(arg)
        
        r_kwargs = {}
        for key, val in kwargs.items():
            if isinstance(val, (list, np.ndarray)):
                with localconverter(ro.default_converter + numpy2ri.converter):
                    r_kwargs[key] = ro.conversion.py2rpy(val)
            else:
                r_kwargs[key] = val
        
        return func(*r_args, **r_kwargs)
    
    def compare_dataframes(
        self, 
        r_df: Any, 
        py_df: pd.DataFrame,
        rtol: float = 1e-10,
        atol: float = 1e-12
    ) -> None:
        """Compare R and Python dataframes.
        
        Args:
            r_df: R dataframe
            py_df: Python pandas DataFrame
            rtol: Relative tolerance for numeric columns
            atol: Absolute tolerance for numeric columns
        """
        # Convert R dataframe to pandas
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_pandas = ro.conversion.rpy2py(r_df)
        
        # Compare shape
        assert r_pandas.shape == py_df.shape, f"Shape mismatch: R {r_pandas.shape} vs Python {py_df.shape}"
        
        # Compare columns
        assert set(r_pandas.columns) == set(py_df.columns), "Column names mismatch"
        
        # Compare values column by column
        for col in py_df.columns:
            if pd.api.types.is_numeric_dtype(py_df[col]):
                np.testing.assert_allclose(
                    r_pandas[col].values, 
                    py_df[col].values, 
                    rtol=rtol, 
                    atol=atol,
                    err_msg=f"Column '{col}' values mismatch"
                )
            else:
                # String/categorical comparison
                assert (r_pandas[col] == py_df[col]).all(), f"Column '{col}' values mismatch"
    
    def get_tolerance(self, test_type: str = 'default') -> Dict[str, float]:
        """Get appropriate tolerance values for different test types.
        
        Args:
            test_type: Type of test ('default', 'matrix', 'sparse', 'large')
            
        Returns:
            Dictionary with rtol and atol values
        """
        tolerances = {
            'default': {'rtol': 1e-10, 'atol': 1e-12},
            'matrix': {'rtol': 1e-8, 'atol': 1e-10},
            'sparse': {'rtol': 1e-6, 'atol': 1e-8},
            'large': {'rtol': 1e-5, 'atol': 1e-7},
        }
        return tolerances.get(test_type, tolerances['default'])


class _RSessionProxy:
    """Small proxy to provide robust assign() conversions for rpy2>=3.6."""

    def __init__(self, r_obj, ro_module, numpy2ri_module, pandas2ri_module, localconverter_fn):
        self._r = r_obj
        self._ro = ro_module
        self._numpy2ri = numpy2ri_module
        self._pandas2ri = pandas2ri_module
        self._localconverter = localconverter_fn

    def __call__(self, code: str):
        return self._r(code)

    def __getitem__(self, key):
        return self._r[key]

    def __getattr__(self, name):
        return getattr(self._r, name)

    def assign(self, name: str, value: Any):
        if isinstance(value, str):
            return self._r.assign(name, value)

        # Robust string-like handling (numpy/pandas object arrays).
        if isinstance(value, np.ndarray):
            if value.dtype.kind in ("i", "u"):
                r_value = self._ro.vectors.IntVector(value.astype(int).ravel().tolist())
                return self._r.assign(name, r_value)
            if value.dtype.kind in ("f",):
                r_value = self._ro.vectors.FloatVector(value.astype(float).ravel().tolist())
                return self._r.assign(name, r_value)
            if value.dtype.kind in ("b",):
                r_value = self._ro.vectors.BoolVector(value.astype(bool).ravel().tolist())
                return self._r.assign(name, r_value)
            if value.dtype.kind in ("U", "S", "O"):
                flat = np.asarray(value).ravel().tolist()
                if all(isinstance(v, (int, np.integer)) for v in flat):
                    r_value = self._ro.vectors.IntVector([int(v) for v in flat])
                    return self._r.assign(name, r_value)
                if all(isinstance(v, (int, float, np.integer, np.floating)) for v in flat):
                    r_value = self._ro.vectors.FloatVector([float(v) for v in flat])
                    return self._r.assign(name, r_value)
                r_value = self._ro.vectors.StrVector([str(v) for v in flat])
                return self._r.assign(name, r_value)
        else:
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return self._r.assign(name, self._ro.vectors.FloatVector([]))
                if all(isinstance(v, (bool, np.bool_)) for v in value):
                    return self._r.assign(name, self._ro.vectors.BoolVector([bool(v) for v in value]))
                if all(isinstance(v, (int, np.integer)) for v in value):
                    return self._r.assign(name, self._ro.vectors.IntVector([int(v) for v in value]))
                if all(isinstance(v, (int, float, np.integer, np.floating)) for v in value):
                    return self._r.assign(name, self._ro.vectors.FloatVector([float(v) for v in value]))
                if all(isinstance(v, str) for v in value):
                    return self._r.assign(name, self._ro.vectors.StrVector(list(value)))
            try:
                arr = np.asarray(value)
                if arr.dtype.kind in ("U", "S", "O"):
                    if arr.ndim == 0:
                        return self._r.assign(name, str(arr.item()))
                    r_value = self._ro.vectors.StrVector(arr.astype(str).ravel().tolist())
                    return self._r.assign(name, r_value)
            except Exception:
                pass

        if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
            r_value = self._ro.vectors.StrVector(list(value))
            return self._r.assign(name, r_value)

        converter = (
            self._ro.default_converter
            + self._numpy2ri.converter
            + self._pandas2ri.converter
        )
        with self._localconverter(converter):
            r_value = self._ro.conversion.py2rpy(value)
        return self._r.assign(name, r_value)
