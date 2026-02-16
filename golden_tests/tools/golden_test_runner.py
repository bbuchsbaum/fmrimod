#!/usr/bin/env python3
"""
Golden Test Runner for Python Implementations

Executes golden test specifications and validates numerical outputs
against expected values with proper tolerance handling.

Usage:
    python golden_test_runner.py [options]
    python golden_test_runner.py --test hrf_spmg1_basic
    python golden_test_runner.py --output results.json
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
import numpy as np
import importlib.util
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import re


@dataclass
class TestResult:
    """Represents the result of a single test validation check"""
    test_id: str
    check_index: int
    check_type: str
    location: str
    status: str  # 'pass', 'fail', 'error', 'skip'
    expected: Optional[Union[float, int, str]] = None
    actual: Optional[Union[float, int, str]] = None
    tolerance: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    property_name: Optional[str] = None
    error_message: Optional[str] = None
    error_magnitude: Optional[float] = None
    execution_time: Optional[float] = None


@dataclass 
class TestSummary:
    """Summary of all tests for a golden test specification"""
    test_id: str
    description: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    error_checks: int
    skipped_checks: int
    overall_status: str  # 'pass', 'fail', 'error', 'partial'
    execution_time: float
    timestamp: str
    implementation_found: bool
    results: List[TestResult]


class NumericalComparator:
    """Handles numerical comparisons with various tolerance types"""
    
    @staticmethod
    def compare_values(actual: Any, expected: Any, tolerance: float = 1e-10,
                      check_type: str = 'approximate') -> Tuple[bool, float]:
        """
        Compare actual vs expected values with tolerance
        
        Returns:
            (is_match, error_magnitude)
        """
        try:
            actual_val = np.asarray(actual, dtype=float)
            expected_val = np.asarray(expected, dtype=float)
            
            if check_type == 'exact_value':
                # For exact values, use very tight tolerance
                diff = np.abs(actual_val - expected_val)
                max_diff = np.max(diff) if diff.size > 0 else 0
                return max_diff <= tolerance, float(max_diff)
                
            elif check_type == 'approximate':
                # Standard absolute tolerance
                diff = np.abs(actual_val - expected_val)
                max_diff = np.max(diff) if diff.size > 0 else 0
                return max_diff <= tolerance, float(max_diff)
                
            elif check_type == 'range':
                # Not applicable for this method
                return False, float('inf')
                
            elif check_type == 'statistical':
                # Compare statistical properties
                if hasattr(actual_val, '__len__') and len(actual_val) > 1:
                    # For arrays, compute the specified statistical property
                    # This will be handled by the caller
                    pass
                else:
                    # For scalars, treat as approximate
                    diff = abs(float(actual_val) - float(expected_val))
                    return diff <= tolerance, diff
                    
        except Exception as e:
            return False, float('inf')
        
        return False, float('inf')
    
    @staticmethod
    def check_range(value: Any, min_val: float, max_val: float) -> Tuple[bool, float]:
        """Check if value is within specified range"""
        try:
            val = float(value)
            if min_val <= val <= max_val:
                return True, 0.0
            else:
                # Calculate how far outside the range
                if val < min_val:
                    error_mag = min_val - val
                else:
                    error_mag = val - max_val
                return False, error_mag
        except Exception:
            return False, float('inf')
    
    @staticmethod
    def compute_statistical_property(data: Any, property_name: str) -> float:
        """Compute statistical property of data array"""
        try:
            arr = np.asarray(data, dtype=float)
            if property_name == 'mean':
                return float(np.mean(arr))
            elif property_name == 'std':
                return float(np.std(arr))
            elif property_name == 'var':
                return float(np.var(arr))
            elif property_name == 'min':
                return float(np.min(arr))
            elif property_name == 'max':
                return float(np.max(arr))
            elif property_name == 'sum':
                return float(np.sum(arr))
            else:
                raise ValueError(f"Unknown statistical property: {property_name}")
        except Exception as e:
            raise ValueError(f"Cannot compute {property_name}: {e}")


class GoldenTestRunner:
    """Executes golden tests and validates results"""
    
    def __init__(self, golden_tests_dir: str, implementation_module: Optional[str] = None):
        self.golden_tests_dir = Path(golden_tests_dir)
        self.implementation_module = implementation_module
        self.namespace = {'gt': 'http://golden-tests.org/schema'}
        self.comparator = NumericalComparator()
        
        # Try to import the implementation module if provided
        self.impl_module = None
        if implementation_module:
            self.impl_module = self._import_implementation_module(implementation_module)
    
    def _import_implementation_module(self, module_path: str):
        """Import implementation module dynamically"""
        try:
            if os.path.exists(module_path):
                # Load from file path
                spec = importlib.util.spec_from_file_location("implementation", module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            else:
                # Try as module name
                return importlib.import_module(module_path)
        except Exception as e:
            print(f"Warning: Could not import implementation module {module_path}: {e}")
            return None
    
    def run_all_tests(self, test_filter: Optional[str] = None) -> List[TestSummary]:
        """Run all golden tests in the directory"""
        results = []
        
        # Find all XML test files
        specs_dir = self.golden_tests_dir / "specs"
        if not specs_dir.exists():
            print(f"No specs directory found in {self.golden_tests_dir}")
            return results
        
        for xml_file in specs_dir.rglob("*.xml"):
            try:
                # Parse test ID from XML
                tree = ET.parse(xml_file)
                root = tree.getroot()
                test_id_elem = root.find('.//gt:id', self.namespace)
                
                if test_id_elem is not None:
                    test_id = test_id_elem.text
                    
                    # Apply filter if provided
                    if test_filter and test_filter not in test_id:
                        continue
                    
                    print(f"Running test: {test_id}")
                    result = self.run_single_test(xml_file)
                    results.append(result)
                    
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                continue
        
        return results
    
    def run_single_test(self, xml_file: Path) -> TestSummary:
        """Run a single golden test from XML specification"""
        start_time = datetime.now()
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract test metadata
            test_id = root.find('.//gt:id', self.namespace).text
            description_elem = root.find('.//gt:description', self.namespace)
            description = description_elem.text if description_elem is not None else test_id
            
            # Check if implementation exists
            impl_code = self._extract_implementation_code(root, 'Python')
            impl_found = impl_code is not None and impl_code.strip()
            
            # Execute implementation if available
            test_context = {}
            if impl_found:
                test_context = self._execute_implementation(impl_code, test_id)
            
            # Run validation checks
            check_results = []
            checks = root.findall('.//gt:check', self.namespace)
            
            for i, check in enumerate(checks):
                result = self._execute_check(check, i, test_id, test_context)
                check_results.append(result)
            
            # Summarize results
            passed = sum(1 for r in check_results if r.status == 'pass')
            failed = sum(1 for r in check_results if r.status == 'fail')
            errors = sum(1 for r in check_results if r.status == 'error')
            skipped = sum(1 for r in check_results if r.status == 'skip')
            
            # Determine overall status
            if not impl_found:
                overall_status = 'skip'
            elif errors > 0:
                overall_status = 'error'
            elif failed > 0:
                overall_status = 'fail'
            elif passed > 0:
                overall_status = 'pass'
            else:
                overall_status = 'skip'
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TestSummary(
                test_id=test_id,
                description=description,
                total_checks=len(check_results),
                passed_checks=passed,
                failed_checks=failed,
                error_checks=errors,
                skipped_checks=skipped,
                overall_status=overall_status,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                implementation_found=impl_found,
                results=check_results
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestSummary(
                test_id=xml_file.stem,
                description=f"Error parsing {xml_file.name}",
                total_checks=0,
                passed_checks=0,
                failed_checks=0,
                error_checks=1,
                skipped_checks=0,
                overall_status='error',
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                implementation_found=False,
                results=[TestResult(
                    test_id=xml_file.stem,
                    check_index=0,
                    check_type='parse_error',
                    location='XML parsing',
                    status='error',
                    error_message=str(e)
                )]
            )
    
    def _extract_implementation_code(self, root: ET.Element, language: str) -> Optional[str]:
        """Extract implementation code for specified language"""
        impl_section = root.find('.//gt:implementations', self.namespace)
        if impl_section is not None:
            lang_elem = impl_section.find(f'gt:{language}', self.namespace)
            if lang_elem is not None:
                return lang_elem.text
        return None
    
    def _execute_implementation(self, impl_code: str, test_id: str) -> Dict[str, Any]:
        """Execute implementation code and return resulting context"""
        context = {
            'np': np,
            'numpy': np,
            # Add other common imports as needed
        }
        
        try:
            # Execute the implementation code
            exec(impl_code, context)
            return context
            
        except Exception as e:
            print(f"Error executing implementation for {test_id}: {e}")
            traceback.print_exc()
            return {}
    
    def _execute_check(self, check_elem: ET.Element, index: int, 
                      test_id: str, context: Dict[str, Any]) -> TestResult:
        """Execute a single validation check"""
        try:
            check_type = check_elem.find('gt:type', self.namespace).text
            location = check_elem.find('gt:location', self.namespace).text
            
            # Extract expected values and tolerances
            expected_elem = check_elem.find('gt:expected', self.namespace)
            tolerance_elem = check_elem.find('gt:tolerance', self.namespace)
            min_elem = check_elem.find('gt:min', self.namespace)
            max_elem = check_elem.find('gt:max', self.namespace)
            property_elem = check_elem.find('gt:property', self.namespace)
            
            expected = float(expected_elem.text) if expected_elem is not None else None
            tolerance = float(tolerance_elem.text) if tolerance_elem is not None else 1e-10
            min_val = float(min_elem.text) if min_elem is not None else None
            max_val = float(max_elem.text) if max_elem is not None else None
            property_name = property_elem.text if property_elem is not None else None
            
            # Execute the test expression
            try:
                start_time = datetime.now()
                actual = eval(location, {"__builtins__": {}}, context)
                execution_time = (datetime.now() - start_time).total_seconds()
            except Exception as e:
                return TestResult(
                    test_id=test_id,
                    check_index=index,
                    check_type=check_type,
                    location=location,
                    status='error',
                    expected=expected,
                    tolerance=tolerance,
                    min_val=min_val,
                    max_val=max_val,
                    property_name=property_name,
                    error_message=f"Execution error: {e}",
                    execution_time=0.0
                )
            
            # Validate the result
            if check_type == 'range':
                if min_val is not None and max_val is not None:
                    is_valid, error_mag = self.comparator.check_range(actual, min_val, max_val)
                    status = 'pass' if is_valid else 'fail'
                else:
                    status = 'error'
                    error_mag = 0.0
                    
            elif check_type == 'statistical':
                if property_name:
                    try:
                        stat_value = self.comparator.compute_statistical_property(actual, property_name)
                        is_valid, error_mag = self.comparator.compare_values(
                            stat_value, expected, tolerance, 'approximate'
                        )
                        actual = stat_value  # Use computed statistical value as actual
                        status = 'pass' if is_valid else 'fail'
                    except Exception as e:
                        status = 'error'
                        error_mag = 0.0
                else:
                    status = 'error'
                    error_mag = 0.0
                    
            else:
                # Standard value comparison
                is_valid, error_mag = self.comparator.compare_values(
                    actual, expected, tolerance, check_type
                )
                status = 'pass' if is_valid else 'fail'
            
            return TestResult(
                test_id=test_id,
                check_index=index,
                check_type=check_type,
                location=location,
                status=status,
                expected=expected,
                actual=actual,
                tolerance=tolerance,
                min_val=min_val,
                max_val=max_val,
                property_name=property_name,
                error_magnitude=error_mag,
                execution_time=execution_time
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_id,
                check_index=index,
                check_type='unknown',
                location='check parsing',
                status='error',
                error_message=f"Check parsing error: {e}"
            )
    
    def save_results(self, results: List[TestSummary], output_file: str):
        """Save test results to JSON file"""
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'runner_version': '1.0.0',
                'total_tests': len(results),
                'implementation_module': self.implementation_module
            },
            'summary': {
                'total_tests': len(results),
                'passed_tests': sum(1 for r in results if r.overall_status == 'pass'),
                'failed_tests': sum(1 for r in results if r.overall_status == 'fail'),
                'error_tests': sum(1 for r in results if r.overall_status == 'error'),
                'skipped_tests': sum(1 for r in results if r.overall_status == 'skip'),
                'total_checks': sum(r.total_checks for r in results),
                'passed_checks': sum(r.passed_checks for r in results),
                'failed_checks': sum(r.failed_checks for r in results)
            },
            'results': [asdict(result) for result in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"Results saved to {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run golden tests and validate implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--golden-tests-dir', '-d', default='.',
                       help='Directory containing golden tests (default: current directory)')
    parser.add_argument('--implementation', '-i',
                       help='Python module or file containing implementations')
    parser.add_argument('--test', '-t',
                       help='Run specific test (by ID substring match)')
    parser.add_argument('--output', '-o', default='golden_test_results.json',
                       help='Output file for results (default: golden_test_results.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = GoldenTestRunner(
        golden_tests_dir=args.golden_tests_dir,
        implementation_module=args.implementation
    )
    
    # Run tests
    print("Starting golden test execution...")
    results = runner.run_all_tests(test_filter=args.test)
    
    # Print summary
    total_tests = len(results)
    passed = sum(1 for r in results if r.overall_status == 'pass')
    failed = sum(1 for r in results if r.overall_status == 'fail')
    errors = sum(1 for r in results if r.overall_status == 'error')
    skipped = sum(1 for r in results if r.overall_status == 'skip')
    
    print(f"\nTest Results Summary:")
    print(f"====================")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    
    if args.verbose:
        for result in results:
            print(f"\n{result.test_id}: {result.overall_status}")
            if result.overall_status in ['fail', 'error']:
                for check in result.results:
                    if check.status in ['fail', 'error']:
                        print(f"  Check {check.check_index}: {check.status}")
                        if check.error_message:
                            print(f"    Error: {check.error_message}")
                        elif check.status == 'fail':
                            print(f"    Expected: {check.expected}, Actual: {check.actual}")
                            if check.error_magnitude:
                                print(f"    Error magnitude: {check.error_magnitude}")
    
    # Save results
    runner.save_results(results, args.output)
    
    # Exit with appropriate code
    if errors > 0:
        sys.exit(2)  # Errors occurred
    elif failed > 0:
        sys.exit(1)  # Tests failed
    else:
        sys.exit(0)  # All good


if __name__ == '__main__':
    main()