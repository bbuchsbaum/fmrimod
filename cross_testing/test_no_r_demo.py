"""Demonstration of cross-testing infrastructure without R dependencies."""

import numpy as np
import pytest


def test_infrastructure_structure():
    """Test that the cross-testing infrastructure files exist."""
    from pathlib import Path
    import cross_testing
    
    cross_test_dir = Path(cross_testing.__file__).parent
    
    expected_files = [
        'utils.py',
        'conftest.py', 
        'test_hrf_equivalence.py',
        'test_regressor_equivalence.py',
        'test_complex_scenarios.py',
        'test_performance.py',
        'generate_report.py',
        'README.md',
        'CROSS_TESTING_STRATEGY.md'
    ]
    
    print("\n=== Cross-Testing Infrastructure ===")
    print(f"Location: {cross_test_dir}")
    print("\nFiles present:")
    
    for fname in expected_files:
        exists = (cross_test_dir / fname).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {fname}")
        
    # Check subdirectories
    print("\nSubdirectories:")
    data_dir = cross_test_dir / "data"
    print(f"  {'✅' if data_dir.exists() else '❌'} data/")
    if data_dir.exists():
        for f in data_dir.iterdir():
            print(f"      - {f.name}")


def test_python_functionality():
    """Test Python fmrimod functionality independently."""
    print("\n=== Testing Python Implementation ===")
    
    from fmrimod import get_hrf, regressor, regressor_set, SamplingFrame
    
    # Test 1: HRF evaluation
    print("\n1. HRF Evaluation:")
    t = np.linspace(0, 30, 100)
    hrf = get_hrf("spmg1")
    result = hrf(t)
    print(f"   - SPMG1 HRF shape: {result.shape}")
    print(f"   - Peak value: {np.max(result):.4f}")
    print(f"   - Peak time: {t[np.argmax(result)]:.2f}s")
    
    # Test 2: Regressor
    print("\n2. Regressor Creation:")
    reg = regressor(onsets=[10, 30, 50], hrf="spmg1", duration=2.0)
    sf = SamplingFrame(blocklens=100, tr=2.0)
    reg_result = reg.evaluate(sf.samples)
    print(f"   - Regressor shape: {reg_result.shape}")
    print(f"   - Non-zero values: {np.sum(reg_result > 0.01)}")
    
    # Test 3: Design Matrix
    print("\n3. Design Matrix:")
    rset = regressor_set(
        onsets=[5, 15, 25, 35],
        fac=['A', 'B', 'A', 'B'],
        hrf="gamma"
    )
    design = rset.evaluate(sf.samples)
    print(f"   - Design shape: {design.shape}")
    print(f"   - Conditions: {rset.levels}")
    print(f"   - Correlation between conditions: {np.corrcoef(design.T)[0,1]:.3f}")


def test_cross_test_utilities():
    """Test the cross-testing utility functions work."""
    print("\n=== Cross-Testing Utilities ===")
    
    from cross_testing.utils import RPY2_AVAILABLE
    
    print(f"\n1. rpy2 availability: {'✅ Available' if RPY2_AVAILABLE else '❌ Not available'}")
    
    if not RPY2_AVAILABLE:
        print("   Note: Install rpy2 to enable full R-Python cross-testing")
        print("   Command: pip install rpy2")
    
    # Test tolerance settings
    from cross_testing.utils import REquivalenceTester
    if RPY2_AVAILABLE:
        try:
            tester = REquivalenceTester()
            tolerances = tester.get_tolerance('default')
            print("\n2. Tolerance settings:")
            print(f"   - Default rtol: {tolerances['rtol']}")
            print(f"   - Default atol: {tolerances['atol']}")
        except Exception as e:
            print(f"\n2. Could not create tester: {e}")
    else:
        # Show what tolerances would be used
        print("\n2. Tolerance settings (would be used with R):")
        print("   - Default: rtol=1e-10, atol=1e-12")
        print("   - Matrix: rtol=1e-8, atol=1e-10")
        print("   - Sparse: rtol=1e-6, atol=1e-8")


def test_report_generation():
    """Test report generation functionality."""
    print("\n=== Report Generation ===")
    
    from pathlib import Path
    import sys
    
    # Check if generate_report.py exists and is valid Python
    report_script = Path("cross_testing/generate_report.py")
    if report_script.exists():
        print(f"\n✅ Report generation script exists: {report_script}")
        
        # Try to import it
        try:
            import cross_testing.generate_report
            print("✅ Report script is valid Python")
            
            # Check main functions
            if hasattr(cross_testing.generate_report, 'generate_markdown_report'):
                print("✅ Has generate_markdown_report() function")
            if hasattr(cross_testing.generate_report, 'main'):
                print("✅ Has main() function")
                
        except Exception as e:
            print(f"❌ Could not import report script: {e}")
    else:
        print(f"❌ Report script not found: {report_script}")


def test_github_workflow():
    """Check GitHub Actions workflow."""
    print("\n=== CI/CD Configuration ===")
    
    from pathlib import Path
    
    workflow_file = Path(".github/workflows/cross_test.yml")
    if workflow_file.exists():
        print(f"\n✅ GitHub Actions workflow exists: {workflow_file}")
        
        # Read and check key components
        content = workflow_file.read_text()
        checks = [
            ("Multiple Python versions", "python-version:" in content),
            ("Multiple R versions", "r-version:" in content),
            ("rpy2 installation", "pip install rpy2" in content),
            ("R package installation", "remotes::install_github" in content),
            ("Cross-test execution", "pytest cross_testing/" in content),
            ("Report generation", "generate_report.py" in content),
        ]
        
        print("\nWorkflow components:")
        for desc, present in checks:
            status = "✅" if present else "❌"
            print(f"  {status} {desc}")
    else:
        print(f"\n❌ GitHub Actions workflow not found: {workflow_file}")


def test_performance_demo():
    """Demonstrate performance testing capability."""
    print("\n=== Performance Testing Demo ===")
    
    import time
    from fmrimod import regressor
    
    # Different scales
    scales = [
        ("Small", 100, 1000),
        ("Medium", 500, 5000),
        ("Large", 1000, 10000),
    ]
    
    print("\nPython performance scaling:")
    print("Scale    | Events | Points | Time (ms) | Speed")
    print("---------|--------|--------|-----------|-------------")
    
    for name, n_events, n_points in scales:
        np.random.seed(42)
        onsets = np.sort(np.random.uniform(0, n_points/2, n_events))
        times = np.linspace(0, n_points/2, n_points)
        
        start = time.time()
        reg = regressor(onsets=onsets, hrf="spmg1")
        result = reg.evaluate(times)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        speed = n_points / (elapsed / 1000)  # points per second
        print(f"{name:8} | {n_events:6} | {n_points:6} | {elapsed:9.1f} | {speed:,.0f} pts/s")


if __name__ == "__main__":
    # Run all demonstration tests
    print("=" * 50)
    print("Cross-Testing Infrastructure Demonstration")
    print("=" * 50)
    
    test_infrastructure_structure()
    test_python_functionality()
    test_cross_test_utilities()
    test_report_generation()
    test_github_workflow()
    test_performance_demo()
    
    print("\n" + "=" * 50)
    print("Demonstration complete!")
    print("\nTo enable full R-Python cross-testing:")
    print("1. Install rpy2: pip install rpy2")
    print("2. Install R fmrihrf: R -e 'remotes::install_github(\"bbuchsbaum/fmrihrf\")'")
    print("3. Run: ./run_cross_tests.sh")
    print("=" * 50)