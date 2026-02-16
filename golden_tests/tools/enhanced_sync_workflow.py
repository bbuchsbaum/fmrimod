#!/usr/bin/env python3
"""
Enhanced Golden Test Sync with Execution and Validation

Extends the basic sync workflow to include:
1. Test execution after sync
2. Results validation
3. Dashboard generation
4. Status reporting with pass/fail rates

Usage:
    python enhanced_sync_workflow.py --source PATH --target PATH --language LANG [options]
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import tempfile


class EnhancedGoldenTestSync:
    """Enhanced sync workflow with execution and validation"""
    
    def __init__(self, source_dir: str, target_dir: str, language: str,
                 execute_tests: bool = True, generate_dashboard: bool = True,
                 sync_bin: str = None, dry_run: bool = False, verbose: bool = False):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.language = language
        self.execute_tests = execute_tests
        self.generate_dashboard = generate_dashboard
        self.sync_bin = sync_bin or str(Path.home() / "bin" / "sync_golden_tests.py")
        self.dry_run = dry_run
        self.verbose = verbose
        
        # Results tracking
        self.sync_results = None
        self.test_results = None
        self.dashboard_path = None
        
    def run_enhanced_sync(self) -> bool:
        """Run the complete enhanced sync workflow"""
        try:
            # Step 1: Run basic sync
            if not self._run_basic_sync():
                print("❌ Basic sync failed")
                return False
            
            if self.dry_run:
                print("✅ Dry run completed - sync would have succeeded")
                return True
            
            # Step 2: Execute tests if requested
            if self.execute_tests:
                if not self._execute_tests():
                    print("❌ Test execution failed")
                    return False
            
            # Step 3: Generate dashboard if requested
            if self.generate_dashboard:
                if not self._generate_dashboard():
                    print("❌ Dashboard generation failed")
                    return False
            
            # Step 4: Report results
            self._report_results()
            
            return True
            
        except Exception as e:
            print(f"❌ Enhanced sync failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def _run_basic_sync(self) -> bool:
        """Run the basic sync using existing sync_golden_tests.py"""
        print(f"🔄 Running basic sync from {self.source_dir} to {self.target_dir}...")
        
        cmd = [
            sys.executable, self.sync_bin,
            "--source", str(self.source_dir),
            "--target", str(self.target_dir),
            "--language", self.language
        ]
        
        if self.dry_run:
            cmd.append("--dry-run")
        if self.verbose:
            cmd.append("--verbose")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if self.verbose:
                print("Sync output:")
                print(result.stdout)
                if result.stderr:
                    print("Sync errors:")
                    print(result.stderr)
            
            if result.returncode == 0:
                print("✅ Basic sync completed successfully")
                return True
            else:
                print(f"❌ Basic sync failed with return code {result.returncode}")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error running basic sync: {e}")
            return False
    
    def _execute_tests(self) -> bool:
        """Execute golden tests in the target directory"""
        print(f"🧪 Executing {self.language} tests...")
        
        # Determine test runner based on language
        if self.language.lower() == "python":
            return self._run_python_tests()
        elif self.language.lower() == "r":
            return self._run_r_tests()
        else:
            print(f"⚠️  Test execution not yet supported for {self.language}")
            return True  # Don't fail the workflow
    
    def _run_python_tests(self) -> bool:
        """Run Python golden tests"""
        # Look in tools/ subdirectory first
        test_runner = self.target_dir / "tools" / "golden_test_runner.py"
        if not test_runner.exists():
            # Try the improved runner in tools/
            test_runner = self.target_dir / "tools" / "improved_test_runner.py"
        if not test_runner.exists():
            # Fallback to root directory
            test_runner = self.target_dir / "golden_test_runner.py"
        if not test_runner.exists():
            # Try the improved runner in root
            test_runner = self.target_dir / "improved_test_runner.py"
        
        if not test_runner.exists():
            print(f"❌ Python test runner not found in {self.target_dir} or {self.target_dir}/tools/")
            return False
        
        results_file = self.target_dir / f"golden_test_results_python_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        cmd = [
            sys.executable, str(test_runner),
            "--golden-tests-dir", str(self.target_dir),
            "--output", str(results_file)
        ]
        
        if self.verbose:
            cmd.append("--verbose")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if self.verbose:
                print("Python test output:")
                print(result.stdout)
                if result.stderr:
                    print("Python test errors:")
                    print(result.stderr)
            
            # Load and store results
            if results_file.exists():
                with open(results_file) as f:
                    self.test_results = {
                        'python': {
                            'file': str(results_file),
                            'data': json.load(f),
                            'returncode': result.returncode
                        }
                    }
                
                # Report summary
                summary = self.test_results['python']['data']['summary']
                total = summary['total_tests']
                passed = summary['passed_tests']
                failed = summary['failed_tests']
                errors = summary['error_tests']
                
                print(f"✅ Python tests executed: {passed}/{total} passed, {failed} failed, {errors} errors")
                return True
            else:
                print("❌ Python test results file not created")
                return False
                
        except Exception as e:
            print(f"❌ Error running Python tests: {e}")
            return False
    
    def _run_r_tests(self) -> bool:
        """Run R golden tests"""
        test_runner = self.target_dir / "golden_test_runner.R"
        if not test_runner.exists():
            print(f"❌ R test runner not found in {self.target_dir}")
            return False
        
        results_file = self.target_dir / f"golden_test_results_r_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        cmd = [
            "Rscript", str(test_runner),
            "--golden-tests-dir", str(self.target_dir),
            "--output", str(results_file)
        ]
        
        if self.verbose:
            cmd.append("--verbose")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if self.verbose:
                print("R test output:")
                print(result.stdout)
                if result.stderr:
                    print("R test errors:")
                    print(result.stderr)
            
            # Load and store results
            if results_file.exists():
                with open(results_file) as f:
                    if not hasattr(self, 'test_results') or self.test_results is None:
                        self.test_results = {}
                    
                    self.test_results['r'] = {
                        'file': str(results_file),
                        'data': json.load(f),
                        'returncode': result.returncode
                    }
                
                # Report summary
                summary = self.test_results['r']['data']['summary']
                total = summary['total_tests']
                passed = summary['passed_tests']
                failed = summary['failed_tests']
                errors = summary['error_tests']
                
                print(f"✅ R tests executed: {passed}/{total} passed, {failed} failed, {errors} errors")
                return True
            else:
                print("❌ R test results file not created")
                return False
                
        except Exception as e:
            print(f"❌ Error running R tests: {e}")
            return False
    
    def _generate_dashboard(self) -> bool:
        """Generate results dashboard"""
        print("📊 Generating results dashboard...")
        
        # Check if we have comparison dashboard generator
        comparison_gen = self.target_dir / "comparison_dashboard_generator.py"
        enhanced_gen = self.target_dir / "enhanced_dashboard_generator.py"
        
        if not self.test_results:
            print("⚠️  No test results available for dashboard generation")
            return True
        
        try:
            if comparison_gen.exists() and len(self.test_results) > 1:
                # Use comparison dashboard for multiple languages
                return self._generate_comparison_dashboard()
            elif enhanced_gen.exists():
                # Use enhanced dashboard for single language
                return self._generate_enhanced_dashboard()
            else:
                print("⚠️  Dashboard generators not found")
                return True
                
        except Exception as e:
            print(f"❌ Error generating dashboard: {e}")
            return False
    
    def _generate_comparison_dashboard(self) -> bool:
        """Generate comparison dashboard with multiple language results"""
        dashboard_gen = self.target_dir / "comparison_dashboard_generator.py"
        dashboard_dir = self.target_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cmd = [
            sys.executable, str(dashboard_gen),
            "--project", str(self.target_dir),
            "--output", str(dashboard_dir)
        ]
        
        # Add result files for each language
        for lang, results in self.test_results.items():
            if lang.lower() == "python":
                cmd.extend(["--python-results", results['file']])
            elif lang.lower() == "r":
                cmd.extend(["--r-results", results['file']])
            elif lang.lower() == "rust":
                cmd.extend(["--rust-results", results['file']])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                self.dashboard_path = dashboard_dir / "index.html"
                print(f"✅ Comparison dashboard generated: {self.dashboard_path}")
                return True
            else:
                print(f"❌ Dashboard generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error generating comparison dashboard: {e}")
            return False
    
    def _generate_enhanced_dashboard(self) -> bool:
        """Generate enhanced dashboard for single language"""
        dashboard_gen = self.target_dir / "enhanced_dashboard_generator.py"
        dashboard_dir = self.target_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Use the first available result file
        result_file = next(iter(self.test_results.values()))['file']
        
        cmd = [
            sys.executable, str(dashboard_gen),
            "--project", str(self.target_dir),
            "--results", result_file,
            "--output", str(dashboard_dir)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                self.dashboard_path = dashboard_dir / "index.html"
                print(f"✅ Enhanced dashboard generated: {self.dashboard_path}")
                return True
            else:
                print(f"❌ Dashboard generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error generating enhanced dashboard: {e}")
            return False
    
    def _report_results(self):
        """Generate comprehensive results report"""
        print("\n" + "="*60)
        print("🎯 ENHANCED SYNC WORKFLOW SUMMARY")
        print("="*60)
        
        print(f"📁 Source: {self.source_dir}")
        print(f"📁 Target: {self.target_dir}")
        print(f"🌐 Language: {self.language}")
        print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.test_results:
            print(f"\n🧪 TEST EXECUTION RESULTS:")
            for lang, results in self.test_results.items():
                summary = results['data']['summary']
                total = summary['total_tests']
                passed = summary['passed_tests']
                failed = summary['failed_tests']
                errors = summary['error_tests']
                pass_rate = (passed / total * 100) if total > 0 else 0
                
                status_emoji = "✅" if errors == 0 and failed == 0 else "⚠️" if errors == 0 else "❌"
                print(f"  {status_emoji} {lang.upper()}: {passed}/{total} passed ({pass_rate:.1f}%), {failed} failed, {errors} errors")
                
                # Show result file
                print(f"     📄 Results: {results['file']}")
        
        if self.dashboard_path:
            print(f"\n📊 DASHBOARD: {self.dashboard_path}")
        
        # Integration guidance
        print(f"\n🔧 INTEGRATION GUIDANCE:")
        print(f"  • Copy test runners to other repositories")
        print(f"  • Run sync workflow: python {self.sync_bin}")
        print(f"  • Execute tests locally: python golden_test_runner.py")
        print(f"  • Generate dashboards: python comparison_dashboard_generator.py")
        
        print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced golden test sync with execution and validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic enhanced sync with test execution
  %(prog)s --source ~/code/reference/golden_tests --target ./golden_tests --language Python

  # Sync without test execution
  %(prog)s --source ~/code/reference/golden_tests --target ./golden_tests --language Python --no-execute

  # Full workflow with dashboard generation
  %(prog)s --source ~/code/reference/golden_tests --target ./golden_tests --language Python --generate-dashboard
        """
    )
    
    parser.add_argument('--source', '-s', required=True,
                        help='Source repository golden tests directory')
    parser.add_argument('--target', '-t', required=True,
                        help='Target repository golden tests directory')
    parser.add_argument('--language', '-l', required=True,
                        help='Target language (Python, R, Rust, etc.)')
    parser.add_argument('--sync-bin', 
                        help='Path to sync_golden_tests.py (default: ~/bin/sync_golden_tests.py)')
    parser.add_argument('--no-execute', action='store_true',
                        help='Skip test execution after sync')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Skip dashboard generation')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Preview changes without applying them')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')
    
    args = parser.parse_args()
    
    # Create enhanced sync workflow
    workflow = EnhancedGoldenTestSync(
        source_dir=args.source,
        target_dir=args.target,
        language=args.language,
        execute_tests=not args.no_execute,
        generate_dashboard=not args.no_dashboard,
        sync_bin=args.sync_bin,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    # Run the workflow
    success = workflow.run_enhanced_sync()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()