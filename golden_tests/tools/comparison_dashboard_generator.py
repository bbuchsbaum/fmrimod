#!/usr/bin/env python3
"""
Comparison Dashboard Generator for Golden Tests
Generates HTML dashboards comparing test results across multiple languages (Python, R)
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import xml.etree.ElementTree as ET


@dataclass
class TestResultSummary:
    """Summary of test execution results for comparison"""
    test_id: str
    description: str
    language: str
    overall_status: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    error_checks: int
    execution_time: float
    timestamp: str
    results: List[Dict[str, Any]]

    @property
    def status_color(self) -> str:
        if self.overall_status == 'pass':
            return '#10b981'  # Green
        elif self.overall_status == 'fail':
            return '#ef4444'  # Red
        elif self.overall_status == 'error':
            return '#f59e0b'  # Orange
        else:
            return '#6b7280'  # Gray


class ComparisonDashboardGenerator:
    """Generate comparison dashboards for golden test results"""
    
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.specs_dir = self.project_dir / "specs"
        self.namespace = {'gt': 'http://golden-tests.org/schema'}
        
    def load_test_results(self, results_file: str, language: str) -> List[TestResultSummary]:
        """Load test results from JSON file"""
        results = []
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                
            for test_result in data.get('results', []):
                summary = TestResultSummary(
                    test_id=test_result['test_id'],
                    description=test_result['description'],
                    language=language,
                    overall_status=test_result['overall_status'],
                    total_checks=test_result['total_checks'],
                    passed_checks=test_result['passed_checks'],
                    failed_checks=test_result['failed_checks'],
                    error_checks=test_result['error_checks'],
                    execution_time=test_result['execution_time'],
                    timestamp=test_result['timestamp'],
                    results=test_result['results']
                )
                results.append(summary)
                
        except Exception as e:
            print(f"Error loading results from {results_file}: {e}")
            
        return results
    
    def load_test_spec(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Load test specification from XML"""
        # Find XML file for this test
        for xml_file in self.specs_dir.rglob("*.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                spec_id = root.find('.//gt:id', self.namespace)
                if spec_id is not None and spec_id.text == test_id:
                    return self.parse_xml_spec(root)
            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
                continue
        return None
    
    def parse_xml_spec(self, root) -> Dict[str, Any]:
        """Parse XML specification into dict"""
        spec = {}
        
        # Metadata
        metadata = {}
        for field in ['id', 'version', 'description']:
            elem = root.find(f'.//gt:{field}', self.namespace)
            if elem is not None:
                metadata[field] = elem.text
        spec['metadata'] = metadata
        
        # Tags
        tags = []
        for tag_elem in root.findall('.//gt:tag', self.namespace):
            tags.append(tag_elem.text)
        spec['tags'] = tags
        
        # Semantic description
        semantic = {}
        for field in ['purpose', 'algorithm', 'mathematical_properties', 'edge_cases']:
            elem = root.find(f'.//gt:{field}', self.namespace)
            if elem is not None:
                semantic[field] = elem.text
        spec['semantic_description'] = semantic
        
        # Expected outputs
        checks = []
        for check_elem in root.findall('.//gt:check', self.namespace):
            check = {}
            for field in ['type', 'location', 'expected', 'tolerance', 'min', 'max', 'property']:
                elem = check_elem.find(f'gt:{field}', self.namespace)
                if elem is not None:
                    check[field] = elem.text
            checks.append(check)
        spec['checks'] = checks
        
        return spec
    
    def generate_comparison_dashboard(self, language_results: Dict[str, List[TestResultSummary]], 
                                    output_dir: str = "comparison_dashboard"):
        """Generate comparison dashboard with multiple language results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Collect all unique test IDs
        all_test_ids = set()
        for results in language_results.values():
            for result in results:
                all_test_ids.add(result.test_id)
        
        # Generate HTML
        html_content = self._generate_comparison_html(language_results, sorted(all_test_ids))
        
        # Write HTML file
        html_file = output_path / "index.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"Comparison dashboard generated: {html_file}")
        return str(html_file)
    
    def _generate_comparison_html(self, language_results: Dict[str, List[TestResultSummary]], 
                                test_ids: List[str]) -> str:
        """Generate HTML content for comparison dashboard"""
        
        # Calculate overall statistics
        total_stats = {}
        for language, results in language_results.items():
            passed = sum(1 for r in results if r.overall_status == 'pass')
            failed = sum(1 for r in results if r.overall_status == 'fail')
            errors = sum(1 for r in results if r.overall_status == 'error')
            total_stats[language] = {
                'total': len(results),
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'pass_rate': (passed / len(results) * 100) if results else 0
            }
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Golden Tests Comparison Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8fafc;
            color: #1f2937;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }}
        .header p {{
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .stats-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
        }}
        .stats-card h3 {{
            margin: 0 0 1rem 0;
            color: #374151;
            font-size: 1.2rem;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 0.5rem 0;
        }}
        .pass-rate {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #10b981;
        }}
        .comparison-table {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .table-header {{
            background: #f3f4f6;
            padding: 1rem;
            font-weight: 600;
            border-bottom: 1px solid #e5e7eb;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
        }}
        .status-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
            color: white;
        }}
        .status-pass {{ background-color: #10b981; }}
        .status-fail {{ background-color: #ef4444; }}
        .status-error {{ background-color: #f59e0b; }}
        .status-skip {{ background-color: #6b7280; }}
        .details-btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.875rem;
        }}
        .details-btn:hover {{
            background: #5a67d8;
        }}
        .details-panel {{
            display: none;
            background: #f8fafc;
            padding: 1rem;
            margin-top: 0.5rem;
            border-radius: 6px;
            border: 1px solid #e5e7eb;
        }}
        .details-panel.active {{
            display: block;
        }}
        .check-result {{
            margin: 0.5rem 0;
            padding: 0.5rem;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.875rem;
        }}
        .check-pass {{
            background: #d1fae5;
            border-left: 3px solid #10b981;
        }}
        .check-fail {{
            background: #fee2e2;
            border-left: 3px solid #ef4444;
        }}
        .timestamp {{
            color: #6b7280;
            font-size: 0.875rem;
        }}
        .language-tag {{
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
            color: white;
        }}
        .lang-python {{ background-color: #3776ab; }}
        .lang-r {{ background-color: #276dc3; }}
        .lang-rust {{ background-color: #ce422b; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Golden Tests Comparison Dashboard</h1>
        <p>Cross-language test execution results • Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="stats-grid">
"""
        
        # Add language statistics
        for language, stats in total_stats.items():
            html += f"""
        <div class="stats-card">
            <h3>{language.upper()} Results</h3>
            <div class="stat-row">
                <span>Total Tests:</span>
                <span>{stats['total']}</span>
            </div>
            <div class="stat-row">
                <span>Passed:</span>
                <span style="color: #10b981">{stats['passed']}</span>
            </div>
            <div class="stat-row">
                <span>Failed:</span>
                <span style="color: #ef4444">{stats['failed']}</span>
            </div>
            <div class="stat-row">
                <span>Errors:</span>
                <span style="color: #f59e0b">{stats['errors']}</span>
            </div>
            <div class="stat-row">
                <span>Pass Rate:</span>
                <span class="pass-rate">{stats['pass_rate']:.1f}%</span>
            </div>
        </div>
"""
        
        html += """
    </div>

    <div class="comparison-table">
        <div class="table-header">Test Results Comparison</div>
        <table>
            <thead>
                <tr>
                    <th>Test ID</th>
                    <th>Language</th>
                    <th>Status</th>
                    <th>Checks</th>
                    <th>Execution Time</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Add test results
        for test_id in test_ids:
            for language, results in language_results.items():
                test_result = next((r for r in results if r.test_id == test_id), None)
                if test_result:
                    html += f"""
                <tr>
                    <td><strong>{test_result.test_id}</strong><br><small>{test_result.description}</small></td>
                    <td><span class="language-tag lang-{language.lower()}">{language.upper()}</span></td>
                    <td><span class="status-badge status-{test_result.overall_status}">{test_result.overall_status.upper()}</span></td>
                    <td>{test_result.passed_checks}/{test_result.total_checks} passed</td>
                    <td>{test_result.execution_time:.3f}s<br><span class="timestamp">{test_result.timestamp}</span></td>
                    <td>
                        <button class="details-btn" onclick="toggleDetails('{test_result.test_id}_{language}')">
                            View Details
                        </button>
                        <div id="{test_result.test_id}_{language}" class="details-panel">
"""
                    
                    # Add individual check results
                    for check in test_result.results:
                        status_class = "check-pass" if check['status'] == 'pass' else "check-fail"
                        html += f"""
                            <div class="check-result {status_class}">
                                <strong>Check {check['check_index']}:</strong> {check['location']}<br>
                                <strong>Status:</strong> {check['status']}
"""
                        if check['status'] == 'fail':
                            html += f"""
                                <br><strong>Expected:</strong> {check['expected']}
                                <br><strong>Actual:</strong> {check['actual']}
                                <br><strong>Error magnitude:</strong> {check.get('error_magnitude', 'N/A')}
"""
                        html += """
                            </div>
"""
                    
                    html += """
                        </div>
                    </td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>

    <script>
        function toggleDetails(panelId) {
            const panel = document.getElementById(panelId);
            panel.classList.toggle('active');
        }
    </script>
</body>
</html>
"""
        
        return html


def main():
    parser = argparse.ArgumentParser(description='Generate comparison dashboard for golden test results')
    parser.add_argument('--project', '-p', required=True, help='Path to golden tests project directory')
    parser.add_argument('--python-results', help='Path to Python test results JSON file')
    parser.add_argument('--r-results', help='Path to R test results JSON file')
    parser.add_argument('--rust-results', help='Path to Rust test results JSON file')
    parser.add_argument('--output', '-o', default='comparison_dashboard', help='Output directory for dashboard')
    
    args = parser.parse_args()
    
    generator = ComparisonDashboardGenerator(args.project)
    language_results = {}
    
    # Load results for each language
    if args.python_results:
        language_results['Python'] = generator.load_test_results(args.python_results, 'Python')
    
    if args.r_results:
        language_results['R'] = generator.load_test_results(args.r_results, 'R')
    
    if args.rust_results:
        language_results['Rust'] = generator.load_test_results(args.rust_results, 'Rust')
    
    if not language_results:
        print("No result files provided. Use --python-results, --r-results, or --rust-results")
        return
    
    # Generate dashboard
    dashboard_path = generator.generate_comparison_dashboard(language_results, args.output)
    print(f"Comparison dashboard generated at: {dashboard_path}")


if __name__ == "__main__":
    main()