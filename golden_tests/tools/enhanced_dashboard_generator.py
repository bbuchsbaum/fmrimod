#!/usr/bin/env python3
"""
Enhanced Golden Tests Dashboard Generator with Test Results

Generates interactive HTML dashboards showing both test specifications
and actual test execution results with pass/fail status and detailed analysis.

Usage:
    python enhanced_dashboard_generator.py --project PATH [options]
    python enhanced_dashboard_generator.py --project PATH --results results.json
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import argparse
from dataclasses import dataclass


@dataclass
class TestExecutionData:
    """Container for test execution results"""
    test_id: str
    overall_status: str
    passed_checks: int
    failed_checks: int
    error_checks: int
    total_checks: int
    execution_time: float
    timestamp: str
    implementation_found: bool
    check_results: List[Dict]
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage"""
        if self.total_checks == 0:
            return 0.0
        return (self.passed_checks / self.total_checks) * 100
    
    @property
    def status_class(self) -> str:
        """Get CSS class for status"""
        if self.overall_status == 'pass':
            return 'status-pass'
        elif self.overall_status == 'fail':
            return 'status-fail'
        elif self.overall_status == 'error':
            return 'status-error'
        else:
            return 'status-skip'
    
    @property
    def status_color(self) -> str:
        """Get color for status indicators"""
        if self.overall_status == 'pass':
            return '#10b981'  # Green
        elif self.overall_status == 'fail':
            return '#ef4444'  # Red
        elif self.overall_status == 'error':
            return '#f59e0b'  # Orange
        else:
            return '#6b7280'  # Gray


class EnhancedDashboardGenerator:
    """Dashboard generator with test results integration"""
    
    def __init__(self, project_path: str, results_file: Optional[str] = None):
        self.project_path = Path(project_path)
        self.results_file = results_file
        self.namespace = {'gt': 'http://golden-tests.org/schema'}
        
        # Load test results if available
        self.test_results = {}
        self.results_metadata = {}
        if results_file and Path(results_file).exists():
            self._load_test_results(results_file)
    
    def _load_test_results(self, results_file: str):
        """Load test execution results from JSON file"""
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            self.results_metadata = data.get('metadata', {})
            
            # Index results by test_id
            for result in data.get('results', []):
                test_id = result['test_id']
                self.test_results[test_id] = TestExecutionData(
                    test_id=test_id,
                    overall_status=result['overall_status'],
                    passed_checks=result['passed_checks'],
                    failed_checks=result['failed_checks'],
                    error_checks=result['error_checks'],
                    total_checks=result['total_checks'],
                    execution_time=result['execution_time'],
                    timestamp=result['timestamp'],
                    implementation_found=result['implementation_found'],
                    check_results=result['results']
                )
                
        except Exception as e:
            print(f"Warning: Could not load test results from {results_file}: {e}")
    
    def generate_dashboard(self, output_dir: str):
        """Generate complete dashboard with test results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all test specifications
        specs_dir = self.project_path / "specs"
        test_specs = []
        
        if specs_dir.exists():
            for xml_file in specs_dir.rglob("*.xml"):
                try:
                    spec_data = self._parse_test_spec(xml_file)
                    if spec_data:
                        test_specs.append(spec_data)
                except Exception as e:
                    print(f"Error parsing {xml_file}: {e}")
        
        # Generate main dashboard
        self._generate_main_dashboard(test_specs, output_path)
        
        # Generate individual test pages
        for spec in test_specs:
            self._generate_test_detail_page(spec, output_path)
        
        # Generate results summary page if we have results
        if self.test_results:
            self._generate_results_summary_page(test_specs, output_path)
        
        # Copy static assets
        self._generate_static_assets(output_path)
        
        print(f"Enhanced dashboard generated in: {output_path}")
        print(f"Open {output_path}/index.html to view the dashboard")
    
    def _parse_test_spec(self, xml_file: Path) -> Optional[Dict]:
        """Parse test specification from XML file"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract basic metadata
            test_id = self._get_text(root, './/gt:id')
            description = self._get_text(root, './/gt:description')
            version = self._get_text(root, './/gt:version')
            
            # Extract tags
            tags = []
            tag_elements = root.findall('.//gt:tag', self.namespace)
            for tag_elem in tag_elements:
                if tag_elem.text:
                    tags.append(tag_elem.text)
            
            # Extract semantic description
            purpose = self._get_text(root, './/gt:purpose')
            algorithm = self._get_text(root, './/gt:algorithm')
            
            # Extract validation checks
            checks = []
            check_elements = root.findall('.//gt:check', self.namespace)
            for i, check in enumerate(check_elements):
                check_data = {
                    'index': i,
                    'type': self._get_text(check, 'gt:type'),
                    'location': self._get_text(check, 'gt:location'),
                    'expected': self._get_text(check, 'gt:expected'),
                    'tolerance': self._get_text(check, 'gt:tolerance'),
                    'min': self._get_text(check, 'gt:min'),
                    'max': self._get_text(check, 'gt:max'),
                    'property': self._get_text(check, 'gt:property')
                }
                checks.append(check_data)
            
            # Get test execution results if available
            execution_data = self.test_results.get(test_id)
            
            return {
                'test_id': test_id,
                'description': description,
                'version': version,
                'tags': tags,
                'purpose': purpose,
                'algorithm': algorithm,
                'checks': checks,
                'file_path': xml_file.relative_to(self.project_path),
                'execution_data': execution_data
            }
            
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            return None
    
    def _get_text(self, element: ET.Element, xpath: str) -> Optional[str]:
        """Get text from element using xpath with canonical namespace"""
        found = element.find(xpath, self.namespace)
        return found.text if found is not None else None
    
    def _generate_main_dashboard(self, test_specs: List[Dict], output_path: Path):
        """Generate main dashboard HTML"""
        # Calculate summary statistics
        total_tests = len(test_specs)
        tests_with_results = sum(1 for spec in test_specs if spec['execution_data'])
        
        if self.test_results:
            passed_tests = sum(1 for data in self.test_results.values() if data.overall_status == 'pass')
            failed_tests = sum(1 for data in self.test_results.values() if data.overall_status == 'fail')
            error_tests = sum(1 for data in self.test_results.values() if data.overall_status == 'error')
            skipped_tests = sum(1 for data in self.test_results.values() if data.overall_status == 'skip')
            
            total_checks = sum(data.total_checks for data in self.test_results.values())
            passed_checks = sum(data.passed_checks for data in self.test_results.values())
        else:
            passed_tests = failed_tests = error_tests = skipped_tests = 0
            total_checks = passed_checks = 0
        
        # Generate test cards HTML
        test_cards_html = ""
        for spec in sorted(test_specs, key=lambda x: x['test_id'] or ''):
            execution_data = spec['execution_data']
            
            # Status indicator
            if execution_data:
                status_html = f'''
                <div class="test-status {execution_data.status_class}">
                    <div class="status-indicator" style="background-color: {execution_data.status_color}"></div>
                    <span class="status-text">{execution_data.overall_status.upper()}</span>
                    <span class="pass-rate">{execution_data.pass_rate:.1f}% passed</span>
                </div>
                '''
                
                summary_html = f'''
                <div class="test-summary">
                    <span class="check-count">{execution_data.passed_checks}/{execution_data.total_checks} checks passed</span>
                    <span class="execution-time">{execution_data.execution_time:.3f}s</span>
                </div>
                '''
            else:
                status_html = '''
                <div class="test-status status-no-results">
                    <div class="status-indicator" style="background-color: #9ca3af"></div>
                    <span class="status-text">NO RESULTS</span>
                </div>
                '''
                summary_html = '<div class="test-summary">Not executed</div>'
            
            tags_html = " ".join(f'<span class="tag">{tag}</span>' for tag in spec['tags'])
            
            test_cards_html += f'''
            <div class="test-card">
                <div class="test-header">
                    <div>
                        <h3 class="test-title">{spec['description']}</h3>
                        <div class="test-id">{spec['test_id']}</div>
                    </div>
                    {status_html}
                </div>
                <div class="test-tags">{tags_html}</div>
                <div class="test-description">{spec['purpose'] or '...'}</div>
                {summary_html}
                <div class="test-actions">
                    <a href="test_{spec['test_id']}.html" class="btn btn-primary">View Details</a>
                </div>
            </div>
            '''
        
        # Generate results indicator
        results_indicator = ""
        if self.test_results:
            timestamp = self.results_metadata.get('timestamp', 'Unknown')
            results_indicator = f'''
            <div class="results-indicator">
                <span class="results-badge">Test Results Available</span>
                <span class="results-timestamp">Last run: {timestamp}</span>
                <a href="results_summary.html" class="btn btn-secondary">View Results Summary</a>
            </div>
            '''
        
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Golden Tests Dashboard</title>
    <link rel="stylesheet" href="static/enhanced_dashboard.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1>Golden Tests Dashboard</h1>
            <p>Cross-language test specifications and execution results</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {results_indicator}
        </div>
    </header>
    
    <main class="container">
        <!-- Statistics Overview -->
        <section class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{total_tests}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat-card stat-pass">
                <div class="stat-number">{passed_tests}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card stat-fail">
                <div class="stat-number">{failed_tests}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card stat-error">
                <div class="stat-number">{error_tests}</div>
                <div class="stat-label">Errors</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{tests_with_results}/{total_tests}</div>
                <div class="stat-label">With Results</div>
            </div>
        </section>
        
        <!-- Tests Grid -->
        <section class="tests-grid">
            {test_cards_html}
        </section>
    </main>
    
    <script src="static/enhanced_dashboard.js"></script>
</body>
</html>'''
        
        with open(output_path / "index.html", 'w') as f:
            f.write(html_content)
    
    def _generate_test_detail_page(self, spec: Dict, output_path: Path):
        """Generate detailed test page with results"""
        test_id = spec['test_id']
        execution_data = spec['execution_data']
        
        # Generate check results table
        checks_html = ""
        for i, check in enumerate(spec['checks']):
            # Find corresponding execution result
            exec_result = None
            if execution_data:
                exec_result = next((r for r in execution_data.check_results if r['check_index'] == i), None)
            
            # Status and result columns
            if exec_result:
                status_class = f"check-status-{exec_result['status']}"
                status_text = exec_result['status'].upper()
                actual_value = exec_result.get('actual', 'N/A')
                error_mag = exec_result.get('error_magnitude', 0)
                
                if exec_result['status'] == 'fail' and error_mag:
                    if error_mag < 0.001:
                        severity = "close"
                        severity_class = "severity-close"
                    elif error_mag < 0.1:
                        severity = "moderate"
                        severity_class = "severity-moderate"
                    else:
                        severity = "major"
                        severity_class = "severity-major"
                    
                    result_html = f'''
                    <div class="check-result {severity_class}">
                        <div class="status {status_class}">{status_text}</div>
                        <div class="actual-value">Actual: {actual_value}</div>
                        <div class="error-magnitude">Error: {error_mag:.6f} ({severity})</div>
                    </div>
                    '''
                else:
                    result_html = f'''
                    <div class="check-result">
                        <div class="status {status_class}">{status_text}</div>
                        <div class="actual-value">Actual: {actual_value}</div>
                    </div>
                    '''
            else:
                result_html = '<div class="check-result no-result">Not executed</div>'
            
            # Check specification
            expected_val = check.get('expected', 'N/A')
            tolerance_val = check.get('tolerance', 'N/A')
            min_val = check.get('min', 'N/A')
            max_val = check.get('max', 'N/A')
            
            checks_html += f'''
            <tr>
                <td>{i + 1}</td>
                <td><code>{check['type']}</code></td>
                <td><code>{check['location']}</code></td>
                <td>{expected_val}</td>
                <td>{tolerance_val}</td>
                <td>{min_val}</td>
                <td>{max_val}</td>
                <td>{result_html}</td>
            </tr>
            '''
        
        # Execution summary
        exec_summary = ""
        if execution_data:
            exec_summary = f'''
            <div class="execution-summary">
                <h3>Execution Results</h3>
                <div class="execution-stats">
                    <div class="stat">
                        <span class="label">Overall Status:</span>
                        <span class="value status-{execution_data.overall_status}">{execution_data.overall_status.upper()}</span>
                    </div>
                    <div class="stat">
                        <span class="label">Pass Rate:</span>
                        <span class="value">{execution_data.pass_rate:.1f}%</span>
                    </div>
                    <div class="stat">
                        <span class="label">Execution Time:</span>
                        <span class="value">{execution_data.execution_time:.3f}s</span>
                    </div>
                    <div class="stat">
                        <span class="label">Last Run:</span>
                        <span class="value">{execution_data.timestamp}</span>
                    </div>
                </div>
            </div>
            '''
        
        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test: {spec['description']}</title>
    <link rel="stylesheet" href="static/enhanced_dashboard.css">
</head>
<body>
    <header class="header">
        <div class="container">
            <h1>Test Details: {spec['description']}</h1>
            <p><a href="index.html" style="color: white;">← Back to Dashboard</a></p>
        </div>
    </header>
    
    <main class="container">
        <section class="test-metadata">
            <h2>Test Metadata</h2>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <span class="label">Test ID:</span>
                    <span class="value"><code>{test_id}</code></span>
                </div>
                <div class="metadata-item">
                    <span class="label">Version:</span>
                    <span class="value">{spec['version']}</span>
                </div>
                <div class="metadata-item">
                    <span class="label">File Path:</span>
                    <span class="value"><code>{spec['file_path']}</code></span>
                </div>
            </div>
        </section>
        
        {exec_summary}
        
        <section class="test-purpose">
            <h2>Purpose</h2>
            <p>{spec['purpose']}</p>
        </section>
        
        <section class="validation-checks">
            <h2>Validation Checks ({len(spec['checks'])} checks)</h2>
            <table class="checks-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Type</th>
                        <th>Location</th>
                        <th>Expected</th>
                        <th>Tolerance</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Result</th>
                    </tr>
                </thead>
                <tbody>
                    {checks_html}
                </tbody>
            </table>
        </section>
    </main>
</body>
</html>'''
        
        with open(output_path / f"test_{test_id}.html", 'w') as f:
            f.write(html_content)
    
    def _generate_results_summary_page(self, test_specs: List[Dict], output_path: Path):
        """Generate results summary page"""
        # Implementation for results summary page
        # This would show detailed failure analysis, trends, etc.
        pass
    
    def _generate_static_assets(self, output_path: Path):
        """Generate CSS and JavaScript files"""
        static_dir = output_path / "static"
        static_dir.mkdir(exist_ok=True)
        
        # Enhanced CSS with test result styling
        css_content = '''
:root {
    --primary-color: #2563eb;
    --success-color: #10b981;
    --error-color: #ef4444;
    --warning-color: #f59e0b;
    --gray-color: #6b7280;
    --light-bg: #f8fafc;
    --border-color: #e2e8f0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #ffffff;
    color: #1f2937;
}

.header {
    background-color: var(--primary-color);
    color: white;
    padding: 1.5rem 0;
    margin-bottom: 2rem;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
}

.header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 600;
}

.header p {
    margin: 0.5rem 0 0 0;
    opacity: 0.9;
}

.results-indicator {
    margin-top: 1rem;
    padding: 1rem;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.results-badge {
    background: var(--success-color);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-size: 0.875rem;
    font-weight: 600;
}

.results-timestamp {
    font-size: 0.875rem;
    opacity: 0.8;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}

.stat-card {
    background: white;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.stat-card.stat-pass {
    border-color: var(--success-color);
    background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
}

.stat-card.stat-fail {
    border-color: var(--error-color);
    background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
}

.stat-card.stat-error {
    border-color: var(--warning-color);
    background: linear-gradient(135deg, #ffffff 0%, #fffbeb 100%);
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--primary-color);
    line-height: 1;
}

.stat-label {
    font-size: 0.875rem;
    color: #6b7280;
    margin-top: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.tests-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    gap: 1.5rem;
}

.test-card {
    background: white;
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: box-shadow 0.2s;
}

.test-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.test-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
}

.test-title {
    margin: 0 0 0.5rem 0;
    font-size: 1.125rem;
    font-weight: 600;
    color: #1f2937;
}

.test-id {
    font-size: 0.875rem;
    color: #6b7280;
    font-family: monospace;
}

.test-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-shrink: 0;
}

.status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
}

.status-text {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
}

.pass-rate {
    font-size: 0.75rem;
    color: #6b7280;
}

.test-status.status-pass .status-text {
    color: var(--success-color);
}

.test-status.status-fail .status-text {
    color: var(--error-color);
}

.test-status.status-error .status-text {
    color: var(--warning-color);
}

.test-status.status-no-results .status-text {
    color: var(--gray-color);
}

.test-tags {
    margin-bottom: 1rem;
}

.tag {
    display: inline-block;
    background: var(--light-bg);
    color: #374151;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    margin-right: 0.5rem;
    margin-bottom: 0.25rem;
}

.test-description {
    color: #6b7280;
    margin-bottom: 1rem;
    line-height: 1.5;
}

.test-summary {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.875rem;
    color: #6b7280;
    margin-bottom: 1rem;
}

.check-count {
    font-weight: 600;
}

.execution-time {
    font-family: monospace;
}

.test-actions {
    display: flex;
    gap: 0.5rem;
}

.btn {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    text-decoration: none;
    font-size: 0.875rem;
    font-weight: 600;
    transition: all 0.2s;
}

.btn-primary {
    background: var(--primary-color);
    color: white;
}

.btn-primary:hover {
    background: #1d4ed8;
}

.btn-secondary {
    background: var(--light-bg);
    color: #374151;
    border: 1px solid var(--border-color);
}

.btn-secondary:hover {
    background: #e5e7eb;
}

/* Test detail page styles */
.test-metadata {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.metadata-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
}

.metadata-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.metadata-item .label {
    font-size: 0.875rem;
    font-weight: 600;
    color: #374151;
}

.metadata-item .value {
    color: #6b7280;
}

.execution-summary {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.execution-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.execution-stats .stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: var(--light-bg);
    border-radius: 6px;
}

.execution-stats .label {
    font-weight: 600;
    color: #374151;
}

.execution-stats .value {
    font-family: monospace;
    color: #6b7280;
}

.execution-stats .status-pass {
    color: var(--success-color);
    font-weight: 600;
}

.execution-stats .status-fail {
    color: var(--error-color);
    font-weight: 600;
}

.execution-stats .status-error {
    color: var(--warning-color);
    font-weight: 600;
}

.test-purpose {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.validation-checks {
    background: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.checks-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
}

.checks-table th,
.checks-table td {
    padding: 0.75rem;
    text-align: left;
    border: 1px solid var(--border-color);
    vertical-align: top;
}

.checks-table th {
    background-color: var(--light-bg);
    font-weight: 600;
    font-size: 0.875rem;
}

.checks-table code {
    background: var(--light-bg);
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.875rem;
}

.check-result {
    min-width: 150px;
}

.check-result .status {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    text-align: center;
    margin-bottom: 0.5rem;
}

.check-status-pass {
    background: #dcfce7;
    color: #166534;
}

.check-status-fail {
    background: #fef2f2;
    color: #dc2626;
}

.check-status-error {
    background: #fef3c7;
    color: #d97706;
}

.check-status-skip {
    background: #f3f4f6;
    color: #6b7280;
}

.actual-value {
    font-size: 0.875rem;
    font-family: monospace;
    color: #6b7280;
    margin-bottom: 0.25rem;
}

.error-magnitude {
    font-size: 0.75rem;
    font-weight: 600;
}

.severity-close {
    color: var(--warning-color);
}

.severity-moderate {
    color: #ea580c;
}

.severity-major {
    color: var(--error-color);
}

.no-result {
    color: var(--gray-color);
    font-style: italic;
}
'''
        
        with open(static_dir / "enhanced_dashboard.css", 'w') as f:
            f.write(css_content)
        
        # Simple JavaScript for interactivity
        js_content = '''
document.addEventListener('DOMContentLoaded', function() {
    // Add any interactive features here
    console.log('Enhanced Golden Tests Dashboard loaded');
});
'''
        
        with open(static_dir / "enhanced_dashboard.js", 'w') as f:
            f.write(js_content)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate enhanced golden tests dashboard with results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--project', '-p', required=True,
                       help='Path to golden tests project directory')
    parser.add_argument('--results', '-r',
                       help='Path to test results JSON file')
    parser.add_argument('--output', '-o', default='enhanced_dashboard_output',
                       help='Output directory for dashboard (default: enhanced_dashboard_output)')
    
    args = parser.parse_args()
    
    # Generate dashboard
    generator = EnhancedDashboardGenerator(
        project_path=args.project,
        results_file=args.results
    )
    
    generator.generate_dashboard(args.output)


if __name__ == '__main__':
    main()