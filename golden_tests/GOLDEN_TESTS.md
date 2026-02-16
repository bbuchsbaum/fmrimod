# Golden Tests: Complete Setup and Usage Guide

## Purpose
Golden tests ensure semantic equivalence across language implementations (R, Python, Rust) of the same software by validating numeric outputs rather than implementation details.

## Quick Setup for Port Repositories

### Prerequisites
- Python 3.7+ (for test runners and dashboards)
- R 4.0+ (for R test execution)
- Required packages: `pip install numpy scipy xml2` and `R -e "install.packages(c('xml2', 'jsonlite'))"`

### Complete Setup Checklist

**Step 1: Install Basic Sync Tool**
```bash
# Copy the basic sync tool to your system
cp /path/to/reference/repo/bin/sync_golden_tests.py ~/bin/
chmod +x ~/bin/sync_golden_tests.py
```

**Step 2: Create Golden Tests Directory**
```bash
mkdir -p golden_tests/specs/core
mkdir -p golden_tests/validators
cd golden_tests
```

**Step 3: Copy All Required Files**
Copy these files from the reference repository's `golden_tests/tools/` directory:
```bash
# Option A: Copy from organized tools directory (recommended)
cp /ref/repo/golden_tests/tools/golden_test_runner.py .
cp /ref/repo/golden_tests/tools/improved_test_runner.py .
cp /ref/repo/golden_tests/tools/golden_test_runner.R .
cp /ref/repo/golden_tests/tools/enhanced_dashboard_generator.py .
cp /ref/repo/golden_tests/tools/comparison_dashboard_generator.py .
cp /ref/repo/golden_tests/tools/enhanced_sync_workflow.py .
cp /ref/repo/golden_tests/tools/golden_tests_config.yaml . 2>/dev/null || true

# Option B: Copy from flat directory (if tools/ doesn't exist)
cp /ref/repo/golden_tests/golden_test_runner.py .
cp /ref/repo/golden_tests/improved_test_runner.py .
cp /ref/repo/golden_tests/golden_test_runner.R .
cp /ref/repo/golden_tests/enhanced_dashboard_generator.py .
cp /ref/repo/golden_tests/comparison_dashboard_generator.py .
cp /ref/repo/golden_tests/enhanced_sync_workflow.py .
cp /ref/repo/golden_tests/golden_tests_config.yaml . 2>/dev/null || true

# Always copy documentation
cp /ref/repo/golden_tests/GOLDEN_TESTS.md .

# Make scripts executable
chmod +x *.py *.R
```

**Step 4: Initial Sync from Reference Repository**
```bash
# Sync test specifications (replace with your language)
python enhanced_sync_workflow.py \
    --source /path/to/reference/repo/golden_tests \
    --target . \
    --language Python \
    --verbose
```

**Step 5: Verify Installation**
```bash
# Test Python runner
python golden_test_runner.py --help

# Test R runner (if using R)
Rscript golden_test_runner.R --help

# Test enhanced workflow
python enhanced_sync_workflow.py --help
```

### File Structure After Setup
```
golden_tests/
├── GOLDEN_TESTS.md                    # This documentation
├── golden_test_runner.py              # Python test execution
├── improved_test_runner.py            # Enhanced Python runner
├── golden_test_runner.R               # R test execution
├── enhanced_dashboard_generator.py    # Single-language dashboards
├── comparison_dashboard_generator.py  # Multi-language comparison
├── enhanced_sync_workflow.py          # Complete sync workflow
├── golden_tests_config.yaml           # Configuration (optional)
└── specs/                             # Test specifications
    └── core/
        └── *.xml                      # Golden test files
```

## Daily Usage Workflows

### Sync and Test Workflow
```bash
# 1. Sync latest specifications and run tests
python enhanced_sync_workflow.py \
    --source /path/to/reference/repo/golden_tests \
    --target . \
    --language Python \
    --verbose

# 2. View results dashboard
open dashboard_*/index.html

# 3. For comparison across languages (if you have both)
python comparison_dashboard_generator.py \
    --project . \
    --python-results golden_test_results_python_*.json \
    --r-results golden_test_results_r_*.json \
    --output comparison_dashboard
```

### Manual Test Execution
```bash
# Run Python tests only
python golden_test_runner.py --output results_python.json --verbose

# Run R tests only  
Rscript golden_test_runner.R --output results_r.json --verbose

# Run specific test
python golden_test_runner.py --test hrf_spmg1_basic --verbose
```

### Development Integration

**Add to your CI/CD (e.g., GitHub Actions):**
```yaml
# .github/workflows/golden-tests.yml
name: Golden Tests
on: [push, pull_request]
jobs:
  golden-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install numpy scipy
      - name: Run Golden Tests
        run: |
          cd golden_tests
          python enhanced_sync_workflow.py \
            --source ${REFERENCE_REPO_PATH}/golden_tests \
            --target . \
            --language Python
```

**Daily Development Routine:**
```bash
# Morning: sync latest specs
python enhanced_sync_workflow.py --source ~/ref/golden_tests --target . --language Python --dry-run

# After changes: validate your implementation  
python golden_test_runner.py --output results.json --verbose

# Before commit: ensure tests pass
python enhanced_sync_workflow.py --source ~/ref/golden_tests --target . --language Python
```

## Core Concepts

### What Golden Tests Are
- Language-agnostic test specifications in XML format
- Focus on WHAT code should do (semantic behavior), not HOW
- Validate using numeric outputs with tolerances
- Each language implements same semantics differently

### Key Principles
1. **Numeric Focus**: All validations based on matrix dimensions, values, statistical properties
2. **Semantic Descriptions**: Purpose + mathematical algorithm in each test
3. **Progressive Enhancement**: Spec → R → Python → Rust implementations
4. **Language Agnosticism**: Describe behavior, not implementation

## XML Test Structure

**CRITICAL**: All golden test XML files MUST follow this exact schema format. The dashboard generator and validators depend on this structure.

### Complete XML Schema Template

```xml
<?xml version="1.0" encoding="UTF-8"?>
<golden_test xmlns="http://golden-tests.org/schema">
  <metadata>
    <id>unique_test_id</id>
    <version>1.0</version>
    <description>Brief description</description>
    <tags>
      <tag>category</tag>
      <tag>component</tag>
    </tags>
  </metadata>
  
  <semantic_description>
    <purpose>What functionality is tested</purpose>
    <algorithm>Step-by-step mathematical description</algorithm>
    <mathematical_properties>Key properties and constraints</mathematical_properties>
    <edge_cases>Special cases and boundary conditions</edge_cases>
  </semantic_description>
  
  <inputs>
    <input name="input_name" type="numeric_vector">
      <description>Description of this input</description>
      <value>c(1, 2, 3, 4, 5)</value>
    </input>
  </inputs>
  
  <expected_outputs>
    <numeric_checks>
      <check>
        <type>exact_value|approximate|range|statistical</type>
        <location>R expression to evaluate</location>
        <expected>expected_value</expected>
        <tolerance>acceptable_deviation</tolerance>
      </check>
      <check>
        <type>range</type>
        <location>R expression to evaluate</location>
        <min>minimum_value</min>
        <max>maximum_value</max>
      </check>
    </numeric_checks>
  </expected_outputs>
  
  <implementations>
    <R><![CDATA[
      # R reference implementation
      # Function calls and test code here
    ]]></R>
    <Python><![CDATA[
      # Python implementation following semantic specification
      # Focus on mathematical behavior, not R syntax
    ]]></Python>
    <Rust><![CDATA[
      // Rust implementation using idiomatic Rust patterns
      // Maintain mathematical equivalence
    ]]></Rust>
  </implementations>
  
  <propagation_status>
    <implementation lang="R" status="completed" date="2024-01-15"/>
    <implementation lang="Python" status="pending"/>
    <implementation lang="Rust" status="pending"/>
  </propagation_status>
</golden_test>
```

### Schema Requirements

**MANDATORY Elements:**
1. **Root Element**: `<golden_test xmlns="http://golden-tests.org/schema">` - MUST include namespace
2. **Metadata Section**: `<metadata>` with `<id>`, `<version>`, `<description>`, `<tags>`
3. **Semantic Description**: `<semantic_description>` with at least `<purpose>` and `<algorithm>`
4. **Expected Outputs**: `<expected_outputs><numeric_checks>` with validation criteria
5. **Implementations**: `<implementations>` with language-specific code sections
6. **Propagation Status**: `<propagation_status>` tracking implementation progress

**Validation Rules:**
- ✅ **Correct Namespace**: `xmlns="http://golden-tests.org/schema"`
- ❌ **Invalid**: `xsi:noNamespaceSchemaLocation` or missing namespace
- ✅ **Element Names**: Use `<id>` not `<test_id>`, `<description>` not `<desc>`
- ✅ **Check Structure**: `<expected_outputs><numeric_checks><check>` hierarchy
- ✅ **Implementation Status**: Use `status="completed|pending|in_progress"`

### Common Schema Errors to Avoid

❌ **Wrong namespace declaration:**
```xml
<!-- WRONG -->
<golden_test xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
             xsi:noNamespaceSchemaLocation="../../schema/golden_test.xsd">
```

❌ **Incorrect element names:**
```xml
<!-- WRONG -->
<metadata>
  <test_id>my_test</test_id>  <!-- Should be <id> -->
  <name>Test Name</name>      <!-- Should be <description> -->
</metadata>
```

❌ **Wrong validation structure:**
```xml
<!-- WRONG -->
<test_case name="basic">
  <outputs>
    <output name="result" type="vector">
```

✅ **Correct validation structure:**
```xml
<!-- CORRECT -->
<expected_outputs>
  <numeric_checks>
    <check>
      <type>approximate</type>
      <location>function_call(inputs)</location>
```

## Directory Structure

```
golden_tests/
├── specs/
│   ├── core/
│   │   ├── event_model/
│   │   │   ├── basic_hrf.xml          # Start here
│   │   │   ├── multiple_conditions.xml
│   │   │   └── continuous_regressors.xml
│   │   ├── baseline_model/
│   │   └── hrf_bases/
│   ├── integration/
│   └── edge_cases/
├── schema/
│   └── golden_test.xsd
└── validators/
    ├── R/validate_specs.R
    ├── Python/validate_specs.py
    └── Rust/validate_specs.rs
```

## HTML Dashboard Visualization

The golden tests methodology includes an interactive HTML dashboard generator for visualizing test specifications, implementation status, and progress tracking across languages.

### Dashboard Generator Script

Location: `~/code/translation/golden/generate_dashboard.py` (also available as `generate_dashboard.py` in `~/bin`)

**Key Features:**
- **Interactive Overview**: Statistics, test grid, search/filter capabilities
- **Test Detail Pages**: Complete semantic descriptions, validation criteria, implementation code
- **Implementation Matrix**: Cross-language status tracking grid
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### Usage Examples

```bash
# Generate dashboard for a single project
generate_dashboard.py --project /path/to/golden_tests

# Generate for multiple projects
generate_dashboard.py --projects /path/to/project1 /path/to/project2

# Custom output directory
generate_dashboard.py --project /path/to/golden_tests --output /path/to/dashboard

# For fmrihrf project specifically
generate_dashboard.py --project ~/code/fmrihrf/golden_tests --output ./docs/dashboard
```

### Dashboard Components

**Generated Files:**
- `index.html` - Main dashboard overview with statistics and test grid
- `implementation_matrix.html` - Cross-language implementation status matrix
- `test_[id].html` - Individual test detail pages with specifications
- `static/` - CSS, JavaScript, and styling assets

**Interactive Features:**
- Search across test descriptions and metadata
- Filter by language, status, or tags
- Sort by various criteria
- Navigate between overview, details, and matrix views

### Integration with Development Workflow

The dashboard is designed to support the golden tests methodology by:

1. **Progress Tracking**: Visual indicators for implementation status across languages
2. **Documentation**: Living documentation of test specifications and algorithms
3. **Cross-Language Coordination**: Clear view of what needs to be implemented where
4. **Quality Assurance**: Easy identification of failing tests and patterns

### CI/CD Integration

The dashboard can be integrated into continuous integration workflows:

```yaml
# Example GitHub Actions workflow
- name: Generate Golden Tests Dashboard
  run: |
    generate_dashboard.py --project ./golden_tests --output ./docs/dashboard
    
- name: Deploy to GitHub Pages
  uses: peaceiris/actions-gh-pages@v3
  with:
    publish_dir: ./docs/dashboard
```

This provides an always up-to-date view of test specifications and implementation progress for development teams.

## Test Execution Framework

The golden tests methodology includes comprehensive test execution tools that can run implementations and validate numerical outputs against the XML specifications. This enables automated testing across language ports to ensure semantic equivalence.

### Python Test Runners

Two Python test runners are available for executing golden tests:

#### 1. Basic Test Runner (`golden_test_runner.py`)
Basic execution framework for validating Python implementations against golden test specifications.

**Usage:**
```bash
# Run all tests
python golden_test_runner.py --golden-tests-dir /path/to/golden_tests

# Run specific test
python golden_test_runner.py --test hrf_spmg1_basic

# Save results to file
python golden_test_runner.py --output test_results.json
```

#### 2. Advanced Test Runner (`improved_test_runner.py`)
Enhanced execution framework with R-to-Python expression translation and comprehensive error analysis.

**Key Features:**
- **R Expression Translation**: Automatically converts R test expressions to Python equivalents
- **Comprehensive Error Analysis**: Calculates error magnitudes and severity classifications
- **Built-in Function Support**: Handles common R functions (`length()`, `seq()`, `which.max()`, etc.)
- **Advanced Context Creation**: Automatically creates test variables and input data

**Usage:**
```bash
# Run all tests with detailed output
python improved_test_runner.py --verbose

# Run specific test
python improved_test_runner.py --test hrf_spmg1_basic --verbose

# Save results with custom filename
python improved_test_runner.py --output comprehensive_results.json
```

### R-to-Python Expression Translation

The improved test runner includes automatic translation of R expressions to Python equivalents:

**Translation Examples:**
```r
# R Expression → Python Translation
length(output) → len(output)
seq(0, 30, by = 0.1) → np.arange(0, 30 + 0.1, 0.1)
which.max(hrf_values) → np.argmax(hrf_values)
c(-5, -1, -0.1) → np.array([-5, -1, -0.1])
max(hrf_result) → np.max(hrf_result)
```

**Supported R Functions:**
- `length()` → `len()`
- `seq(from, to, by=step)` → `np.arange(from, to+step, step)`
- `c(...)` → `np.array([...])`
- `which.max()` → `np.argmax()`
- `sum()`, `max()`, `min()`, `mean()` → `np.sum()`, `np.max()`, `np.min()`, `np.mean()`

### Numerical Comparison Engine

The test execution framework includes a sophisticated numerical comparison engine that handles multiple tolerance types:

#### Tolerance Types

1. **exact_value**: For precise integer or categorical comparisons
   ```xml
   <check>
     <type>exact_value</type>
     <location>length(output)</location>
     <expected>301</expected>
     <tolerance>0</tolerance>
   </check>
   ```

2. **approximate**: For floating-point comparisons with absolute tolerance
   ```xml
   <check>
     <type>approximate</type>
     <location>max(hrf_result)</location>
     <expected>0.05504</expected>
     <tolerance>1e-4</tolerance>
   </check>
   ```

3. **range**: For values that must fall within bounds
   ```xml
   <check>
     <type>range</type>
     <location>peak_time</location>
     <min>5.0</min>
     <max>6.0</max>
   </check>
   ```

4. **statistical**: For computed statistical properties
   ```xml
   <check>
     <type>statistical</type>
     <property>mean</property>
     <location>hrf_values</location>
     <expected>0.01836</expected>
     <tolerance>1e-4</tolerance>
   </check>
   ```

#### Error Magnitude Analysis

The comparison engine calculates precise error magnitudes and classifies failure severity:

- **Close**: error < 0.001 (minor numerical differences, often acceptable)
- **Moderate**: 0.001 ≤ error < 0.1 (significant differences requiring investigation)
- **Major**: error ≥ 0.1 (substantial implementation differences requiring fixes)

### JSON Result Storage Format

Test execution results are stored in structured JSON format with comprehensive metadata:

```json
{
  "metadata": {
    "timestamp": "2025-07-08T16:33:47.325937",
    "runner_version": "2.0.0",
    "total_tests": 10
  },
  "summary": {
    "total_tests": 10,
    "passed_tests": 0,
    "failed_tests": 3,
    "error_tests": 7,
    "skipped_tests": 0,
    "total_checks": 156,
    "passed_checks": 23,
    "failed_checks": 45
  },
  "results": [
    {
      "test_id": "hrf_spmg1_basic",
      "description": "Basic SPM canonical double gamma HRF function", 
      "overall_status": "fail",
      "passed_checks": 5,
      "failed_checks": 7,
      "error_checks": 0,
      "execution_time": 0.090833,
      "implementation_found": true,
      "results": [
        {
          "test_id": "hrf_spmg1_basic",
          "check_index": 3,
          "check_type": "approximate",
          "location": "max(hrf_spmg1(seq(0, 30, by = 0.1)))",
          "status": "fail",
          "expected": 0.05504,
          "actual": 0.054913083767180196,
          "tolerance": 1e-4,
          "error_magnitude": 0.00012691623281980285,
          "execution_time": 0.002
        }
      ]
    }
  ]
}
```

**Result Fields:**
- **metadata**: Execution timestamp, runner version, test counts
- **summary**: Aggregate statistics across all tests
- **results**: Individual test results with check-level details
- **error_magnitude**: Precise numerical difference for failed checks
- **execution_time**: Performance metrics for each test and check

## Enhanced HTML Dashboard with Test Results

The enhanced dashboard generator (`enhanced_dashboard_generator.py`) provides comprehensive visualization of both test specifications and execution results, enabling teams to see exactly where implementations are failing and how severely.

### Enhanced Dashboard Features

**Visual Status Indicators:**
- 🟢 **Pass**: All checks passed within tolerance
- 🔴 **Fail**: One or more checks failed with numerical differences  
- 🟠 **Error**: Execution errors (missing functions, syntax issues)
- ⚫ **Skip**: No implementation found or test skipped

**Error Severity Classification:**
- **Close failures**: Minor numerical differences (yellow indicator)
- **Moderate failures**: Significant differences requiring investigation (orange indicator)
- **Major failures**: Substantial implementation issues (red indicator)

### Enhanced Dashboard Usage

```bash
# Generate dashboard with test results
python enhanced_dashboard_generator.py --project /path/to/golden_tests --results test_results.json --output dashboard_with_results

# Generate specs-only dashboard (original functionality preserved)
python enhanced_dashboard_generator.py --project /path/to/golden_tests --output specs_dashboard
```

### Dashboard Components with Results

**Generated Files:**
- `index.html` - Main dashboard with execution statistics and pass/fail status
- `test_[id].html` - Individual test pages with detailed result analysis
- `results_summary.html` - Cross-test failure analysis and trends
- `static/` - Enhanced CSS and JavaScript with result visualization

**New Interactive Features:**
- **Execution Statistics**: Total tests, pass/fail rates, error counts
- **Result Timestamps**: When tests were last executed and results updated
- **Failure Analysis**: Error magnitude visualization and severity classification
- **Check-Level Details**: Individual validation failures with precise error values
- **Implementation Status**: Whether code was found and executed successfully

**Example Dashboard Output:**
```
Golden Tests Dashboard
======================
📊 Statistics:
   Total Tests: 10
   ✅ Passed: 0    ❌ Failed: 3    🟠 Errors: 7    ⚫ Skipped: 0
   📈 Pass Rate: 0%    🕐 Last Run: 2025-07-08 16:33:47

🧪 Test Results:
   hrf_spmg1_basic: ❌ FAIL (5/12 checks passed, close failures)
   hrf_lwu_basic: ❌ FAIL (0/17 checks passed, major failures)  
   hrf_gamma_vs_gaussian: ❌ FAIL (16/25 checks passed, moderate failures)
```

## Multi-Repository Testing Workflow

The test execution framework integrates with the existing `~/bin/sync_golden_tests.py` workflow to enable comprehensive cross-language testing across repository boundaries.

### Complete Multi-Repository Workflow

#### 1. **Specification Propagation** (via sync_golden_tests.py)
```bash
# In the reference repository (e.g., fmrihrf R package)
~/bin/sync_golden_tests.py --push-to /path/to/python-port/golden_tests
~/bin/sync_golden_tests.py --push-to /path/to/rust-port/golden_tests
```

#### 2. **Implementation and Testing** (in each port repository)
```bash
# In Python port repository
cd /path/to/python-port
python golden_tests/improved_test_runner.py --verbose --output python_results.json

# In Rust port repository  
cd /path/to/rust-port
cargo run --bin golden_test_runner -- --output rust_results.json
```

#### 3. **Dashboard Generation with Results**
```bash
# Generate comprehensive dashboard with results
python golden_tests/enhanced_dashboard_generator.py \
  --project ./golden_tests \
  --results python_results.json \
  --output dashboard_with_results

# View results
open dashboard_with_results/index.html
```

#### 4. **Result Analysis and Implementation Improvement**
- Review failed tests and error magnitudes
- Identify close vs. major failures for prioritization
- Fix implementations and re-run tests
- Update dashboards to track progress

### Cross-Language Result Comparison

The framework enables systematic comparison of implementation quality across languages:

**Result Comparison Workflow:**
1. **Execute tests in each port** using language-specific runners
2. **Compare error magnitudes** across implementations  
3. **Identify patterns** in failures (systematic vs. isolated issues)
4. **Prioritize fixes** based on error severity and test importance
5. **Track progress** through dashboard updates

**Example Cross-Language Analysis:**
```json
{
  "test_id": "hrf_spmg1_basic",
  "language_results": {
    "Python": {"status": "fail", "max_error": 0.000127, "severity": "close"},
    "Rust": {"status": "fail", "max_error": 0.045, "severity": "moderate"}, 
    "R": {"status": "pass", "max_error": 0.0, "severity": "reference"}
  }
}
```

## Test Result Analysis and Interpretation

Understanding test execution results is crucial for improving implementation quality and ensuring semantic equivalence across language ports.

### Interpreting Error Magnitudes

**Error Magnitude Significance:**
- **< 1e-10**: Machine precision differences (usually acceptable)
- **1e-10 to 1e-6**: Small numerical differences (investigate if systematic)
- **1e-6 to 1e-3**: Moderate differences (likely algorithmic variations)
- **1e-3 to 1e-1**: Significant differences (implementation issues)
- **> 1e-1**: Major differences (substantial implementation problems)

**Error Classification Examples:**
```json
{
  "check_results": [
    {
      "description": "HRF peak value",
      "expected": 0.05504,
      "actual": 0.05491,
      "error_magnitude": 0.000127,
      "severity": "close",
      "interpretation": "Excellent agreement - minor parameter differences"
    },
    {
      "description": "Statistical mean",
      "expected": 0.01836,
      "actual": 0.00867,
      "error_magnitude": 0.00969,
      "severity": "moderate", 
      "interpretation": "Moderate difference - check algorithm implementation"
    },
    {
      "description": "Area under curve",
      "expected": 0.5509,
      "actual": 0.2618,
      "error_magnitude": 0.2891,
      "severity": "major",
      "interpretation": "Major difference - significant implementation issue"
    }
  ]
}
```

### Using Results for Implementation Improvement

**Iterative Improvement Process:**
1. **Run comprehensive test suite** to establish baseline
2. **Identify highest-priority failures** (major errors in core functionality)
3. **Analyze specific failure patterns** (systematic vs. isolated issues)
4. **Fix implementations incrementally** starting with major errors
5. **Re-run tests and validate improvements**
6. **Update dashboards to track progress**

**Common Failure Patterns and Solutions:**
- **All values scaled by constant factor**: Check parameter values or scaling factors
- **Shape correct but magnitude wrong**: Verify normalization or amplitude settings
- **Systematic offset**: Check baseline or zero-point handling
- **Random scatter**: Investigate floating-point precision or algorithm stability

## Multi-Repository Sync Integration

### Enhanced Sync Workflow

The golden tests ecosystem supports multi-repository development with automated sync, execution, and validation. Use the **Enhanced Sync Workflow** for complete integration.

**Core Components:**
- **Basic Sync**: `~/bin/sync_golden_tests.py` - Synchronizes specifications while preserving implementations
- **Enhanced Workflow**: `enhanced_sync_workflow.py` - Adds test execution and dashboard generation
- **Cross-Language Validation**: Comparison dashboards for multi-language verification

### Quick Start Multi-Repository Workflow

```bash
# 1. Sync specifications and execute tests
python enhanced_sync_workflow.py \
    --source ~/code/reference/golden_tests \
    --target ./golden_tests \
    --language Python \
    --verbose

# 2. Run multiple languages for comparison
python enhanced_sync_workflow.py --source ~/ref/golden_tests --target . --language R
python enhanced_sync_workflow.py --source ~/ref/golden_tests --target . --language Python

# 3. Generate cross-language comparison dashboard
python comparison_dashboard_generator.py \
    --project . \
    --python-results golden_test_results_python.json \
    --r-results golden_test_results_r.json \
    --output comparison_dashboard
```

**Enhanced Workflow Features:**
- **Automated Sync**: Pulls latest specifications from reference repository
- **Preserve Local Code**: Keeps language-specific implementations intact
- **Test Execution**: Runs golden tests using appropriate language runner
- **Results Validation**: Generates JSON results with detailed failure analysis
- **Dashboard Generation**: Creates visual comparison reports
- **Integration Guidance**: Provides setup instructions for multi-repo deployment

### Integration Patterns

**Pattern 1: Continuous Integration**
```yaml
# .github/workflows/golden-tests.yml
- name: Validate Golden Tests
  run: |
    cd golden_tests
    python enhanced_sync_workflow.py \
      --source ${REFERENCE_REPO}/golden_tests \
      --target . \
      --language Python
```

**Pattern 2: Multi-Language Repository**
```bash
# Sync each language implementation
cd python/golden_tests && python enhanced_sync_workflow.py --source ~/ref --target . --language Python
cd r/golden_tests && python enhanced_sync_workflow.py --source ~/ref --target . --language R
cd rust/golden_tests && python enhanced_sync_workflow.py --source ~/ref --target . --language Rust
```

**Pattern 3: Development Workflow**
```bash
# Daily development sync
python enhanced_sync_workflow.py --source ~/ref --target . --language Python --dry-run  # Preview
python enhanced_sync_workflow.py --source ~/ref --target . --language Python           # Execute
```

For complete integration instructions, see [SYNC_INTEGRATION.md](SYNC_INTEGRATION.md).

### When Failures Indicate Specification Issues

Not all test failures indicate implementation problems. Sometimes they reveal issues with the test specifications themselves:

**Specification Issues to Consider:**
- **Unrealistic tolerances**: Too tight for the algorithm's inherent variability
- **Incorrect expected values**: Based on outdated or incorrect reference implementation
- **Missing context**: Test expressions assume variables not created by implementation
- **Cross-language incompatibilities**: R-specific behaviors that don't translate directly

**Resolution Process:**
1. **Verify reference implementation** produces expected values
2. **Check tolerance appropriateness** for the mathematical operation
3. **Validate test expression translation** from R to target language
4. **Consider algorithmic alternatives** that achieve semantic equivalence
5. **Update specifications if needed** and re-propagate to all ports

## Validation Types

1. **Dimensional**: Matrix/array dimensions, shape consistency
2. **Value Checks**:
   - Exact: For integers/categorical
   - Approximate: Floating-point with tolerance
   - Range: Within bounds
   - Relative: Percentage-based
3. **Statistical**: Sum, mean, std dev, min/max, percentiles
4. **Structural**: Column/row names, ordering, sparsity

## Workflow for New Tests

1. **Create XML spec** in appropriate directory using the EXACT schema format above
2. **Validate XML structure** - check namespace, element names, and hierarchy
3. **Write comprehensive semantic description** (purpose + algorithm + properties + edge cases)
4. **Define clear validation checks** with appropriate tolerances
5. **Implement in R** and validate outputs
6. **Test with dashboard generator** to ensure parsing works correctly
7. **Execute tests with Python runner** to validate implementation and check translations:
   ```bash
   python improved_test_runner.py --test your_test_id --verbose
   ```
8. **Review test results** and adjust tolerances or fix implementations as needed
9. **Generate dashboard with results** to visualize pass/fail status
10. **Document propagation status** with initial implementation status
11. **Other languages** see failing tests, implement, and validate with test runners
12. **Update XML** with their implementations and re-test
13. **Track cross-language progress** through enhanced dashboards

### XML Validation Checklist

Before committing any golden test XML file, verify:

- [ ] **Namespace**: Uses `xmlns="http://golden-tests.org/schema"`
- [ ] **Root Element**: `<golden_test>` (not `<test>` or other)
- [ ] **Metadata Complete**: Has `<id>`, `<version>`, `<description>`, `<tags>`
- [ ] **ID Format**: Uses underscores, no spaces (e.g., `hrf_spmg1_basic`)
- [ ] **Semantic Description**: Has `<purpose>` and `<algorithm>` at minimum
- [ ] **Validation Structure**: Uses `<expected_outputs><numeric_checks><check>`
- [ ] **Check Types**: Uses valid types (`exact_value`, `approximate`, `range`, `statistical`)
- [ ] **Implementation Sections**: Has `<R>`, `<Python>`, `<Rust>` (even if empty)
- [ ] **Status Tracking**: Has `<propagation_status>` with proper attributes
- [ ] **Dashboard Test**: File parses correctly with `generate_dashboard.py`

### Dashboard Integration Test

After creating or modifying XML files, always test with the dashboard:

```bash
# Test that your XML file parses correctly
generate_dashboard.py --project ./golden_tests --output ./test_dashboard

# Check for parsing errors in the output
# Verify your test appears with complete information
```

If your test appears blank or incomplete in the dashboard, it indicates a schema formatting issue.

## Workflow for New Language Implementation

1. **Get test specs** (submodule/copy from R repo)
2. **Create validator** in your test suite:
   ```python
   class GoldenTestValidator:
       def parse_golden_test(xml_path)
       def perform_numeric_check(matrix, check)
       def validate_test(xml_path)
   ```
3. **Start with basic_hrf.xml** - simplest test
4. **Implement required functionality** based on semantic descriptions
5. **Add your code to XML** implementations section
6. **Submit PR** to share implementation

## Test Sharing Methods

1. **Git Submodule** (recommended):
   ```bash
   git submodule add https://github.com/user/fmridesign.git fmridesign-r
   ln -s fmridesign-r/golden_tests golden_tests
   ```

2. **Separate Repository**: Dedicated golden-tests repo

3. **Package Distribution**: Include in package data

## Best Practices

### Adding Tests
- One behavior per test
- Start simple, minimal inputs
- Document WHY test exists
- Set appropriate tolerances (tighter for deterministic, looser for iterative)

### Implementing in New Language
- Match semantics, not syntax
- Focus on numeric equivalence
- Document any deviations in `<implementation_notes>`
- Use idiomatic code for your language

### Handling Differences
```xml
<implementation_notes>
  <note lang="Python">
    Uses scipy.linalg, tolerance 1e-6 for eigenvalues
  </note>
</implementation_notes>
```

## Current Test Status

Based on the latest test execution results (run comprehensive dashboard for real-time status):

| Test ID | R | Python | Rust | Last Execution | Status Details |
|---------|---|--------|------|----------------|----------------|
| hrf_spmg1_basic | ✅ | ❌ | ⏳ | 2025-07-08 | Python: 5/12 checks passed, close failures |
| hrf_lwu_basic | ✅ | ❌ | ⏳ | 2025-07-08 | Python: 0/17 checks passed, major failures |
| hrf_gamma_vs_gaussian | ✅ | ❌ | ⏳ | 2025-07-08 | Python: 18/25 checks passed, mixed failures |
| hrf_bspline_basis | ✅ | 🔧 | ⏳ | 2025-07-08 | Python: Missing R-to-Python translations |
| empirical_hrf_interpolation | ✅ | 🔧 | ⏳ | 2025-07-08 | Python: Missing context variables |
| evaluate_hrf_duration | ✅ | 🔧 | ⏳ | 2025-07-08 | Python: Missing evaluate() function |
| regressor_construction | ✅ | 🔧 | ⏳ | 2025-07-08 | Python: Complex R expressions need translation |
| bind_basis_combination | ✅ | 🔧 | ⏳ | 2025-07-08 | Python: Missing implementation |
| hrf_fir_generator | ✅ | 🔧 | ⏳ | 2025-07-08 | Python: Missing implementation |
| block_hrf_decorator | ✅ | ❌ | ⏳ | 2025-07-08 | Python: 1/13 checks passed, missing dependencies |

**Legend:** 
- ✅ **Complete**: All tests passing
- ❌ **Failing**: Implementation found but tests failing  
- 🔧 **Needs Work**: Missing implementations or translation issues
- ⏳ **Pending**: Not yet implemented
- 🚧 **In Progress**: Currently being developed

**Summary Statistics:**
- **Total Tests**: 10
- **Python Status**: 0 passing, 3 failing, 7 needing work
- **Overall Pass Rate**: 0% (Python port needs significant implementation work)

**View Live Results:**
```bash
# Generate current dashboard with latest results
python improved_test_runner.py --output latest_results.json
python enhanced_dashboard_generator.py --project . --results latest_results.json --output live_dashboard
open live_dashboard/index.html
```

## Key Files to Read for Context

### Test Specifications and Examples
1. **Test examples**: Look at `specs/core/hrf_spmg1_basic.xml` for a complete working example
2. **Schema**: Review `schema/golden_test.xsd` for XML structure requirements

### Test Execution Framework
3. **Advanced Test Runner**: `improved_test_runner.py` - R-to-Python translation and comprehensive execution
4. **Basic Test Runner**: `golden_test_runner.py` - Simple validation framework
5. **Test Results**: `all_tests_results.json` - Example of comprehensive result format

### Dashboard and Visualization
6. **Enhanced Dashboard Generator**: `enhanced_dashboard_generator.py` - Dashboard with test results integration
7. **Generated Dashboards**: `comprehensive_dashboard/index.html` - Example dashboard with results
8. **Original Dashboard Generator**: `~/code/translation/golden/generate_dashboard.py` - Specs-only visualization

### Validation and Implementation
9. **R Validator**: `validators/R/validate_golden_tests.R` - R-specific validation patterns
10. **Configuration**: `golden_tests_config.yaml` - Test execution configuration options

## Critical Points for AI Understanding

### Core Principles
1. **Tests define behavior, not implementation** - focus on mathematical equivalence
2. **All validation is numeric** - no string comparisons, UI testing, or performance metrics
3. **Progressive workflow** - R implements first, others follow semantic spec using test execution results
4. **Tolerances matter** - document why specific values chosen, validated through test execution
5. **Cross-language collaboration** - implementations shared via XML updates and validated through test runners

### Test Execution Framework
6. **Automated validation** - test runners provide immediate feedback on implementation quality
7. **Error magnitude analysis** - precise numerical differences help prioritize fixes
8. **R-to-Python translation** - automatic conversion enables cross-language test execution
9. **Severity classification** - close/moderate/major errors guide implementation priorities
10. **Result-driven development** - dashboards with execution results enable data-driven improvement

### Multi-Repository Workflow
11. **Sync-test-dashboard cycle** - specifications propagate via sync, validate via test runners, visualize via enhanced dashboards
12. **Cross-language comparison** - test results enable systematic comparison of implementation quality across ports
13. **Incremental improvement** - error magnitude tracking shows progress over time
14. **Live documentation** - enhanced dashboards provide real-time view of test status and implementation progress

## Common Pitfalls to Avoid

### Specification Development
- Don't modify existing specs when adding new language
- Don't test implementation details (data structures, variable names)
- Don't assume specific libraries available
- Don't use language-specific features in semantic descriptions
- Don't forget to update propagation_status
- **Don't use alternative XML schemas** - stick to the standard format
- **Don't skip dashboard validation** - always test XML parsing

### Test Execution and Results
- **Don't ignore test execution results** - they reveal real implementation issues
- **Don't dismiss close failures** - they may indicate systematic parameter differences
- **Don't assume R expressions translate automatically** - verify with test runner
- **Don't skip error magnitude analysis** - it guides fix prioritization
- **Don't test without running validation** - implementations may appear correct but fail numerically
- **Don't ignore missing implementations** - track coverage through dashboard results
- **Don't update specifications without re-testing** - changes may break existing implementations

### Dashboard and Workflow
- **Don't generate dashboards without results** - they provide incomplete information
- **Don't skip cross-language result comparison** - it reveals implementation quality differences  
- **Don't ignore execution errors** - they often indicate missing R-to-Python translations
- **Don't assume tolerance values are correct** - validate through actual test execution

## Troubleshooting and Common Issues

### Setup Problems

**Problem: "sync_golden_tests.py not found"**
```bash
# Solution: Install the basic sync tool
cp /path/to/reference/repo/bin/sync_golden_tests.py ~/bin/
chmod +x ~/bin/sync_golden_tests.py

# Or specify custom location
python enhanced_sync_workflow.py --sync-bin /custom/path/sync_golden_tests.py ...
```

**Problem: "Permission denied" when running scripts**
```bash
# Solution: Make scripts executable
chmod +x *.py *.R
```

**Problem: Missing Python dependencies**
```bash
# Solution: Install required packages
pip install numpy scipy xml2 jsonlite

# For conda users
conda install numpy scipy libxml2
```

**Problem: Missing R dependencies**
```bash
# Solution: Install R packages
R -e "install.packages(c('xml2', 'jsonlite'))"
```

**Problem: "Test runner not found"**
```bash
# Solution: Copy test runners to current directory
cp /ref/repo/golden_tests/golden_test_runner.py .
cp /ref/repo/golden_tests/golden_test_runner.R .
chmod +x *.py *.R
```

### Execution Problems

**Problem: Python tests fail with "could not find function"**
- **Cause**: Test requires functions from your actual package
- **Solution**: Update test runner to load your package:
```python
# In execute_implementation() function
import your_package
exec_env['your_function'] = your_package.your_function
```

**Problem: R tests fail with "object not found"**
- **Cause**: Test calls functions not defined in XML
- **Solution**: Either update XML implementation or load your R package:
```r
# In execute_implementation() function
library(your_package)
```

**Problem: Tests pass but values are very different**
- **Cause**: Different mathematical implementation
- **Solution**: Check if you're using correct algorithm formulation
- **Debug**: Compare intermediate calculations step by step

**Problem: "No test results generated"**
```bash
# Debug steps:
# 1. Check if specs directory exists
ls -la specs/

# 2. Run with verbose output
python golden_test_runner.py --verbose

# 3. Check for XML parsing errors
python -c "import xml.etree.ElementTree as ET; ET.parse('specs/core/test.xml')"
```

### Dashboard Generation Problems

**Problem: "Dashboard generator not found"**
```bash
# Solution: Copy dashboard generators
cp /ref/repo/golden_tests/enhanced_dashboard_generator.py .
cp /ref/repo/golden_tests/comparison_dashboard_generator.py .
```

**Problem: Dashboard shows no results**
- **Cause**: No test results JSON files found
- **Solution**: Run tests first to generate results:
```bash
python golden_test_runner.py --output results.json
python enhanced_dashboard_generator.py --project . --results results.json
```

**Problem: Comparison dashboard missing languages**
- **Cause**: Result files for other languages not provided
- **Solution**: Generate results for each language:
```bash
python golden_test_runner.py --output results_python.json
Rscript golden_test_runner.R --output results_r.json
python comparison_dashboard_generator.py --project . --python-results results_python.json --r-results results_r.json
```

### Integration Problems

**Problem: Sync workflow fails in CI/CD**
- **Check**: Environment variables are set correctly
- **Check**: Reference repository path is accessible
- **Solution**: Use absolute paths and verify permissions

**Problem: Tests pass locally but fail in CI**
- **Cause**: Different package versions or missing dependencies
- **Solution**: Pin dependency versions and ensure consistent environment

### Performance Issues

**Problem: Test execution is very slow**
- **Cause**: Large numerical computations or inefficient implementations
- **Solution**: Profile your code and optimize bottlenecks
- **Workaround**: Use `--test specific_test` to run individual tests

**Problem: Dashboard generation is slow**
- **Cause**: Many test results or large result files
- **Solution**: This is normal for comprehensive test suites

### Validation and Debugging

**Quick Health Check:**
```bash
# Verify all components work
python enhanced_sync_workflow.py --help                    # Should show help
python golden_test_runner.py --help                       # Should show help  
Rscript golden_test_runner.R --help                      # Should show help
ls specs/core/*.xml                                       # Should list XML files
python -c "import numpy, scipy; print('Dependencies OK')" # Should print OK
```

**Debug Failing Tests:**
```bash
# 1. Run single test with maximum verbosity
python golden_test_runner.py --test failing_test_name --verbose

# 2. Check XML structure
python -c "
import xml.etree.ElementTree as ET
doc = ET.parse('specs/core/failing_test.xml')
print('XML parsed successfully')
print('Test ID:', doc.find('.//{http://golden-tests.org/schema}id').text)
"

# 3. Manually execute implementation
python -c "
# Copy implementation code from XML and run manually
# to debug step-by-step
"
```

## Troubleshooting Schema Issues

### Problem: Test appears blank in dashboard

**Symptoms:**
- Test card shows title but no description or "..." 
- "View Details" shows empty sections
- Validation checks table is empty or shows "N/A"

**Common Causes & Solutions:**

1. **Wrong Namespace**
   ```xml
   <!-- WRONG -->
   <golden_test xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
   
   <!-- CORRECT -->
   <golden_test xmlns="http://golden-tests.org/schema">
   ```

2. **Wrong Element Names**
   ```xml
   <!-- WRONG -->
   <metadata>
     <test_id>my_test</test_id>
   
   <!-- CORRECT -->
   <metadata>
     <id>my_test</id>
   ```

3. **Wrong Validation Structure**
   ```xml
   <!-- WRONG -->
   <outputs>
     <output name="result">
   
   <!-- CORRECT -->
   <expected_outputs>
     <numeric_checks>
       <check>
   ```

4. **Missing Required Elements**
   - Every test MUST have `<metadata>`, `<semantic_description>`, `<expected_outputs>`, `<implementations>`, `<propagation_status>`

### Problem: Dashboard parsing errors

**Debugging Steps:**
1. Check the console output when running `generate_dashboard.py`
2. Look for XML parsing error messages
3. Validate XML structure against the template above
4. Compare with working test files (e.g., `hrf_spmg1_basic.xml`)

### Problem: Validation checks not displaying

**Check:**
- Uses `<expected_outputs><numeric_checks><check>` hierarchy
- Each `<check>` has `<type>`, `<location>`, and either `<expected>` or `<min>/<max>`
- Check types are valid: `exact_value`, `approximate`, `range`, `statistical`

## Lessons Learned and Best Practices

### 1. Write Idiomatic Code, Not Syntax Emulation
- **Principle**: Each language should use its natural patterns to achieve semantic equivalence
- **Why it matters**: Trying to emulate R syntax in Python (or vice versa) leads to unnatural, hard-to-maintain code
- **Best practice**:
  - Focus on producing the same numeric results, not mimicking syntax
  - Use language-appropriate data structures and patterns
  - Example: Use numpy arrays naturally in Python rather than trying to make them behave like R matrices

### 2. Handle Cross-Language Indexing Differences
- **Challenge**: R uses 1-based indexing, Python/Rust use 0-based
- **Solution patterns**:
  - Create wrapper classes for test compatibility when needed
  - Convert indices at the boundary between test framework and implementation
  - Document index conversions clearly
- **Example**: For R's `vol[1,1,1]` use Python's `vol[0,0,0]` internally, but provide a wrapper for tests

### 3. Make Attributes Callable for Test Framework
- **Challenge**: Test expressions like `dim(obj)` expect functions, but Python often uses attributes
- **Solution**: Create helper functions that wrap attribute access
- **Example**:
  ```python
  def dim(obj):
      return obj.dim if hasattr(obj, 'dim') else obj.shape
  ```

### 4. Handle Complex Test Expressions
- **Challenge**: R-style expressions like `sum(vec4d[,,,1])` don't translate directly
- **Solutions**:
  1. Compute values directly and store in variables the test can find
  2. Extend validators to handle language-specific patterns
  3. Use wrapper classes that understand special slicing syntax
- **Best practice**: Start simple - compute the expected values directly rather than over-engineering

### 5. Memory Layout Matters
- **Principle**: R uses column-major (Fortran) order, Python defaults to row-major (C) order
- **Why it matters**: Array reshaping and flattening produce different results
- **Best practice**: Use `order='F'` in numpy operations when matching R behavior
- **Example**: `np.arange(1, 28).reshape((3, 3, 3), order='F')`

### 6. Validator Limitations and Workarounds
- **Common validator issues**:
  - R-style array slicing syntax (e.g., `[,1]` or `[,,,1]`)
  - Function calls vs attribute access
  - Complex expressions in test locations
- **Workarounds**:
  - Store intermediate results in named variables
  - Extend validators to handle common patterns
  - Use wrapper classes for complex indexing

### 7. Test Development Workflow (Updated)
1. **Understand the R implementation**: Read and run the R code first
2. **Identify semantic goals**: What numeric results must match?
3. **Write idiomatic implementation**: Use natural patterns for your language
4. **Handle impedance mismatches**: Add wrappers/helpers for test compatibility
5. **Iterate on failures**: Each failure teaches something about differences
6. **Document solutions**: Comment why certain patterns were needed

### 8. Common Cross-Language Pitfalls
- **Don't assume function names match**: `concat` in R might be `concatenate` or a method in Python
- **Check import availability**: Not all functions may be exported (`from package import *` might miss some)
- **Verify method vs function**: R's `as.logical(roi)` might be Python's `roi.as_logical()`
- **Test sparse representations**: Different languages handle sparse data differently

### 9. Debugging Cross-Language Tests
- **When tests fail mysteriously**:
  1. Check if the function/method exists and is imported
  2. Verify the exact shape and type of data structures
  3. Print intermediate values to understand transformations
  4. Compare memory layouts (row vs column major)
  5. Check if wrapper classes are preserving necessary attributes

### 10. Architecture Patterns for Test Compatibility
- **Wrapper classes**: Bridge between language-specific implementations and test expectations
- **Helper functions**: Make attributes callable, handle special operations
- **Strategic computation**: Pre-compute complex expressions that validators struggle with
- **Clear separation**: Keep test compatibility code separate from core implementation

### Previous Lessons (Still Valid)

### 11. Always Execute Code Before Writing Tests
- **Principle**: Never assume API behavior - verify through execution
- **Why it matters**: Function signatures, return types, and data structures often differ from expectations
- **Best practice**: 
  - Run code interactively before writing test specifications
  - Verify actual outputs match your mental model
  - Document any surprising behaviors in test comments

### 12. Be Aware of Function Polymorphism
- **Principle**: Many functions have multiple signatures with different behaviors
- **Why it matters**: The same function name may process arguments differently based on type or count
- **Best practice**:
  - Test all relevant function signatures
  - Read documentation for overloaded methods
  - Example: A function might treat `func(vec)` vs `func(x, y, z)` completely differently

### 13. Handle Object Systems Appropriately
- **Principle**: Different languages use different object models (S3/S4/R6 in R, classes in Python, structs in Rust)
- **Why it matters**: Direct field access, type coercion, and method calls vary by system
- **Best practice**:
  - Use appropriate accessor methods rather than direct field access
  - Test type conversions explicitly
  - Don't assume automatic coercion will work

### 14. XML Encoding Requirements
- **Principle**: XML has reserved characters that must be escaped
- **Common escapes**:
  - `<` → `&lt;`
  - `>` → `&gt;`
  - `&` → `&amp;`
  - `"` → `&quot;`
  - `'` → `&apos;`
- **Best practice**: Always escape comparison operators and special characters in test expressions

### 15. Verify Data Structure Internals
- **Principle**: Don't assume field names or structure without verification
- **Why it matters**: Internal representations often differ from external documentation
- **Best practice**:
  - Inspect objects programmatically (e.g., `str()` in R, `dir()` in Python)
  - Check field names exactly as they appear
  - Verify nested structure assumptions

---

## Quick Reference

### Essential Commands
```bash
# Complete setup from scratch
mkdir golden_tests && cd golden_tests
cp /ref/repo/golden_tests/tools/*.py . 2>/dev/null || cp /ref/repo/golden_tests/*.py .
cp /ref/repo/golden_tests/tools/*.R . 2>/dev/null || cp /ref/repo/golden_tests/*.R . 
cp /ref/repo/golden_tests/GOLDEN_TESTS.md .
chmod +x *.py *.R

# Daily sync and test
python enhanced_sync_workflow.py --source /ref/repo/golden_tests --target . --language Python --verbose

# Manual test execution
python golden_test_runner.py --output results.json --verbose
Rscript golden_test_runner.R --output results.json --verbose

# Generate dashboards
python enhanced_dashboard_generator.py --project . --results results.json
python comparison_dashboard_generator.py --project . --python-results r1.json --r-results r2.json
```

### File Checklist for Port Repository
```
✅ ~/bin/sync_golden_tests.py                                  # Basic sync tool (system-wide)
✅ golden_tests/GOLDEN_TESTS.md                                # This documentation
✅ golden_tests/golden_test_runner.py                          # Python test execution (copied from tools/)
✅ golden_tests/golden_test_runner.R                           # R test execution (copied from tools/)
✅ golden_tests/enhanced_sync_workflow.py                      # Complete workflow (copied from tools/)
✅ golden_tests/enhanced_dashboard_generator.py                # Single language dashboard (copied from tools/)
✅ golden_tests/comparison_dashboard_generator.py              # Multi-language comparison (copied from tools/)
✅ golden_tests/golden_tests_config.yaml                       # Configuration (copied from tools/)
✅ golden_tests/specs/                                         # Test specifications (synced)
```

### Recommended Reference Repository Structure
```
reference-repo/
├── golden_tests/
│   ├── GOLDEN_TESTS.md                     # Complete documentation
│   ├── tools/                              # Helper scripts (organized)
│   │   ├── golden_test_runner.py           # Python test execution
│   │   ├── improved_test_runner.py         # Enhanced Python runner
│   │   ├── golden_test_runner.R            # R test execution
│   │   ├── enhanced_dashboard_generator.py # Single-language dashboards
│   │   ├── comparison_dashboard_generator.py # Multi-language comparison
│   │   ├── enhanced_sync_workflow.py       # Complete sync workflow
│   │   └── golden_tests_config.yaml        # Configuration template
│   ├── specs/                              # Test specifications
│   │   ├── core/                           # Core functionality tests
│   │   ├── edge_cases/                     # Edge case tests
│   │   └── integration/                    # Integration tests
│   └── validators/                         # Language-specific validators
└── bin/
    └── sync_golden_tests.py                # Basic sync tool
```

### Dependencies Checklist
```bash
# Python
pip install numpy scipy xml2

# R  
R -e "install.packages(c('xml2', 'jsonlite'))"

# Verify
python -c "import numpy, scipy; print('Python OK')"
Rscript -e "library(xml2); library(jsonlite); cat('R OK\n')"
```

### Integration Patterns
```bash
# Pattern 1: Single repo, multiple languages
cd python/golden_tests && python enhanced_sync_workflow.py --source ~/ref --target . --language Python
cd r/golden_tests && python enhanced_sync_workflow.py --source ~/ref --target . --language R

# Pattern 2: Separate repos per language  
cd my-python-port/golden_tests && python enhanced_sync_workflow.py --source ~/ref --target . --language Python
cd my-r-port/golden_tests && python enhanced_sync_workflow.py --source ~/ref --target . --language R

# Pattern 3: CI/CD integration
# Add to .github/workflows/golden-tests.yml
python enhanced_sync_workflow.py --source ${REF_REPO}/golden_tests --target ./golden_tests --language Python
```

### Common Issues Quick Fix
```bash
# "sync tool not found" 
cp /ref/repo/bin/sync_golden_tests.py ~/bin/ && chmod +x ~/bin/sync_golden_tests.py

# "permission denied"
chmod +x *.py *.R

# "test runner not found"  
cp /ref/repo/golden_tests/tools/golden_test_runner.* . 2>/dev/null || cp /ref/repo/golden_tests/golden_test_runner.* .

# "dependencies missing"
pip install numpy scipy && R -e "install.packages(c('xml2', 'jsonlite'))"

# "no test results"
python golden_test_runner.py --verbose  # Check for errors

# "dashboard empty"
python golden_test_runner.py --output results.json  # Generate results first
```

This documentation contains everything needed to set up and use golden tests in any port repository. For additional details on specific components, refer to the relevant sections above.

This consolidated reference provides everything needed to understand and work with golden tests efficiently.