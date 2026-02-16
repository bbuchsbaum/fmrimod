# Schema Fragmentation Prevention Guide

## Overview

This document outlines the measures implemented to prevent schema fragmentation issues in the Golden Tests system and ensure consistent test discovery and execution.

## The Problem We Solved

**Original Issue:** Test runners could only discover 1 out of 11 tests due to schema fragmentation:
- Some XML files used `xmlns="http://golden-tests.org/schema"` (canonical)
- Others used `xsi:noNamespaceSchemaLocation` (deprecated)
- Test runners had fallback parsing logic that masked schema violations
- Silent parsing failures caused tests to be skipped

**Root Cause:** Lack of schema enforcement allowed inconsistent XML formats to proliferate.

## Prevention Measures Implemented

### 1. **Automated Schema Validation**

**Schema Validator Tool:** `tools/validate_golden_tests.py`
- Validates all XML files against canonical schema
- Enforces namespace compliance: `xmlns="http://golden-tests.org/schema"`
- Detects deprecated elements and structures
- Provides detailed error messages

**Usage:**
```bash
# Validate all files
python tools/validate_golden_tests.py --project .

# Validate with verbose output
python tools/validate_golden_tests.py --project . --verbose
```

### 2. **Automated Enforcement (Git Integration)**

**Pre-commit Hook:** `.pre-commit-config.yaml`
- Validates schema before every commit
- Prevents invalid XML from entering repository
- Fails commit if schema violations found

**CI/CD Integration:** `.github/workflows/validate-schema.yml`
- Runs on every push and pull request
- Blocks merging if schema violations detected
- Provides immediate feedback on schema issues

**Installation:**
```bash
# Enable pre-commit hooks
pre-commit install

# Test pre-commit validation
pre-commit run --all-files
```

### 3. **Simplified Test Runner Logic**

**Removed Dual-Namespace Support:**
- Enhanced dashboard generator: Only uses canonical namespace
- All test runners: Removed fallback parsing logic
- Simplified XML parsing: Single namespace path only

**Result:** Test runners now reliably discover all 11 tests because all XML files follow identical schema format.

### 4. **Developer Tools and Workflows**

**Template Generator:** `scripts/new_test_template.py`
- Creates new tests with canonical schema format
- Prevents manual creation of invalid XML files
- Includes TODOs for all required sections

**Makefile Commands:**
```bash
make validate        # Validate schema compliance
make test-all        # Run all tests with validation
make dashboard       # Generate dashboard with validation
make new-test ID=... DESC="..." # Create new test template
```

**Development Workflow:**
1. Use template generator for new tests
2. Run `make validate` before committing
3. Use `make test-all` to verify test discovery
4. Pre-commit hooks prevent invalid commits

### 5. **Enhanced Documentation**

**Updated Documentation:** `~/code/translation/golden/GOLDEN_TESTS.md`
- Clear canonical template with examples
- Common schema errors and how to avoid them
- Validation tools and enforcement strategies
- Error prevention checklist

**Key Sections Added:**
- Schema Validation and Enforcement
- Prevention Strategies
- Error Prevention Checklist
- Development Workflow

## Current Status

✅ **All 11 tests use canonical schema format**
✅ **Schema validation tool operational**
✅ **Pre-commit hooks configured**
✅ **CI/CD integration ready**
✅ **Test runners simplified (no dual-namespace)**
✅ **Template generator available**
✅ **Documentation enhanced**

## Validation Results

```bash
$ make validate
Golden Tests Schema Validation Results
==================================================
Total files: 11
Valid files: 11
Invalid files: 0
Total errors: 0
Total warnings: 0
```

**Test Discovery Results:**
- R test runner: 11/11 tests discovered and run
- Python test runner: 11/11 tests discovered and run
- Dashboard: Shows all 11 tests with execution results

## How to Use These Tools

### For New Test Creation
```bash
# Create new test with canonical schema
make new-test ID=my_new_test DESC="Test description"

# Edit the generated file (replace TODO items)
# Validate before committing
make validate
```

### For Daily Development
```bash
# Before committing changes
make validate

# Run all tests
make test-all

# Generate dashboard
make dashboard
```

### For Code Reviews
1. Verify XML files use canonical namespace
2. Check that `make validate` passes
3. Ensure test discovery works for new tests

## Future Maintenance

**Regular Tasks:**
- Monitor CI/CD for schema validation failures
- Update template generator if schema evolves
- Review and update documentation as needed

**When Adding New Features:**
- Update schema validator for new requirements
- Maintain canonical template in documentation
- Test with both R and Python runners

**Emergency Response:**
If schema fragmentation occurs again:
1. Use `make validate` to identify issues
2. Fix XML files to use canonical namespace
3. Update test runners if needed
4. Verify all tests are discovered

## Success Metrics

- **Test Discovery Rate:** 11/11 tests (100%) consistently discovered
- **Schema Compliance:** 11/11 files (100%) pass validation
- **Prevention Effectiveness:** 0 schema violations since implementation
- **Developer Experience:** Simplified workflow with automated validation