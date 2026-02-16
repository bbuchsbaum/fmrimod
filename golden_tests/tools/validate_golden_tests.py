#!/usr/bin/env python3
"""
Golden Tests Schema Validator

Validates that all golden test XML files conform to the canonical schema
defined in ~/code/translation/golden/GOLDEN_TESTS.md

Usage:
    python validate_golden_tests.py [--project PATH] [--verbose]
    python validate_golden_tests.py --project /path/to/golden_tests --verbose
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import argparse
from datetime import datetime


@dataclass
class ValidationError:
    """Represents a schema validation error"""
    file_path: str
    error_type: str
    element: str
    message: str
    line_number: Optional[int] = None


@dataclass
class ValidationResult:
    """Results of schema validation"""
    file_path: str
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]


class GoldenTestSchemaValidator:
    """Validates golden test XML files against canonical schema"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.canonical_namespace = "http://golden-tests.org/schema"
        self.namespace = {'gt': self.canonical_namespace}
        self.required_elements = {
            'metadata': ['id', 'version', 'description', 'tags'],
            'semantic_description': ['purpose', 'algorithm'],
            'expected_outputs': ['numeric_checks'],
            'implementations': ['R'],
            'propagation_status': []
        }
        self.valid_check_types = {'exact_value', 'approximate', 'range', 'statistical'}
        self.valid_implementation_status = {'completed', 'pending', 'in_progress'}
    
    def validate_file(self, xml_file: Path) -> ValidationResult:
        """Validate a single XML file"""
        errors = []
        warnings = []
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Validate root element and namespace
            errors.extend(self._validate_root_element(root, str(xml_file)))
            
            # Validate required sections
            errors.extend(self._validate_metadata(root, str(xml_file)))
            errors.extend(self._validate_semantic_description(root, str(xml_file)))
            errors.extend(self._validate_expected_outputs(root, str(xml_file)))
            errors.extend(self._validate_implementations(root, str(xml_file)))
            errors.extend(self._validate_propagation_status(root, str(xml_file)))
            
            # Check for deprecated elements
            warnings.extend(self._check_deprecated_elements(root, str(xml_file)))
            
        except ET.ParseError as e:
            errors.append(ValidationError(
                file_path=str(xml_file),
                error_type="parse_error",
                element="root",
                message=f"XML parsing failed: {e}",
                line_number=getattr(e, 'lineno', None)
            ))
        except Exception as e:
            errors.append(ValidationError(
                file_path=str(xml_file),
                error_type="validation_error",
                element="root",
                message=f"Validation failed: {e}"
            ))
        
        return ValidationResult(
            file_path=str(xml_file),
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_root_element(self, root: ET.Element, file_path: str) -> List[ValidationError]:
        """Validate root element and namespace"""
        errors = []
        
        # Check root element name - handle both namespaced and non-namespaced
        tag_name = root.tag
        if tag_name.startswith(f'{{{self.canonical_namespace}}}'):
            tag_name = tag_name[len(f'{{{self.canonical_namespace}}}'):]
        
        if tag_name != 'golden_test':
            errors.append(ValidationError(
                file_path=file_path,
                error_type="root_element",
                element="root",
                message=f"Root element should be 'golden_test', found '{tag_name}'"
            ))
        
        # Check namespace - look at the tag itself for namespace information
        namespace = root.get('xmlns')
        if namespace != self.canonical_namespace:
            # Check if the tag itself has the namespace
            if not root.tag.startswith(f'{{{self.canonical_namespace}}}'):
                errors.append(ValidationError(
                    file_path=file_path,
                    error_type="namespace",
                    element="root",
                    message=f"Root element must use canonical namespace '{self.canonical_namespace}', found '{namespace}'"
                ))
        
        # Check for deprecated namespace attributes
        if 'xsi:noNamespaceSchemaLocation' in root.attrib:
            errors.append(ValidationError(
                file_path=file_path,
                error_type="deprecated_namespace",
                element="root",
                message="xsi:noNamespaceSchemaLocation is deprecated, use canonical namespace instead"
            ))
        
        return errors
    
    def _validate_metadata(self, root: ET.Element, file_path: str) -> List[ValidationError]:
        """Validate metadata section"""
        errors = []
        
        metadata = root.find('gt:metadata', self.namespace)
        if metadata is None:
            errors.append(ValidationError(
                file_path=file_path,
                error_type="missing_section",
                element="metadata",
                message="Missing required <metadata> section"
            ))
            return errors
        
        # Check required elements
        required_elements = ['id', 'version', 'description', 'tags']
        for elem_name in required_elements:
            elem = metadata.find(f'gt:{elem_name}', self.namespace)
            if elem is None:
                errors.append(ValidationError(
                    file_path=file_path,
                    error_type="missing_element",
                    element=f"metadata/{elem_name}",
                    message=f"Missing required element <{elem_name}> in metadata"
                ))
            elif elem_name != 'tags' and (elem.text is None or elem.text.strip() == ''):
                errors.append(ValidationError(
                    file_path=file_path,
                    error_type="empty_element",
                    element=f"metadata/{elem_name}",
                    message=f"Element <{elem_name}> cannot be empty"
                ))
        
        # Check for deprecated elements
        if metadata.find('gt:test_id', self.namespace) is not None:
            errors.append(ValidationError(
                file_path=file_path,
                error_type="deprecated_element",
                element="metadata/test_id",
                message="<test_id> is deprecated, use <id> instead"
            ))
        
        # Validate tags structure
        tags = metadata.find('gt:tags', self.namespace)
        if tags is not None:
            tag_elements = tags.findall('gt:tag', self.namespace)
            if not tag_elements:
                errors.append(ValidationError(
                    file_path=file_path,
                    error_type="empty_element",
                    element="metadata/tags",
                    message="<tags> element should contain at least one <tag>"
                ))
            
            for tag in tag_elements:
                tag_name = tag.tag
                if tag_name.startswith(f'{{{self.canonical_namespace}}}'):
                    tag_name = tag_name[len(f'{{{self.canonical_namespace}}}'):]
                if tag_name != 'tag':
                    errors.append(ValidationError(
                        file_path=file_path,
                        error_type="invalid_element",
                        element=f"metadata/tags/{tag_name}",
                        message=f"Invalid element <{tag_name}> in tags, only <tag> allowed"
                    ))
        
        return errors
    
    def _validate_semantic_description(self, root: ET.Element, file_path: str) -> List[ValidationError]:
        """Validate semantic description section"""
        errors = []
        
        semantic_desc = root.find('gt:semantic_description', self.namespace)
        if semantic_desc is None:
            errors.append(ValidationError(
                file_path=file_path,
                error_type="missing_section",
                element="semantic_description",
                message="Missing required <semantic_description> section"
            ))
            return errors
        
        # Check required elements
        required_elements = ['purpose', 'algorithm']
        for elem_name in required_elements:
            elem = semantic_desc.find(f'gt:{elem_name}', self.namespace)
            if elem is None:
                errors.append(ValidationError(
                    file_path=file_path,
                    error_type="missing_element",
                    element=f"semantic_description/{elem_name}",
                    message=f"Missing required element <{elem_name}> in semantic_description"
                ))
            elif elem.text is None or elem.text.strip() == '':
                errors.append(ValidationError(
                    file_path=file_path,
                    error_type="empty_element",
                    element=f"semantic_description/{elem_name}",
                    message=f"Element <{elem_name}> cannot be empty"
                ))
        
        return errors
    
    def _validate_expected_outputs(self, root: ET.Element, file_path: str) -> List[ValidationError]:
        """Validate expected outputs section"""
        errors = []
        
        expected_outputs = root.find('gt:expected_outputs', self.namespace)
        if expected_outputs is None:
            errors.append(ValidationError(
                file_path=file_path,
                error_type="missing_section",
                element="expected_outputs",
                message="Missing required <expected_outputs> section"
            ))
            return errors
        
        # Check for numeric_checks
        numeric_checks = expected_outputs.find('gt:numeric_checks', self.namespace)
        if numeric_checks is None:
            errors.append(ValidationError(
                file_path=file_path,
                error_type="missing_element",
                element="expected_outputs/numeric_checks",
                message="Missing required <numeric_checks> element"
            ))
            return errors
        
        # Validate individual checks
        checks = numeric_checks.findall('gt:check', self.namespace)
        if not checks:
            errors.append(ValidationError(
                file_path=file_path,
                error_type="empty_element",
                element="expected_outputs/numeric_checks",
                message="<numeric_checks> must contain at least one <check>"
            ))
        
        for i, check in enumerate(checks):
            errors.extend(self._validate_check(check, i, file_path))
        
        return errors
    
    def _validate_check(self, check: ET.Element, index: int, file_path: str) -> List[ValidationError]:
        """Validate a single check element"""
        errors = []
        
        # Check type
        check_type = check.find('gt:type', self.namespace)
        if check_type is None:
            errors.append(ValidationError(
                file_path=file_path,
                error_type="missing_element",
                element=f"check[{index}]/type",
                message=f"Missing required <type> element in check {index}"
            ))
        elif check_type.text not in self.valid_check_types:
            errors.append(ValidationError(
                file_path=file_path,
                error_type="invalid_value",
                element=f"check[{index}]/type",
                message=f"Invalid check type '{check_type.text}', must be one of {self.valid_check_types}"
            ))
        
        # Check location
        location = check.find('gt:location', self.namespace)
        if location is None:
            errors.append(ValidationError(
                file_path=file_path,
                error_type="missing_element",
                element=f"check[{index}]/location",
                message=f"Missing required <location> element in check {index}"
            ))
        elif location.text is None or location.text.strip() == '':
            errors.append(ValidationError(
                file_path=file_path,
                error_type="empty_element",
                element=f"check[{index}]/location",
                message=f"<location> element cannot be empty in check {index}"
            ))
        
        # Validate based on check type
        if check_type is not None and check_type.text == 'range':
            # Range checks need min/max
            if check.find('gt:min', self.namespace) is None:
                errors.append(ValidationError(
                    file_path=file_path,
                    error_type="missing_element",
                    element=f"check[{index}]/min",
                    message=f"Range check {index} missing required <min> element"
                ))
            if check.find('gt:max', self.namespace) is None:
                errors.append(ValidationError(
                    file_path=file_path,
                    error_type="missing_element",
                    element=f"check[{index}]/max",
                    message=f"Range check {index} missing required <max> element"
                ))
        else:
            # Other checks need expected value
            if check.find('gt:expected', self.namespace) is None:
                errors.append(ValidationError(
                    file_path=file_path,
                    error_type="missing_element",
                    element=f"check[{index}]/expected",
                    message=f"Check {index} missing required <expected> element"
                ))
        
        return errors
    
    def _validate_implementations(self, root: ET.Element, file_path: str) -> List[ValidationError]:
        """Validate implementations section"""
        errors = []
        
        implementations = root.find('gt:implementations', self.namespace)
        if implementations is None:
            errors.append(ValidationError(
                file_path=file_path,
                error_type="missing_section",
                element="implementations",
                message="Missing required <implementations> section"
            ))
            return errors
        
        # Check for R implementation (required)
        r_impl = implementations.find('gt:R', self.namespace)
        if r_impl is None:
            errors.append(ValidationError(
                file_path=file_path,
                error_type="missing_element",
                element="implementations/R",
                message="Missing required <R> implementation"
            ))
        
        return errors
    
    def _validate_propagation_status(self, root: ET.Element, file_path: str) -> List[ValidationError]:
        """Validate propagation status section"""
        errors = []
        
        propagation_status = root.find('gt:propagation_status', self.namespace)
        if propagation_status is None:
            errors.append(ValidationError(
                file_path=file_path,
                error_type="missing_section",
                element="propagation_status",
                message="Missing required <propagation_status> section"
            ))
            return errors
        
        # Validate implementation status entries
        implementations = propagation_status.findall('gt:implementation', self.namespace)
        for impl in implementations:
            lang = impl.get('lang')
            status = impl.get('status')
            
            if lang is None:
                errors.append(ValidationError(
                    file_path=file_path,
                    error_type="missing_attribute",
                    element="propagation_status/implementation",
                    message="Missing required 'lang' attribute in implementation status"
                ))
            
            if status is None:
                errors.append(ValidationError(
                    file_path=file_path,
                    error_type="missing_attribute",
                    element="propagation_status/implementation",
                    message="Missing required 'status' attribute in implementation status"
                ))
            elif status not in self.valid_implementation_status:
                errors.append(ValidationError(
                    file_path=file_path,
                    error_type="invalid_value",
                    element="propagation_status/implementation",
                    message=f"Invalid status '{status}', must be one of {self.valid_implementation_status}"
                ))
        
        return errors
    
    def _check_deprecated_elements(self, root: ET.Element, file_path: str) -> List[ValidationError]:
        """Check for deprecated elements and structures"""
        warnings = []
        
        # Check for old test_case structure (both namespaced and non-namespaced)
        test_cases = root.findall('gt:test_case', self.namespace) + root.findall('test_case')
        if test_cases:
            warnings.append(ValidationError(
                file_path=file_path,
                error_type="deprecated_structure",
                element="test_case",
                message="<test_case> structure is deprecated, use <expected_outputs><numeric_checks> instead"
            ))
        
        # Check for old output structure (both namespaced and non-namespaced)
        outputs = root.findall('gt:outputs', self.namespace) + root.findall('outputs')
        if outputs:
            warnings.append(ValidationError(
                file_path=file_path,
                error_type="deprecated_structure",
                element="outputs",
                message="<outputs> structure is deprecated, use <expected_outputs><numeric_checks> instead"
            ))
        
        return warnings
    
    def validate_project(self, project_path: Path) -> Tuple[List[ValidationResult], Dict[str, int]]:
        """Validate all XML files in a project"""
        specs_dir = project_path / "specs"
        if not specs_dir.exists():
            raise FileNotFoundError(f"No specs directory found in {project_path}")
        
        results = []
        xml_files = list(specs_dir.rglob("*.xml"))
        
        if self.verbose:
            print(f"Found {len(xml_files)} XML files to validate")
        
        for xml_file in xml_files:
            if self.verbose:
                print(f"Validating {xml_file.relative_to(project_path)}...")
            
            result = self.validate_file(xml_file)
            results.append(result)
        
        # Calculate summary statistics
        stats = {
            'total_files': len(results),
            'valid_files': sum(1 for r in results if r.is_valid),
            'invalid_files': sum(1 for r in results if not r.is_valid),
            'total_errors': sum(len(r.errors) for r in results),
            'total_warnings': sum(len(r.warnings) for r in results)
        }
        
        return results, stats
    
    def print_results(self, results: List[ValidationResult], stats: Dict[str, int]):
        """Print validation results"""
        print(f"\nGolden Tests Schema Validation Results")
        print("=" * 50)
        print(f"Total files: {stats['total_files']}")
        print(f"Valid files: {stats['valid_files']}")
        print(f"Invalid files: {stats['invalid_files']}")
        print(f"Total errors: {stats['total_errors']}")
        print(f"Total warnings: {stats['total_warnings']}")
        
        # Print file-by-file results
        for result in results:
            if result.is_valid:
                status = "✅ VALID"
            else:
                status = "❌ INVALID"
            
            print(f"\n{status}: {Path(result.file_path).name}")
            
            # Print errors
            for error in result.errors:
                print(f"  ERROR [{error.error_type}] {error.element}: {error.message}")
            
            # Print warnings
            for warning in result.warnings:
                print(f"  WARNING [{warning.error_type}] {warning.element}: {warning.message}")
        
        # Return status code
        return 0 if stats['invalid_files'] == 0 else 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Validate Golden Tests XML Schema')
    parser.add_argument('--project', '-p', 
                       default='.', 
                       help='Path to golden tests project directory')
    parser.add_argument('--verbose', '-v', 
                       action='store_true', 
                       help='Verbose output')
    
    args = parser.parse_args()
    
    project_path = Path(args.project)
    if not project_path.exists():
        print(f"Error: Project directory {project_path} does not exist")
        return 1
    
    validator = GoldenTestSchemaValidator(verbose=args.verbose)
    
    try:
        results, stats = validator.validate_project(project_path)
        exit_code = validator.print_results(results, stats)
        return exit_code
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())