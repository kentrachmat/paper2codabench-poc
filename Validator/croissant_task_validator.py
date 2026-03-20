#!/usr/bin/env python3
"""
Validate Croissant Task (cr:TaskProblem) JSON-LD files.

Checks LLM-generated Croissant Tasks against:
1. Formal Croissant Task Vocabulary spec (JSON-LD structure)
2. Component completeness (cr:input, cr:output, cr:evaluation, etc.)
3. Type correctness (dataType values, numeric types)
4. FILL IN THE BLANK placeholder detection
5. Pipeline compatibility (descriptions for bundle generation)

Usage:
    python Validator/croissant_task_validator.py croissant_tasks/paper1.croissant_task.json
    python Validator/croissant_task_validator.py croissant_tasks/
    python Validator/croissant_task_validator.py croissant_tasks/ --json
"""
import json
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


class CroissantTaskValidationError(Exception):
    """Base exception for validation errors."""
    pass


@dataclass
class ValidationResult:
    """Result of validating a Croissant Task."""
    file: str
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fitb_count: int = 0
    fitb_paths: List[str] = field(default_factory=list)
    completeness: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "file": self.file,
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "fitb_count": self.fitb_count,
            "fitb_paths": self.fitb_paths,
            "completeness": self.completeness,
        }


class CroissantTaskValidator:
    """Validates a Croissant Task (cr:TaskProblem) JSON-LD structure."""

    # Valid dataType values from schema.org
    VALID_DATA_TYPES = {
        "sc:Text",
        "sc:Integer",
        "sc:Float",
        "sc:Boolean",
        "sc:URL",
        "sc:ImageObject",
        # Also accept schema: variants
        "schema:Text",
        "schema:Integer",
        "schema:Float",
        "schema:Boolean",
        "schema:URL",
        "schema:ImageObject",
    }

    # Pattern to detect FILL IN THE BLANK placeholders
    FITB_PATTERN = re.compile(r'\[FILL IN THE BLANK[^\]]*\]', re.IGNORECASE)

    def __init__(self, data: Dict[str, Any]):
        """Initialize validator with parsed JSON data."""
        self.data = data
        self.result = ValidationResult(file="<in-memory>")

    def validate(self) -> ValidationResult:
        """Run all validation checks."""
        # Level 1: JSON-LD / Structural
        self._validate_structure()

        # Level 2: Component Completeness
        self._validate_components()

        # Level 3: Type and Value Correctness
        self._validate_types()

        # Level 4: FILL IN THE BLANK Detection
        self._detect_fitb()

        # Level 5: Pipeline Compatibility
        self._validate_pipeline_compatibility()

        # Determine overall validity
        self.result.valid = len(self.result.errors) == 0

        return self.result

    def _validate_structure(self):
        """Level 1: Validate JSON-LD structure and required fields."""
        # Check @context
        context = self.data.get("@context")
        if not context:
            self.result.errors.append("Missing required field: @context")
        elif not isinstance(context, dict):
            self.result.errors.append("@context must be an object")
        else:
            # Check for required namespaces
            if "cr" not in context:
                self.result.errors.append("@context missing 'cr' namespace")
            if "sc" not in context and "schema" not in context:
                self.result.warnings.append("@context missing 'sc' or 'schema' namespace")

        # Check @type
        at_type = self.data.get("@type")
        if not at_type:
            self.result.errors.append("Missing required field: @type")
        elif at_type != "cr:TaskProblem":
            self.result.errors.append(f"@type must be 'cr:TaskProblem', got '{at_type}'")

        # Check conformsTo
        conforms_to = self.data.get("conformsTo")
        if not conforms_to:
            self.result.errors.append("Missing required field: conformsTo")
        elif not isinstance(conforms_to, str) or "croissant" not in conforms_to.lower():
            self.result.warnings.append(f"conformsTo should reference Croissant spec, got '{conforms_to}'")

        # Check @id
        at_id = self.data.get("@id")
        if not at_id:
            self.result.errors.append("Missing required field: @id")
        elif not isinstance(at_id, str) or not at_id.strip():
            self.result.errors.append("@id must be a non-empty string")

        # Check name
        name = self.data.get("name")
        if not name:
            self.result.errors.append("Missing required field: name")
        elif not isinstance(name, str) or not name.strip():
            self.result.errors.append("name must be a non-empty string")

        # Check description
        description = self.data.get("description")
        if not description:
            self.result.errors.append("Missing required field: description")
        elif not isinstance(description, str) or not description.strip():
            self.result.errors.append("description must be a non-empty string")

    def _validate_components(self):
        """Level 2: Validate component completeness."""
        # Track which components are present
        self.result.completeness = {
            "cr:input": False,
            "cr:output": False,
            "cr:implementation": False,
            "cr:execution": False,
            "cr:evaluation": False,
        }

        # Validate cr:input
        cr_input = self.data.get("cr:input")
        if cr_input is None:
            self.result.errors.append("Missing required component: cr:input")
        else:
            self.result.completeness["cr:input"] = True
            if not isinstance(cr_input, list):
                self.result.errors.append("cr:input must be a list")
            elif len(cr_input) == 0:
                self.result.errors.append("cr:input must contain at least one entry")
            else:
                for i, input_item in enumerate(cr_input):
                    if not isinstance(input_item, dict):
                        self.result.errors.append(f"cr:input[{i}] must be an object")
                    else:
                        if not input_item.get("name"):
                            self.result.warnings.append(f"cr:input[{i}] missing 'name'")
                        if not input_item.get("description"):
                            self.result.warnings.append(f"cr:input[{i}] missing 'description'")

        # Validate cr:output
        cr_output = self.data.get("cr:output")
        if cr_output is None:
            self.result.errors.append("Missing required component: cr:output")
        else:
            self.result.completeness["cr:output"] = True
            if not isinstance(cr_output, dict):
                self.result.errors.append("cr:output must be an object")
            else:
                schema = cr_output.get("cr:schema")
                if not schema:
                    self.result.errors.append("cr:output missing 'cr:schema'")
                elif not isinstance(schema, dict):
                    self.result.errors.append("cr:output.cr:schema must be an object")
                else:
                    # Check RecordSet type
                    schema_type = schema.get("@type")
                    if schema_type and schema_type != "cr:RecordSet":
                        self.result.warnings.append(f"cr:output.cr:schema @type should be 'cr:RecordSet', got '{schema_type}'")

                    # Check fields
                    fields = schema.get("field")
                    if not fields:
                        self.result.errors.append("cr:output.cr:schema must have at least one field")
                    elif not isinstance(fields, list):
                        self.result.errors.append("cr:output.cr:schema.field must be a list")
                    else:
                        for i, field_item in enumerate(fields):
                            if not isinstance(field_item, dict):
                                self.result.errors.append(f"cr:output.cr:schema.field[{i}] must be an object")
                            else:
                                if not field_item.get("name"):
                                    self.result.errors.append(f"cr:output.cr:schema.field[{i}] missing 'name'")
                                if not field_item.get("dataType"):
                                    self.result.errors.append(f"cr:output.cr:schema.field[{i}] missing 'dataType'")

        # Validate cr:evaluation
        cr_evaluation = self.data.get("cr:evaluation")
        if cr_evaluation is None:
            self.result.errors.append("Missing required component: cr:evaluation")
        else:
            self.result.completeness["cr:evaluation"] = True
            if not isinstance(cr_evaluation, dict):
                self.result.errors.append("cr:evaluation must be an object")
            else:
                primary_metric = cr_evaluation.get("primaryMetric")
                if not primary_metric:
                    self.result.errors.append("cr:evaluation missing 'primaryMetric'")
                elif not isinstance(primary_metric, str) or not primary_metric.strip():
                    self.result.errors.append("cr:evaluation.primaryMetric must be a non-empty string")

                metrics = cr_evaluation.get("metrics")
                if not metrics:
                    self.result.errors.append("cr:evaluation missing 'metrics'")
                elif not isinstance(metrics, list) or len(metrics) == 0:
                    self.result.errors.append("cr:evaluation.metrics must be a non-empty list")

                higher_is_better = cr_evaluation.get("higherIsBetter")
                if higher_is_better is None:
                    self.result.warnings.append("cr:evaluation missing 'higherIsBetter'")
                elif not isinstance(higher_is_better, bool):
                    self.result.errors.append("cr:evaluation.higherIsBetter must be a boolean")

        # Validate cr:implementation (optional but recommended)
        cr_implementation = self.data.get("cr:implementation")
        if cr_implementation is None:
            self.result.warnings.append("Missing recommended component: cr:implementation")
        else:
            self.result.completeness["cr:implementation"] = True
            if isinstance(cr_implementation, dict):
                if not cr_implementation.get("entryPoint"):
                    self.result.warnings.append("cr:implementation missing 'entryPoint'")

        # Validate cr:execution (optional but recommended)
        cr_execution = self.data.get("cr:execution")
        if cr_execution is None:
            self.result.warnings.append("Missing recommended component: cr:execution")
        else:
            self.result.completeness["cr:execution"] = True
            if isinstance(cr_execution, dict):
                # These will be checked in _validate_types, but note if missing
                if "runtimeLimitSec" not in cr_execution:
                    self.result.warnings.append("cr:execution missing 'runtimeLimitSec'")
                if "memoryLimitMb" not in cr_execution:
                    self.result.warnings.append("cr:execution missing 'memoryLimitMb'")

    def _validate_types(self):
        """Level 3: Validate type correctness."""
        # Validate dataType values in output schema fields
        cr_output = self.data.get("cr:output")
        if cr_output and isinstance(cr_output, dict):
            schema = cr_output.get("cr:schema")
            if schema and isinstance(schema, dict):
                fields = schema.get("field")
                if fields and isinstance(fields, list):
                    for i, field_item in enumerate(fields):
                        if isinstance(field_item, dict):
                            data_type = field_item.get("dataType")
                            if data_type and data_type not in self.VALID_DATA_TYPES:
                                self.result.warnings.append(
                                    f"cr:output.cr:schema.field[{i}].dataType '{data_type}' not in standard set. "
                                    f"Expected one of: {', '.join(sorted(self.VALID_DATA_TYPES))}"
                                )

        # Validate cr:execution numeric fields
        cr_execution = self.data.get("cr:execution")
        if cr_execution and isinstance(cr_execution, dict):
            runtime_limit = cr_execution.get("runtimeLimitSec")
            if runtime_limit is not None:
                if isinstance(runtime_limit, str):
                    # Check if it's a FITB placeholder
                    if self.FITB_PATTERN.search(runtime_limit):
                        # This will be caught by FITB detection, but also flag as type error
                        self.result.errors.append("cr:execution.runtimeLimitSec must be numeric, not a string")
                    else:
                        self.result.errors.append("cr:execution.runtimeLimitSec must be numeric, not a string")
                elif not isinstance(runtime_limit, (int, float)):
                    self.result.errors.append("cr:execution.runtimeLimitSec must be numeric")

            memory_limit = cr_execution.get("memoryLimitMb")
            if memory_limit is not None:
                if isinstance(memory_limit, str):
                    # Check if it's a FITB placeholder
                    if self.FITB_PATTERN.search(memory_limit):
                        # This will be caught by FITB detection, but also flag as type error
                        self.result.errors.append("cr:execution.memoryLimitMb must be numeric, not a string")
                    else:
                        self.result.errors.append("cr:execution.memoryLimitMb must be numeric, not a string")
                elif not isinstance(memory_limit, (int, float)):
                    self.result.errors.append("cr:execution.memoryLimitMb must be numeric")

        # Validate primaryMetric appears in metrics list
        cr_evaluation = self.data.get("cr:evaluation")
        if cr_evaluation and isinstance(cr_evaluation, dict):
            primary_metric = cr_evaluation.get("primaryMetric")
            metrics = cr_evaluation.get("metrics")
            if primary_metric and metrics and isinstance(metrics, list):
                if primary_metric not in metrics:
                    self.result.warnings.append(
                        f"cr:evaluation.primaryMetric '{primary_metric}' not found in metrics list. "
                        f"This may cause issues in bundle generation."
                    )

    def _detect_fitb(self):
        """Level 4: Detect FILL IN THE BLANK placeholders."""
        fitb_paths = []
        self._scan_for_fitb(self.data, "", fitb_paths)
        self.result.fitb_count = len(fitb_paths)
        self.result.fitb_paths = fitb_paths

        if fitb_paths:
            self.result.warnings.append(f"Found {len(fitb_paths)} FILL IN THE BLANK placeholder(s)")

        # Cross-reference with fill_in_the_blank array
        fill_in_blank_array = self.data.get("fill_in_the_blank", [])
        if fill_in_blank_array:
            if not isinstance(fill_in_blank_array, list):
                self.result.warnings.append("fill_in_the_blank should be a list")
            else:
                # Check if all FITB paths are documented
                documented_paths = set()
                for entry in fill_in_blank_array:
                    if isinstance(entry, str):
                        # Extract path from entry (format: "path - reason")
                        path_part = entry.split(" - ")[0].strip()
                        documented_paths.add(path_part)

                # Check if any FITB in data is not documented
                for path in fitb_paths:
                    if path not in documented_paths:
                        self.result.warnings.append(
                            f"FILL IN THE BLANK at '{path}' not documented in fill_in_the_blank array"
                        )

                # Check if documented entries don't have corresponding FITB
                for doc_path in documented_paths:
                    if doc_path not in fitb_paths:
                        self.result.warnings.append(
                            f"fill_in_the_blank documents '{doc_path}' but no FITB found at that path"
                        )

    def _scan_for_fitb(self, obj: Any, path: str, fitb_paths: List[str]):
        """Recursively scan for FILL IN THE BLANK patterns."""
        if isinstance(obj, str):
            if self.FITB_PATTERN.search(obj):
                fitb_paths.append(path if path else "<root>")
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                self._scan_for_fitb(value, new_path, fitb_paths)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                self._scan_for_fitb(item, new_path, fitb_paths)

    def _validate_pipeline_compatibility(self):
        """Level 5: Validate pipeline compatibility."""
        # Check output schema field descriptions (needed for toy data generation)
        cr_output = self.data.get("cr:output")
        if cr_output and isinstance(cr_output, dict):
            schema = cr_output.get("cr:schema")
            if schema and isinstance(schema, dict):
                fields = schema.get("field")
                if fields and isinstance(fields, list):
                    for i, field_item in enumerate(fields):
                        if isinstance(field_item, dict):
                            if not field_item.get("description"):
                                self.result.warnings.append(
                                    f"cr:output.cr:schema.field[{i}].{field_item.get('name', '?')} missing description. "
                                    f"Descriptions are needed for toy data generation."
                                )
                            elif len(field_item.get("description", "").strip()) < 10:
                                self.result.warnings.append(
                                    f"cr:output.cr:schema.field[{i}].{field_item.get('name', '?')} description is too short. "
                                    f"More detail needed for toy data generation."
                                )

        # Check input descriptions are concrete
        cr_input = self.data.get("cr:input")
        if cr_input and isinstance(cr_input, list):
            for i, input_item in enumerate(cr_input):
                if isinstance(input_item, dict):
                    desc = input_item.get("description", "")
                    if not desc or len(desc.strip()) < 20:
                        self.result.warnings.append(
                            f"cr:input[{i}] description is too short or missing. "
                            f"Concrete descriptions needed for bundle generation."
                        )

        # Check evaluation notes for non-standard metrics
        cr_evaluation = self.data.get("cr:evaluation")
        if cr_evaluation and isinstance(cr_evaluation, dict):
            metrics = cr_evaluation.get("metrics", [])
            if metrics and len(metrics) > 0:
                standard_metrics = {"accuracy", "precision", "recall", "f1", "f1-score", "mse", "rmse", "mae", "auc", "roc-auc"}
                has_non_standard = any(
                    m.lower() not in standard_metrics
                    for m in metrics
                    if isinstance(m, str)
                )
                if has_non_standard and not cr_evaluation.get("notes"):
                    self.result.warnings.append(
                        "cr:evaluation has non-standard metrics but missing 'notes' field. "
                        "Notes are helpful for bundle generation."
                    )


def validate_file(path: Path) -> ValidationResult:
    """Load and validate a single Croissant Task JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result = ValidationResult(file=str(path))
        result.valid = False
        result.errors.append(f"Invalid JSON: {e}")
        return result
    except Exception as e:
        result = ValidationResult(file=str(path))
        result.valid = False
        result.errors.append(f"Error reading file: {e}")
        return result

    validator = CroissantTaskValidator(data)
    result = validator.validate()
    result.file = str(path)
    return result


def validate_directory(path: Path) -> List[ValidationResult]:
    """Validate all *.croissant_task.json files in a directory."""
    results = []
    pattern = "*.croissant_task.json"
    files = sorted(path.glob(pattern))
    
    if not files:
        # Try alternative pattern
        files = sorted(path.glob("*.json"))
    
    for file_path in files:
        results.append(validate_file(file_path))
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Croissant Task (cr:TaskProblem) JSON-LD files"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a Croissant Task JSON file or directory containing *.croissant_task.json files"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path does not exist: {args.path}", file=sys.stderr)
        sys.exit(1)

    # Determine if path is file or directory
    if args.path.is_file():
        results = [validate_file(args.path)]
    elif args.path.is_dir():
        results = validate_directory(args.path)
    else:
        print(f"Error: Path is neither a file nor directory: {args.path}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print(f"No Croissant Task files found in {args.path}", file=sys.stderr)
        sys.exit(1)

    # Output results
    if args.json:
        output = {
            "results": [r.to_dict() for r in results],
            "summary": {
                "total": len(results),
                "valid": sum(1 for r in results if r.valid),
                "total_errors": sum(len(r.errors) for r in results),
                "total_warnings": sum(len(r.warnings) for r in results),
                "total_fitb": sum(r.fitb_count for r in results),
            }
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        for result in results:
            status = "VALID" if result.valid else "INVALID"
            print(f"\n[{status}] {result.file}")
            
            if result.errors:
                print("  Errors:")
                for error in result.errors:
                    print(f"    - {error}")
            
            if result.warnings:
                print("  Warnings:")
                for warning in result.warnings[:10]:  # Limit to first 10
                    print(f"    - {warning}")
                if len(result.warnings) > 10:
                    print(f"    ... and {len(result.warnings) - 10} more warnings")
            
            if result.fitb_count > 0:
                print(f"  FILL IN THE BLANK: {result.fitb_count} placeholder(s)")
                if result.fitb_paths:
                    print("    Locations:")
                    for path in result.fitb_paths[:5]:  # Show first 5
                        print(f"      - {path}")
                    if len(result.fitb_paths) > 5:
                        print(f"      ... and {len(result.fitb_paths) - 5} more")
            
            if result.completeness:
                present = [k for k, v in result.completeness.items() if v]
                missing = [k for k, v in result.completeness.items() if not v]
                if present:
                    print(f"  Components present: {', '.join(present)}")
                if missing:
                    print(f"  Components missing: {', '.join(missing)}")

        # Summary
        valid_count = sum(1 for r in results if r.valid)
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        total_fitb = sum(r.fitb_count for r in results)
        
        print(f"\n{'='*60}")
        print(f"Summary: {valid_count}/{len(results)} valid | "
              f"{total_errors} errors | {total_warnings} warnings | "
              f"{total_fitb} FITB placeholders")
        print(f"{'='*60}")

    # Exit code: 0 if all valid, 1 if any errors
    has_errors = any(not r.valid for r in results)
    sys.exit(0 if not has_errors else 1)


if __name__ == "__main__":
    main()
