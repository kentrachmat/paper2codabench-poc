#!/usr/bin/env python3
"""
Codabench competition bundle validator.

Ported from: https://github.com/kentrachmat/paper2codabench-poc/tree/bundle_validator/Validator

Validates that a bundle directory has all required files and correct
structure for upload to Codabench.

Checks:
  - competition.yaml required fields: title, image, tasks, phases, leaderboards, pages, terms, docker_image
  - Image file exists and is JPG/JPEG/PNG
  - Pages have title + file, files exist and are non-empty
  - Tasks have unique indexes, scoring_program required, referenced files exist
  - Terms file exists and is non-empty
  - Phases are sequential (date validation), reference valid tasks
  - Leaderboards have index, title, key, submission_rule, columns with index/title/key

Usage:
    python src/bundle_validator.py bundles/paper1
"""
import argparse
import datetime
import os
import sys

import yaml
from dateutil import parser as date_parser


class BundleValidationError(Exception):
    pass


def get_datetime(field):
    """Parse a date/datetime field from competition.yaml."""
    if not field:
        return None
    if isinstance(field, (datetime.date, datetime.datetime)):
        if isinstance(field, datetime.date) and not isinstance(field, datetime.datetime):
            field = datetime.datetime.combine(field, datetime.time())
        return field
    try:
        return date_parser.parse(str(field))
    except (ValueError, TypeError):
        raise BundleValidationError(f"Invalid date format: {field}")


class BundleValidator:
    """Validates a Codabench competition bundle directory."""

    def __init__(self, bundle_path):
        self.bundle_path = bundle_path
        self.yaml_path = os.path.join(bundle_path, "competition.yaml")
        self.competition_yaml = None

    def validate(self):
        """Run all validation checks. Raises BundleValidationError on failure."""
        print(f"[*] Validating bundle at: {self.bundle_path}")

        if not os.path.isdir(self.bundle_path):
            raise BundleValidationError(
                f"Bundle path does not exist or is not a directory: {self.bundle_path}"
            )

        if not os.path.exists(self.yaml_path):
            raise BundleValidationError("competition.yaml not found in bundle root.")

        with open(self.yaml_path, "r") as f:
            try:
                self.competition_yaml = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise BundleValidationError(f"Error parsing competition.yaml: {e}")

        self._check_required_fields()
        self._validate_image()
        self._validate_pages()
        self._validate_tasks()
        self._validate_terms()
        self._validate_phases()
        self._validate_leaderboards()

        print("[+] Bundle is valid!")

    def _check_required_fields(self):
        required = [
            "title", "image", "tasks", "phases",
            "leaderboards", "pages", "terms", "docker_image",
        ]
        for field in required:
            if field not in self.competition_yaml:
                raise BundleValidationError(f"Missing required field: {field}")

    def _validate_image(self):
        image = self.competition_yaml.get("image")
        if not image:
            raise BundleValidationError("Image field is missing in competition YAML")

        allowed_extensions = {".jpg", ".jpeg", ".png"}
        _, ext = os.path.splitext(image.lower())
        if ext not in allowed_extensions:
            raise BundleValidationError(
                f"Invalid image format: {image}. Only JPG, JPEG and PNG are allowed."
            )

        image_path = os.path.join(self.bundle_path, image)
        if not os.path.exists(image_path):
            raise BundleValidationError(f"Logo/Image not found: {image}")

    def _validate_pages(self):
        pages = self.competition_yaml.get("pages")
        if not pages:
            raise BundleValidationError("At least one page must be defined in 'pages'.")

        if not isinstance(pages, list):
            raise BundleValidationError("'pages' must be a list.")

        for i, page in enumerate(pages):
            title = page.get("title")
            file_name = page.get("file")

            if not title:
                raise BundleValidationError(
                    f"Page at index {i} missing or empty 'title' field."
                )
            if not file_name:
                raise BundleValidationError(
                    f"Page at index {i} missing or empty 'file' field."
                )

            page_path = os.path.join(self.bundle_path, file_name)
            if not os.path.exists(page_path):
                raise BundleValidationError(f"Page file not found: {file_name}")

            if os.path.getsize(page_path) == 0:
                raise BundleValidationError(f"Page file is empty: {file_name}")

    def _validate_tasks(self):
        tasks = self.competition_yaml.get("tasks")
        if not tasks:
            raise BundleValidationError("No tasks listed in competition.yaml")

        seen_indexes = set()
        for task in tasks:
            index = task.get("index")
            if index is None:
                raise BundleValidationError(
                    f"Task '{task.get('name')}' missing 'index'."
                )

            if index in seen_indexes:
                raise BundleValidationError(f"Duplicate task index: {index}")
            seen_indexes.add(index)

            if "key" in task:
                continue  # Existing task reference

            # Scoring program is required
            scoring_program = task.get("scoring_program")
            if not scoring_program:
                raise BundleValidationError(
                    f"Task {index} missing scoring_program."
                )

            scoring_path = os.path.join(self.bundle_path, scoring_program)
            if not os.path.exists(scoring_path):
                raise BundleValidationError(
                    f"Task {index} scoring_program not found: {scoring_program}"
                )

            # Optional fields — must exist if specified
            for file_type in ["ingestion_program", "input_data", "reference_data"]:
                file_name = task.get(file_type)
                if file_name:
                    file_path = os.path.join(self.bundle_path, file_name)
                    if not os.path.exists(file_path):
                        raise BundleValidationError(
                            f"Task {index} {file_type} not found: {file_name}"
                        )

    def _validate_terms(self):
        terms_path = self.competition_yaml.get("terms")
        if not terms_path:
            raise BundleValidationError("Missing 'terms' field in competition.yaml")

        full_path = os.path.join(self.bundle_path, terms_path)
        if not os.path.exists(full_path):
            raise BundleValidationError(f"Terms file not found: {terms_path}")

        if os.path.getsize(full_path) == 0:
            raise BundleValidationError(f"Terms file is empty: {terms_path}")

    def _validate_phases(self):
        phases = self.competition_yaml.get("phases", [])
        if not phases:
            raise BundleValidationError("Competition must have at least one phase.")

        sorted_phases = sorted(phases, key=lambda p: p.get("index", 0))

        for i, phase in enumerate(sorted_phases):
            if "tasks" not in phase:
                raise BundleValidationError(
                    f"Phase {phase.get('index')} missing 'tasks'."
                )

            # Check public_data and starting_kit
            for key in ["public_data", "starting_kit"]:
                f_name = phase.get(key)
                if f_name:
                    if not os.path.exists(os.path.join(self.bundle_path, f_name)):
                        raise BundleValidationError(
                            f"Phase {i} {key} file not found: {f_name}"
                        )

            # Date validation — sequential phases
            start = get_datetime(phase.get("start"))
            end = get_datetime(phase.get("end"))

            if i > 0:
                prev_phase = sorted_phases[i - 1]
                prev_end = get_datetime(prev_phase.get("end"))

                if prev_end is None:
                    raise BundleValidationError(
                        f"Phase {prev_phase.get('index')} must have an end date "
                        f"because there is a phase after it."
                    )

                if start < prev_end:
                    raise BundleValidationError(
                        f"Phases must be sequential. Phase {phase.get('index')} "
                        f"starts before Phase {prev_phase.get('index')} ends."
                    )
                if start == prev_end:
                    raise BundleValidationError(
                        f"Phase {phase.get('index')} should start after "
                        f"Phase {prev_phase.get('index')} has ended (dates conflict)."
                    )

    def _validate_leaderboards(self):
        leaderboards = self.competition_yaml.get("leaderboards")
        if not leaderboards:
            raise BundleValidationError(
                "Missing or empty 'leaderboards' in competition.yaml"
            )

        if not isinstance(leaderboards, list):
            raise BundleValidationError("'leaderboards' must be a list")

        leaderboard_indexes = set()

        for i, leaderboard in enumerate(leaderboards):
            if not isinstance(leaderboard, dict):
                raise BundleValidationError(
                    f"Leaderboard at index {i} must be an object"
                )

            index = leaderboard.get("index")
            title = leaderboard.get("title")
            key = leaderboard.get("key")
            submission_rule = leaderboard.get("submission_rule")
            columns = leaderboard.get("columns")

            if index is None:
                raise BundleValidationError(
                    f"Leaderboard at position {i} missing 'index'"
                )

            if index in leaderboard_indexes:
                raise BundleValidationError(f"Duplicate leaderboard index: {index}")
            leaderboard_indexes.add(index)

            if not title:
                raise BundleValidationError(
                    f"Leaderboard {index} missing or empty 'title'"
                )

            if not key:
                raise BundleValidationError(
                    f"Leaderboard {index} missing or empty 'key'"
                )

            if not submission_rule:
                raise BundleValidationError(
                    f"Leaderboard {index} missing or empty 'submission_rule'"
                )

            if not columns:
                raise BundleValidationError(
                    f"Leaderboard {index} must have at least one column"
                )

            if not isinstance(columns, list):
                raise BundleValidationError(
                    f"'columns' in leaderboard {index} must be a list"
                )

            column_indexes = set()

            for col_i, column in enumerate(columns):
                if not isinstance(column, dict):
                    raise BundleValidationError(
                        f"Column at position {col_i} in leaderboard {index} "
                        f"must be an object"
                    )

                col_index = column.get("index")
                col_title = column.get("title")
                col_key = column.get("key")

                if col_index is None:
                    raise BundleValidationError(
                        f"Column at position {col_i} in leaderboard {index} "
                        f"missing 'index'"
                    )

                if col_index in column_indexes:
                    raise BundleValidationError(
                        f"Duplicate column index {col_index} in leaderboard {index}"
                    )
                column_indexes.add(col_index)

                if not col_title:
                    raise BundleValidationError(
                        f"Column {col_index} in leaderboard {index} "
                        f"missing or empty 'title'"
                    )

                if not col_key:
                    raise BundleValidationError(
                        f"Column {col_index} in leaderboard {index} "
                        f"missing or empty 'key'"
                    )


# ── Lightweight structure check ───────────────────────────────────────────────

REQUIRED_DIRS = [
    "ingestion_program",
    "scoring_program",
    "reference_data",
    "input_data",
]

REQUIRED_FILES = [
    "ingestion_program/ingestion.py",
    "scoring_program/score.py",
    "scoring_program/metrics.py",
    "input_data/input.csv",
    "reference_data/reference.csv",
    "competition.yaml",
]


def validate_bundle_structure(bundle_path) -> list:
    """
    Lightweight check that required directories and files exist.

    No YAML parsing — suitable for fast pre-flight checks in local_run
    and package_bundle.

    Args:
        bundle_path: Path to the bundle directory (str or Path)

    Returns:
        List of missing items (empty if valid)
    """
    bundle_path = str(bundle_path)
    missing = []
    for d in REQUIRED_DIRS:
        if not os.path.isdir(os.path.join(bundle_path, d)):
            missing.append(f"directory: {d}")
    for f in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(bundle_path, f)):
            missing.append(f"file: {f}")
    return missing


# ── Convenience wrapper ──────────────────────────────────────────────────────

def validate_bundle(bundle_path) -> tuple[bool, list[str]]:
    """
    Validate a Codabench competition bundle.

    Convenience wrapper around BundleValidator that returns a tuple
    instead of raising exceptions.

    Args:
        bundle_path: Path to the bundle directory

    Returns:
        (is_valid, errors) — True if valid, list of error strings
    """
    validator = BundleValidator(str(bundle_path))
    try:
        validator.validate()
        return True, []
    except BundleValidationError as e:
        return False, [str(e)]


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate Codabench competition bundle"
    )
    parser.add_argument("bundle_path", help="Path to bundle directory")
    args = parser.parse_args()

    if not os.path.exists(args.bundle_path):
        print(f"Error: Bundle not found: {args.bundle_path}")
        sys.exit(1)

    validator = BundleValidator(args.bundle_path)
    try:
        validator.validate()
    except BundleValidationError as e:
        print(f"[-] Validation Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[-] Unexpected Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
