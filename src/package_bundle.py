#!/usr/bin/env python3
"""
Package a Codabench bundle directory into a zip file for upload.

Usage:
    python src/package_bundle.py bundles/paper1
    python src/package_bundle.py bundles/paper1 --output paper1_bundle.zip
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

from bundle_validator import validate_bundle_structure

EXCLUDE_DIRS = {"__pycache__", ".git"}


def package_bundle(bundle_path: Path, output_path: Path = None) -> Path:
    """
    Zip a bundle directory for Codabench upload.

    Excludes seals/ and __pycache__/ directories.

    Args:
        bundle_path: Path to bundle directory
        output_path: Output zip path (default: <bundle_name>.zip next to bundle)

    Returns:
        Path to created zip file
    """
    bundle_path = bundle_path.resolve()

    if output_path is None:
        output_path = bundle_path.parent / f"{bundle_path.name}.zip"
    output_path = Path(output_path).resolve()

    # Validate
    missing = validate_bundle_structure(bundle_path)
    if missing:
        print("Bundle validation failed. Missing:")
        for item in missing:
            print(f"  - {item}")
        raise ValueError("Invalid bundle structure")

    # Create zip
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(bundle_path.rglob("*")):
            if file_path.is_file():
                rel = file_path.relative_to(bundle_path)
                # Skip excluded directories
                if any(part in EXCLUDE_DIRS for part in rel.parts):
                    continue
                zf.write(file_path, rel)

    size_kb = output_path.stat().st_size / 1024
    print(f"Packaged bundle: {output_path} ({size_kb:.1f} KB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Package Codabench bundle as zip")
    parser.add_argument("bundle_path", type=Path, help="Path to bundle directory")
    parser.add_argument("--output", "-o", type=Path, help="Output zip path")
    args = parser.parse_args()

    if not args.bundle_path.exists():
        print(f"Error: Bundle not found: {args.bundle_path}")
        sys.exit(1)

    try:
        zip_path = package_bundle(args.bundle_path, args.output)
        print(f"Success: {zip_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
