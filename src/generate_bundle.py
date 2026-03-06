#!/usr/bin/env python3
"""
Generate Codabench competition bundle from Croissant Task.

Usage:
    python generate_bundle.py croissant_tasks/paper1.croissant_task.json
    python generate_bundle.py croissant_tasks/paper1.croissant_task.json --output bundles/custom_name
"""
import argparse
import copy
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from config import Config
from llm_client import call_azure_openai_for_code
from utils import validate_python_syntax
from prompts import (
    CODE_GENERATION_SYSTEM,
    TOY_DATA_SYSTEM,
    SOLUTION_GENERATION_SYSTEM,
    create_metrics_generation_prompt,
    create_ingestion_generation_prompt,
    create_score_generation_prompt,
    create_toy_data_generation_prompt,
    create_sample_solution_prompt,
    infer_task_type
)


def load_croissant_task(croissant_task_path: Path) -> dict:
    """Load Croissant Task JSON-LD from file"""
    with open(croissant_task_path, 'r') as f:
        return json.load(f)


def load_template(template_name: str) -> str:
    """Load template file content"""
    template_path = Config.TEMPLATES_DIR / template_name
    with open(template_path, 'r') as f:
        return f.read()


def truncate_croissant_for_llm(croissant_task: dict, max_desc_chars: int = 15000) -> dict:
    """
    Deep-copy the Croissant Task and truncate long text fields to avoid token overflow.
    """
    task = copy.deepcopy(croissant_task)

    # Truncate top-level description
    desc = task.get('description', '')
    if len(desc) > max_desc_chars:
        task['description'] = desc[:max_desc_chars] + '\n... [TRUNCATED]'

    # Truncate each input description
    for inp in task.get('cr:input', []):
        inp_desc = inp.get('description', '')
        if len(inp_desc) > max_desc_chars:
            inp['description'] = inp_desc[:max_desc_chars] + '\n... [TRUNCATED]'

    return task


def generate_toy_data_minimal_fallback(croissant_task: dict, num_samples: int = 20) -> tuple:
    """
    Minimal fallback toy data generation when all other methods fail.
    Creates safe DataFrames with id + prediction columns.
    """
    print(f"  Generating {num_samples} minimal fallback toy data samples...")

    output_spec = croissant_task.get('cr:output', {})
    schema = output_spec.get('cr:schema', {})
    fields = schema.get('field', [])
    columns = [f.get('name', '') for f in fields] if fields else ['id', 'pred']
    target_column = columns[-1] if len(columns) > 1 else 'pred'

    ids = [f"sample_{i:03d}" for i in range(num_samples)]

    input_df = pd.DataFrame({'id': ids})
    reference_df = pd.DataFrame({'id': ids, target_column: [0] * num_samples})
    sample_submission_df = pd.DataFrame({'id': ids, target_column: [0] * num_samples})

    print(f"  Generated {num_samples} minimal samples (id + {target_column})")
    return input_df, reference_df, sample_submission_df


def fill_template(template_content: str, variables: Dict[str, Any]) -> str:
    """
    Fill template placeholders with variables.

    Args:
        template_content: Template string with {{placeholders}}
        variables: Dictionary of variable names to values

    Returns:
        Filled template string
    """
    result = template_content

    for key, value in variables.items():
        placeholder = f"{{{{{key}}}}}"

        # Convert value to string representation
        if isinstance(value, list):
            value_str = json.dumps(value)
        elif isinstance(value, bool):
            value_str = str(value).lower()
        elif value is None:
            value_str = "null"
        else:
            value_str = str(value)

        result = result.replace(placeholder, value_str)

    return result


_FITB_PATTERN = re.compile(r'\[FILL IN THE BLANK[^\]]*\]', re.IGNORECASE)


def sanitize_bundle_value(
    value: Any,
    field_name: str,
    default: Any,
    fallback_log: List[str],
) -> Any:
    """
    Check if *value* is None, empty string, or contains a [FILL IN THE BLANK...]
    placeholder.  Return *default* when invalid and append a note to *fallback_log*
    for traceability.
    """
    if value is None:
        fallback_log.append(f"{field_name}: was None, using default '{default}'")
        return default

    if isinstance(value, str):
        if value.strip() == '':
            fallback_log.append(f"{field_name}: was empty string, using default '{default}'")
            return default
        if _FITB_PATTERN.search(value):
            fallback_log.append(
                f"{field_name}: contained '[FILL IN THE BLANK]', using default '{default}'"
            )
            return default

    return value


def extract_bundle_vars(croissant_task: dict) -> Tuple[Dict[str, Any], List[str]]:
    """
    Convert Croissant Task JSON-LD to flat dict for template filling.

    Derives task_type, columns, target_column, etc. from Croissant structure.

    Returns:
        (template_vars, fallback_log) — the variable dict and a list of
        fallback substitutions that were applied.
    """
    fallback_log: List[str] = []

    task_name = sanitize_bundle_value(
        croissant_task.get('name'), 'task_name', 'Unnamed Competition', fallback_log
    )
    task_type = infer_task_type(croissant_task)

    evaluation = croissant_task.get('cr:evaluation', {})
    execution = croissant_task.get('cr:execution', {})

    problem_statement = sanitize_bundle_value(
        croissant_task.get('description'), 'problem_statement',
        'No description provided', fallback_log
    )
    primary_metric = sanitize_bundle_value(
        evaluation.get('primaryMetric'), 'primary_metric', 'accuracy', fallback_log
    )

    higher_is_better = evaluation.get('higherIsBetter')
    if higher_is_better is None:
        fallback_log.append("higherIsBetter: was None, defaulting to True")
        higher_is_better = True

    output_spec = croissant_task.get('cr:output', {})
    schema = output_spec.get('cr:schema', {})
    fields = schema.get('field', [])
    columns = [f.get('name', '') for f in fields] if fields else []
    # Filter empty column names
    columns = [c for c in columns if c.strip()]
    if not columns:
        columns = ['id', 'pred']
        fallback_log.append("required_columns: no valid columns found, using ['id', 'pred']")

    target_column = columns[-1] if len(columns) > 1 else 'pred'

    template_vars = {
        'task_name': task_name,
        'problem_statement': problem_statement,
        'task_type': task_type,
        'primary_metric': primary_metric,
        'sort_order': 'desc' if higher_is_better else 'asc',
        'runtime_limit_sec': execution.get('runtimeLimitSec', 600),
        'memory_limit_mb': execution.get('memoryLimitMb', 4096),
        'submission_filename': 'solution.py',
        'required_columns': columns,
        'target_column': target_column,
    }

    return template_vars, fallback_log


def generate_toy_data_with_llm(croissant_task: dict, num_samples: int = 20) -> tuple:
    """
    Generate paper-specific toy data using LLM.

    Returns:
        (input_df, reference_df, sample_submission_df) tuple of DataFrames
    """
    print(f"  Generating {num_samples} paper-specific toy examples with LLM...")

    user_prompt = create_toy_data_generation_prompt(croissant_task, num_samples)

    try:
        response = call_azure_openai_for_code(TOY_DATA_SYSTEM, user_prompt)

        # Parse the response to extract CSVs
        parts = response.split('```')

        input_csv = None
        reference_csv = None
        sample_csv = None

        for part in parts:
            part = part.strip()
            if part.upper().startswith('INPUT.CSV'):
                input_csv = '\n'.join(part.split('\n')[1:])
            elif part.upper().startswith('REFERENCE.CSV'):
                reference_csv = '\n'.join(part.split('\n')[1:])
            elif part.upper().startswith('SAMPLE_SUBMISSION.CSV') or part.upper().startswith('SAMPLE SUBMISSION.CSV'):
                sample_csv = '\n'.join(part.split('\n')[1:])

        from io import StringIO

        if input_csv:
            input_df = pd.read_csv(StringIO(input_csv))
        else:
            raise ValueError("Could not parse input.csv from LLM response")

        if reference_csv:
            reference_df = pd.read_csv(StringIO(reference_csv))
        else:
            raise ValueError("Could not parse reference.csv from LLM response")

        if sample_csv:
            sample_submission_df = pd.read_csv(StringIO(sample_csv))
        else:
            raise ValueError("Could not parse sample_submission.csv from LLM response")

        print(f"  Generated {len(input_df)} paper-specific examples")
        return input_df, reference_df, sample_submission_df

    except Exception as e:
        print(f"  LLM toy data generation failed: {e}")
        print("  Falling back to generic toy data...")
        return generate_toy_data_generic(croissant_task, num_samples)


def generate_toy_data_generic(croissant_task: dict, num_samples: int = 20) -> tuple:
    """
    Generate generic toy data (fallback).

    Returns:
        (input_df, reference_df, sample_submission_df) tuple of DataFrames
    """
    print(f"  Generating {num_samples} generic toy data samples...")

    output_spec = croissant_task.get('cr:output', {})
    schema = output_spec.get('cr:schema', {})
    fields = schema.get('field', [])
    columns = [f.get('name', '') for f in fields] if fields else ['id', 'pred']

    task_type = infer_task_type(croissant_task)

    # Generate IDs
    ids = [f"sample_{i:03d}" for i in range(num_samples)]

    # Generate ground truth based on task type
    if task_type == 'classification':
        true_labels = np.random.randint(0, 2, size=num_samples)
        pred_labels = np.random.randint(0, 2, size=num_samples)
    elif task_type == 'ranking':
        true_labels = np.random.randint(1, 11, size=num_samples)
        pred_labels = np.random.randint(1, 11, size=num_samples)
    elif task_type == 'generation':
        true_labels = [f"generated_text_{i}" for i in range(num_samples)]
        pred_labels = [f"predicted_text_{i}" for i in range(num_samples)]
    elif task_type == 'segmentation':
        true_labels = np.random.randint(0, 2, size=num_samples)
        pred_labels = np.random.randint(0, 2, size=num_samples)
    else:
        true_labels = np.random.rand(num_samples)
        pred_labels = np.random.rand(num_samples)

    # Create DataFrames with ALL columns from schema
    # input_df gets all columns EXCEPT the prediction target (last column)
    # reference_df and sample_submission_df get ALL columns
    target_column = columns[-1] if len(columns) > 1 else 'pred'

    # Determine if schema already has an id-like first column
    has_id_column = len(columns) > 0 and columns[0].lower() == 'id'

    input_data = {}
    reference_data = {}
    sample_data = {}

    if has_id_column:
        # Schema already has 'id' as first column — use it directly
        input_data[columns[0]] = ids
        reference_data[columns[0]] = ids
        sample_data[columns[0]] = ids
        # Add middle columns (between id and target)
        for i in range(1, len(columns) - 1):
            col_name = columns[i]
            metadata_values = [f"value_{j}" for j in range(num_samples)]
            input_data[col_name] = metadata_values
            reference_data[col_name] = metadata_values
            sample_data[col_name] = metadata_values
    else:
        # Schema doesn't have 'id' — add synthetic id + ALL schema fields except target
        input_data['id'] = ids
        reference_data['id'] = ids
        sample_data['id'] = ids
        for i in range(0, len(columns) - 1):
            col_name = columns[i]
            metadata_values = [f"value_{j}" for j in range(num_samples)]
            input_data[col_name] = metadata_values
            reference_data[col_name] = metadata_values
            sample_data[col_name] = metadata_values

    # Add the final prediction column (only to reference and sample, NOT input)
    reference_data[target_column] = true_labels
    sample_data[target_column] = pred_labels

    input_df = pd.DataFrame(input_data)
    reference_df = pd.DataFrame(reference_data)
    sample_submission_df = pd.DataFrame(sample_data)

    print(f"  Generated {num_samples} samples")
    return input_df, reference_df, sample_submission_df


def create_bundle_structure(bundle_path: Path):
    """Create bundle directory structure"""
    print(f"  Creating bundle structure at {bundle_path}...")

    directories = [
        bundle_path / "ingestion_program",
        bundle_path / "scoring_program",
        bundle_path / "input_data",
        bundle_path / "reference_data",
        bundle_path / "examples",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("  Bundle structure created")


def generate_metrics_py_with_llm(croissant_task: dict) -> str:
    """Generate metrics.py using LLM."""
    print("  Generating metrics.py with LLM...")
    user_prompt = create_metrics_generation_prompt(croissant_task)
    return call_azure_openai_for_code(CODE_GENERATION_SYSTEM, user_prompt)


def generate_ingestion_py_with_llm(croissant_task: dict) -> str:
    """Generate ingestion.py using LLM."""
    print("  Generating ingestion.py with LLM...")
    user_prompt = create_ingestion_generation_prompt(croissant_task)
    return call_azure_openai_for_code(CODE_GENERATION_SYSTEM, user_prompt)


def generate_score_py_with_llm(croissant_task: dict) -> str:
    """Generate score.py using LLM."""
    print("  Generating score.py with LLM...")
    user_prompt = create_score_generation_prompt(croissant_task)
    return call_azure_openai_for_code(CODE_GENERATION_SYSTEM, user_prompt)


def generate_solution_py_with_llm(croissant_task: dict) -> str:
    """Generate sample solution.py using LLM."""
    print("  Generating sample solution.py with LLM...")
    user_prompt = create_sample_solution_prompt(croissant_task)
    return call_azure_openai_for_code(SOLUTION_GENERATION_SYSTEM, user_prompt)


def generate_bundle_files_with_llm(croissant_task: dict) -> Dict[str, str]:
    """Generate all bundle files using LLM."""
    print("  Generating bundle files with LLM...")

    generated_files = {}

    # Generate metrics.py
    try:
        metrics_code = generate_metrics_py_with_llm(croissant_task)
        is_valid, error = validate_python_syntax(metrics_code)
        if not is_valid:
            raise ValueError(f"Generated metrics.py has syntax error: {error}")
        generated_files['metrics.py'] = metrics_code
        print("    metrics.py generated and validated")
    except Exception as e:
        raise RuntimeError(f"Failed to generate metrics.py: {e}")

    # Generate ingestion.py
    try:
        ingestion_code = generate_ingestion_py_with_llm(croissant_task)
        is_valid, error = validate_python_syntax(ingestion_code)
        if not is_valid:
            raise ValueError(f"Generated ingestion.py has syntax error: {error}")
        generated_files['ingestion.py'] = ingestion_code
        print("    ingestion.py generated and validated")
    except Exception as e:
        raise RuntimeError(f"Failed to generate ingestion.py: {e}")

    # Generate score.py
    try:
        score_code = generate_score_py_with_llm(croissant_task)
        is_valid, error = validate_python_syntax(score_code)
        if not is_valid:
            raise ValueError(f"Generated score.py has syntax error: {error}")
        generated_files['score.py'] = score_code
        print("    score.py generated and validated")
    except Exception as e:
        raise RuntimeError(f"Failed to generate score.py: {e}")

    # Generate sample solution.py
    try:
        solution_code = generate_solution_py_with_llm(croissant_task)
        is_valid, error = validate_python_syntax(solution_code)
        if not is_valid:
            raise ValueError(f"Generated solution.py has syntax error: {error}")
        generated_files['solution.py'] = solution_code
        print("    solution.py generated and validated")
    except Exception as e:
        print(f"  Warning: Failed to generate solution.py: {e}")
        # Non-fatal: generate a minimal fallback
        generated_files['solution.py'] = generate_fallback_solution(croissant_task)
        print("    solution.py generated (fallback)")

    return generated_files


def generate_fallback_solution(croissant_task: dict) -> str:
    """Generate a minimal fallback solution.py"""
    output_spec = croissant_task.get('cr:output', {})
    schema = output_spec.get('cr:schema', {})
    fields = schema.get('field', [])
    columns = [f.get('name', '') for f in fields] if fields else ['id', 'pred']
    target_column = columns[-1] if len(columns) > 1 else 'pred'

    return f'''import pandas as pd
import numpy as np
from pathlib import Path

def predict(input_dir, output_dir):
    """Generate baseline predictions."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input data (contains all columns except the prediction target)
    input_df = pd.read_csv(input_dir / 'input.csv')

    # Copy input columns and add baseline prediction
    predictions = input_df.copy()
    predictions['{target_column}'] = 0

    predictions.to_csv(output_dir / 'predictions.csv', index=False)

if __name__ == "__main__":
    predict("input_data", "output")
'''


def generate_bundle(croissant_task_path: Path, output_dir: Path = None) -> Path:
    """
    Main bundle generation function.

    Args:
        croissant_task_path: Path to Croissant Task JSON-LD file
        output_dir: Output directory for bundle (default: bundles/<paper_id>)

    Returns:
        Path to generated bundle
    """
    print(f"\n{'='*60}")
    print(f"Generating Codabench Bundle")
    print(f"{'='*60}\n")

    # Load Croissant Task
    print(f"  Loading Croissant Task from {croissant_task_path.name}...")
    croissant_task = load_croissant_task(croissant_task_path)
    paper_id = croissant_task.get('paper_id', croissant_task_path.stem.replace('.croissant_task', ''))
    task_name = croissant_task.get('name', 'Unknown Task')
    task_type = infer_task_type(croissant_task)

    print(f"  Loaded: {task_name}")
    print(f"  Paper ID: {paper_id}")
    print(f"  Task Type: {task_type}")

    # Determine output directory
    if output_dir is None:
        output_dir = Config.BUNDLES_DIR / paper_id

    bundle_path = Path(output_dir)

    if bundle_path.exists():
        print(f"\n  Bundle directory already exists: {bundle_path}")
        response = input("  Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(bundle_path)

    # Create bundle structure
    create_bundle_structure(bundle_path)

    # Extract template variables from Croissant Task
    template_vars, fallback_log = extract_bundle_vars(croissant_task)
    if fallback_log:
        print("\n  Sanitization fallbacks applied:")
        for entry in fallback_log:
            print(f"    - {entry}")

    # Truncate Croissant Task for LLM calls to avoid token overflow
    croissant_task_for_llm = truncate_croissant_for_llm(croissant_task)

    # Try LLM-based generation first, fall back to templates
    use_llm = True
    llm_generation_successful = False

    if use_llm:
        try:
            generated_files = generate_bundle_files_with_llm(croissant_task_for_llm)

            # Write LLM-generated files
            print("\n  Writing LLM-generated files...")
            (bundle_path / "ingestion_program" / "ingestion.py").write_text(generated_files['ingestion.py'])
            (bundle_path / "scoring_program" / "score.py").write_text(generated_files['score.py'])
            (bundle_path / "scoring_program" / "metrics.py").write_text(generated_files['metrics.py'])
            (bundle_path / "examples" / "solution.py").write_text(generated_files['solution.py'])

            llm_generation_successful = True
            print("  LLM generation successful!")

        except Exception as e:
            print(f"\n  LLM generation failed: {e}")
            print("  Falling back to template-based generation...")
            use_llm = False

    if not llm_generation_successful:
        # Fallback to template-based generation
        print("\n  Generating with templates...")

        # Generate ingestion program
        ingestion_py = fill_template(load_template('ingestion.py.template'), template_vars)
        (bundle_path / "ingestion_program" / "ingestion.py").write_text(ingestion_py)

        # Generate scoring program
        score_py = fill_template(load_template('score.py.template'), template_vars)
        (bundle_path / "scoring_program" / "score.py").write_text(score_py)

        # Generate metrics module
        metrics_py = fill_template(load_template('metrics.py.template'), template_vars)
        (bundle_path / "scoring_program" / "metrics.py").write_text(metrics_py)

        # Generate fallback solution.py
        solution_py = generate_fallback_solution(croissant_task)
        (bundle_path / "examples" / "solution.py").write_text(solution_py)

        print("  Template-based generation completed")

    # Generate competition.yaml (always use template for metadata)
    print("\n  Generating competition.yaml...")
    comp_yaml = fill_template(load_template('competition.yaml.template'), template_vars)
    (bundle_path / "competition.yaml").write_text(comp_yaml)
    print("  competition.yaml created")

    # Create metadata files
    (bundle_path / "ingestion_program" / "metadata").write_text("command: python ingestion.py\n")
    (bundle_path / "scoring_program" / "metadata").write_text("command: python score.py\n")
    print("  Metadata files created")

    # Generate ancillary HTML files and logo
    print("\n  Generating ancillary files...")
    overview_html = fill_template(load_template('overview.html.template'), template_vars)
    evaluation_html = fill_template(load_template('evaluation.html.template'), template_vars)
    terms_html = fill_template(load_template('terms.html.template'), template_vars)
    (bundle_path / "overview.html").write_text(overview_html)
    (bundle_path / "evaluation.html").write_text(evaluation_html)
    (bundle_path / "terms.html").write_text(terms_html)

    # Copy logo
    logo_src = Config.TEMPLATES_DIR / "logo.png"
    if logo_src.exists():
        shutil.copy(logo_src, bundle_path / "logo.png")
    print("  Ancillary files created")

    # Generate toy data
    print("\n  Generating toy data...")
    try:
        input_df, reference_df, sample_submission_df = generate_toy_data_with_llm(croissant_task_for_llm, num_samples=20)
    except Exception as e:
        print(f"  All toy data generation failed: {e}")
        input_df, reference_df, sample_submission_df = generate_toy_data_minimal_fallback(croissant_task)

    # Save toy data to bundle directories
    (bundle_path / "input_data" / "input.csv").write_text(input_df.to_csv(index=False))
    (bundle_path / "reference_data" / "reference.csv").write_text(reference_df.to_csv(index=False))
    (bundle_path / "examples" / "sample_submission.csv").write_text(sample_submission_df.to_csv(index=False))

    print("  Toy data created")

    evaluation = croissant_task.get('cr:evaluation', {})
    execution = croissant_task.get('cr:execution', {})

    # Generate README
    print("\n  Generating README...")
    readme_content = f"""# {task_name}

Auto-generated Codabench bundle from Croissant Task.

## Task Overview

**Type:** {task_type}
**Primary Metric:** {evaluation.get('primaryMetric', 'N/A')}

{croissant_task.get('description', 'No description provided')}

## Submission Format

**Type:** Code execution
**File:** solution.py
**Interface:** `predict(input_dir, output_dir)`

Your solution must define a `predict(input_dir, output_dir)` function that:
1. Reads input data from `input_dir/input.csv`
2. Generates predictions
3. Writes `predictions.csv` to `output_dir`

## Evaluation

Primary metric: {evaluation.get('primaryMetric', 'N/A')}
Higher is better: {evaluation.get('higherIsBetter', True)}

## Resource Limits

- Runtime: {execution.get('runtimeLimitSec', 600)} seconds
- Memory: {execution.get('memoryLimitMb', 4096)} MB

## Directory Structure

```
{paper_id}/
├── competition.yaml          # Competition configuration
├── ingestion_program/        # Executes submitted code
│   ├── ingestion.py
│   └── metadata
├── scoring_program/          # Computes metrics
│   ├── score.py
│   ├── metrics.py
│   └── metadata
├── input_data/              # Input data for predictions
│   └── input.csv
├── reference_data/          # Ground truth (hidden)
│   └── reference.csv
└── examples/                # Example submission
    ├── solution.py
    └── sample_submission.csv
```

## Usage

```bash
# Run local simulation with sample solution
python src/local_run.py bundles/{paper_id} bundles/{paper_id}/examples/solution.py

# Run with CSV submission (backward compatible)
python src/local_run.py bundles/{paper_id} bundles/{paper_id}/examples/sample_submission.csv
```

---

Generated by Paper2Codabench
"""
    (bundle_path / "README.md").write_text(readme_content)
    print("  README created")

    print(f"\n{'='*60}")
    print(f"Bundle generation complete!")
    print(f"   Location: {bundle_path}")
    print(f"   Task: {task_name}")
    print(f"   Type: {task_type}")
    print(f"{'='*60}\n")

    # Validate bundle
    from bundle_validator import validate_bundle as _validate_bundle
    is_valid, validation_errors = _validate_bundle(bundle_path)
    if is_valid:
        print("  Bundle validation: PASSED")
    else:
        print(f"  Bundle validation: {len(validation_errors)} issues")
        for err in validation_errors:
            print(f"    - {err}")

    # Show bundle contents
    print("\nBundle contents:")
    for item in sorted(bundle_path.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(bundle_path)
            size = item.stat().st_size
            print(f"  {rel_path} ({size} bytes)")

    return bundle_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Codabench bundle from Croissant Task"
    )
    parser.add_argument(
        "croissant_task_path",
        type=Path,
        help="Path to Croissant Task JSON-LD file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for bundle (default: bundles/<paper_id>)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.croissant_task_path.exists():
        print(f"  Error: Croissant Task file not found: {args.croissant_task_path}")
        sys.exit(1)

    # Generate bundle
    try:
        bundle_path = generate_bundle(args.croissant_task_path, args.output)
        print(f"\n  Success! Bundle ready at: {bundle_path}")
    except Exception as e:
        print(f"\n  Error generating bundle: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
