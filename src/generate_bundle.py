#!/usr/bin/env python3
"""
Generate Codabench competition bundle from TaskSpec.

Usage:
    python generate_bundle.py taskspec/paper1.taskspec.json
    python generate_bundle.py taskspec/paper1.taskspec.json --output bundles/custom_name
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from openai import OpenAI

from config import Config
from seal import create_bundle_seal
from prompts import (
    CODE_GENERATION_SYSTEM,
    TOY_DATA_SYSTEM,
    create_metrics_generation_prompt,
    create_ingestion_generation_prompt,
    create_score_generation_prompt,
    create_toy_data_generation_prompt
)


def load_taskspec(taskspec_path: Path) -> dict:
    """Load TaskSpec JSON from file"""
    with open(taskspec_path, 'r') as f:
        return json.load(f)


def load_template(template_name: str) -> str:
    """Load template file content"""
    template_path = Config.TEMPLATES_DIR / template_name
    with open(template_path, 'r') as f:
        return f.read()


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


def generate_toy_data_with_llm(taskspec: dict, num_samples: int = 20) -> tuple:
    """
    Generate paper-specific toy data using LLM.

    Args:
        taskspec: TaskSpec dictionary
        num_samples: Number of samples to generate

    Returns:
        (input_df, reference_df, sample_submission_df) tuple of DataFrames
    """
    print(f"📊 Generating {num_samples} paper-specific toy examples with LLM...")

    user_prompt = create_toy_data_generation_prompt(taskspec, num_samples)

    try:
        response = call_azure_openai_for_code(TOY_DATA_SYSTEM, user_prompt)

        # Parse the response to extract CSVs
        parts = response.split('```')

        # Extract CSVs from response
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

        # Parse CSVs
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

        print(f"✅ Generated {len(input_df)} paper-specific examples")
        return input_df, reference_df, sample_submission_df

    except Exception as e:
        print(f"⚠️  LLM toy data generation failed: {e}")
        print("📝 Falling back to generic toy data...")
        return generate_toy_data_generic(taskspec, num_samples)


def generate_toy_data_generic(taskspec: dict, num_samples: int = 20) -> tuple:
    """
    Generate generic toy data (fallback).

    Args:
        taskspec: TaskSpec dictionary
        num_samples: Number of samples to generate

    Returns:
        (input_df, reference_df, sample_submission_df) tuple of DataFrames
    """
    print(f"📊 Generating {num_samples} generic toy data samples...")

    submission_format = taskspec.get('submission_format', {})
    columns = submission_format.get('columns', ['id', 'pred'])
    task_type = taskspec.get('task_type', 'classification')

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

    # Create DataFrames with ALL columns from TaskSpec
    input_df = pd.DataFrame({'id': ids})

    # For reference and sample_submission, include ALL columns
    reference_data = {'id': ids}
    sample_data = {'id': ids}

    # Add middle columns (metadata columns between id and prediction)
    for i in range(1, len(columns) - 1):
        col_name = columns[i]
        # Generate simple placeholder values for metadata columns
        reference_data[col_name] = [f"value_{j}" for j in range(num_samples)]
        sample_data[col_name] = [f"value_{j}" for j in range(num_samples)]

    # Add the final prediction column
    target_column = columns[-1] if len(columns) > 1 else 'pred'
    reference_data[target_column] = true_labels
    sample_data[target_column] = pred_labels

    reference_df = pd.DataFrame(reference_data)
    sample_submission_df = pd.DataFrame(sample_data)

    print(f"✓ Generated {num_samples} samples")
    return input_df, reference_df, sample_submission_df


def create_bundle_structure(bundle_path: Path):
    """Create bundle directory structure"""
    print(f"📁 Creating bundle structure at {bundle_path}...")

    directories = [
        bundle_path / "ingestion_program",
        bundle_path / "scoring_program",
        bundle_path / "input_data",
        bundle_path / "reference_data",
        bundle_path / "seals",
        bundle_path / "examples"  # Add examples directory inside bundle
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("✓ Bundle structure created")


def validate_python_syntax(code: str) -> tuple[bool, str]:
    """
    Validate Python syntax.

    Args:
        code: Python code string

    Returns:
        (is_valid, error_message) tuple
    """
    try:
        compile(code, '<string>', 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def call_azure_openai_for_code(system_prompt: str, user_prompt: str, max_retries: int = 2) -> str:
    """
    Call Azure OpenAI for code generation.

    Args:
        system_prompt: System message
        user_prompt: User prompt
        max_retries: Max retry attempts

    Returns:
        Generated code string
    """
    try:
        Config.validate()
    except ValueError as e:
        raise RuntimeError(f"Configuration error: {e}")

    client = OpenAI(
        base_url=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_KEY
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=Config.AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.2,  # Low temperature for more deterministic code
            )

            code = response.choices[0].message.content.strip()

            # Remove markdown code fences if present
            if code.startswith("```python"):
                code = code[9:]  # Remove ```python
            if code.startswith("```"):
                code = code[3:]  # Remove ```
            if code.endswith("```"):
                code = code[:-3]  # Remove trailing ```

            return code.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  Attempt {attempt + 1} failed, retrying: {e}")
            else:
                raise RuntimeError(f"LLM call failed after {max_retries} attempts: {e}")


def generate_metrics_py_with_llm(taskspec: dict) -> str:
    """
    Generate metrics.py using LLM.

    Args:
        taskspec: TaskSpec dictionary

    Returns:
        Generated Python code
    """
    print("  🤖 Generating metrics.py with LLM...")

    user_prompt = create_metrics_generation_prompt(taskspec)
    return call_azure_openai_for_code(CODE_GENERATION_SYSTEM, user_prompt)


def generate_ingestion_py_with_llm(taskspec: dict) -> str:
    """
    Generate ingestion.py using LLM.

    Args:
        taskspec: TaskSpec dictionary

    Returns:
        Generated Python code
    """
    print("  🤖 Generating ingestion.py with LLM...")

    user_prompt = create_ingestion_generation_prompt(taskspec)
    return call_azure_openai_for_code(CODE_GENERATION_SYSTEM, user_prompt)


def generate_score_py_with_llm(taskspec: dict) -> str:
    """
    Generate score.py using LLM.

    Args:
        taskspec: TaskSpec dictionary

    Returns:
        Generated Python code
    """
    print("  🤖 Generating score.py with LLM...")

    user_prompt = create_score_generation_prompt(taskspec)
    return call_azure_openai_for_code(CODE_GENERATION_SYSTEM, user_prompt)


def generate_bundle_files_with_llm(taskspec: dict, paper_text: str = None) -> Dict[str, str]:
    """
    Generate all bundle files using LLM.

    Args:
        taskspec: TaskSpec dictionary
        paper_text: Optional paper text for additional context

    Returns:
        Dictionary mapping filenames to generated code
    """
    print("🤖 Generating bundle files with LLM...")

    generated_files = {}

    # Generate metrics.py
    try:
        metrics_code = generate_metrics_py_with_llm(taskspec)
        is_valid, error = validate_python_syntax(metrics_code)
        if not is_valid:
            raise ValueError(f"Generated metrics.py has syntax error: {error}")
        generated_files['metrics.py'] = metrics_code
        print("  ✅ metrics.py generated and validated")
    except Exception as e:
        raise RuntimeError(f"Failed to generate metrics.py: {e}")

    # Generate ingestion.py
    try:
        ingestion_code = generate_ingestion_py_with_llm(taskspec)
        is_valid, error = validate_python_syntax(ingestion_code)
        if not is_valid:
            raise ValueError(f"Generated ingestion.py has syntax error: {error}")
        generated_files['ingestion.py'] = ingestion_code
        print("  ✅ ingestion.py generated and validated")
    except Exception as e:
        raise RuntimeError(f"Failed to generate ingestion.py: {e}")

    # Generate score.py
    try:
        score_code = generate_score_py_with_llm(taskspec)
        is_valid, error = validate_python_syntax(score_code)
        if not is_valid:
            raise ValueError(f"Generated score.py has syntax error: {error}")
        generated_files['score.py'] = score_code
        print("  ✅ score.py generated and validated")
    except Exception as e:
        raise RuntimeError(f"Failed to generate score.py: {e}")

    return generated_files


def generate_bundle(taskspec_path: Path, output_dir: Path = None) -> Path:
    """
    Main bundle generation function.

    Args:
        taskspec_path: Path to TaskSpec JSON file
        output_dir: Output directory for bundle (default: bundles/<paper_id>)

    Returns:
        Path to generated bundle
    """
    print(f"\n{'='*60}")
    print(f"Generating Codabench Bundle")
    print(f"{'='*60}\n")

    # Load TaskSpec
    print(f"📄 Loading TaskSpec from {taskspec_path.name}...")
    taskspec = load_taskspec(taskspec_path)
    paper_id = taskspec.get('paper_id', taskspec_path.stem.replace('.taskspec', ''))
    task_name = taskspec.get('task_name', 'Unknown Task')
    task_type = taskspec.get('task_type', 'other')

    print(f"✓ Loaded: {task_name}")
    print(f"  Paper ID: {paper_id}")
    print(f"  Task Type: {task_type}")

    # Determine output directory
    if output_dir is None:
        output_dir = Config.BUNDLES_DIR / paper_id

    bundle_path = Path(output_dir)

    if bundle_path.exists():
        print(f"\n⚠️  Bundle directory already exists: {bundle_path}")
        response = input("  Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(bundle_path)

    # Create bundle structure
    create_bundle_structure(bundle_path)

    # Extract template variables
    submission_format = taskspec.get('submission_format', {})
    evaluation = taskspec.get('evaluation', {})
    constraints = taskspec.get('constraints', {})

    # Determine target column (second column in submission format)
    columns = submission_format.get('columns', ['id', 'pred'])
    target_column = columns[1] if len(columns) > 1 else 'label'

    template_vars = {
        'task_name': task_name,
        'problem_statement': taskspec.get('problem_statement', 'No description provided'),
        'task_type': task_type,
        'primary_metric': evaluation.get('primary_metric', 'accuracy'),
        'sort_order': 'desc' if evaluation.get('higher_is_better', True) else 'asc',
        'runtime_limit_sec': constraints.get('runtime_limit_sec', 600),
        'memory_limit_mb': constraints.get('memory_limit_mb', 4096),
        'submission_filename': submission_format.get('filename', 'submission.csv'),
        'required_columns': columns,
        'target_column': target_column,
    }

    # Try LLM-based generation first, fall back to templates
    use_llm = True
    llm_generation_successful = False

    if use_llm:
        try:
            # Check if paper text exists for additional context
            paper_text_path = Config.TASKSPEC_DIR / f"{paper_id}.pdf_text.txt"
            paper_text = None
            if paper_text_path.exists():
                with open(paper_text_path) as f:
                    paper_text = f.read()

            # Generate files with LLM
            generated_files = generate_bundle_files_with_llm(taskspec, paper_text)

            # Write LLM-generated files
            print("\n📝 Writing LLM-generated files...")
            (bundle_path / "ingestion_program" / "ingestion.py").write_text(generated_files['ingestion.py'])
            (bundle_path / "scoring_program" / "score.py").write_text(generated_files['score.py'])
            (bundle_path / "scoring_program" / "metrics.py").write_text(generated_files['metrics.py'])

            llm_generation_successful = True
            print("✅ LLM generation successful!")

        except Exception as e:
            print(f"\n⚠️  LLM generation failed: {e}")
            print("📝 Falling back to template-based generation...")
            use_llm = False

    if not llm_generation_successful:
        # Fallback to template-based generation
        print("\n📝 Generating with templates...")

        # Generate ingestion program
        ingestion_py = fill_template(load_template('ingestion.py.template'), template_vars)
        (bundle_path / "ingestion_program" / "ingestion.py").write_text(ingestion_py)

        # Generate scoring program
        score_py = fill_template(load_template('score.py.template'), template_vars)
        (bundle_path / "scoring_program" / "score.py").write_text(score_py)

        # Generate metrics module
        metrics_py = fill_template(load_template('metrics.py.template'), template_vars)
        (bundle_path / "scoring_program" / "metrics.py").write_text(metrics_py)

        print("✓ Template-based generation completed")

    # Generate competition.yaml (always use template for metadata)
    print("\n📝 Generating competition.yaml...")
    comp_yaml = fill_template(load_template('competition.yaml.template'), template_vars)
    (bundle_path / "competition.yaml").write_text(comp_yaml)
    print("✓ competition.yaml created")

    # Create metadata files
    (bundle_path / "ingestion_program" / "metadata").write_text("command: python ingestion.py\n")
    (bundle_path / "scoring_program" / "metadata").write_text("command: python score.py\n")
    print("✓ Metadata files created")

    # Generate paper-specific toy data and examples
    print("\n📊 Generating paper-specific toy data and examples...")
    input_df, reference_df, sample_submission_df = generate_toy_data_with_llm(taskspec, num_samples=20)

    # Save input and reference data
    (bundle_path / "input_data" / "input.csv").write_text(input_df.to_csv(index=False))
    (bundle_path / "reference_data" / "reference.csv").write_text(reference_df.to_csv(index=False))

    # Save sample submission inside bundle/examples/
    (bundle_path / "examples" / "sample_submission.csv").write_text(sample_submission_df.to_csv(index=False))

    # Create examples README
    examples_readme = f"""# Example Submissions for {task_name}

This directory contains example submission files to help you get started.

## sample_submission.csv

A valid submission file with {len(sample_submission_df)} example predictions.

Format:
- Columns: {', '.join(sample_submission_df.columns.tolist())}
- Rows: {len(sample_submission_df)}

To test this bundle locally:
```bash
python src/local_run.py bundles/{paper_id} bundles/{paper_id}/examples/sample_submission.csv
```
"""
    (bundle_path / "examples" / "README.md").write_text(examples_readme)

    print("✓ Toy data and examples created")

    # Generate README
    print("\n📝 Generating README...")
    readme_content = f"""# {task_name}

Auto-generated Codabench bundle from TaskSpec.

## Task Overview

**Type:** {task_type}
**Primary Metric:** {evaluation.get('primary_metric', 'N/A')}

{taskspec.get('problem_statement', 'No description provided')}

## Submission Format

**File:** {submission_format.get('filename', 'submission.csv')}
**Columns:** {', '.join(columns)}

## Dataset

- Input samples: 20 (toy data)
- Reference labels: 20 (toy data)

## Evaluation

Primary metric: {evaluation.get('primary_metric', 'N/A')}
Higher is better: {evaluation.get('higher_is_better', True)}

## Resource Limits

- Runtime: {constraints.get('runtime_limit_sec', 600)} seconds
- Memory: {constraints.get('memory_limit_mb', 4096)} MB

## Directory Structure

```
{paper_id}/
├── competition.yaml          # Competition configuration
├── ingestion_program/        # Processes submissions
│   ├── ingestion.py
│   └── metadata
├── scoring_program/          # Computes metrics
│   ├── score.py
│   ├── metrics.py
│   └── metadata
├── input_data/              # Test inputs (toy data)
│   └── input.csv
├── reference_data/          # Ground truth (toy data)
│   └── reference.csv
├── seals/                   # Verification seals
└── README.md
```

## Usage

See main project README for instructions on running local simulations.

---

Generated by Paper2Codabench POC
"""
    (bundle_path / "README.md").write_text(readme_content)
    print("✓ README created")

    # Create verification seal
    print("\n🔒 Creating verification seal...")
    seal_metadata = {
        'task_name': task_name,
        'task_type': task_type,
        'paper_id': paper_id,
        'primary_metric': evaluation.get('primary_metric', 'N/A'),
    }
    seal = create_bundle_seal(bundle_path, seal_metadata)
    print(f"✓ Seal created: {seal.get('seal_id', 'N/A')}")

    print(f"\n{'='*60}")
    print(f"✅ Bundle generation complete!")
    print(f"   Location: {bundle_path}")
    print(f"   Task: {task_name}")
    print(f"   Type: {task_type}")
    print(f"{'='*60}\n")

    # Show bundle contents
    print("Bundle contents:")
    for item in sorted(bundle_path.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(bundle_path)
            size = item.stat().st_size
            print(f"  {rel_path} ({size} bytes)")

    return bundle_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Codabench bundle from TaskSpec"
    )
    parser.add_argument(
        "taskspec_path",
        type=Path,
        help="Path to TaskSpec JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for bundle (default: bundles/<paper_id>)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.taskspec_path.exists():
        print(f"✗ Error: TaskSpec file not found: {args.taskspec_path}")
        sys.exit(1)

    # Generate bundle
    try:
        bundle_path = generate_bundle(args.taskspec_path, args.output)
        print(f"\n✅ Success! Bundle ready at: {bundle_path}")
    except Exception as e:
        print(f"\n✗ Error generating bundle: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
