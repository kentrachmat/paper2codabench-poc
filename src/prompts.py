"""
LLM Prompts for Paper2Codabench Pipeline

All prompts used for Azure OpenAI GPT-4 calls are centralized here
for easier debugging, versioning, and maintenance.
"""

# ============================================================================
# TASKSPEC EXTRACTION PROMPTS
# ============================================================================

TASKSPEC_SCHEMA = """{
  "paper_id": "paper1",
  "task_name": "",
  "task_type": "classification|ranking|generation|detection|segmentation|other",
  "problem_statement": "",
  "input_description": "IMPORTANT: Include CONCRETE EXAMPLES from the paper. E.g., 'SMILES strings like CC(=O)Oc1ccccc1C(=O)O for molecules' or 'satellite images of 256x256 pixels' or 'Python function signatures'",
  "output_description": "IMPORTANT: Include CONCRETE EXAMPLES and value ranges. E.g., 'Binary labels: 0=no binding, 1=binding' or 'Confidence scores between 0.0-1.0' or 'Generated Python code as string'",
  "submission_format": {
    "type": "predictions_file|code|docker",
    "filename": "submission.csv",
    "columns": ["id", "pred"],
    "example_rows": [
      ["molecule_001", "0"],
      ["molecule_002", "1"],
      ["molecule_003", "0.85"]
    ],
    "INSTRUCTION": "Provide 3-5 REALISTIC examples with domain-appropriate IDs. For molecules use 'molecule_XXX', for weather use dates/locations, for code use 'problem_XXX', etc."
  },
  "dataset": {
    "public_train_available": false,
    "public_dev_available": false,
    "test_is_hidden": true,
    "notes": ""
  },
  "evaluation": {
    "primary_metric": "",
    "metrics": [""],
    "higher_is_better": true,
    "notes": ""
  },
  "constraints": {
    "runtime_limit_sec": 600,
    "memory_limit_mb": 4096,
    "allowed_external_data": "unknown"
  },
  "codabench_mapping": {
    "ingestion_io": "read predictions file; align by id",
    "scoring_steps": [
      "load reference",
      "load submission",
      "validate",
      "compute metric",
      "write scores.json"
    ],
    "edge_cases": [
      "missing ids",
      "extra ids",
      "ties"
    ]
  },
  "open_questions": [
    "list unclear details here"
  ]
}"""


TASKSPEC_EXTRACTION_SYSTEM = """You are a senior ML engineer and reproducibility reviewer.
Your job is to extract a TaskSpec from a scientific paper so that it can be implemented as a Codabench competition bundle.

You must:
- Be precise
- Avoid guessing
- Explicitly list uncertainties in open_questions
- Output MUST be valid JSON only"""


def create_taskspec_extraction_prompt(paper_text: str) -> str:
    """Create user prompt for TaskSpec extraction"""
    return f"""Extract a TaskSpec JSON from the paper below.

Rules:
- Output ONLY valid JSON matching the schema.
- Do NOT invent dataset details.
- If unknown, use null or "unknown" and add explanation to open_questions.
- Focus on what is required to build:
  1) ingestion_program
  2) scoring_program
- Prefer file-based submission unless clearly code-based.

CRITICAL for toy data generation:
- For input_description: Extract CONCRETE examples from the paper (SMILES strings, image dimensions, code format, etc.)
- For output_description: Extract CONCRETE value ranges and formats (0/1, 0.0-1.0, text strings, etc.)
- For example_rows: Create 3-5 REALISTIC examples with domain-appropriate IDs:
  * Chemistry: molecule_001, molecule_002, ...
  * Weather: location_date_001, 2024-01-01_NYC, ...
  * Code: problem_001, challenge_A1, ...
  * Images: image_001, sample_001, ...
- These examples will be used to generate realistic toy data!

TaskSpec schema:
{TASKSPEC_SCHEMA}

Paper text:
{paper_text}"""


# ============================================================================
# BUNDLE GENERATION PROMPTS
# ============================================================================

CODE_GENERATION_SYSTEM = """You are an expert Python developer creating Codabench evaluation code.
Output ONLY valid Python code with no markdown formatting or explanations."""


def create_metrics_generation_prompt(taskspec: dict) -> str:
    """Create prompt for generating metrics.py"""
    import json

    evaluation = taskspec.get('evaluation', {})
    primary_metric = evaluation.get('primary_metric', 'accuracy')
    all_metrics = evaluation.get('metrics', [primary_metric])
    task_type = taskspec.get('task_type', 'classification')
    submission_format = taskspec.get('submission_format', {})
    codabench_mapping = taskspec.get('codabench_mapping', {})

    return f"""Create a complete metrics.py module for this Codabench competition:

TASK SPECIFICATION:
- Task Type: {task_type}
- Primary Metric: {primary_metric}
- All Metrics: {', '.join(all_metrics)}
- Metric Details: {evaluation.get('notes', 'N/A')}
- Higher is Better: {evaluation.get('higher_is_better', True)}

SUBMISSION FORMAT:
{json.dumps(submission_format, indent=2)}

CODABENCH CONVENTIONS:
1. Must define: compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]
2. Return dictionary with metric names as keys, float scores as values
3. Handle edge cases: empty predictions, mismatched shapes, NaN values
4. Use sklearn, numpy, scipy for standard metric implementations
5. Primary metric should be first in returned dict
6. Include comprehensive docstrings
7. Add error handling for edge cases
8. CRITICAL: Handle both 1D and 2D arrays - reshape 1D arrays to 2D (single target) at the start:
   ```python
   if y_true.ndim == 1:
       y_true = y_true.reshape(-1, 1)
   if y_pred.ndim == 1:
       y_pred = y_pred.reshape(-1, 1)
   ```

EDGE CASES TO HANDLE:
{json.dumps(codabench_mapping.get('edge_cases', []), indent=2)}

REFERENCE STRUCTURE (follow this general pattern):
```python
import numpy as np
from typing import Dict
from sklearn.metrics import ...

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
    \"\"\"Compute evaluation metrics.\"\"\"
    # IMPORTANT: Reshape 1D to 2D first
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Validation
    # Metric computation
    # Return dict with primary metric first
    pass
```

Generate the complete metrics.py code now. Include all necessary imports."""


def create_ingestion_generation_prompt(taskspec: dict) -> str:
    """Create prompt for generating ingestion.py"""
    task_name = taskspec.get('task_name', 'Unknown Task')
    submission_format = taskspec.get('submission_format', {})
    columns = submission_format.get('columns', ['id', 'pred'])

    return f"""Create a complete ingestion.py for this competition:

TASK: {task_name}

SUBMISSION FORMAT:
- Type: {submission_format.get('type', 'predictions_file')}
- Filename: {submission_format.get('filename', 'submission.csv')}
- Required Columns: {columns}

VALIDATION REQUIREMENTS:
- Check required columns exist
- Verify no missing IDs
- Check predictions are valid format (no NaN, correct data types)
- Reject submissions with duplicate IDs
- Handle both Codabench paths (/app/...) and local paths

CODABENCH CONVENTIONS:
- Read from: submission_dir = Path('/app/ingested_program') or Path('submission')
- Write to: output_dir = Path('/app/output') or Path('output')
- Output file: output_dir / 'predictions.csv'
- Print progress and validation messages
- Exit with sys.exit(1) on validation failure

REFERENCE STRUCTURE:
```python
import sys
import pandas as pd
from pathlib import Path

def main():
    # Set up paths (Codabench or local)
    # Find submission file
    # Load and validate
    # Write to output/predictions.csv
    pass

if __name__ == "__main__":
    main()
```

Generate the complete ingestion.py code now."""


def create_score_generation_prompt(taskspec: dict) -> str:
    """Create prompt for generating score.py"""
    task_name = taskspec.get('task_name', 'Unknown Task')
    task_type = taskspec.get('task_type', 'classification')
    submission_format = taskspec.get('submission_format', {})
    columns = submission_format.get('columns', ['id', 'pred'])

    # The prediction column is always the LAST column
    target_column = columns[-1] if len(columns) > 1 else 'label'

    # Merge keys are all columns except the last (prediction) column
    merge_columns = columns[:-1] if len(columns) > 1 else ['id']

    return f"""Create a complete score.py for this competition:

TASK: {task_name}
TASK TYPE: {task_type}

SUBMISSION FORMAT:
- Columns: {columns}
- IMPORTANT: The prediction column is '{target_column}' (the LAST column)
- Merge keys (for alignment): {merge_columns}
- Reference CSV format: {','.join(columns)}
- Predictions CSV format: {','.join(columns)}

CODABENCH CONVENTIONS:
- Import metrics from: from metrics import compute_metrics
- Read from:
  - output_dir = Path('/app/output') or Path('output')
  - reference_dir = Path('/app/reference_data') or Path('reference_data')
- Load predictions from: output_dir / 'predictions.csv'
- Load reference from: reference_dir / 'reference.csv'
- CRITICAL: Both files must have ALL columns: {columns}
- Write scores to: scores_dir / 'scores.json' (Path('/app/scores') or Path('scores'))
- Scores JSON format: {{"metric_name": float_value, ...}}
- Handle missing files with clear error messages

REFERENCE STRUCTURE:
```python
import json
import sys
import pandas as pd
from pathlib import Path
from metrics import compute_metrics

def main():
    # Set up paths
    # Load predictions and reference with pd.read_csv()
    # Validate columns match: {columns}
    # Merge on: {merge_columns}
    #   merged = pd.merge(reference, predictions, on={merge_columns}, suffixes=['_true', '_pred'])
    # IMPORTANT: Extract prediction column as numpy arrays using .values
    #   y_true = merged['{target_column}_true'].values
    #   y_pred = merged['{target_column}_pred'].values
    # Call: scores = compute_metrics(y_true, y_pred, task_type="{task_type}")
    # Write scores.json
    pass

if __name__ == "__main__":
    main()
```

CRITICAL REQUIREMENTS:
1. Validate that both files have ALL required columns: {columns}

2. Merge on the correct columns: {merge_columns}
   ```python
   merged = pd.merge(reference, predictions, on={merge_columns}, suffixes=['_true', '_pred'])
   ```

3. Extract the PREDICTION column '{target_column}' as numpy arrays:
   ```python
   y_true = merged['{target_column}_true'].values
   y_pred = merged['{target_column}_pred'].values
   ```

4. Call compute_metrics with all 3 arguments:
   ```python
   scores = compute_metrics(y_true, y_pred, task_type="{task_type}")
   ```

Generate the complete score.py code now."""


# ============================================================================
# TOY DATA GENERATION PROMPTS
# ============================================================================

TOY_DATA_SYSTEM = """You are generating realistic toy examples for a machine learning competition.
Output ONLY valid CSV data with no markdown formatting or explanations."""


def create_toy_data_generation_prompt(taskspec: dict, num_samples: int = 20) -> str:
    """Create prompt for generating toy data"""
    import json

    task_name = taskspec.get('task_name', 'Unknown Task')
    problem_statement = taskspec.get('problem_statement', '')
    input_description = taskspec.get('input_description', '')
    output_description = taskspec.get('output_description', '')
    submission_format = taskspec.get('submission_format', {})
    columns = submission_format.get('columns', ['id', 'pred'])
    example_rows = submission_format.get('example_rows', [])

    return f"""Generate {num_samples} realistic toy examples for this competition:

TASK: {task_name}
PROBLEM: {problem_statement}

INPUT FORMAT (with concrete examples): {input_description}
OUTPUT FORMAT (with concrete examples): {output_description}

SUBMISSION FORMAT:
- Columns: {columns}
- REAL EXAMPLES from the paper: {json.dumps(example_rows, indent=2)}
  ↑ USE THESE AS A GUIDE for ID naming and value formats!

Generate THREE CSV files with EXACT column names from the TaskSpec:

CRITICAL: All CSV files MUST use these EXACT columns from TaskSpec:
{', '.join(columns)}

1. INPUT DATA (input.csv):
   - Column: id
   - {num_samples} rows with realistic IDs matching the pattern from example_rows above
   - Follow the domain-specific ID format shown in the examples

2. REFERENCE DATA (reference.csv):
   - Columns: {', '.join(columns)}  (ALL columns, including id)
   - Ground truth labels/values
   - Use the same value range/format as shown in OUTPUT FORMAT above
   - MUST have EXACTLY the same columns as sample_submission.csv

3. SAMPLE SUBMISSION (sample_submission.csv):
   - Columns: {', '.join(columns)}  (ALL columns, including id)
   - Example predictions (not all correct, mix of right/wrong)
   - MUST have EXACTLY the same columns as reference.csv

Output format:
```
INPUT.CSV:
id
[rows...]

REFERENCE.CSV:
{','.join(columns)}
[rows...]

SAMPLE_SUBMISSION.CSV:
{','.join(columns)}
[rows...]
```

CRITICAL: Use the EXACT column names from above. Do not rename or skip any columns.

Generate the CSV data now."""
