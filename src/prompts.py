"""
LLM Prompts for Paper2Codabench Pipeline

All prompts used for Azure OpenAI GPT-4 calls are centralized here
for easier debugging, versioning, and maintenance.

Uses Croissant Task (cr:TaskProblem) JSON-LD as the intermediate format.
"""

# ============================================================================
# CROISSANT TASK EXTRACTION PROMPTS
# ============================================================================

CROISSANT_TASK_SCHEMA = """{
  "@context": {
    "@vocab": "http://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "sc": "http://schema.org/"
  },
  "@type": "cr:TaskProblem",
  "@id": "paper1-task",
  "name": "Molecule Binding Prediction",
  "description": "Predict whether a molecule binds to a target protein based on its SMILES representation.",
  "paper_id": "paper1",
  "cr:input": [
    {
      "name": "competition_dataset",
      "description": "IMPORTANT: Include CONCRETE EXAMPLES from the paper. E.g., 'CSV with columns: id (string), smiles (SMILES string like CC(=O)Oc1ccccc1C(=O)O), protein_name (one of BRD4, HSA, sEH)'",
      "url": "[FILL IN THE BLANK]"
    }
  ],
  "cr:output": {
    "name": "predictions",
    "description": "IMPORTANT: Include CONCRETE EXAMPLES and value ranges. E.g., 'Binary binding predictions: 0=no binding, 1=binding'",
    "cr:schema": {
      "name": "prediction_schema",
      "field": [
        {"name": "id", "dataType": "sc:Text", "description": "Unique sample identifier, e.g. molecule_001"},
        {"name": "pred", "dataType": "sc:Float", "description": "Predicted binding probability between 0.0 and 1.0"}
      ]
    }
  },
  "cr:implementation": {
    "cr:environment": {
      "language": "Python",
      "packages": ["pandas", "numpy", "scikit-learn"]
    },
    "entryPoint": "solution.py",
    "interface": "predict(input_dir, output_dir)"
  },
  "cr:evaluation": {
    "primaryMetric": "mean_average_precision",
    "metrics": ["mean_average_precision", "auc_roc"],
    "higherIsBetter": true,
    "notes": "MAP computed per protein target, then averaged"
  },
  "cr:execution": {
    "runtimeLimitSec": 600,
    "memoryLimitMb": 4096,
    "allowedExternalData": "unknown"
  },
  "open_questions": [
    "list unclear details here"
  ],
  "fill_in_the_blank": [
    "cr:input[0].url - dataset URL not specified in paper"
  ]
}"""


# ============================================================================
# PDF CHUNKED SUMMARIZATION PROMPTS (for long papers)
# ============================================================================

CHUNK_SUMMARIZE_SYSTEM = """You are a senior ML engineer extracting key information from a research paper.
Your job is to summarize a SEGMENT of a paper, focusing on details needed to create a machine learning competition.
Output a structured summary in plain text (NOT JSON)."""


def create_chunk_summarize_prompt(chunk_text: str, chunk_index: int, total_chunks: int) -> str:
    """Create prompt for summarizing a single chunk of a long paper"""
    return f"""Summarize this segment ({chunk_index + 1} of {total_chunks}) of a research paper.

Focus on extracting these details (if present in this segment):
1. TASK DEFINITION: What ML task is being proposed/evaluated? What is the problem statement?
2. DATASET: Dataset name, format, columns, features, data types, number of samples, concrete examples (e.g., SMILES strings, image dimensions)
3. INPUT/OUTPUT: What does the model receive as input? What does it produce as output? Exact formats and value ranges.
4. EVALUATION METRICS: Which metrics are used? Primary metric? Higher is better? Any special computation details.
5. BASELINE METHODS: What baseline approaches are described?
6. IMPLEMENTATION DETAILS: Programming language, packages, runtime constraints, memory limits.
7. ANY CONCRETE EXAMPLES: Sample data rows, example predictions, value ranges, ID formats.

If this segment does not contain information about a category, skip it.
Be precise and include specific numbers, names, and formats — do NOT paraphrase vaguely.

--- PAPER SEGMENT {chunk_index + 1}/{total_chunks} ---
{chunk_text}"""


COMBINE_SUMMARIES_SYSTEM = """You are a senior ML engineer consolidating information from a long research paper.
Your job is to combine segment summaries into a single coherent description of the ML task.
Output a comprehensive plain-text summary (NOT JSON)."""


def create_combine_summaries_prompt(summaries: list[str]) -> str:
    """Create prompt for combining chunk summaries into one unified summary"""
    combined = ""
    for i, summary in enumerate(summaries):
        combined += f"\n\n--- SEGMENT {i + 1} SUMMARY ---\n{summary}"

    return f"""Below are summaries from different segments of the same research paper.
Combine them into ONE comprehensive description that covers:

1. Task name and problem statement
2. Dataset description with concrete examples (column names, data types, sample values)
3. Input format (what participants receive)
4. Output format (what participants must produce — exact columns, value ranges)
5. Evaluation metrics (primary metric, all metrics, higher-is-better, computation notes)
6. Implementation requirements (language, packages, entry point, runtime/memory limits)
7. Any baseline approaches described

RULES:
- Deduplicate information that appears in multiple segments
- Prefer specific/concrete details over vague descriptions
- Preserve ALL concrete examples (data samples, value ranges, ID formats)
- If segments conflict, note the conflict

{combined}"""


CROISSANT_TASK_EXTRACTION_SYSTEM = """You are a senior ML engineer and reproducibility reviewer.
Your job is to extract a Croissant Task (cr:TaskProblem) from a scientific paper so that it can be implemented as a Codabench competition bundle.

You must:
- Be precise
- Avoid guessing
- Use [FILL IN THE BLANK] for any information you cannot determine from the paper (dataset URLs, exact environment specs, etc.)
- Track all [FILL IN THE BLANK] fields in the fill_in_the_blank array
- Explicitly list uncertainties in open_questions
- Output MUST be valid JSON only (JSON-LD format with Croissant vocabulary)"""


def create_croissant_task_extraction_prompt(paper_text: str) -> str:
    """Create user prompt for Croissant Task extraction"""
    return f"""Extract a Croissant Task (cr:TaskProblem) JSON-LD from the paper below.

Rules:
- Output ONLY valid JSON matching the schema.
- Do NOT invent dataset details.
- Use [FILL IN THE BLANK] for any information not found in the paper (dataset URLs, Docker images, exact package versions, etc.)
- Track ALL [FILL IN THE BLANK] usages in the fill_in_the_blank array
- If unknown, use null or "unknown" and add explanation to open_questions.
- Focus on what is required to build:
  1) ingestion_program (code-execution: runs user's solution.py)
  2) scoring_program (computes metrics)

CRITICAL for toy data generation:
- For cr:input description: Extract CONCRETE examples from the paper (SMILES strings, image dimensions, code format, etc.)
- For cr:output description: Extract CONCRETE value ranges and formats (0/1, 0.0-1.0, text strings, etc.)
- For cr:output cr:schema fields: Define the exact output columns with realistic example values
  * Use domain-appropriate field names and IDs:
    - Chemistry: molecule_001, molecule_002, ...
    - Weather: location_date_001, 2024-01-01_NYC, ...
    - Code: problem_001, challenge_A1, ...
    - Images: image_001, sample_001, ...
- These examples will be used to generate realistic toy data!

IMPORTANT for cr:evaluation:
- Extract the EXACT metric name from the paper (e.g., "Mean Average Precision", not just "accuracy")
- List ALL metrics mentioned in the paper
- Note whether higher values are better

Croissant Task schema:
{CROISSANT_TASK_SCHEMA}

Paper text:
{paper_text}"""


# ============================================================================
# BUNDLE GENERATION PROMPTS
# ============================================================================

CODE_GENERATION_SYSTEM = """You are an expert Python developer creating Codabench evaluation code.
Output ONLY valid Python code with no markdown formatting or explanations."""


def create_metrics_generation_prompt(croissant_task: dict) -> str:
    """Create prompt for generating metrics.py"""
    import json

    evaluation = croissant_task.get('cr:evaluation', {})
    primary_metric = evaluation.get('primaryMetric', 'accuracy')
    all_metrics = evaluation.get('metrics', [primary_metric])
    higher_is_better = evaluation.get('higherIsBetter', True)
    notes = evaluation.get('notes', 'N/A')

    # Infer task type from metrics/description
    task_type = infer_task_type(croissant_task)

    output_spec = croissant_task.get('cr:output', {})
    schema = output_spec.get('cr:schema', {})
    fields = schema.get('field', [])

    return f"""Create a complete metrics.py module for this Codabench competition:

TASK SPECIFICATION:
- Task Type: {task_type}
- Primary Metric: {primary_metric}
- All Metrics: {', '.join(all_metrics)}
- Metric Details: {notes}
- Higher is Better: {higher_is_better}

OUTPUT SCHEMA FIELDS:
{json.dumps(fields, indent=2)}

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


def create_ingestion_generation_prompt(croissant_task: dict) -> str:
    """Create prompt for generating ingestion.py (code-execution pattern with CSV fallback)"""
    task_name = croissant_task.get('name', 'Unknown Task')

    implementation = croissant_task.get('cr:implementation', {})
    entry_point = implementation.get('entryPoint', 'solution.py')
    interface = implementation.get('interface', 'predict(input_dir, output_dir)')

    output_spec = croissant_task.get('cr:output', {})
    schema = output_spec.get('cr:schema', {})
    fields = schema.get('field', [])
    field_names = [f.get('name', '') for f in fields] if fields else ['id', 'pred']

    return f"""Create a complete ingestion.py for this CODE-EXECUTION competition:

TASK: {task_name}

PRIMARY SUBMISSION TYPE: Code execution
- Users submit a Python file: {entry_point}
- The file must define: {interface}
- Ingestion runs the submitted code, which produces predictions

FALLBACK SUBMISSION TYPE: CSV predictions
- If NO .py file is found, check for .csv files
- If a .csv file is found, copy it directly to output/predictions.csv
- This provides backward compatibility for CSV-only submissions

OUTPUT SCHEMA:
- Expected output columns: {field_names}
- Output file: predictions.csv

CODE-EXECUTION INGESTION REQUIREMENTS:
1. Look for {entry_point} in submission directory
2. If not found, look for any .py file
3. If a .py file is found: dynamically import it using importlib.util, call solution.predict(input_dir, output_dir)
4. If NO .py file is found: look for .csv files and copy the first one to output/predictions.csv
5. If neither .py nor .csv found: print error + sys.exit(1)
6. Verify output/predictions.csv was created
7. On failure: print clear error message + sys.exit(1)

CODABENCH PATH CONVENTIONS:
- Submission dir: Path('/app/ingested_program') or Path('submission')
- Input data dir: Path('/app/input_data') or Path('input_data')
- Output dir: Path('/app/output') or Path('output')
- Output file: output_dir / 'predictions.csv'

REFERENCE STRUCTURE (follow this closely):
```python
import sys
import shutil
import importlib.util
from pathlib import Path

def main():
    # Set up paths (Codabench or local)
    submission_dir = Path('/app/ingested_program')
    input_dir = Path('/app/input_data')
    output_dir = Path('/app/output')

    if not submission_dir.exists():
        submission_dir = Path('submission')
        input_dir = Path('input_data')
        output_dir = Path('output')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find solution.py
    solution_file = submission_dir / '{entry_point}'
    if not solution_file.exists():
        py_files = list(submission_dir.glob('*.py'))
        if py_files:
            solution_file = py_files[0]
        else:
            # CSV fallback: no .py file found, check for .csv
            csv_files = list(submission_dir.glob('*.csv'))
            if csv_files:
                print(f"CSV submission detected: {{csv_files[0].name}}")
                shutil.copy(csv_files[0], output_dir / 'predictions.csv')
                print("Copied CSV to output/predictions.csv")
                return
            print("ERROR: No solution.py or CSV file found")
            sys.exit(1)

    # Import and run solution
    spec = importlib.util.spec_from_file_location('solution', str(solution_file))
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    if not hasattr(solution, 'predict'):
        print("ERROR: solution.py does not define a 'predict' function")
        sys.exit(1)

    solution.predict(str(input_dir), str(output_dir))

    # Verify output was created
    predictions_file = output_dir / 'predictions.csv'
    if not predictions_file.exists():
        print("ERROR: Solution did not create predictions.csv")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

CRITICAL: You MUST include the CSV fallback logic. If no .py file is found, check for .csv files and copy them to output.

Generate the complete ingestion.py code now."""


def create_score_generation_prompt(croissant_task: dict) -> str:
    """Create prompt for generating score.py"""
    task_name = croissant_task.get('name', 'Unknown Task')
    task_type = infer_task_type(croissant_task)

    output_spec = croissant_task.get('cr:output', {})
    schema = output_spec.get('cr:schema', {})
    fields = schema.get('field', [])
    field_names = [f.get('name', '') for f in fields] if fields else ['id', 'pred']

    # The prediction column is always the LAST column
    target_column = field_names[-1] if len(field_names) > 1 else 'pred'

    # Merge keys are all columns except the last (prediction) column
    merge_columns = field_names[:-1] if len(field_names) > 1 else ['id']

    return f"""Create a complete score.py for this competition:

TASK: {task_name}
TASK TYPE: {task_type}

OUTPUT SCHEMA:
- Columns: {field_names}
- IMPORTANT: The prediction column is '{target_column}' (the LAST column)
- Merge keys (for alignment): {merge_columns}
- Reference CSV format: {','.join(field_names)}
- Predictions CSV format: {','.join(field_names)}

CODABENCH CONVENTIONS:
- Import metrics from: from metrics import compute_metrics
- Read from:
  - output_dir = Path('/app/output') or Path('output')
  - reference_dir = Path('/app/reference_data') or Path('reference_data')
- Load predictions from: output_dir / 'predictions.csv'
- Load reference from: reference_dir / 'reference.csv'
- CRITICAL: Both files must have ALL columns: {field_names}
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
    # Validate columns match: {field_names}
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
1. Validate that both files have ALL required columns: {field_names}

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


def create_toy_data_generation_prompt(croissant_task: dict, num_samples: int = 20) -> str:
    """Create prompt for generating toy data"""
    import json

    task_name = croissant_task.get('name', 'Unknown Task')
    description = croissant_task.get('description', '')

    inputs = croissant_task.get('cr:input', [])
    input_description = inputs[0].get('description', '') if inputs else ''

    output_spec = croissant_task.get('cr:output', {})
    output_description = output_spec.get('description', '')
    schema = output_spec.get('cr:schema', {})
    fields = schema.get('field', [])
    field_names = [f.get('name', '') for f in fields] if fields else ['id', 'pred']

    # Input columns = all columns EXCEPT the last (prediction target)
    input_columns = field_names[:-1] if len(field_names) > 1 else ['id']
    target_column = field_names[-1] if len(field_names) > 1 else 'pred'

    return f"""Generate {num_samples} realistic toy examples for this competition:

TASK: {task_name}
PROBLEM: {description}

INPUT FORMAT (with concrete examples): {input_description}
OUTPUT FORMAT (with concrete examples): {output_description}

OUTPUT SCHEMA FIELDS:
{json.dumps(fields, indent=2)}

Generate THREE CSV files:

1. INPUT DATA (input.csv):
   - Columns: {', '.join(input_columns)} (all columns EXCEPT the prediction target '{target_column}')
   - {num_samples} rows with realistic IDs matching the domain
   - Follow domain-specific ID format
   - This is what participants see as input

2. REFERENCE DATA (reference.csv):
   - Columns: {', '.join(field_names)} (ALL columns including '{target_column}')
   - Ground truth labels/values
   - Use the same value range/format as shown in OUTPUT FORMAT above
   - MUST have EXACTLY the same columns as sample_submission.csv

3. SAMPLE SUBMISSION (sample_submission.csv):
   - Columns: {', '.join(field_names)} (ALL columns including '{target_column}')
   - Example predictions (not all correct, mix of right/wrong)
   - MUST have EXACTLY the same columns as reference.csv

Output format:
```
INPUT.CSV:
{','.join(input_columns)}
[rows...]

REFERENCE.CSV:
{','.join(field_names)}
[rows...]

SAMPLE_SUBMISSION.CSV:
{','.join(field_names)}
[rows...]
```

CRITICAL: input.csv must NOT contain the '{target_column}' column. reference.csv and sample_submission.csv MUST contain it.

Generate the CSV data now."""


# ============================================================================
# SAMPLE SOLUTION GENERATION PROMPT
# ============================================================================

SOLUTION_GENERATION_SYSTEM = """You are an expert Python developer creating a sample solution for a machine learning competition.
Output ONLY valid Python code with no markdown formatting or explanations.
The solution should be simple and demonstrate the correct interface."""


def create_sample_solution_prompt(croissant_task: dict) -> str:
    """Create prompt for generating a sample solution.py"""
    import json

    task_name = croissant_task.get('name', 'Unknown Task')
    description = croissant_task.get('description', '')

    output_spec = croissant_task.get('cr:output', {})
    schema = output_spec.get('cr:schema', {})
    fields = schema.get('field', [])
    field_names = [f.get('name', '') for f in fields] if fields else ['id', 'pred']

    # Input columns = all output columns EXCEPT the last (prediction target)
    input_columns = field_names[:-1] if len(field_names) > 1 else ['id']
    target_column = field_names[-1] if len(field_names) > 1 else 'pred'

    implementation = croissant_task.get('cr:implementation', {})
    interface = implementation.get('interface', 'predict(input_dir, output_dir)')

    return f"""Create a simple sample solution.py for this competition:

TASK: {task_name}
DESCRIPTION: {description}

INTERFACE: The solution MUST define a function: {interface}
- input_dir: string path to directory containing input data (input.csv)
- output_dir: string path to directory where predictions.csv should be written

INPUT DATA (input.csv):
- Contains columns: {input_columns}
- Does NOT contain the prediction column '{target_column}'
- The solution must read these columns and generate predictions

OUTPUT PREDICTIONS (predictions.csv):
- Must contain ALL columns: {field_names}
- Copy the input columns ({input_columns}) from input.csv
- Add the prediction column '{target_column}' with your predictions

REQUIREMENTS:
1. Define predict(input_dir, output_dir) function
2. Read input.csv from input_dir (columns: {input_columns})
3. Generate predictions (can be random/simple baseline)
4. Write predictions.csv to output_dir with exact columns: {field_names}
5. ONLY use columns that exist in input.csv: {input_columns}
6. Keep it simple - this is a baseline/example solution
7. Use only standard libraries: pandas, numpy, os, pathlib

REFERENCE STRUCTURE:
```python
import pandas as pd
import numpy as np
from pathlib import Path

def predict(input_dir, output_dir):
    \"\"\"Generate predictions for the competition.\"\"\"
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read input data (columns: {input_columns})
    input_df = pd.read_csv(input_dir / 'input.csv')

    # Build predictions dataframe with all required columns
    predictions = input_df.copy()
    predictions['{target_column}'] = 0  # Simple baseline

    # Save predictions with exact columns: {field_names}
    predictions[{field_names}].to_csv(output_dir / 'predictions.csv', index=False)

if __name__ == "__main__":
    predict("input_data", "output")
```

Generate the complete solution.py code now."""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def infer_task_type(croissant_task: dict) -> str:
    """Infer task type from Croissant Task evaluation and description"""
    evaluation = croissant_task.get('cr:evaluation', {})
    primary_metric = evaluation.get('primaryMetric', '').lower()
    description = croissant_task.get('description', '').lower()

    # Map metrics to task types
    classification_metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc', 'roc',
                              'log_loss', 'cross_entropy']
    ranking_metrics = ['mrr', 'ndcg', 'map', 'mean_average_precision', 'average_precision']
    generation_metrics = ['bleu', 'rouge', 'meteor', 'perplexity', 'cer', 'wer']
    segmentation_metrics = ['iou', 'dice', 'pixel_accuracy', 'jaccard']

    for m in classification_metrics:
        if m in primary_metric:
            return 'classification'
    for m in ranking_metrics:
        if m in primary_metric:
            return 'ranking'
    for m in generation_metrics:
        if m in primary_metric:
            return 'generation'
    for m in segmentation_metrics:
        if m in primary_metric:
            return 'segmentation'

    # Fallback: check description
    if 'classif' in description:
        return 'classification'
    if 'rank' in description:
        return 'ranking'
    if 'generat' in description:
        return 'generation'
    if 'segment' in description:
        return 'segmentation'

    return 'other'
