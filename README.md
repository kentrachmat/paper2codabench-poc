# Paper2Codabench

Convert research papers (PDFs) into executable [Codabench](https://www.codabench.org/) competition bundles using LLMs.

Uses **Croissant Task (`cr:TaskProblem`)** from [MLCommons](https://mlcommons.org/croissant/) as the intermediate metadata format and **code-execution ingestion** — participants submit a `solution.py` that the platform runs, rather than uploading pre-computed predictions.

## Pipeline Overview

```
PDF  ──[LLM]──>  Croissant Task JSON-LD  ──[LLM]──>  Codabench Bundle  ──>  Local Simulation
                 (croissant_tasks/)                   (bundles/)              (scores.json)
```

| Step | Script | LLM? | Description |
|------|--------|------|-------------|
| 1 | `extract_croissant_task.py` | Yes | Extract structured task metadata from paper |
| 2 | `generate_bundle.py` | Yes | Generate evaluation code, sample solution, toy data |
| 3 | `local_run.py` | No | Run the bundle locally to validate it works |

---

## Setup

```bash
pip install -r requirements.txt

cp .env.example .env
```

---

## Step-by-Step Usage

### Step 1: Extract Croissant Task from Paper

Reads a PDF and uses the LLM to extract a Croissant Task (`cr:TaskProblem`) JSON-LD.

```bash
python src/extract_croissant_task.py papers/paper1.pdf

# With custom paper ID:
python src/extract_croissant_task.py papers/paper2.pdf --paper-id paper2
```

**What it does:**
- Extracts text from PDF (via PyMuPDF, up to 50 pages)
- Sends to Azure OpenAI to extract a `cr:TaskProblem` containing:
  - `cr:input` — dataset descriptions with concrete examples
  - `cr:output` — output schema with field names, types, and value ranges
  - `cr:evaluation` — paper-specific metrics (e.g., "Mean Average Precision", not just "accuracy")
  - `cr:execution` — runtime/memory constraints
  - `cr:implementation` — environment and interface spec (`predict(input_dir, output_dir)`)
  - `fill_in_the_blank` — tracks fields the LLM couldn't determine (dataset URLs, etc.)
- Validates with Pydantic (continues with warnings if validation fails)
- Saves to `croissant_tasks/paper1.croissant_task.json`

**Output files:**
```
croissant_tasks/
  paper1.croissant_task.json   # Validated Croissant Task
  paper1.raw.json              # Raw LLM response
  paper1.pdf_text.txt          # Extracted PDF text
```

---

### Step 2: Generate Codabench Bundle

Uses the LLM to generate paper-specific evaluation code, a sample solution, and toy data.

```bash
python src/generate_bundle.py croissant_tasks/paper1.croissant_task.json

# With custom output directory:
python src/generate_bundle.py croissant_tasks/paper1.croissant_task.json --output bundles/custom_name
```

**What it does:**
1. **Generates `metrics.py`** — implements the exact metric from the paper
2. **Generates `ingestion.py`** — code-execution ingestion that dynamically imports and runs the user's `solution.py`
3. **Generates `score.py`** — evaluation pipeline: loads predictions + reference, computes metrics, writes `scores.json`
4. **Generates `solution.py`** — sample solution with `predict(input_dir, output_dir)` interface
5. **Generates toy data** — realistic examples with domain-specific IDs:
   - `input_data/input.csv`
   - `reference_data/reference.csv`
   - `examples/sample_submission.csv`
6. Creates `competition.yaml` from template
7. Creates cryptographic verification seal

If LLM code generation fails, falls back to template-based generation (`src/templates/`). All generated Python code is syntax-validated with `compile()` before writing.

**Output structure:**
```
bundles/paper1/
  competition.yaml              # Competition configuration
  README.md                     # Bundle documentation
  ingestion_program/
    ingestion.py                # Executes submitted solution.py
    metadata                    # Codabench metadata
  scoring_program/
    score.py                    # Evaluation pipeline
    metrics.py                  # Paper-specific metrics
    metadata                    # Codabench metadata
  input_data/
    input.csv                   # Toy input data
  reference_data/
    reference.csv               # Toy ground truth (hidden from participants)
  examples/
    solution.py                 # Sample solution
    sample_submission.csv       # CSV example (backward compatible)
  seals/
    bundle_creation_*.json      # Cryptographic verification seal
```

---

### Step 3: Run Local Simulation

Runs the generated bundle locally to validate it works end-to-end. No LLM needed.

```bash
# Code submission (primary — runs solution.py):
python src/local_run.py bundles/paper1 bundles/paper1/examples/solution.py

# CSV submission (backward compatible):
python src/local_run.py bundles/paper1 bundles/paper1/examples/sample_submission.csv

# Verbose mode (keeps temp files, prints all subprocess output):
python src/local_run.py bundles/paper1 bundles/paper1/examples/solution.py --verbose
```

**What it does:**
1. Validates bundle structure (checks for required directories and files)
2. Creates a temp directory mimicking the Codabench directory layout
3. Runs `ingestion.py` — for `.py` submissions, imports and calls `solution.predict()`; for `.csv`, copies to output
4. Runs `score.py` — loads predictions and reference, calls `compute_metrics()`, writes `scores.json`
5. Displays scores and creates an evaluation verification seal

--- 

## Quick Commands: All 4 Papers

```bash
# Extract all Croissant Tasks (uses LLM)
python src/extract_croissant_task.py papers/paper1.pdf
python src/extract_croissant_task.py papers/paper2.pdf
python src/extract_croissant_task.py papers/paper3.pdf
python src/extract_croissant_task.py papers/paper4.pdf

# Generate all bundles (uses LLM)
python src/generate_bundle.py croissant_tasks/paper1.croissant_task.json
python src/generate_bundle.py croissant_tasks/paper2.croissant_task.json
python src/generate_bundle.py croissant_tasks/paper3.croissant_task.json
python src/generate_bundle.py croissant_tasks/paper4.croissant_task.json

# Test all bundles (code submission)
python src/local_run.py bundles/paper1 bundles/paper1/examples/solution.py
python src/local_run.py bundles/paper2 bundles/paper2/examples/solution.py
python src/local_run.py bundles/paper3 bundles/paper3/examples/solution.py
python src/local_run.py bundles/paper4 bundles/paper4/examples/solution.py
```

---

## Included Papers

| Paper ID | Title | Task Type |
|----------|-------|-----------|
| paper1 | [BELKA — Big Encoded Library for Chemical Assessment](https://openreview.net/pdf?id=zwppB4butE) | Classification |
| paper2 | [FAIR Universe — Handling Uncertainties in Fundamental Science](https://openreview.net/pdf?id=aiYrZONlqy) | Ranking |
| paper3 | [HAC — The Hacker-Cup AI Competition](https://openreview.net/pdf?id=nZB55Omfc3) | Generation |
| paper4 | [Weather4cast 2024 — Multi-task Rain Movie Prediction](https://openreview.net/pdf?id=AZ9WzDxoTf) | Segmentation |

---

## Architecture
 
### Key Design Decisions

**Croissant Task as intermediate format.**, the pipeline uses `cr:TaskProblem` JSON-LD from the MLCommons Croissant vocabulary. This standardizes the metadata format and makes it interoperable with the broader ML ecosystem.

**Code-execution ingestion.** Participants submit `solution.py` with a `predict(input_dir, output_dir)` function. The ingestion script dynamically imports it via `importlib.util` and calls `predict()`. This is more flexible than CSV-only submission — participants can use any approach (ML models, heuristics, ensembles) as long as they produce `predictions.csv`.

**`[FILL IN THE BLANK]` placeholders.** When the LLM cannot determine something from the paper (dataset URLs, exact Docker images, package versions), it uses `[FILL IN THE BLANK]` and tracks these in the `fill_in_the_blank` array. This makes unknowns explicit rather than hallucinated.

**Template fallback.** If LLM code generation fails (network error, syntax error, etc.), `generate_bundle.py` falls back to `src/templates/*.template` files using `{{placeholder}}` substitution. This ensures a bundle is always produced.

**Cryptographic seals.** RSA-2048 seals verify bundle integrity and evaluation results. Keys are stored in `.keys/`. Each seal includes a timestamp, SHA-256 bundle hash, and RSA-PSS signature.

### Submission Interface

Participants implement:

```python
# solution.py
def predict(input_dir: str, output_dir: str) -> None:
    """
    Read input data, generate predictions, write output.

    Args:
        input_dir: Path to directory containing input.csv
        output_dir: Path to directory where predictions.csv should be written
    """
    import pandas as pd
    from pathlib import Path

    input_df = pd.read_csv(Path(input_dir) / 'input.csv')

    # Your prediction logic here
    predictions = pd.DataFrame({
        'id': input_df['id'],
        'pred': 0  # Replace with actual predictions
    })

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    predictions.to_csv(Path(output_dir) / 'predictions.csv', index=False)
```
