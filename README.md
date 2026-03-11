# Paper2Codabench

Convert research papers (PDFs) into executable [Codabench](https://www.codabench.org/) competition bundles using LLMs.

Uses **Croissant Task (`cr:TaskProblem`)** from [MLCommons](https://mlcommons.org/croissant/) as the intermediate metadata format and **code-execution ingestion** — participants submit a `solution.py` that the platform runs.

## Pipelines

```
Pipeline A:  PDF ──[LLM]──> Croissant Task JSON-LD ──[LLM]──> Codabench Bundle ──> Local Simulation
Pipeline B:  PDF ──[LLM]──> Codabench Bundle (direct) ──> Local Simulation
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in Azure OpenAI credentials
```

## Usage

### Pipeline A: Two-Stage (via Croissant Task)

```bash
# Step 1: Extract Croissant Task from Paper
python src/extract_croissant_task.py papers/paper1.pdf --paper-id paper1

# Step 2: Generate Codabench Bundle
python src/generate_bundle.py croissant_tasks/paper1.croissant_task.json

# Step 3: Run Local Simulation
python src/local_run.py bundles/paper1 bundles/paper1/examples/solution.py --verbose
```

### Pipeline B: Direct (PDF to Bundle)

```bash
python src/generate_bundle_direct.py papers/paper1.pdf --paper-id paper1
python src/local_run.py bundles_pipelineB/paper1 bundles_pipelineB/paper1/examples/solution.py --verbose
```

### Full Pipeline (all 8 papers)

```bash
for i in 1 2 3 4 5 6 7 9; do
    # Pipeline A
    python src/extract_croissant_task.py papers/paper${i}.pdf --paper-id paper${i}
    python src/generate_bundle.py croissant_tasks/paper${i}.croissant_task.json

    # Pipeline B
    python src/generate_bundle_direct.py papers/paper${i}.pdf --paper-id paper${i}
done
```

> Paper 8 has no PDF — skipped.

## Experiments

```bash
# Bundle verification (both pipelines)
python experiments/verify_bundles.py

# Hallucination check (requires LLM)
python experiments/hallucination_check.py

# Fill-in-the-blank analysis
python experiments/fitb_check.py

# Data consistency check
python experiments/compare_csv_data.py

# Croissant schema validation
python experiments/validate_croissant.py

# Plots (individual PNGs + optional combined)
python experiments/plot_data_overview.py --combined
python experiments/plot_results.py --combined
python experiments/plot_hallucination.py
```

## Bundle Structure

```
bundles/paper1/
  competition.yaml
  logo.png
  overview.html
  evaluation.html
  terms.html
  ingestion_program/ingestion.py    # Executes submitted solution.py
  scoring_program/score.py          # Evaluation pipeline
  scoring_program/metrics.py        # Paper-specific metrics
  input_data/input.csv              # Toy input data
  reference_data/reference.csv      # Toy ground truth
  examples/solution.py              # Sample solution
  examples/sample_submission.csv    # CSV example
```

## Key Modules

| Module | Description |
|--------|-------------|
| `src/config.py` | Azure OpenAI credentials, project paths |
| `src/prompts.py` | All LLM prompts (extraction, code gen, data gen) |
| `src/llm_client.py` | Centralized Azure OpenAI client |
| `src/extract_croissant_task.py` | PDF → Croissant Task JSON-LD |
| `src/generate_bundle.py` | Croissant Task → Codabench bundle (Pipeline A) |
| `src/generate_bundle_direct.py` | PDF → Codabench bundle (Pipeline B) |
| `src/local_run.py` | Local bundle simulation |
| `src/bundle_validator.py` | Codabench bundle structure validator |
| `src/utils.py` | Shared utilities (syntax validation, task type inference) |
| `src/croissant_schema.py` | Pydantic models for Croissant Task |

## Submission Interface

Participants implement a `solution.py` with:

```python
def predict(input_dir: str, output_dir: str) -> None:
    import pandas as pd
    from pathlib import Path

    input_df = pd.read_csv(Path(input_dir) / 'input.csv')

    predictions = pd.DataFrame({
        'id': input_df['id'],
        'pred': 0  # Replace with actual predictions
    })

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    predictions.to_csv(Path(output_dir) / 'predictions.csv', index=False)
```


## Data Sources

Two CSV files track NeurIPS 2025 Datasets & Benchmarks paper metadata:

| Source | File | Rows | Description |
|--------|------|------|-------------|
| Root CSV | `neurips2025_db_croissants.csv` | 506 | All papers with Croissant URLs (includes rejected) |
| Data CSV | `data/neurips_2025_db_papers.csv` | 497 | Accepted D&B papers (with additional fields) |

### Data Consistency Summary

| Metric | Count |
|--------|-------|
| Papers in both CSVs | 426 |
| Only in root CSV (rejected papers with croissant) | 80 |
| Only in data CSV (no croissant URL) | 71 |
| Title mismatches | 1 |
| Croissant URL mismatches | 0 |

- **Root CSV** contains 506 papers that have Croissant metadata URLs — including 80 rejected papers not in the data CSV.
- **Data CSV** contains 497 accepted D&B papers — 71 of which lack Croissant URLs (not in root CSV).
- **1 title mismatch**: paper `mORzRZaqT4` — "PolyGuard" (root) vs "GuardSet-X" (data), likely a paper rename.
- Full comparison: `experiments/results/csv_comparison.json`
