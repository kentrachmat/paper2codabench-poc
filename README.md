# Paper2Codabench

Convert research papers (PDFs) into executable [Codabench](https://www.codabench.org/) competition bundles using LLMs.

Uses **Croissant Task (`cr:TaskProblem`)** from [MLCommons](https://mlcommons.org/croissant/) as the intermediate metadata format and **code-execution ingestion** — participants submit a `solution.py` that the platform runs.

## Pipeline

```
PDF  ──[LLM]──>  Croissant Task JSON-LD  ──[LLM]──>  Codabench Bundle  ──>  Local Simulation
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in Azure OpenAI credentials
```

## Usage

### Step 1: Extract Croissant Task from Paper

```bash
python src/extract_croissant_task.py papers/paper1.pdf
python src/extract_croissant_task.py papers/paper2.pdf --paper-id paper2
```

### Step 2: Generate Codabench Bundle

```bash
python src/generate_bundle.py croissant_tasks/paper1.croissant_task.json
python src/generate_bundle.py croissant_tasks/paper1.croissant_task.json --output bundles/custom_name
```

### Step 3: Run Local Simulation

```bash
python src/local_run.py bundles/paper1 bundles/paper1/examples/solution.py
python src/local_run.py bundles/paper1 bundles/paper1/examples/sample_submission.csv
python src/local_run.py bundles/paper1 bundles/paper1/examples/solution.py --verbose
```

## Bundle Structure

```
bundles/paper1/
  competition.yaml
  ingestion_program/ingestion.py    # Executes submitted solution.py
  scoring_program/score.py          # Evaluation pipeline
  scoring_program/metrics.py        # Paper-specific metrics
  input_data/input.csv              # Toy input data
  reference_data/reference.csv      # Toy ground truth
  examples/solution.py              # Sample solution
  examples/sample_submission.csv    # CSV example
  seals/                            # Cryptographic verification
```

## Included Papers

| Paper ID | Title | Task Type |
|----------|-------|-----------|
| paper1 | [BELKA — Big Encoded Library for Chemical Assessment](https://openreview.net/pdf?id=zwppB4butE) | Classification |
| paper2 | [FAIR Universe — Handling Uncertainties in Fundamental Science](https://openreview.net/pdf?id=aiYrZONlqy) | Ranking |
| paper3 | [HAC — The Hacker-Cup AI Competition](https://openreview.net/pdf?id=nZB55Omfc3) | Generation |
| paper4 | [Weather4cast 2024 — Multi-task Rain Movie Prediction](https://openreview.net/pdf?id=AZ9WzDxoTf) | Segmentation |

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
