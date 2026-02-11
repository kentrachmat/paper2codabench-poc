# Paper → Codabench POC

Convert research papers into executable Codabench competition bundles using LLMs.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure Azure OpenAI
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

---

## Pipeline: Step-by-Step

### Step 1: Extract TaskSpec from Paper ***

**Uses LLM** to read the PDF and extract structured specifications.

```bash
python src/extract_taskspec.py papers/paper1.pdf
```

**What it does:**
- Extracts text from PDF
- *** Sends to Azure OpenAI GPT-4 to extract:
  - Task type (classification/ranking/generation/segmentation)
  - **Concrete input examples** (e.g., "SMILES strings like CC(=O)Oc1ccccc1C(=O)O")
  - **Concrete output examples** (e.g., "Binary labels: 0=no binding, 1=binding")
  - **Paper-specific metrics** (e.g., "Mean Average Precision" not just "accuracy")
  - **Domain-specific example rows** (e.g., `[["molecule_001", "0"]]`)
- Validates schema with Pydantic
- Saves to `taskspec/paper1.taskspec.json`

**Check the output:**
```bash
cat taskspec/paper1.taskspec.json
```

Look for:
- `"primary_metric": "Mean Average Precision"` (paper-specific, not generic!)
- `"input_description": "SMILES strings like..."` (concrete examples)
- `"example_rows": [["molecule_001", "0"], ...]` (domain-specific IDs)

---

### Step 2: Generate Codabench Bundle ***

**Uses LLM** to generate paper-specific evaluation code, metrics, and toy data.

```bash
python src/generate_bundle.py taskspec/paper1.taskspec.json
```

**What it does:**
1. *** **Generates metrics.py** - Implements the exact metric from the paper (e.g., Mean Average Precision, not generic accuracy)
2. *** **Generates ingestion.py** - Creates submission validation code
3. *** **Generates score.py** - Creates evaluation pipeline code
4. *** **Generates toy data** - Creates realistic examples:
   - `input.csv` with domain-specific IDs (e.g., molecule_001)
   - `reference.csv` with appropriate labels/values
   - `sample_submission.csv` in `bundles/paper1/examples/`
5. Creates competition.yaml (from template)
6. Creates README.md
7. Creates cryptographic seal

---

### Step 3: Run Local Simulation

**No LLM** - Executes the generated code to test the bundle.

```bash
python src/local_run.py bundles/paper1 bundles/paper1/examples/sample_submission.csv
```

**What it does:**
1. Validates bundle structure
2. Runs **ingestion.py** (processes submission)
3. Runs **score.py** (computes metrics using metrics.py)
4. Creates scores.json
5. Creates evaluation seal

---

### Step 4: Web Interface (Optional)

```bash
python web/app.py
# Visit http://localhost:5000
```

View dashboard with all papers, TaskSpecs, bundles, and verification seals.

---

## Quick Commands: All 4 Papers

```bash
# Extract all TaskSpecs (*** uses LLM)
python src/extract_taskspec.py papers/paper1.pdf 
python src/extract_taskspec.py papers/paper2.pdf
python src/extract_taskspec.py papers/paper3.pdf 
python src/extract_taskspec.py papers/paper4.pdf 

# Generate all bundles (*** uses LLM)
python src/generate_bundle.py taskspec/paper1.taskspec.json
python src/generate_bundle.py taskspec/paper2.taskspec.json
python src/generate_bundle.py taskspec/paper3.taskspec.json
python src/generate_bundle.py taskspec/paper4.taskspec.json

# Test all bundles
python src/local_run.py bundles/paper1 bundles/paper1/examples/sample_submission.csv
python src/local_run.py bundles/paper2 bundles/paper2/examples/sample_submission.csv
python src/local_run.py bundles/paper3 bundles/paper3/examples/sample_submission.csv
python src/local_run.py bundles/paper4 bundles/paper4/examples/sample_submission.csv
```