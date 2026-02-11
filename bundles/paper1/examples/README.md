# Example Submissions for Small Molecule-Protein Binding Prediction

This directory contains example submission files to help you get started.

## sample_submission.csv

A valid submission file with 20 example predictions.

Format:
- Columns: id, protein, pred
- Rows: 20

To test this bundle locally:
```bash
python src/local_run.py bundles/paper1 bundles/paper1/examples/sample_submission.csv
```
