# Troubleshooting & Common Issues

### 1. `ModuleNotFoundError: No module named 'rdkit'`
- **Symptom**: RDKit is not found in standard system Python.
- **Solution**: Install RDKit via conda (`conda install -c conda-forge rdkit`) or pre-built pip wheel (`pip install rdkit`). The codebase includes fallback canonicalization for testing environments where RDKit is being provisioned.

### 2. Kaggle Submission Evaluation Time Limit Exceeded
- **Symptom**: Notebook exceeds the 9-hour execution window.
- **Solution**:
  - Precompute and store candidate database indices (e.g., SQLite / Parquet indexed by rounded neutral mass).
  - Use vector similarity search with `faiss` or binned cosine dot products.
  - Limit candidate retrieval to top-1000 precursor mass matches prior to reranking.

### 3. Submission Rejection on Kaggle
- **Symptom**: `Submission Scoring Error` or `NaN values detected`.
- **Checks**:
  - Run `python scripts/generate_submission.py --verify` locally.
  - Verify header is exactly `molecule_id,smiles`.
  - Ensure every molecule has between 1 and 25 SMILES, separated by semicolons.
  - Check for trailing semicolons or unquoted commas.
