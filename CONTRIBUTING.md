# Contributing to Molecule-AI

Thank you for your interest in contributing to **Molecule-AI: Deep Spectral Representation & Candidate Retrieval Platform for 2D Chemical Structure Prediction**!

This project is developed under faculty supervision at the **Centre for Artificial Intelligence (CAI) & Department of Computer Science & Engineering (CSE)**, Islamic University of Science and Technology (IUST), Kashmir, India.

---

## 🛠️ Development Workflow

1. **Fork the Repository**:
   Click the "Fork" button at the top right of this repository to create your own copy.

2. **Clone & Set Up Environment**:
   ```bash
   git clone https://github.com/<your-username>/Molecules.git
   cd Molecules
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Create a Feature Branch**:
   ```bash
   git checkout -b feat/your-feature-name
   ```

4. **Code Standards & Quality**:
   - Write clean, modular, and type-annotated Python code following **PEP 8** style guidelines.
   - Ensure all chemical transformations maintain canonical SMILES representations and valid InChIKey14 hashes via RDKit.
   - Keep deep learning layers and tensor transformations strictly deterministic where applicable.

5. **Run the Automated Test Suite**:
   Before committing, verify that all unit and integration tests pass cleanly:
   ```bash
   PYTHONPATH=. pytest tests/ -v
   ```

6. **Generate Diagnostics & Visuals**:
   If modifying neural models, evaluation metrics, or preprocessing, refresh the publication diagnostics:
   ```bash
   python scripts/generate_all_10_lab_visualizations.py
   ```

7. **Submit a Pull Request (PR)**:
   - Commit your changes with descriptive, conventional commit messages (`feat: ...`, `fix: ...`, `docs: ...`).
   - Push to your fork:
     ```bash
     git push origin feat/your-feature-name
     ```
   - Open a Pull Request against the `main` branch with a clear summary of your changes, motivation, and test output.

---

## 🔬 Research & Academic Code of Conduct

- Maintain complete scientific integrity and avoid data leakage between training and evaluation partitions.
- Respect chemical reproducibility standards and benchmark protocols defined for the Enveda CASMI 2026 challenge.
- For questions or research collaborations, reach out to the principal researchers and faculty supervisors listed in the [README.md](README.md).
