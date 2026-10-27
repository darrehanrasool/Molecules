# Installation & Environment Setup

## Prerequisites
- macOS, Linux, or Windows (WSL2 recommended)
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14)
- (Optional) CUDA-capable GPU or Apple Silicon (Metal/MPS)

## Quick Installation

```bash
# 1. Clone repository
git clone https://github.com/darrehanrasool/Molecules.git
cd Molecules

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Install local package in editable mode
pip install -e .

# 5. Run test suite
pytest tests/ -v
```

## Conda Environment Alternative (with RDKit)

```bash
conda create -n molecules python=3.11 -y
conda activate molecules
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
pip install -e .
```
