# Temporal Decay for Knowledge Graph Retrieval: Replication Package

This package contains all code and data needed to reproduce the experiments from our paper on hierarchical Bayesian temporal decay models for knowledge graph retrieval.

## Repository Structure

```
replication/
├── src/                    # Source code
│   ├── models/            
│   │   ├── bayesian/       # Survival analysis models (Weibull, Log-Normal)
│   │   └── gradient/       # Gradient-based decay learning
│   ├── clustering/         # Temporal pattern clustering (HDBSCAN)
│   ├── retrieval/          # Temporal-aware retrieval scoring
│   ├── wikipedia/          # Wikipedia revision extraction pipeline
│   ├── synthetic/          # Synthetic data generator
│   ├── synthea/            # Clinical data pipeline
│   ├── evaluation/         # Experiment runners and figure generation
│   └── embeddings/         # Text embedding utilities
├── data/
│   ├── wikipedia/          # Wikipedia article revisions (JSON + Parquet)
│   └── synthea/            # Synthetic clinical records
├── tests/                  # Unit tests
├── results/                # Experiment outputs (JSON)
└── requirements.txt        # Python dependencies
```

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** JAX installation may require platform-specific steps. See https://jax.readthedocs.io/en/latest/installation.html

## Running Experiments

### 1. Synthetic Data Experiment

Tests temporal decay model fitting on controlled synthetic data with known ground truth:

```bash
python -m src.evaluation.run_synthetic
```

Output: `results/synthetic_experiment.json`

### 2. Wikipedia Experiment

Validates temporal decay patterns on real Wikipedia revision histories:

```bash
python -m src.evaluation.run_wikipedia
```

Output: `results/wikipedia_experiment.json`

### 3. Sensitivity Analysis

Evaluates robustness across hyperparameter ranges:

```bash
python -m src.evaluation.sensitivity_analysis
```

Output: `results/sensitivity_analysis.json`

### 4. Generate Figures

Reproduces all paper figures from experiment results:

```bash
python -m src.evaluation.generate_figures
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Results to Verify

After running experiments, check `results/` JSON files for:

1. **Log-Normal vs Weibull**: Log-Normal should show better fit (lower AIC/BIC)
2. **Lindy Effect**: Older stable facts should have lower decay rates
3. **Cluster Separation**: HDBSCAN should identify 3-4 distinct temporal patterns
4. **Retrieval Improvement**: Temporal scoring should improve NDCG@10 over static baselines

## Citation

```bibtex
@misc{temporal-decay-kg,
  title={Hierarchical Bayesian Temporal Decay for Knowledge Graph Retrieval},
  author={Karhade, Mandar},
  email={mandar.karhade@citingale.com},
  year={2026},
  url={https://arxiv.org/abs/2604.26970},
  eprint={2604.26970},
  archivePrefix={arXiv}
}
```
