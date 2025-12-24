# Hybrid 4D-Var Conditioning Analysis

Code to reproduce figures from "Conditioning and Convergence of Localised Hybrid 4D-Var"

## Files

### Main Scripts
- `generate_paper_figures.py` - Main script to generate all paper figures
- `core.py` - Core implementation of Hybrid 4D-Var (L96 model, CVT, bounds)
- `analysis.py` - Analysis utilities (K matrix construction, metrics)

### Output Figures
- `fig_bounds_comparison.pdf` - Theorem 4 vs Theorem 6 bounds comparison
- `fig_cpu_time.pdf` - Computational cost comparison  
- `fig_preconditioned_analysis.pdf` - 4-panel preconditioned Hessian analysis
- `fig_unpreconditioned_analysis.pdf` - 4-panel unpreconditioned Hessian analysis
- `fig_spectral_analysis.pdf` - 6-panel spectral spread metrics
- `fig_eigenvalue_clustering.pdf` - Eigenvalue clustering analysis

## Requirements

```
numpy
scipy
matplotlib
```

## Usage

```bash
python generate_paper_figures.py
```

This will generate all figures in the `figures/` directory.

## Configuration

Edit the `CONFIG` dictionary in `generate_paper_figures.py` to change:
- `n`: State dimension (default: 40)
- `M`: Ensemble size (default: 10)
- `L_Bc`: Climatological correlation length
- `L_cut_values`: Localisation length scales to test
- `beta_values`: Hybrid weights to test
- `seed`: Random seed for reproducibility

## Citation

[Add paper citation here]

## License

[Add license here]
