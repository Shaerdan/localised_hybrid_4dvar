"""
Analysis module for fair comparison of localized vs non-localized Hybrid 4D-Var.

Contains:
1. Unpreconditioned Hessian analysis (Option A)
2. Spectral spread metrics (Option B)
"""
import numpy as np
from scipy.sparse.linalg import cg
from scipy.linalg import eigvalsh
from dataclasses import dataclass
from typing import Tuple, Dict, List
import time

from core import (
    l96_step, l96_integrate, compute_tlm_trajectory,
    soar_correlation, localisation_matrix, ensemble_covariance, matrix_sqrt,
    obs_operator, build_cvt_matrix, build_hessian
)


# =============================================================================
# OPTION A: UNPRECONDITIONED HESSIAN ANALYSIS
# =============================================================================

def build_K_matrix(H: np.ndarray, R_inv: np.ndarray, TLM: np.ndarray, 
                   obs_times: np.ndarray) -> np.ndarray:
    """Build observation contribution K = Σ_j M_j^T H^T R^{-1} H M_j."""
    n = TLM.shape[1]
    K = np.zeros((n, n))
    for t_idx in obs_times:
        HM = H @ TLM[t_idx]
        K += HM.T @ R_inv @ HM
    return K


def build_unpreconditioned_hessian(Bc: np.ndarray, Pf: np.ndarray, 
                                    K: np.ndarray, beta: float,
                                    rho: np.ndarray = None) -> np.ndarray:
    """
    Build unpreconditioned Hessian A = B^{-1} + K.
    
    B = β*Bc + (1-β)*Pf  (without localization)
    B = β*Bc + (1-β)*(ρ∘Pf)  (with localization)
    
    Returns n×n matrix regardless of localization choice.
    """
    if rho is None:
        Pf_used = Pf
    else:
        Pf_used = rho * Pf  # Schur product
    
    B = beta * Bc + (1 - beta) * Pf_used
    B_inv = np.linalg.inv(B)
    A = B_inv + K
    return A


@dataclass
class UnpreconditionedResult:
    """Results from unpreconditioned analysis."""
    beta: float
    L_cut: float
    kappa: float
    iterations: int
    eigenvalues: np.ndarray
    lambda_max: float
    lambda_min: float


def analyze_unpreconditioned(Bc: np.ndarray, Pf: np.ndarray, K: np.ndarray,
                              beta: float, L_cut: float = None,
                              rng: np.random.Generator = None) -> UnpreconditionedResult:
    """
    Analyze unpreconditioned Hessian for given parameters.
    
    Args:
        Bc: Climatological covariance
        Pf: Ensemble covariance (un-localized)
        K: Observation operator contribution
        beta: Hybrid weight
        L_cut: Localization length (None or 0 for no localization)
        rng: Random generator for RHS vector
    """
    n = Bc.shape[0]
    
    if L_cut is None or L_cut == 0:
        rho = None
    else:
        rho = localisation_matrix(n, L_cut)
    
    A = build_unpreconditioned_hessian(Bc, Pf, K, beta, rho)
    
    # Eigenvalue analysis
    eigs = eigvalsh(A)
    eigs = np.sort(eigs)[::-1]
    kappa = eigs[0] / eigs[-1]
    
    # CG iterations
    if rng is None:
        rng = np.random.default_rng(42)
    b = rng.standard_normal(n)
    
    iters = [0]
    def callback(xk):
        iters[0] += 1
    
    cg(A, b, rtol=1e-10, atol=0, maxiter=500, callback=callback)
    
    return UnpreconditionedResult(
        beta=beta,
        L_cut=L_cut if L_cut else 0,
        kappa=kappa,
        iterations=iters[0],
        eigenvalues=eigs,
        lambda_max=eigs[0],
        lambda_min=eigs[-1]
    )


def run_unpreconditioned_sweep(Bc: np.ndarray, Pf: np.ndarray, K: np.ndarray,
                                beta_values: np.ndarray, 
                                L_cut_values: np.ndarray) -> Dict:
    """
    Run parameter sweep for unpreconditioned analysis.
    
    Returns dict with results for each (L_cut, beta) combination.
    """
    results = {
        'beta': beta_values,
        'L_cut': L_cut_values,
        'kappa': {},      # L_cut -> array of kappa values
        'iterations': {}, # L_cut -> array of iteration counts
        'eigenvalues': {} # L_cut -> [n_beta, n] array of eigenvalues
    }
    
    rng = np.random.default_rng(42)
    
    for L_cut in L_cut_values:
        kappas = []
        iters = []
        all_eigs = []
        
        for beta in beta_values:
            r = analyze_unpreconditioned(Bc, Pf, K, beta, L_cut, rng)
            kappas.append(r.kappa)
            iters.append(r.iterations)
            all_eigs.append(r.eigenvalues)
        
        results['kappa'][L_cut] = np.array(kappas)
        results['iterations'][L_cut] = np.array(iters)
        results['eigenvalues'][L_cut] = np.array(all_eigs)
    
    return results


# =============================================================================
# OPTION B: SPECTRAL SPREAD METRICS
# =============================================================================

def compute_spectral_spread_metrics(eigs: np.ndarray) -> Dict[str, float]:
    """
    Compute various spectral spread metrics beyond condition number.
    
    Args:
        eigs: Eigenvalues (sorted descending)
    
    Returns:
        Dictionary of metrics
    """
    eigs = np.sort(eigs)[::-1]
    n = len(eigs)
    
    # Basic metrics
    kappa = eigs[0] / eigs[-1]
    lambda_max = eigs[0]
    lambda_min = eigs[-1]
    
    # Metric 1: Number of "active" eigenvalues (significantly different from 1)
    # For preconditioned systems where λ_min ≈ 1
    n_active_01 = np.sum(np.abs(eigs - 1) > 0.1)  # 10% threshold
    n_active_05 = np.sum(np.abs(eigs - 1) > 0.5)  # 50% threshold
    
    # Metric 2: Effective spectral dimension (using entropy)
    # Normalized eigenvalues as probabilities
    eigs_pos = np.maximum(eigs, 1e-15)
    p = eigs_pos / eigs_pos.sum()
    entropy = -np.sum(p * np.log(p))
    n_eff_entropy = np.exp(entropy)
    
    # Metric 3: Number of eigenvalue clusters (gaps > 20%)
    n_clusters = 1
    for i in range(1, n):
        if eigs[i-1] / eigs[i] > 1.2:
            n_clusters += 1
    
    # Metric 4: Spectral spread ratio (interquartile range in log scale)
    log_eigs = np.log10(eigs_pos)
    q75, q25 = np.percentile(log_eigs, [75, 25])
    iqr_log = q75 - q25
    
    # Metric 5: Eigenvalues above threshold (for preconditioned systems)
    n_above_10 = np.sum(eigs > 10)
    n_above_100 = np.sum(eigs > 100)
    
    # Metric 6: Modified condition number using percentiles
    # More robust to outliers
    kappa_90 = np.percentile(eigs, 95) / np.percentile(eigs, 5)
    
    # Metric 7: Spectral gap ratio (largest gap in spectrum)
    ratios = eigs[:-1] / eigs[1:]
    max_gap_ratio = np.max(ratios)
    max_gap_idx = np.argmax(ratios)
    
    # Metric 8: CG-relevant metric - effective number of distinct eigenvalues
    # Based on clustering with relative tolerance
    def count_distinct(eigs, rtol=0.1):
        distinct = [eigs[0]]
        for e in eigs[1:]:
            if all(abs(e - d) / max(abs(d), 1e-10) > rtol for d in distinct):
                distinct.append(e)
        return len(distinct)
    
    n_distinct = count_distinct(eigs, rtol=0.1)
    
    return {
        'kappa': kappa,
        'lambda_max': lambda_max,
        'lambda_min': lambda_min,
        'n_active_01': n_active_01,
        'n_active_05': n_active_05,
        'n_eff_entropy': n_eff_entropy,
        'n_clusters': n_clusters,
        'iqr_log': iqr_log,
        'n_above_10': n_above_10,
        'n_above_100': n_above_100,
        'kappa_90': kappa_90,
        'max_gap_ratio': max_gap_ratio,
        'max_gap_idx': max_gap_idx,
        'n_distinct': n_distinct
    }


@dataclass
class SpectralAnalysisResult:
    """Complete spectral analysis result."""
    beta: float
    L_cut: float
    dim: int
    iterations: int
    metrics: Dict[str, float]
    eigenvalues: np.ndarray


def analyze_preconditioned_spectral(Bc_sq: np.ndarray, Xpert: np.ndarray,
                                     H: np.ndarray, R_inv: np.ndarray,
                                     TLM: np.ndarray, obs_times: np.ndarray,
                                     beta: float, L_cut: float,
                                     method: str = 'direct') -> SpectralAnalysisResult:
    """
    Full spectral analysis of preconditioned Hessian S.
    """
    n = Bc_sq.shape[0]
    
    if L_cut == 0:
        rho = np.ones((n, n))
    else:
        rho = localisation_matrix(n, L_cut)
    
    U_h = build_cvt_matrix(Bc_sq, Xpert, rho, beta, method=method)
    S = build_hessian(U_h, H, R_inv, TLM, obs_times)
    
    eigs = eigvalsh(S)
    eigs = np.sort(eigs)[::-1]
    
    metrics = compute_spectral_spread_metrics(eigs)
    
    # CG iterations
    b = np.random.randn(S.shape[0])
    iters = [0]
    def callback(xk):
        iters[0] += 1
    cg(S, b, rtol=1e-10, atol=0, maxiter=500, callback=callback)
    
    return SpectralAnalysisResult(
        beta=beta,
        L_cut=L_cut,
        dim=S.shape[0],
        iterations=iters[0],
        metrics=metrics,
        eigenvalues=eigs
    )


def find_best_iteration_predictor(results: List[SpectralAnalysisResult]) -> Dict:
    """
    Find which spectral metric best predicts CG iterations.
    
    Returns correlation coefficients for each metric.
    """
    iterations = np.array([r.iterations for r in results])
    
    metric_names = list(results[0].metrics.keys())
    correlations = {}
    
    for name in metric_names:
        values = np.array([r.metrics[name] for r in results])
        # Handle potential NaN/inf
        mask = np.isfinite(values) & np.isfinite(iterations)
        if mask.sum() > 2:
            corr = np.corrcoef(values[mask], iterations[mask])[0, 1]
            correlations[name] = corr
        else:
            correlations[name] = np.nan
    
    return correlations


def run_spectral_sweep(Bc_sq: np.ndarray, Xpert: np.ndarray,
                        H: np.ndarray, R_inv: np.ndarray,
                        TLM: np.ndarray, obs_times: np.ndarray,
                        beta_values: np.ndarray, 
                        L_cut_values: np.ndarray) -> List[SpectralAnalysisResult]:
    """Run full spectral analysis sweep."""
    results = []
    
    for L_cut in L_cut_values:
        for beta in beta_values:
            r = analyze_preconditioned_spectral(
                Bc_sq, Xpert, H, R_inv, TLM, obs_times, beta, L_cut
            )
            results.append(r)
    
    return results


# =============================================================================
# THEORETICAL BOUNDS FOR UNPRECONDITIONED CASE
# =============================================================================

def bound_kappa_unpreconditioned(Bc: np.ndarray, Pf_loc: np.ndarray,
                                  K: np.ndarray, beta: float) -> Dict[str, float]:
    """
    Compute bounds on κ(A) for unpreconditioned Hessian A = B^{-1} + K.
    
    B = β*Bc + (1-β)*Pf_loc
    
    Uses eigenvalue bounds:
    λ_max(A) ≤ λ_max(B^{-1}) + λ_max(K) = 1/λ_min(B) + λ_max(K)
    λ_min(A) ≥ λ_min(B^{-1}) + λ_min(K) = 1/λ_max(B) + λ_min(K)
    """
    # Eigenvalues of B = β*Bc + (1-β)*Pf_loc
    # Using Weyl: λ_i(β*Bc + (1-β)*Pf_loc) ≥ β*λ_n(Bc) + (1-β)*λ_n(Pf_loc)
    
    eigs_Bc = eigvalsh(Bc)
    eigs_Pf = eigvalsh(Pf_loc)
    eigs_K = eigvalsh(K)
    
    lambda_max_Bc = eigs_Bc.max()
    lambda_min_Bc = eigs_Bc.min()
    lambda_max_Pf = eigs_Pf.max()
    lambda_min_Pf = eigs_Pf.min()
    lambda_max_K = eigs_K.max()
    lambda_min_K = max(eigs_K.min(), 0)  # K is positive semidefinite
    
    # Bounds on B
    lambda_max_B_upper = beta * lambda_max_Bc + (1 - beta) * lambda_max_Pf
    lambda_min_B_lower = beta * lambda_min_Bc + (1 - beta) * lambda_min_Pf
    
    # Bounds on B^{-1}
    lambda_max_Binv_upper = 1.0 / lambda_min_B_lower if lambda_min_B_lower > 0 else np.inf
    lambda_min_Binv_lower = 1.0 / lambda_max_B_upper
    
    # Bounds on A = B^{-1} + K
    lambda_max_A_upper = lambda_max_Binv_upper + lambda_max_K
    lambda_min_A_lower = lambda_min_Binv_lower + lambda_min_K
    
    # Condition number bound
    kappa_upper = lambda_max_A_upper / lambda_min_A_lower if lambda_min_A_lower > 0 else np.inf
    
    return {
        'lambda_max_A_upper': lambda_max_A_upper,
        'lambda_min_A_lower': lambda_min_A_lower,
        'kappa_upper': kappa_upper,
        'lambda_max_B_upper': lambda_max_B_upper,
        'lambda_min_B_lower': lambda_min_B_lower,
        'lambda_max_K': lambda_max_K,
        'lambda_min_K': lambda_min_K
    }


# =============================================================================
# ITERATION PREDICTION MODEL
# =============================================================================

def predict_iterations_from_spectrum(eigs: np.ndarray, tol: float = 1e-10) -> Dict[str, float]:
    """
    Predict CG iterations using different models.
    
    Model 1: Classical bound k ≈ 0.5 * sqrt(κ) * log(2/tol)
    Model 2: Cluster-based: k ≈ n_clusters * log(κ)
    Model 3: Active eigenvalue count
    """
    eigs = np.sort(eigs)[::-1]
    kappa = eigs[0] / eigs[-1]
    
    # Model 1: Classical
    k_classical = 0.5 * np.sqrt(kappa) * np.log(2 / tol)
    
    # Model 2: Cluster-based
    n_clusters = 1
    for i in range(1, len(eigs)):
        if eigs[i-1] / eigs[i] > 1.2:
            n_clusters += 1
    k_cluster = n_clusters * np.log(kappa)
    
    # Model 3: Active eigenvalues (simplified)
    n_active = np.sum(np.abs(eigs - 1) > 0.1)
    k_active = n_active * np.log10(kappa)
    
    # Model 4: Distinct eigenvalues
    def count_distinct(eigs, rtol=0.1):
        distinct = [eigs[0]]
        for e in eigs[1:]:
            if all(abs(e - d) / max(abs(d), 1e-10) > rtol for d in distinct):
                distinct.append(e)
        return len(distinct)
    n_distinct = count_distinct(eigs)
    k_distinct = n_distinct
    
    return {
        'classical': k_classical,
        'cluster': k_cluster,
        'active': k_active,
        'distinct': k_distinct,
        'n_clusters': n_clusters,
        'n_active': n_active,
        'n_distinct': n_distinct,
        'kappa': kappa
    }
