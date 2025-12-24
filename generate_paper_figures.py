"""
Plotting scripts for "Conditioning and Convergence of Localised Hybrid 4D-Var"

This module generates all figures for the Results section of the paper.

Figures generated:
    - fig_bounds_comparison.pdf: Theorem 4 vs Theorem 6 bounds comparison
    - fig_cpu_time.pdf: Computational cost comparison
    - fig_preconditioned_analysis.pdf: 4-panel preconditioned Hessian analysis
    - fig_unpreconditioned_analysis.pdf: 4-panel unpreconditioned Hessian analysis
    - fig_spectral_analysis.pdf: 6-panel spectral spread metrics
    - fig_eigenvalue_clustering.pdf: Eigenvalue clustering analysis

Usage:
    python generate_paper_figures.py

Requirements:
    - numpy
    - scipy
    - matplotlib
    - core.py (Hybrid 4D-Var implementation)
    - analysis.py (Analysis utilities)

Author: [Your Name]
Date: December 2024
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh, solve
from scipy.sparse.linalg import cg
import time
import os

# Import from core module
from core import (
    l96_step, l96_integrate, compute_tlm_trajectory,
    soar_correlation, localisation_matrix, ensemble_covariance, matrix_sqrt,
    obs_operator, build_cvt_matrix, build_hessian,
    bound_theorem4, bound_theorem6
)
from analysis import build_K_matrix


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'n': 40,                    # State dimension
    'M': 10,                    # Ensemble size
    'dt': 0.025,                # Time step
    'L_Bc': 0.5,                # Climatological correlation length
    'L_cut_values': [0, 0.1, 0.25, 0.5],  # Localisation length scales
    'beta_values': np.linspace(0.1, 0.9, 17),  # Hybrid weights
    'obs_fraction': 0.5,        # Fraction of state observed
    'sigma_obs': 0.1,           # Observation error std dev
    'nsteps': 4,                # Number of time steps
    'obs_times': np.array([2, 4]),  # Observation times
    'seed': 42,                 # Random seed for reproducibility
    'output_dir': 'figures',    # Output directory
    'dpi': 150,                 # Figure resolution
}


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_experiment(config):
    """Set up the experimental configuration."""
    np.random.seed(config['seed'])
    
    n = config['n']
    M = config['M']
    dt = config['dt']
    
    # Initial condition and trajectory
    u0 = 8 * np.ones(n)
    u0[0] += 5
    for _ in range(100):
        u0 = l96_step(u0, dt)
    
    x_guess = l96_integrate(u0, config['nsteps'], dt)[config['nsteps']//2]
    
    # Observation operator
    p = int(n * config['obs_fraction'])
    H = obs_operator(n, p, 'every_other')
    R_inv = np.eye(p) / config['sigma_obs']**2
    
    # Tangent linear model
    _, TLM = compute_tlm_trajectory(x_guess, config['nsteps'], dt)
    
    # Covariance matrices
    Bc = soar_correlation(n, config['L_Bc'])
    Bc_sq = matrix_sqrt(Bc)
    X_ens = Bc_sq @ np.random.randn(n, M)
    Pf, Xpert = ensemble_covariance(X_ens)
    
    # Observation information matrix
    K = build_K_matrix(H, R_inv, TLM, config['obs_times'])
    
    return {
        'n': n, 'M': M, 'dt': dt,
        'x_guess': x_guess,
        'H': H, 'R_inv': R_inv, 'TLM': TLM,
        'Bc': Bc, 'Bc_sq': Bc_sq,
        'Pf': Pf, 'Xpert': Xpert, 'K': K,
        'obs_times': config['obs_times'],
    }


def compute_cg_iterations(A, b, tol=1e-10, maxiter=1000):
    """Count CG iterations to solve Ax = b."""
    iter_count = [0]
    def callback(xk):
        iter_count[0] += 1
    # Use rtol for newer scipy versions, fall back to tol for older versions
    try:
        _, info = cg(A, b, rtol=tol, maxiter=maxiter, callback=callback)
    except TypeError:
        _, info = cg(A, b, tol=tol, maxiter=maxiter, callback=callback)
    return iter_count[0]


# =============================================================================
# FIGURE 1: BOUNDS COMPARISON
# =============================================================================

def generate_bounds_comparison(setup, config, output_dir):
    """Generate Theorem 4 vs Theorem 6 bounds comparison figure."""
    print("Generating bounds comparison figure...")
    
    # Use specific parameters for clearer gap
    np.random.seed(123)
    n, M = config['n'], config['M']
    
    # Recompute with short correlation length for better visual separation
    L_Bc_plot = 0.15
    Bc = soar_correlation(n, L_Bc_plot)
    Bc_sq = matrix_sqrt(Bc)
    X_ens = Bc_sq @ np.random.randn(n, M)
    Pf, Xpert = ensemble_covariance(X_ens)
    
    L_cut = 0.25
    rho = localisation_matrix(n, L_cut)
    Pf_loc = rho * Pf
    
    beta_values = config['beta_values']
    
    kappas = []
    bound4s = []
    bound6s = []
    
    for beta in beta_values:
        U_h = build_cvt_matrix(Bc_sq, Xpert, rho, beta, method='direct')
        S = build_hessian(U_h, setup['H'], setup['R_inv'], setup['TLM'], setup['obs_times'])
        eigs = eigvalsh(S)
        kappas.append(eigs.max() / eigs.min())
        bound4s.append(bound_theorem4(Bc, setup['K'], Pf_loc, beta))
        bound6s.append(bound_theorem6(Bc, setup['K'], Pf_loc, beta))
    
    kappas = np.array(kappas)
    bound4s = np.array(bound4s)
    bound6s = np.array(bound6s)
    
    # Set y-axis limits
    y_min = 300
    y_max = 4000
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel (a): Theorem 4
    ax = axes[0]
    ax.semilogy(beta_values, kappas, 'k-', linewidth=2.5, label=r'$\kappa(S)$ actual')
    ax.semilogy(beta_values, bound4s, 'b--', linewidth=2.5, label='Theorem 4 bound')
    ax.set_xlabel(r'$\beta$', fontsize=14)
    ax.set_ylabel('Condition number', fontsize=12)
    ax.set_title(r'Theorem 4 bound ($L_{cut} = 0.25$)', fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([y_min, y_max])
    
    # Panel (b): Theorem 6
    ax = axes[1]
    ax.semilogy(beta_values, kappas, 'k-', linewidth=2.5, label=r'$\kappa(S)$ actual')
    ax.semilogy(beta_values, bound6s, 'r--', linewidth=2.5, label='Theorem 6 bound')
    ax.set_xlabel(r'$\beta$', fontsize=14)
    ax.set_ylabel('Condition number', fontsize=12)
    ax.set_title(r'Theorem 6 bound ($L_{cut} = 0.25$)', fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([y_min, y_max])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_bounds_comparison.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig_bounds_comparison.png'), dpi=config['dpi'])
    plt.close()
    
    print(f"  Gap (Bound6/Bound4): {bound6s.mean()/bound4s.mean():.2f}x")


# =============================================================================
# FIGURE 2: CPU TIME COMPARISON
# =============================================================================

def generate_cpu_time_figure(setup, config, output_dir):
    """Generate CPU time comparison figure."""
    print("Generating CPU time figure...")
    
    L_cut = 0.25
    rho = localisation_matrix(config['n'], L_cut)
    Pf_loc = rho * setup['Pf']
    
    beta_values = np.linspace(0.05, 0.95, 30)
    
    times_thm4 = []
    times_thm6 = []
    
    for beta in beta_values:
        # Theorem 4 (eigenvalue-based)
        t0 = time.perf_counter()
        for _ in range(10):
            _ = bound_theorem4(setup['Bc'], setup['K'], Pf_loc, beta)
        times_thm4.append((time.perf_counter() - t0) / 10)
        
        # Theorem 6 (infinity norm)
        t0 = time.perf_counter()
        for _ in range(10):
            _ = bound_theorem6(setup['Bc'], setup['K'], Pf_loc, beta)
        times_thm6.append((time.perf_counter() - t0) / 10)
    
    times_thm4 = np.array(times_thm4)
    times_thm6 = np.array(times_thm6)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.semilogy(beta_values, times_thm4, 'o-', color='#1f77b4', linewidth=2, markersize=6,
                label='Theorem 4 (eigenvalue-based)')
    ax.semilogy(beta_values, times_thm6, 's-', color='#d62728', linewidth=2, markersize=6,
                label='Theorem 6 (infinity norm)')
    
    ax.set_xlabel(r'$\beta$', fontsize=14)
    ax.set_ylabel('CPU time (seconds)', fontsize=14)
    ax.set_title('Computational Cost of Bound Evaluation', fontsize=13)
    ax.legend(loc='center right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    avg_ratio = np.mean(times_thm4 / times_thm6)
    ax.text(0.05, 0.15, f'Average speedup: {avg_ratio:.0f}x',
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_cpu_time.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig_cpu_time.png'), dpi=config['dpi'])
    plt.close()
    
    print(f"  Average speedup: {avg_ratio:.0f}x")


# =============================================================================
# FIGURE 3: PRECONDITIONED HESSIAN ANALYSIS
# =============================================================================

def generate_preconditioned_analysis(setup, config, output_dir):
    """Generate 4-panel preconditioned Hessian analysis figure."""
    print("Generating preconditioned analysis figure...")
    
    beta_values = config['beta_values']
    L_cut_values = [0, 0.25, 0.5]
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    markers = ['o', 's', '^']
    labels = ['No localisation', r'$L_{cut}=0.25$', r'$L_{cut}=0.5$']
    
    results = {}
    
    for L_cut in L_cut_values:
        if L_cut == 0:
            rho = np.ones((config['n'], config['n']))
        else:
            rho = localisation_matrix(config['n'], L_cut)
        
        kappas = []
        iterations = []
        all_eigs = []
        
        for beta in beta_values:
            U_h = build_cvt_matrix(setup['Bc_sq'], setup['Xpert'], rho, beta, method='direct')
            S = build_hessian(U_h, setup['H'], setup['R_inv'], setup['TLM'], setup['obs_times'])
            eigs = eigvalsh(S)
            kappas.append(eigs.max() / eigs.min())
            
            # CG iterations
            b = np.random.randn(S.shape[0])
            n_iter = compute_cg_iterations(S, b)
            iterations.append(n_iter)
            all_eigs.append(eigs)
        
        results[L_cut] = {
            'kappa': np.array(kappas),
            'iterations': np.array(iterations),
            'eigs_mid': all_eigs[len(beta_values)//2]
        }
    
    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel (a): Condition number vs beta
    ax = axes[0, 0]
    for i, L_cut in enumerate(L_cut_values):
        ax.semilogy(beta_values, results[L_cut]['kappa'],
                    color=colors[i], marker=markers[i], markevery=3,
                    linewidth=2, markersize=7, label=labels[i])
    ax.set_xlabel(r'$\beta$', fontsize=12)
    ax.set_ylabel(r'$\kappa(S)$', fontsize=12)
    ax.set_title('(a) Condition number vs $\\beta$', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel (b): Iterations vs beta
    ax = axes[0, 1]
    for i, L_cut in enumerate(L_cut_values):
        ax.plot(beta_values, results[L_cut]['iterations'],
                color=colors[i], marker=markers[i], markevery=3,
                linewidth=2, markersize=7, label=labels[i])
    ax.set_xlabel(r'$\beta$', fontsize=12)
    ax.set_ylabel('CG iterations', fontsize=12)
    ax.set_title('(b) CG iterations vs $\\beta$', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel (c): Scatter plot
    ax = axes[1, 0]
    all_kappa = []
    all_iter = []
    for i, L_cut in enumerate(L_cut_values):
        ax.scatter(results[L_cut]['kappa'], results[L_cut]['iterations'],
                   color=colors[i], marker=markers[i], s=60, label=labels[i], alpha=0.7)
        all_kappa.extend(results[L_cut]['kappa'])
        all_iter.extend(results[L_cut]['iterations'])
    
    corr = np.corrcoef(all_kappa, all_iter)[0, 1]
    ax.set_xlabel(r'$\kappa(S)$', fontsize=12)
    ax.set_ylabel('CG iterations', fontsize=12)
    ax.set_title(f'(c) Iterations vs $\\kappa(S)$ (r = {corr:.2f})', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel (d): Eigenvalue distribution
    ax = axes[1, 1]
    for i, L_cut in enumerate(L_cut_values):
        eigs = results[L_cut]['eigs_mid']
        ax.semilogy(range(1, len(eigs)+1), np.sort(eigs)[::-1],
                    color=colors[i], marker=markers[i], markevery=max(1, len(eigs)//10),
                    linewidth=2, markersize=5, label=labels[i])
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title(r'(d) Eigenvalue distribution ($\beta = 0.5$)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_preconditioned_analysis.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig_preconditioned_analysis.png'), dpi=config['dpi'])
    plt.close()
    
    print(f"  Correlation kappa vs iterations: {corr:.2f}")


# =============================================================================
# FIGURE 4: UNPRECONDITIONED HESSIAN ANALYSIS
# =============================================================================

def generate_unpreconditioned_analysis(setup, config, output_dir):
    """Generate 4-panel unpreconditioned Hessian analysis figure."""
    print("Generating unpreconditioned analysis figure...")
    
    beta_values = config['beta_values']
    L_cut_values = [0, 0.1, 0.25, 0.5]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 'd', 's', '^']
    labels = ['No loc', r'$L_{cut}=0.1$', r'$L_{cut}=0.25$', r'$L_{cut}=0.5$']
    
    n = config['n']
    results = {}
    
    for L_cut in L_cut_values:
        if L_cut == 0:
            rho = np.ones((n, n))
        else:
            rho = localisation_matrix(n, L_cut)
        Pf_loc = rho * setup['Pf']
        
        kappas = []
        iterations = []
        all_eigs = []
        
        for beta in beta_values:
            # Unpreconditioned Hessian A = B^{-1} + K
            B = beta * Pf_loc + (1 - beta) * setup['Bc']
            B_inv = np.linalg.inv(B)
            A = B_inv + setup['K']
            
            eigs = eigvalsh(A)
            kappas.append(eigs.max() / eigs.min())
            
            # CG iterations
            b = np.random.randn(n)
            n_iter = compute_cg_iterations(A, b)
            iterations.append(n_iter)
            all_eigs.append(eigs)
        
        results[L_cut] = {
            'kappa': np.array(kappas),
            'iterations': np.array(iterations),
            'eigs_mid': all_eigs[len(beta_values)//2]
        }
    
    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel (a): Condition number vs beta
    ax = axes[0, 0]
    for i, L_cut in enumerate(L_cut_values):
        ax.semilogy(beta_values, results[L_cut]['kappa'],
                    color=colors[i], marker=markers[i], markevery=3,
                    linewidth=2, markersize=7, label=labels[i])
    ax.set_xlabel(r'$\beta$', fontsize=12)
    ax.set_ylabel(r'$\kappa(A)$', fontsize=12)
    ax.set_title('(a) Condition number vs $\\beta$', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel (b): Iterations vs beta
    ax = axes[0, 1]
    for i, L_cut in enumerate(L_cut_values):
        ax.plot(beta_values, results[L_cut]['iterations'],
                color=colors[i], marker=markers[i], markevery=3,
                linewidth=2, markersize=7, label=labels[i])
    ax.set_xlabel(r'$\beta$', fontsize=12)
    ax.set_ylabel('CG iterations', fontsize=12)
    ax.set_title('(b) CG iterations vs $\\beta$', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel (c): Scatter plot
    ax = axes[1, 0]
    all_kappa = []
    all_iter = []
    for i, L_cut in enumerate(L_cut_values):
        ax.scatter(results[L_cut]['kappa'], results[L_cut]['iterations'],
                   color=colors[i], marker=markers[i], s=60, label=labels[i], alpha=0.7)
        all_kappa.extend(results[L_cut]['kappa'])
        all_iter.extend(results[L_cut]['iterations'])
    
    corr = np.corrcoef(all_kappa, all_iter)[0, 1]
    ax.set_xlabel(r'$\kappa(A)$', fontsize=12)
    ax.set_ylabel('CG iterations', fontsize=12)
    ax.set_title(f'(c) Iterations vs $\\kappa(A)$ (r = {corr:.2f})', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel (d): Eigenvalue distribution
    ax = axes[1, 1]
    for i, L_cut in enumerate(L_cut_values):
        eigs = results[L_cut]['eigs_mid']
        ax.semilogy(range(1, len(eigs)+1), np.sort(eigs)[::-1],
                    color=colors[i], marker=markers[i], markevery=max(1, len(eigs)//10),
                    linewidth=2, markersize=5, label=labels[i])
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title(r'(d) Eigenvalue distribution ($\beta = 0.5$)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_unpreconditioned_analysis.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig_unpreconditioned_analysis.png'), dpi=config['dpi'])
    plt.close()
    
    print(f"  Correlation kappa vs iterations: {corr:.2f}")


# =============================================================================
# FIGURE 5: SPECTRAL SPREAD METRICS
# =============================================================================

def compute_spectral_metrics(eigs, delta=0.5):
    """Compute spectral spread metrics."""
    n_active = np.sum(np.abs(eigs - 1) > delta)
    
    # Distinct eigenvalues (within tolerance)
    sorted_eigs = np.sort(eigs)
    tol = 0.01 * (sorted_eigs.max() - sorted_eigs.min())
    n_distinct = 1
    for i in range(1, len(sorted_eigs)):
        if sorted_eigs[i] - sorted_eigs[i-1] > tol:
            n_distinct += 1
    
    # Clusters (gaps > 10% of range)
    gap_threshold = 0.1 * (sorted_eigs.max() - sorted_eigs.min())
    n_clusters = 1
    for i in range(1, len(sorted_eigs)):
        if sorted_eigs[i] - sorted_eigs[i-1] > gap_threshold:
            n_clusters += 1
    
    return n_active, n_distinct, n_clusters


def generate_spectral_analysis(setup, config, output_dir):
    """Generate 6-panel spectral spread analysis figure."""
    print("Generating spectral analysis figure...")
    
    beta_values = config['beta_values']
    L_cut_values = [0, 0.25, 0.5]
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    markers = ['o', 's', '^']
    labels = ['No loc', r'$L_{cut}=0.25$', r'$L_{cut}=0.5$']
    
    all_data = []
    
    for L_cut in L_cut_values:
        if L_cut == 0:
            rho = np.ones((config['n'], config['n']))
        else:
            rho = localisation_matrix(config['n'], L_cut)
        
        for beta in beta_values:
            U_h = build_cvt_matrix(setup['Bc_sq'], setup['Xpert'], rho, beta, method='direct')
            S = build_hessian(U_h, setup['H'], setup['R_inv'], setup['TLM'], setup['obs_times'])
            eigs = eigvalsh(S)
            
            kappa = eigs.max() / eigs.min()
            b = np.random.randn(S.shape[0])
            n_iter = compute_cg_iterations(S, b)
            n_active, n_distinct, n_clusters = compute_spectral_metrics(eigs)
            
            all_data.append({
                'L_cut': L_cut, 'beta': beta, 'kappa': kappa,
                'iterations': n_iter, 'n_active': n_active,
                'n_distinct': n_distinct, 'n_clusters': n_clusters,
                'eigs': eigs
            })
    
    # Extract arrays for plotting
    kappas = np.array([d['kappa'] for d in all_data])
    iterations = np.array([d['iterations'] for d in all_data])
    n_actives = np.array([d['n_active'] for d in all_data])
    n_distincts = np.array([d['n_distinct'] for d in all_data])
    n_clusterss = np.array([d['n_clusters'] for d in all_data])
    
    # Create 6-panel figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel (a): Iterations vs kappa
    ax = axes[0, 0]
    for i, L_cut in enumerate(L_cut_values):
        mask = [d['L_cut'] == L_cut for d in all_data]
        ax.scatter(kappas[mask], iterations[mask], color=colors[i],
                   marker=markers[i], s=60, label=labels[i], alpha=0.7)
    corr_kappa = np.corrcoef(kappas, iterations)[0, 1]
    ax.set_xlabel(r'$\kappa(S)$', fontsize=12)
    ax.set_ylabel('CG iterations', fontsize=12)
    ax.set_title(f'(a) Iterations vs $\\kappa$ (r={corr_kappa:.2f})', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel (b): Iterations vs n_distinct
    ax = axes[0, 1]
    for i, L_cut in enumerate(L_cut_values):
        mask = [d['L_cut'] == L_cut for d in all_data]
        ax.scatter(n_distincts[mask], iterations[mask], color=colors[i],
                   marker=markers[i], s=60, label=labels[i], alpha=0.7)
    corr_distinct = np.corrcoef(n_distincts, iterations)[0, 1]
    ax.set_xlabel(r'$n_{distinct}$', fontsize=12)
    ax.set_ylabel('CG iterations', fontsize=12)
    ax.set_title(f'(b) Iterations vs $n_{{distinct}}$ (r={corr_distinct:.2f})', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel (c): Iterations vs n_active
    ax = axes[0, 2]
    for i, L_cut in enumerate(L_cut_values):
        mask = [d['L_cut'] == L_cut for d in all_data]
        ax.scatter(n_actives[mask], iterations[mask], color=colors[i],
                   marker=markers[i], s=60, label=labels[i], alpha=0.7)
    corr_active = np.corrcoef(n_actives, iterations)[0, 1]
    ax.set_xlabel(r'$n_{active}$', fontsize=12)
    ax.set_ylabel('CG iterations', fontsize=12)
    ax.set_title(f'(c) Iterations vs $n_{{active}}$ (r={corr_active:.2f})', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel (d): Iterations vs n_clusters
    ax = axes[1, 0]
    for i, L_cut in enumerate(L_cut_values):
        mask = [d['L_cut'] == L_cut for d in all_data]
        ax.scatter(n_clusterss[mask], iterations[mask], color=colors[i],
                   marker=markers[i], s=60, label=labels[i], alpha=0.7)
    corr_clusters = np.corrcoef(n_clusterss, iterations)[0, 1]
    ax.set_xlabel(r'$n_{clusters}$', fontsize=12)
    ax.set_ylabel('CG iterations', fontsize=12)
    ax.set_title(f'(d) Iterations vs $n_{{clusters}}$ (r={corr_clusters:.2f})', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel (e): Prediction model
    ax = axes[1, 1]
    slope, intercept = np.polyfit(n_actives, iterations, 1)
    predicted = slope * n_actives + intercept
    for i, L_cut in enumerate(L_cut_values):
        mask = [d['L_cut'] == L_cut for d in all_data]
        ax.scatter(n_actives[mask], iterations[mask], color=colors[i],
                   marker=markers[i], s=60, label=labels[i], alpha=0.7)
    ax.plot([n_actives.min(), n_actives.max()],
            [slope*n_actives.min()+intercept, slope*n_actives.max()+intercept],
            'k--', linewidth=2, label=f'Fit: {slope:.2f}x + {intercept:.1f}')
    ax.set_xlabel(r'$n_{active}$', fontsize=12)
    ax.set_ylabel('CG iterations', fontsize=12)
    ax.set_title('(e) Linear prediction model', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel (f): Eigenvalue distributions
    ax = axes[1, 2]
    mid_idx = len(beta_values) // 2
    for i, L_cut in enumerate(L_cut_values):
        data_idx = i * len(beta_values) + mid_idx
        eigs = all_data[data_idx]['eigs']
        ax.semilogy(range(1, len(eigs)+1), np.sort(eigs)[::-1],
                    color=colors[i], marker=markers[i],
                    markevery=max(1, len(eigs)//10),
                    linewidth=2, markersize=5, label=labels[i])
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title(r'(f) Eigenvalue distribution ($\beta=0.5$)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_spectral_analysis.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig_spectral_analysis.png'), dpi=config['dpi'])
    plt.close()
    
    print(f"  Correlations: kappa={corr_kappa:.2f}, n_active={corr_active:.2f}, n_distinct={corr_distinct:.2f}")


# =============================================================================
# FIGURE 6: EIGENVALUE CLUSTERING
# =============================================================================

def generate_eigenvalue_clustering(setup, config, output_dir):
    """Generate eigenvalue clustering figure."""
    print("Generating eigenvalue clustering figure...")
    
    beta_values = config['beta_values']
    L_cut_values = [0, 0.25, 0.5]
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    markers = ['o', 's', '^']
    labels = ['No loc', r'$L_{cut}=0.25$', r'$L_{cut}=0.5$']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel (a): Eigenvalue distribution at beta=0.5
    ax = axes[0]
    beta_mid = 0.5
    
    for i, L_cut in enumerate(L_cut_values):
        if L_cut == 0:
            rho = np.ones((config['n'], config['n']))
        else:
            rho = localisation_matrix(config['n'], L_cut)
        
        U_h = build_cvt_matrix(setup['Bc_sq'], setup['Xpert'], rho, beta_mid, method='direct')
        S = build_hessian(U_h, setup['H'], setup['R_inv'], setup['TLM'], setup['obs_times'])
        eigs = np.sort(eigvalsh(S))[::-1]
        
        ax.semilogy(range(1, len(eigs)+1), eigs,
                    color=colors[i], marker=markers[i],
                    markevery=max(1, len(eigs)//10),
                    linewidth=2, markersize=6, label=labels[i])
    
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title(r'(a) Eigenvalue distribution ($\beta = 0.5$)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel (b): Number of eigenvalues near 1 vs beta
    ax = axes[1]
    threshold = 0.1
    
    for i, L_cut in enumerate(L_cut_values):
        if L_cut == 0:
            rho = np.ones((config['n'], config['n']))
        else:
            rho = localisation_matrix(config['n'], L_cut)
        
        n_clustered = []
        for beta in beta_values:
            U_h = build_cvt_matrix(setup['Bc_sq'], setup['Xpert'], rho, beta, method='direct')
            S = build_hessian(U_h, setup['H'], setup['R_inv'], setup['TLM'], setup['obs_times'])
            eigs = eigvalsh(S)
            n_clustered.append(np.sum(np.abs(eigs - 1) < threshold))
        
        ax.plot(beta_values, n_clustered,
                color=colors[i], marker=markers[i], markevery=3,
                linewidth=2, markersize=7, label=labels[i])
    
    ax.set_xlabel(r'$\beta$', fontsize=12)
    ax.set_ylabel(f'Eigenvalues with $|\\lambda - 1| < {threshold}$', fontsize=12)
    ax.set_title('(b) Eigenvalue clustering near 1', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_eigenvalue_clustering.pdf'))
    plt.savefig(os.path.join(output_dir, 'fig_eigenvalue_clustering.png'), dpi=config['dpi'])
    plt.close()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Generate all paper figures."""
    print("=" * 60)
    print("Generating figures for Hybrid 4D-Var paper")
    print("=" * 60)
    
    # Create output directory
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup experiment
    print("\nSetting up experiment...")
    setup = setup_experiment(CONFIG)
    
    # Generate all figures
    print("\n" + "-" * 60)
    generate_bounds_comparison(setup, CONFIG, output_dir)
    
    print("-" * 60)
    generate_cpu_time_figure(setup, CONFIG, output_dir)
    
    print("-" * 60)
    generate_preconditioned_analysis(setup, CONFIG, output_dir)
    
    print("-" * 60)
    generate_unpreconditioned_analysis(setup, CONFIG, output_dir)
    
    print("-" * 60)
    generate_spectral_analysis(setup, CONFIG, output_dir)
    
    print("-" * 60)
    generate_eigenvalue_clustering(setup, CONFIG, output_dir)
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
