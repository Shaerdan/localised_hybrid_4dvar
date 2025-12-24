"""
Core module for Hybrid 4D-Var with localisation.
Contains: Lorenz96 model, covariance matrices, DA methods, theoretical bounds.
"""
import numpy as np
from scipy.linalg import sqrtm, cholesky, solve_triangular
from scipy.sparse.linalg import cg
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

# =============================================================================
# LORENZ 96 MODEL
# =============================================================================

def l96_rhs(u: np.ndarray, F: float = 8.0) -> np.ndarray:
    """Lorenz 96 right-hand side: du/dt = (u[i+1] - u[i-2]) * u[i-1] - u[i] + F"""
    n = len(u)
    return (np.roll(u, -1) - np.roll(u, 2)) * np.roll(u, 1) - u + F

def l96_step(u: np.ndarray, dt: float, F: float = 8.0) -> np.ndarray:
    """Single RK4 step for Lorenz 96."""
    k1 = l96_rhs(u, F)
    k2 = l96_rhs(u + 0.5*dt*k1, F)
    k3 = l96_rhs(u + 0.5*dt*k2, F)
    k4 = l96_rhs(u + dt*k3, F)
    return u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def l96_integrate(u0: np.ndarray, nsteps: int, dt: float, F: float = 8.0) -> np.ndarray:
    """Integrate Lorenz 96 for nsteps, return trajectory [nsteps+1, nx]."""
    traj = np.zeros((nsteps + 1, len(u0)))
    traj[0] = u0
    u = u0.copy()
    for i in range(nsteps):
        u = l96_step(u, dt, F)
        traj[i+1] = u
    return traj

def l96_tlm(u: np.ndarray) -> np.ndarray:
    """Tangent Linear Model (Jacobian) of Lorenz 96 at state u."""
    n = len(u)
    F = np.zeros((n, n))
    for j in range(n):
        F[j, (j-2) % n] = -u[(j-1) % n]
        F[j, (j-1) % n] = u[(j+1) % n] - u[(j-2) % n]
        F[j, j] = -1
        F[j, (j+1) % n] = u[(j-1) % n]
    return F

def l96_tlm_step(u: np.ndarray, M: np.ndarray, dt: float, F: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate state and TLM together using RK4."""
    def rhs_combined(u, M):
        return l96_rhs(u, F), l96_tlm(u) @ M
    
    k1_u, k1_M = rhs_combined(u, M)
    k2_u, k2_M = rhs_combined(u + 0.5*dt*k1_u, M + 0.5*dt*k1_M)
    k3_u, k3_M = rhs_combined(u + 0.5*dt*k2_u, M + 0.5*dt*k2_M)
    k4_u, k4_M = rhs_combined(u + dt*k3_u, M + dt*k3_M)
    
    u_new = u + dt/6 * (k1_u + 2*k2_u + 2*k3_u + k4_u)
    M_new = M + dt/6 * (k1_M + 2*k2_M + 2*k3_M + k4_M)
    return u_new, M_new

def compute_tlm_trajectory(u0: np.ndarray, nsteps: int, dt: float, F: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute trajectory and TLM matrices M[i] mapping x0 -> xi."""
    nx = len(u0)
    traj = np.zeros((nsteps + 1, nx))
    TLM = np.zeros((nsteps + 1, nx, nx))
    
    traj[0] = u0
    TLM[0] = np.eye(nx)
    
    u, M = u0.copy(), np.eye(nx)
    for i in range(nsteps):
        u, M = l96_tlm_step(u, M, dt, F)
        traj[i+1] = u
        TLM[i+1] = M
    return traj, TLM

# =============================================================================
# COVARIANCE MATRICES
# =============================================================================

def soar_correlation(nx: int, L: float, grid_type: str = 'greatcircle') -> np.ndarray:
    """Second-Order Auto-Regressive correlation matrix."""
    if grid_type == 'greatcircle':
        theta = np.linspace(0, 2*np.pi*(nx-1)/nx, nx)
        C = np.zeros((nx, nx))
        for i in range(nx):
            for j in range(nx):
                d = 2 * np.sin(np.abs(theta[i] - theta[j]) / 2)  # Chord distance
                C[i, j] = (1 + d/L) * np.exp(-d/L)
    else:  # flat/periodic
        C = np.zeros((nx, nx))
        for i in range(nx):
            for j in range(nx):
                d = min(abs(i-j), nx - abs(i-j))
                C[i, j] = (1 + d/L) * np.exp(-d/L)
    return C

def gaspari_cohn(z: float, c: float) -> float:
    """Gaspari-Cohn correlation function with half-width c."""
    r = abs(z) / c
    if r <= 1:
        return 1 - 5/3*r**2 + 5/8*r**3 + r**4/2 - r**5/4
    elif r <= 2:
        return 4 - 5*r + 5/3*r**2 + 5/8*r**3 - r**4/2 + r**5/12 - 2/(3*r)
    return 0.0

def localisation_matrix(nx: int, L_cut: float, grid_type: str = 'greatcircle') -> np.ndarray:
    """Gaspari-Cohn localisation matrix."""
    if L_cut == 0:
        return np.ones((nx, nx))
    
    c = L_cut / np.sqrt(0.3)  # Convert to GC half-width
    rho = np.zeros((nx, nx))
    
    if grid_type == 'greatcircle':
        theta = np.linspace(0, 2*np.pi*(nx-1)/nx, nx)
        for i in range(nx):
            for j in range(nx):
                d = 2 * np.sin(np.abs(theta[i] - theta[j]) / 2)
                rho[i, j] = gaspari_cohn(d, c)
    else:
        for i in range(nx):
            for j in range(nx):
                d = min(abs(i-j), nx - abs(i-j))
                rho[i, j] = gaspari_cohn(d, c)
    return rho

def ensemble_covariance(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ensemble covariance Pf = Xpert @ Xpert.T and perturbation matrix."""
    M = X.shape[1]
    x_mean = X.mean(axis=1, keepdims=True)
    Xpert = (X - x_mean) / np.sqrt(M - 1)
    Pf = Xpert @ Xpert.T
    return Pf, Xpert

def matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Matrix square root via eigendecomposition (handles semi-definite)."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0)  # Ensure non-negative
    return eigvecs @ np.diag(np.sqrt(eigvals))

# =============================================================================
# OBSERVATION OPERATORS
# =============================================================================

def obs_operator(nx: int, nx_obs: int, obs_type: str = 'every_other', seed: int = None) -> np.ndarray:
    """Create observation operator H [nx_obs, nx]."""
    if obs_type == 'every_other':
        H = np.zeros((nx_obs, nx))
        obs_idx = np.arange(0, nx, nx // nx_obs)[:nx_obs]
        for i, j in enumerate(obs_idx):
            H[i, j] = 1
    elif obs_type == 'random':
        rng = np.random.default_rng(seed)
        obs_idx = rng.choice(nx, nx_obs, replace=False)
        obs_idx.sort()
        H = np.zeros((nx_obs, nx))
        for i, j in enumerate(obs_idx):
            H[i, j] = 1
    elif obs_type == 'first_half':
        H = np.zeros((nx_obs, nx))
        for i in range(nx_obs):
            H[i, i] = 1
    return H

def generate_observations(truth: np.ndarray, H: np.ndarray, R_sqrt: np.ndarray, 
                          period: int, seed: int = None) -> np.ndarray:
    """Generate noisy observations from truth trajectory."""
    nsteps = truth.shape[0] - 1
    n_obs_times = nsteps // period
    nx_obs = H.shape[0]
    
    rng = np.random.default_rng(seed)
    y = np.zeros((n_obs_times, nx_obs))
    
    for i in range(n_obs_times):
        t_idx = (i + 1) * period
        y[i] = H @ truth[t_idx] + R_sqrt @ rng.standard_normal(nx_obs)
    return y

# =============================================================================
# DATA ASSIMILATION - HYBRID 4D-VAR
# =============================================================================

@dataclass
class DAResult:
    """Results from DA experiment."""
    condition_number: float
    n_iterations: int
    eigenvalues: np.ndarray
    beta: float
    residuals: np.ndarray = None

def build_cvt_matrix(Bc_sqrt: np.ndarray, Xpert: np.ndarray, rho: np.ndarray, 
                     beta: float, method: str = 'direct') -> np.ndarray:
    """Build CVT matrix U_h = [sqrt(beta)*Bc_sqrt, sqrt(1-beta)*Xpert_loc].
    
    Methods:
        'direct': Directly factorize (rho ∘ Pf) - keeps dimensions small
        'modulated': Use modulated ensemble (paper Eq. 13) - dimensions explode
        'truncated_k': Use top k eigenvectors of rho (e.g., 'truncated_10')
    """
    nx, M = Xpert.shape
    
    if np.allclose(rho, 1):  # No localisation
        Xpert_loc = Xpert
    elif method == 'direct':
        # Direct factorization of localised covariance: (rho ∘ Pf)^{1/2}
        Pf = Xpert @ Xpert.T
        Pf_loc = rho * Pf  # Schur product
        Xpert_loc = matrix_sqrt(Pf_loc)
    elif method == 'modulated':
        # Modulated ensemble (paper Eq. 13) - WARNING: dimension explodes!
        rho_sqrt = matrix_sqrt(rho)
        Xpert_loc = np.zeros((nx, M * nx))
        for m in range(M):
            Xpert_loc[:, m*nx:(m+1)*nx] = np.diag(Xpert[:, m]) @ rho_sqrt
    elif method.startswith('truncated_'):
        # Truncated: keep top k eigenvectors of rho
        k = int(method.split('_')[1])
        eigvals, eigvecs = np.linalg.eigh(rho)
        idx = np.argsort(eigvals)[::-1][:k]
        rho_sqrt_trunc = eigvecs[:, idx] @ np.diag(np.sqrt(eigvals[idx]))
        Xpert_loc = np.zeros((nx, M * k))
        for m in range(M):
            Xpert_loc[:, m*k:(m+1)*k] = np.diag(Xpert[:, m]) @ rho_sqrt_trunc
    else:
        raise ValueError(f"Unknown method: {method}")
    
    U_h = np.hstack([np.sqrt(beta) * Bc_sqrt, np.sqrt(1 - beta) * Xpert_loc])
    return U_h

def build_hessian(U_h: np.ndarray, H: np.ndarray, R_inv: np.ndarray, 
                  TLM: np.ndarray, obs_times: np.ndarray) -> np.ndarray:
    """Build preconditioned Hessian S = I + U_h^T @ H_hat^T @ R_hat^{-1} @ H_hat @ U_h."""
    n_ctrl = U_h.shape[1]
    S = np.eye(n_ctrl)
    
    for t_idx in obs_times:
        M_t = TLM[t_idx]  # TLM from t=0 to t=t_idx
        HMU = H @ M_t @ U_h
        S += HMU.T @ R_inv @ HMU
    return S

def solve_da(S: np.ndarray, b: np.ndarray, tol: float = 1e-16, 
             maxiter: int = 100) -> Tuple[np.ndarray, int, np.ndarray]:
    """Solve S @ v = b using CG with iteration tracking."""
    residuals = []
    def callback(xk):
        res = np.linalg.norm(b - S @ xk)
        residuals.append(res)
    
    # scipy.sparse.linalg.cg uses 'atol' and 'rtol' in newer versions
    try:
        v, info = cg(S, b, rtol=tol, atol=0, maxiter=maxiter, callback=callback)
    except TypeError:
        v, info = cg(S, b, tol=tol, maxiter=maxiter, callback=callback)
    return v, len(residuals), np.array(residuals)

def run_hybrid_4dvar(x_b: np.ndarray, y: np.ndarray, H: np.ndarray, R_inv: np.ndarray,
                     Bc_sqrt: np.ndarray, Xpert: np.ndarray, rho: np.ndarray, beta: float,
                     dt: float, period_obs: int, solver_tol: float = 1e-16,
                     maxiter: int = 100, method: str = 'direct') -> DAResult:
    """Run one cycle of Hybrid 4D-Var and return diagnostics."""
    nx = len(x_b)
    n_obs = y.shape[0]
    nsteps = n_obs * period_obs
    obs_times = np.arange(1, n_obs + 1) * period_obs
    
    # Compute trajectory and TLM
    traj, TLM = compute_tlm_trajectory(x_b, nsteps, dt)
    
    # Build CVT matrix
    U_h = build_cvt_matrix(Bc_sqrt, Xpert, rho, beta, method=method)
    
    # Build innovation vector and RHS
    n_ctrl = U_h.shape[1]
    b = np.zeros(n_ctrl)
    HT_Rinv = H.T @ R_inv
    
    for i, t_idx in enumerate(obs_times):
        d = y[i] - H @ traj[t_idx]  # Innovation
        b += U_h.T @ TLM[t_idx].T @ HT_Rinv @ d
    
    # Build and solve Hessian system
    S = build_hessian(U_h, H, R_inv, TLM, obs_times)
    v, n_iter, residuals = solve_da(S, b, solver_tol, maxiter)
    
    # Compute eigenvalues and condition number
    eigvals = np.linalg.eigvalsh(S)
    cond = eigvals.max() / eigvals.min()
    
    return DAResult(
        condition_number=cond,
        n_iterations=n_iter,
        eigenvalues=np.sort(eigvals)[::-1],
        beta=beta,
        residuals=residuals
    )

# =============================================================================
# THEORETICAL BOUNDS (Theorems 4 and 6)
# =============================================================================

def bound_theorem4(Bc: np.ndarray, K: np.ndarray, Pf_loc: np.ndarray, beta: float) -> float:
    """Upper bound using eigenvalues (Theorem 4)."""
    lam1_Bc = np.linalg.eigvalsh(Bc).max()
    lam1_K = np.linalg.eigvalsh(K).max()
    lam1_K2 = np.linalg.eigvalsh(K @ K).max()
    lam1_Pf = np.linalg.eigvalsh(Pf_loc).max()
    
    A1 = max((1 - beta) * lam1_Bc * lam1_K, beta * lam1_Pf * lam1_K)
    A2 = np.sqrt((beta - beta**2) * lam1_Bc * lam1_K2 * lam1_Pf)
    return 1 + A1 + A2

def bound_theorem6(Bc: np.ndarray, K: np.ndarray, Pf_loc: np.ndarray, beta: float) -> float:
    """Upper bound using infinity norms (Theorem 6) - faster but looser."""
    norm_Bc = np.linalg.norm(Bc, np.inf)
    norm_K = np.linalg.norm(K, np.inf)
    norm_K2 = np.linalg.norm(K @ K, np.inf)
    norm_Pf = np.linalg.norm(Pf_loc, np.inf)
    
    A1 = max((1 - beta) * norm_Bc * norm_K, beta * norm_Pf * norm_K)
    A2 = np.sqrt((beta - beta**2) * norm_Bc * norm_K2 * norm_Pf)
    return 1 + A1 + A2

def compute_K_matrix(H: np.ndarray, R_inv: np.ndarray, TLM: np.ndarray, 
                     obs_times: np.ndarray) -> np.ndarray:
    """Compute K = H_hat^T @ R_hat^{-1} @ H_hat for bound calculations."""
    nx = TLM.shape[1]
    K = np.zeros((nx, nx))
    for t_idx in obs_times:
        HM = H @ TLM[t_idx]
        K += HM.T @ R_inv @ HM
    return K
