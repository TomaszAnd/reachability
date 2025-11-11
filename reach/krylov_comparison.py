"""
Krylov criterion comparison tests for quantum reachability analysis.

This module implements experimental comparisons between three reachability criteria:
1. Spectral criterion: max_λ S(λ) where S is spectral overlap
2. Moment criterion: Positive-definiteness of moment matrix
3. Krylov criterion: Target state membership in Krylov subspace

Key Questions:
- Does Krylov dimension m = dim(K_m(H(λ), ψ)) depend on λ weights?
- How do the three criteria compare in practice for small dimensions?

Test Hamiltonian Generation:
- Method (a): Random Pauli operators on random qubit pairs (canonical basis)
- Method (b): Random projectors from Haar-distributed states
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import qutip
from scipy.stats import unitary_group

from . import mathematics, models, optimize, settings

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    logger.warning("matplotlib not available, plotting functions will fail")

logger = logging.getLogger(__name__)


def _qubit_index_to_sites(n_qubits: int, i: int, j: int) -> Tuple[int, int]:
    """
    Validate and return qubit site indices.

    Args:
        n_qubits: Total number of qubits
        i: First qubit index
        j: Second qubit index

    Returns:
        Tuple of validated site indices

    Raises:
        ValueError: If indices are invalid
    """
    if not (0 <= i < n_qubits and 0 <= j < n_qubits):
        raise ValueError(f"Qubit indices ({i}, {j}) out of range for {n_qubits} qubits")
    if i == j:
        raise ValueError(f"Qubit indices must be different, got i=j={i}")
    return i, j


def generate_canonical_pauli_hamiltonian(
    dim: int,
    K: int,
    seed: Optional[int] = None,
    allow_single_site: bool = True
) -> List[qutip.Qobj]:
    """
    Generate K Hamiltonians using canonical Pauli basis (Method a).

    For d = 2^n dimensions, generates random Pauli operators on random qubit sites.
    Each H_k = sigma_alpha acting on qubits (i,j) with identity elsewhere, where:
    - Alpha ∈ {X, Y, Z}, randomly chosen
    - Sites i,j randomly chosen from n qubits
    - If allow_single_site=True, can also generate single-site terms sigma_alpha ⊗ I ⊗ ... ⊗ I

    Mathematical form:
        H_k = I ⊗ ... ⊗ I ⊗ σ_α(i) ⊗ I ⊗ ... ⊗ I ⊗ σ_β(j) ⊗ I ⊗ ... ⊗ I

    Args:
        dim: Hilbert space dimension (must be power of 2)
        K: Number of Hamiltonians to generate
        seed: Random seed for reproducibility
        allow_single_site: If True, allow single-site Pauli terms in addition to two-site

    Returns:
        List of K Hermitian Pauli operators

    Raises:
        ValueError: If dim is not a power of 2 or K < 1

    Examples:
        >>> hams = generate_canonical_pauli_hamiltonian(dim=8, K=4, seed=42)
        >>> len(hams)
        4
        >>> hams[0].dims
        [[8], [8]]
    """
    # Validate dimension is power of 2
    if dim < 2 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Dimension {dim} must be a power of 2 for Pauli basis")

    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    # Number of qubits
    n_qubits = int(np.log2(dim))

    if seed is None:
        seed = settings.SEED
    rng = np.random.RandomState(seed)

    # Pauli operators
    pauli_ops = [qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]
    pauli_names = ['X', 'Y', 'Z']
    identity = qutip.qeye(2)

    hamiltonians = []

    for k in range(K):
        # Randomly decide: single-site or two-site term
        if allow_single_site and rng.random() < 0.3:  # 30% chance of single-site
            # Single-site term: sigma_alpha ⊗ I ⊗ ... ⊗ I
            site = rng.randint(0, n_qubits)
            alpha_idx = rng.randint(0, 3)

            ops = [identity] * n_qubits
            ops[site] = pauli_ops[alpha_idx]

            H = qutip.tensor(ops)
            logger.debug(f"H_{k}: {pauli_names[alpha_idx]}_{site} (single-site)")
        else:
            # Two-site term: sigma_alpha(i) ⊗ sigma_beta(j)
            # Choose two different random sites
            sites = rng.choice(n_qubits, size=2, replace=False)
            i, j = int(sites[0]), int(sites[1])

            # Choose random Pauli operators for each site
            alpha_idx = rng.randint(0, 3)
            beta_idx = rng.randint(0, 3)

            # Build tensor product
            ops = [identity] * n_qubits
            ops[i] = pauli_ops[alpha_idx]
            ops[j] = pauli_ops[beta_idx]

            H = qutip.tensor(ops)
            logger.debug(
                f"H_{k}: {pauli_names[alpha_idx]}_{i} ⊗ {pauli_names[beta_idx]}_{j} (two-site)"
            )

        # Flatten dims to match pipeline expectations
        H.dims = [[dim], [dim]]
        hamiltonians.append(H)

    logger.info(f"Generated {K} canonical Pauli Hamiltonians for d={dim} (n={n_qubits} qubits)")
    return hamiltonians


def generate_random_projector_hamiltonian(
    dim: int,
    K: int,
    seed: Optional[int] = None,
    rank: int = 1
) -> List[qutip.Qobj]:
    """
    Generate K Hamiltonians as random projectors (Method b).

    Generates random rank-r projectors of the form:
        H_k = U |ψ⟩⟨ψ| U†

    where:
    - |ψ⟩ is a random state from Haar measure
    - U is a random Haar-distributed unitary
    - For rank > 1, uses sum of rank-1 projectors

    Mathematical form:
        H_k = U (Σᵢ |ψᵢ⟩⟨ψᵢ|) U†

    where {|ψᵢ⟩} are orthonormal Haar-random states.

    Args:
        dim: Hilbert space dimension
        K: Number of Hamiltonians to generate
        seed: Random seed for reproducibility
        rank: Rank of each projector (1 ≤ rank ≤ dim)

    Returns:
        List of K Hermitian projector operators

    Raises:
        ValueError: If parameters are invalid

    Examples:
        >>> hams = generate_random_projector_hamiltonian(dim=8, K=4, seed=42)
        >>> len(hams)
        4
        >>> np.allclose(hams[0] @ hams[0], hams[0])  # Projector property: P^2 = P
        True
    """
    if dim < 2:
        raise ValueError(f"Dimension must be >= 2, got {dim}")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if not (1 <= rank <= dim):
        raise ValueError(f"Rank {rank} must be in range [1, {dim}]")

    if seed is None:
        seed = settings.SEED
    rng = np.random.RandomState(seed)

    hamiltonians = []

    for k in range(K):
        # Generate random Haar-distributed unitary
        U_seed = rng.randint(0, 2**31 - 1)
        U_matrix = unitary_group.rvs(dim, random_state=U_seed)
        U = qutip.Qobj(U_matrix, dims=[[dim], [dim]])

        # Generate rank-r projector
        if rank == 1:
            # Single rank-1 projector: |ψ⟩⟨ψ|
            psi_seed = rng.randint(0, 2**31 - 1)
            np.random.seed(psi_seed)
            psi = qutip.rand_ket(dim)
            P = psi * psi.dag()
        else:
            # Sum of rank-1 projectors from orthonormal basis
            # Generate random unitary and take first 'rank' columns
            V_seed = rng.randint(0, 2**31 - 1)
            V_matrix = unitary_group.rvs(dim, random_state=V_seed)

            # Build projector: P = Σᵢ |vᵢ⟩⟨vᵢ| for i = 0, ..., rank-1
            P = qutip.Qobj(np.zeros((dim, dim), dtype=complex), dims=[[dim], [dim]])
            for i in range(rank):
                vi = qutip.Qobj(V_matrix[:, i], dims=[[dim], [1]])
                P += vi * vi.dag()

        # Apply random unitary rotation: H = U P U†
        H = U * P * U.dag()

        # Ensure Hermitian (numerical cleanup)
        H_matrix = H.full()
        H_matrix = (H_matrix + H_matrix.conj().T) / 2.0
        H = qutip.Qobj(H_matrix, dims=[[dim], [dim]])

        hamiltonians.append(H)
        logger.debug(f"H_{k}: Random rank-{rank} projector")

    logger.info(f"Generated {K} random projector Hamiltonians for d={dim} (rank={rank})")
    return hamiltonians


def compute_krylov_dimension(
    H: qutip.Qobj,
    psi: qutip.Qobj,
    tol: float = settings.KRYLOV_BREAKDOWN_TOL
) -> int:
    """
    Compute the dimension of Krylov subspace K(H, ψ).

    The Krylov subspace is K_m(H, ψ) = span{ψ, Hψ, H²ψ, ..., H^(m-1)ψ}.
    This function determines the rank m, which is the smallest integer such that
    H^m ψ lies in the span of {ψ, Hψ, ..., H^(m-1)ψ}.

    Note: m satisfies 1 ≤ m ≤ d (not d²), where d is the Hilbert space dimension.

    Args:
        H: Hamiltonian operator (d×d matrix)
        psi: Initial state vector (d-dimensional)
        tol: Breakdown tolerance for rank detection

    Returns:
        Krylov dimension m ∈ [1, d]

    Raises:
        ValueError: If inputs are invalid
    """
    d = H.shape[0]

    # Try building Krylov basis for full dimension d
    # The function will break early if the space degenerates
    V = mathematics.krylov_basis(H, psi, m=d, tol=tol)

    # Determine actual rank by counting non-zero columns
    # A column is considered zero if its norm is below tolerance
    ranks = []
    for i in range(V.shape[1]):
        col_norm = np.linalg.norm(V[:, i])
        if col_norm > tol:
            ranks.append(i)

    # Krylov dimension is the number of non-zero columns
    m = len(ranks) if ranks else 1

    return m


def test_krylov_lambda_dependence(
    dim: int,
    K: int,
    method: str = 'canonical',
    num_trials: int = 100,
    seed: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Test whether Krylov dimension depends on lambda weights.

    For a fixed set of K Hamiltonians {H₁, ..., Hₖ}, this function:
    1. Samples multiple random lambda vectors from N(0,1)
    2. Computes H(λ) = Σₖ λₖ Hₖ for each lambda
    3. Computes Krylov dimension m for each H(λ)
    4. Analyzes variance of m across different lambdas

    If std(m) ≈ 0, the Krylov dimension is lambda-independent.
    If std(m) > 0, the dimension depends on lambda weights.

    Args:
        dim: Hilbert space dimension
        K: Number of Hamiltonians
        method: 'canonical' (Pauli basis) or 'projector' (random projectors)
        num_trials: Number of random lambda samples to test
        seed: Random seed for reproducibility

    Returns:
        Dictionary with statistics:
        {
            method: {
                'mean': mean Krylov dimension,
                'std': standard deviation,
                'min': minimum dimension observed,
                'max': maximum dimension observed,
                'lambda_independent': bool (std < 0.01)
            }
        }

    Examples:
        >>> results = test_krylov_lambda_dependence(d=16, K=8, num_trials=100)
        >>> print(f"Mean Krylov dim: {results['canonical']['mean']:.2f}")
        >>> print(f"Lambda-independent: {results['canonical']['lambda_independent']}")
    """
    if seed is None:
        seed = settings.SEED
    rng = np.random.RandomState(seed)

    logger.info(
        f"Testing Krylov lambda dependence: d={dim}, K={K}, "
        f"method={method}, trials={num_trials}"
    )

    # Generate Hamiltonians using specified method
    if method == 'canonical':
        hams = generate_canonical_pauli_hamiltonian(dim, K, seed=rng.randint(0, 2**31 - 1))
    elif method == 'projector':
        hams = generate_random_projector_hamiltonian(dim, K, seed=rng.randint(0, 2**31 - 1))
    else:
        raise ValueError(f"Unknown method '{method}', expected 'canonical' or 'projector'")

    # Initial state: |0⟩
    psi = models.fock_state(dim, 0)

    # Sample random lambda vectors and compute Krylov dimensions
    krylov_dims = []

    for trial in range(num_trials):
        # Sample lambda from N(0, 1)
        lambdas = rng.randn(K)

        # Construct H(lambda)
        H_lambda = mathematics.construct_hamiltonian(lambdas, hams)

        # Compute Krylov dimension
        m = compute_krylov_dimension(H_lambda, psi)
        krylov_dims.append(m)

        if trial % 20 == 0:
            logger.debug(f"Trial {trial}/{num_trials}: λ={lambdas[:3]}..., m={m}")

    # Compute statistics
    krylov_dims = np.array(krylov_dims)
    mean_dim = float(np.mean(krylov_dims))
    std_dim = float(np.std(krylov_dims))
    min_dim = int(np.min(krylov_dims))
    max_dim = int(np.max(krylov_dims))

    # Criterion for lambda-independence: std < 0.01
    lambda_independent = std_dim < 0.01

    results = {
        method: {
            'mean': mean_dim,
            'std': std_dim,
            'min': min_dim,
            'max': max_dim,
            'lambda_independent': lambda_independent,
            'num_trials': num_trials
        }
    }

    logger.info(
        f"Results ({method}): mean={mean_dim:.2f}, std={std_dim:.4f}, "
        f"range=[{min_dim}, {max_dim}], lambda_independent={lambda_independent}"
    )

    return results


def moment_criterion(
    psi: qutip.Qobj,
    phi: qutip.Qobj,
    hams: List[qutip.Qobj],
    tol: float = 1e-10
) -> bool:
    """
    Check reachability using moment matrix criterion.

    The moment criterion checks positive semi-definiteness of a matrix constructed
    from moments ⟨ψ|HⁱHʲ|ψ⟩ and target overlaps ⟨φ|Hⁱ|ψ⟩.

    For K Hamiltonians, constructs the (K+1)×(K+1) moment matrix:
        M[0,0] = ⟨ψ|ψ⟩ = 1
        M[0,i] = M[i,0] = ⟨φ|Hᵢ|ψ⟩  for i = 1, ..., K
        M[i,j] = ⟨ψ|HᵢHⱼ|ψ⟩  for i,j = 1, ..., K

    The state is reachable if M is positive semi-definite (all eigenvalues ≥ -tol).

    Args:
        psi: Initial quantum state |ψ⟩
        phi: Target quantum state |φ⟩
        hams: List of K Hamiltonian operators
        tol: Tolerance for eigenvalue positivity check

    Returns:
        True if reachable (M is PSD), False if unreachable

    Note:
        This criterion is threshold-free (does not depend on τ).
    """
    K = len(hams)

    # Build (K+1) × (K+1) moment matrix
    M = np.zeros((K + 1, K + 1), dtype=complex)

    # M[0,0] = ⟨ψ|ψ⟩ = 1 (normalized)
    M[0, 0] = 1.0

    # M[0,i] = M[i,0] = ⟨φ|Hᵢ|ψ⟩
    for i, Hi in enumerate(hams, start=1):
        result = phi.dag() * Hi * psi
        # Extract scalar: could be Qobj or already complex
        if isinstance(result, qutip.Qobj):
            M[0, i] = result.full()[0, 0]
        else:
            M[0, i] = result
        M[i, 0] = M[0, i].conj()

    # M[i,j] = ⟨ψ|HᵢHⱼ|ψ⟩
    for i, Hi in enumerate(hams, start=1):
        for j, Hj in enumerate(hams, start=1):
            result = psi.dag() * Hi * Hj * psi
            # Extract scalar: could be Qobj or already complex
            if isinstance(result, qutip.Qobj):
                M[i, j] = result.full()[0, 0]
            else:
                M[i, j] = result

    # Check positive semi-definiteness via eigenvalues
    eigenvalues = np.linalg.eigvalsh(M)
    min_eigenvalue = np.min(eigenvalues)

    # Reachable if all eigenvalues ≥ -tol
    is_reachable = min_eigenvalue >= -tol

    logger.debug(
        f"Moment criterion: min_eigenvalue={min_eigenvalue:.6e}, "
        f"reachable={is_reachable}"
    )

    return is_reachable


def krylov_criterion(
    psi: qutip.Qobj,
    phi: qutip.Qobj,
    hams: List[qutip.Qobj],
    lambdas: Optional[np.ndarray] = None,
    m: Optional[int] = None
) -> bool:
    """
    Check reachability using Krylov subspace criterion.

    The Krylov criterion checks if the target state |φ⟩ lies in the
    Krylov subspace K_m(H(λ), ψ) = span{ψ, H(λ)ψ, ..., H(λ)^(m-1)ψ}.

    If lambdas is provided, constructs H(λ) = Σₖ λₖ Hₖ.
    If lambdas is None, uses uniform weights (1/K, ..., 1/K).

    If m is provided, uses that Krylov dimension.
    If m is None, computes m = dim(K(H(λ), ψ)) automatically.

    Args:
        psi: Initial quantum state |ψ⟩
        phi: Target quantum state |φ⟩
        hams: List of K Hamiltonian operators
        lambdas: Optional parameter vector (default: uniform weights)
        m: Optional Krylov dimension (default: compute automatically)

    Returns:
        True if reachable (φ ∈ K_m), False if unreachable (φ ∉ K_m)
    """
    K = len(hams)
    d = psi.shape[0]

    # Default: uniform weights
    if lambdas is None:
        lambdas = np.ones(K) / K

    # Construct H(lambda)
    H_lambda = mathematics.construct_hamiltonian(lambdas, hams)

    # Determine Krylov dimension
    if m is None:
        m = compute_krylov_dimension(H_lambda, psi)

    # Use existing is_unreachable_krylov function (returns True if UNREACHABLE)
    is_unreachable = mathematics.is_unreachable_krylov(H_lambda, psi, phi, m)

    # Return True if REACHABLE (invert the unreachability result)
    return not is_unreachable


def compare_all_criteria(
    d_values: List[int],
    K_values: List[int],
    ensemble: str = 'GUE',
    tau: float = 0.95,
    trials_per_point: int = 50,
    seed: Optional[int] = None,
    method_type: str = 'ensemble'
) -> Dict[str, Dict[Tuple[int, int], Dict[str, float]]]:
    """
    Compare spectral, moment, and Krylov criteria across (d, K) parameter space.

    For each (d, K) pair, estimates unreachability probabilities using three criteria:
    1. Spectral: P[max_λ S(λ) < τ]
    2. Moment: P[moment matrix not PSD]
    3. Krylov: P[φ ∉ K_m(H, ψ)] with uniform lambda weights

    Args:
        d_values: List of dimensions to test (e.g., [8, 16, 32])
        K_values: List of K values to test (e.g., [4, 8, 12])
        ensemble: 'GUE', 'GOE', or method type for test Hamiltonians
        tau: Threshold for spectral criterion
        trials_per_point: Number of Monte Carlo samples per (d, K) point
        seed: Random seed for reproducibility
        method_type: 'ensemble' (use GOE/GUE), 'canonical', or 'projector'

    Returns:
        Nested dictionary:
        {
            'spectral': {(d, K): {'prob': p, 'sem': err}},
            'moment': {(d, K): {'prob': p, 'sem': err}},
            'krylov': {(d, K): {'prob': p, 'sem': err}}
        }

    Examples:
        >>> results = compare_all_criteria(
        ...     d_values=[8, 16, 32],
        ...     K_values=[4, 8, 12],
        ...     trials_per_point=50
        ... )
        >>> spectral_8_4 = results['spectral'][(8, 4)]['prob']
    """
    if seed is None:
        seed = settings.SEED
    rng = np.random.RandomState(seed)

    logger.info(
        f"Comparing criteria: d={d_values}, K={K_values}, "
        f"tau={tau}, trials={trials_per_point}, method={method_type}"
    )

    # Initialize result storage
    results = {
        'spectral': {},
        'moment': {},
        'krylov': {}
    }

    for d in d_values:
        for K in K_values:
            # Skip invalid combinations
            if K < 2:
                logger.warning(f"Skipping K={K} (K < 2)")
                continue

            logger.info(f"Testing (d={d}, K={K})")

            # Counters for each criterion
            spectral_unreachable = 0
            moment_unreachable = 0
            krylov_unreachable = 0

            for trial in range(trials_per_point):
                # Generate Hamiltonians
                if method_type == 'ensemble':
                    hams = models.random_hamiltonian_ensemble(
                        d, K, ensemble, seed=rng.randint(0, 2**31 - 1)
                    )
                elif method_type == 'canonical':
                    hams = generate_canonical_pauli_hamiltonian(
                        d, K, seed=rng.randint(0, 2**31 - 1)
                    )
                elif method_type == 'projector':
                    hams = generate_random_projector_hamiltonian(
                        d, K, seed=rng.randint(0, 2**31 - 1)
                    )
                else:
                    raise ValueError(f"Unknown method_type '{method_type}'")

                # Initial and target states
                psi = models.fock_state(d, 0)  # |0⟩
                phi = models.fock_state(d, d - 1)  # |d-1⟩

                # --- Spectral criterion ---
                opt_result = optimize.maximize_spectral_overlap(
                    psi, phi, hams,
                    method='L-BFGS-B',
                    restarts=1,
                    maxiter=100,
                    seed=rng.randint(0, 2**31 - 1)
                )
                if opt_result['best_value'] < tau:
                    spectral_unreachable += 1

                # --- Moment criterion ---
                is_reachable_moment = moment_criterion(psi, phi, hams)
                if not is_reachable_moment:
                    moment_unreachable += 1

                # --- Krylov criterion ---
                is_reachable_krylov = krylov_criterion(psi, phi, hams)
                if not is_reachable_krylov:
                    krylov_unreachable += 1

            # Compute probabilities and standard errors
            n = trials_per_point

            for criterion_name, count in [
                ('spectral', spectral_unreachable),
                ('moment', moment_unreachable),
                ('krylov', krylov_unreachable)
            ]:
                prob = count / n
                sem = mathematics.compute_binomial_sem(prob, n)

                results[criterion_name][(d, K)] = {
                    'prob': prob,
                    'sem': sem,
                    'count': count,
                    'trials': n
                }

            logger.info(
                f"  (d={d}, K={K}): spectral={spectral_unreachable}/{n}, "
                f"moment={moment_unreachable}/{n}, krylov={krylov_unreachable}/{n}"
            )

    return results


def plot_criteria_comparison(
    results: Dict[str, Dict[Tuple[int, int], Dict[str, float]]],
    d_values: List[int],
    K_values: List[int],
    tau: float = 0.95,
    outfile: str = 'criteria_comparison.png',
    figsize: Tuple[int, int] = (16, 12),
    dpi: int = 150
) -> None:
    """
    Create 3x3 grid visualization comparing three criteria across K values.

    Grid layout:
    - Rows: Different criteria (spectral, moment, Krylov)
    - Columns: Different K values
    - Each subplot shows unreachability probability vs dimension d

    Args:
        results: Results dictionary from compare_all_criteria()
        d_values: List of dimensions tested
        K_values: List of K values tested
        tau: Threshold used for spectral criterion (for title)
        outfile: Output filename for plot
        figsize: Figure size in inches (width, height)
        dpi: Resolution in dots per inch

    Saves:
        PNG file with 3×3 grid of comparison plots
    """
    criteria = ['spectral', 'moment', 'krylov']
    criterion_labels = {
        'spectral': f'Spectral (τ={tau:.2f})',
        'moment': 'Moment Matrix',
        'krylov': 'Krylov Subspace'
    }

    n_criteria = len(criteria)
    n_k = len(K_values)

    # Create figure with subplots
    fig, axes = plt.subplots(n_criteria, n_k, figsize=figsize, dpi=dpi)

    # Ensure axes is 2D array
    if n_criteria == 1:
        axes = axes.reshape(1, -1)
    if n_k == 1:
        axes = axes.reshape(-1, 1)

    # Color scheme
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(d_values)))

    for i, criterion in enumerate(criteria):
        for j, K in enumerate(K_values):
            ax = axes[i, j]

            # Extract data for this (criterion, K) pair
            probs = []
            errs = []
            d_plot = []

            for d in d_values:
                if (d, K) in results[criterion]:
                    data = results[criterion][(d, K)]
                    probs.append(data['prob'])
                    errs.append(data['sem'])
                    d_plot.append(d)

            if not probs:
                # No data for this K
                ax.text(
                    0.5, 0.5, 'No data',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=12
                )
                ax.set_xlabel('Dimension d', fontsize=10, fontweight='bold')
                ax.set_ylabel('Unreachability', fontsize=10, fontweight='bold')
                ax.set_title(f'{criterion_labels[criterion]}, K={K}', fontsize=11, fontweight='bold')
                continue

            probs = np.array(probs)
            errs = np.array(errs)
            d_plot = np.array(d_plot)

            # Plot with error bars
            ax.errorbar(
                d_plot, probs, yerr=errs,
                fmt='o-',
                linewidth=2,
                markersize=8,
                capsize=4,
                capthick=2,
                label=f'K={K}',
                color=colors[j % len(colors)]
            )

            # Styling
            ax.set_xlabel('Dimension d', fontsize=10, fontweight='bold')
            ax.set_ylabel('Unreachability Prob.', fontsize=10, fontweight='bold')
            ax.set_title(f'{criterion_labels[criterion]}, K={K}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(-0.05, 1.05)

            # Ticks
            ax.set_xticks(d_plot)
            ax.tick_params(labelsize=9)

    # Overall title
    fig.suptitle(
        'Reachability Criteria Comparison: Spectral vs Moment vs Krylov',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved criteria comparison plot to {outfile}")


def plot_krylov_lambda_dependence(
    canonical_results: Dict[str, Dict[str, float]],
    projector_results: Dict[str, Dict[str, float]],
    outfile: str = 'krylov_lambda_dependence.png',
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 150
) -> None:
    """
    Visualize Krylov dimension statistics for lambda dependence test.

    Creates side-by-side bar plots showing:
    - Left: Canonical Pauli basis results
    - Right: Random projector results

    Each plot shows mean, min, max Krylov dimensions with std deviation error bars.

    Args:
        canonical_results: Results from test_krylov_lambda_dependence(method='canonical')
        projector_results: Results from test_krylov_lambda_dependence(method='projector')
        outfile: Output filename for plot
        figsize: Figure size in inches (width, height)
        dpi: Resolution in dots per inch

    Saves:
        PNG file with side-by-side bar plots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    results_list = [canonical_results, projector_results]
    titles = ['Canonical Pauli Basis', 'Random Projectors']
    method_keys = ['canonical', 'projector']

    for idx, (results, title, method) in enumerate(zip(results_list, titles, method_keys)):
        ax = axes[idx]

        if method not in results:
            ax.text(
                0.5, 0.5, 'No data',
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=12
            )
            ax.set_title(title, fontsize=12, fontweight='bold')
            continue

        data = results[method]
        mean_dim = data['mean']
        std_dim = data['std']
        min_dim = data['min']
        max_dim = data['max']
        lambda_indep = data['lambda_independent']

        # Create bar plot
        x_pos = np.arange(3)
        values = [min_dim, mean_dim, max_dim]
        labels = ['Min', 'Mean', 'Max']
        colors_bar = ['#2E86AB', '#A23B72', '#F18F01']

        bars = ax.bar(x_pos, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add error bar on mean
        ax.errorbar(
            [1], [mean_dim], yerr=[std_dim],
            fmt='none',
            ecolor='black',
            capsize=8,
            capthick=2,
            linewidth=2
        )

        # Styling
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
        ax.set_ylabel('Krylov Dimension m', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Add text annotation
        status_text = 'λ-independent' if lambda_indep else 'λ-dependent'
        status_color = 'green' if lambda_indep else 'red'
        ax.text(
            0.5, 0.95,
            f'Status: {status_text}\nstd = {std_dim:.4f}',
            ha='center',
            va='top',
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.2)
        )

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{val:.1f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

    # Overall title
    fig.suptitle(
        'Krylov Dimension Dependence on Lambda Weights',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved lambda dependence plot to {outfile}")
