"""
Smoke tests for reach package with minimal runtime.

These tests verify basic functionality without extensive computation.
Fast execution (<10 seconds total) for CI/CD integration.
"""

import numpy as np

from reach import analysis, mathematics, models, optimize, settings


def test_package_imports():
    """Test that all modules import successfully."""
    assert settings.SEED == 42
    assert settings.DEFAULT_TAU == 0.95
    assert settings.DEFAULT_METHOD == "L-BFGS-B"


def test_goe_generation():
    """Test GOE Hamiltonian generation."""
    models.setup_environment(settings.SEED)
    hams = models.random_hamiltonian_ensemble(d=4, k=2, ensemble="GOE", seed=42)

    assert len(hams) == 2
    assert hams[0].shape == (4, 4)
    assert hams[1].shape == (4, 4)

    # Verify Hermiticity
    assert mathematics.validate_hermitian(hams[0])
    assert mathematics.validate_hermitian(hams[1])


def test_gue_generation():
    """Test GUE Hamiltonian generation."""
    models.setup_environment(settings.SEED)
    hams = models.random_hamiltonian_ensemble(d=4, k=2, ensemble="GUE", seed=42)

    assert len(hams) == 2
    assert hams[0].shape == (4, 4)
    assert mathematics.validate_hermitian(hams[0])


def test_random_states():
    """Test random state generation."""
    models.setup_environment(settings.SEED)
    states = models.random_states(n=3, dim=4, seed=42)

    assert len(states) == 3
    for state in states:
        assert state.shape == (4, 1)
        # Verify normalization
        norm = np.abs((state.dag() * state).tr())
        assert np.abs(norm - 1.0) < 1e-10


def test_fock_state():
    """Test computational basis state generation."""
    psi = models.fock_state(d=5, n=2)
    assert psi.shape == (5, 1)

    # Verify it's the correct basis state
    vec = psi.full().flatten()
    assert vec[2] == 1.0
    assert np.sum(np.abs(vec) ** 2) == 1.0


def test_spectral_overlap_bounds():
    """Test that spectral overlap is in [0,1]."""
    models.setup_environment(settings.SEED)
    d, k = 4, 2
    hams = models.random_hamiltonian_ensemble(d, k, "GOE", seed=42)
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=43)[0]

    # Test several lambda values
    for _ in range(5):
        lambdas = np.random.uniform(-1, 1, size=k)
        S = mathematics.spectral_overlap(lambdas, psi, phi, hams)
        assert 0 <= S <= 1, f"S={S} out of bounds [0,1]"


def test_eigendecomposition():
    """Test safe eigendecomposition."""
    models.setup_environment(settings.SEED)
    H = models.random_hermitian_matrix(d=5, real=True, seed=42)

    eigenvalues, eigenvectors = mathematics.eigendecompose(H)

    # Check shapes
    assert eigenvalues.shape == (5,)
    assert eigenvectors.shape == (5, 5)

    # Check eigenvalues are real
    assert np.all(np.isreal(eigenvalues))

    # Check that eigenvectors form orthonormal basis
    U = eigenvectors
    UTU = U.conj().T @ U
    assert np.allclose(UTU, np.eye(5), atol=1e-10)


def test_binomial_sem():
    """Test binomial SEM calculation."""
    sem = mathematics.compute_binomial_sem(p=0.5, n=100)
    expected = np.sqrt(0.5 * 0.5 / 100)
    assert np.abs(sem - expected) < 1e-10

    # Edge cases
    sem_zero = mathematics.compute_binomial_sem(p=0.0, n=100)
    assert sem_zero == 0.0

    sem_one = mathematics.compute_binomial_sem(p=1.0, n=100)
    assert sem_one == 0.0


def test_optimizer_registry():
    """Test optimizer registry access."""
    registry = optimize.get_optimizer_registry()

    assert "L-BFGS-B" in registry
    assert "CG" in registry
    assert "Powell" in registry

    assert registry["L-BFGS-B"]["supports_bounds"] is True
    assert registry["CG"]["supports_bounds"] is False


def test_maximize_spectral_overlap():
    """Test optimization with minimal problem."""
    models.setup_environment(settings.SEED)
    d, k = 4, 2
    hams = models.random_hamiltonian_ensemble(d, k, "GOE", seed=42)
    psi = models.fock_state(d, 0)
    phi = models.random_states(1, d, seed=43)[0]

    result = optimize.maximize_spectral_overlap(
        psi, phi, hams, method="L-BFGS-B", restarts=1, maxiter=50, seed=42
    )

    # Check result structure
    assert "best_value" in result
    assert "best_x" in result
    assert "nfev" in result
    assert "success" in result
    assert "runtime_s" in result

    # Check bounds
    assert 0 <= result["best_value"] <= 1
    assert len(result["best_x"]) == k
    assert np.all(np.abs(result["best_x"]) <= 1.0)


def test_monte_carlo_unreachability_small():
    """Test Monte Carlo analysis on tiny problem (fast)."""
    models.setup_environment(settings.SEED)

    # Very small sampling for speed
    results = analysis.monte_carlo_unreachability(
        dims=[3, 4], ks=[2], ensemble="GOE", tau=0.95, nks=3, nst=2, seed=42
    )

    # Check result structure
    assert (3, 2) in results
    assert (4, 2) in results

    # Check bounds
    for prob in results.values():
        assert 0 <= prob <= 1


def test_landscape_shapes():
    """Test landscape generation with minimal grid."""
    models.setup_environment(settings.SEED)

    L1, L2, S = analysis.landscape_spectral_overlap(
        d=4, k=2, ensemble="GOE", grid=5, n_targets=2, seed=42
    )

    # Check shapes
    assert L1.shape == (5, 5)
    assert L2.shape == (5, 5)
    assert S.shape == (5, 5)

    # Check bounds
    assert np.all(S >= 0)
    assert np.all(S <= 1)


def test_clip_to_bounds():
    """Test parameter clipping."""
    x = np.array([1.5, -0.5, -2.0, 0.3])
    bounds = [(-1.0, 1.0)] * 4

    x_clipped = mathematics.clip_to_bounds(x, bounds)

    assert np.all(x_clipped >= -1.0)
    assert np.all(x_clipped <= 1.0)
    assert x_clipped[0] == 1.0  # Clipped from 1.5
    assert x_clipped[1] == -0.5  # Unchanged
    assert x_clipped[2] == -1.0  # Clipped from -2.0
    assert x_clipped[3] == 0.3  # Unchanged


def test_deterministic_seeding():
    """Test that seeding produces reproducible results."""
    # First run
    models.setup_environment(42)
    hams1 = models.random_hamiltonian_ensemble(d=4, k=2, ensemble="GOE", seed=42)
    states1 = models.random_states(n=2, dim=4, seed=42)

    # Second run with same seed
    models.setup_environment(42)
    hams2 = models.random_hamiltonian_ensemble(d=4, k=2, ensemble="GOE", seed=42)
    states2 = models.random_states(n=2, dim=4, seed=42)

    # Check reproducibility
    for h1, h2 in zip(hams1, hams2):
        assert np.allclose(h1.full(), h2.full())

    for s1, s2 in zip(states1, states2):
        assert np.allclose(s1.full(), s2.full())
