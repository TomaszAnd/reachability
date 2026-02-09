#!/usr/bin/env python3
"""
Publication Experiment Configuration

Defines optimal sampling parameters for generating clean, publication-quality data.

Design principles:
1. Higher trial counts (200+) for smooth curves
2. Denser sampling near ρ_c transitions
3. Consistent settings across ensembles
4. Reproducible with fixed seeds
"""

import numpy as np

# =============================================================================
# OPTIMIZER SETTINGS (from audit findings)
# =============================================================================

OPTIMIZER_SETTINGS = {
    'canonical': {
        'maxiter': 100,
        'restarts': 3,
        'ftol': 1e-6,
    },
    'GUE': {
        'maxiter': 100,
        'restarts': 3,
        'ftol': 1e-6,
    },
    'GEO2': {
        'maxiter': 100,  # d≤32
        'restarts': 3,
        'ftol': 1e-6,
    },
    'GEO2_d64': {
        'maxiter': 150,  # d=64 needs more iterations
        'restarts': 3,
        'ftol': 1e-6,
    },
}

# =============================================================================
# SAMPLING CONFIGURATION
# =============================================================================

# Transition density estimates from previous runs (ρ_c at τ=0.99)
TRANSITION_ESTIMATES = {
    'canonical': {
        8: 0.15,
        16: 0.10,
        32: 0.06,
        64: 0.03,
    },
    'GEO2': {
        8: 0.08,
        16: 0.035,
        32: 0.02,
        64: 0.015,  # Estimated
    },
}


def get_rho_points(ensemble: str, d: int, n_transition: int = 20) -> np.ndarray:
    """
    Generate ρ sampling points with dense sampling near transition.

    Args:
        ensemble: 'canonical' or 'GEO2'
        d: Hilbert space dimension
        n_transition: Number of points in transition region

    Returns:
        Sorted array of ρ values
    """
    rho_c = TRANSITION_ESTIMATES.get(ensemble, {}).get(d, 0.05)

    # Region before transition (coarse)
    rho_low = np.linspace(0.005, max(0.01, rho_c - 0.05), 5)

    # Transition region (dense)
    rho_trans = np.linspace(
        max(0.01, rho_c - 0.03),
        min(0.20, rho_c + 0.05),
        n_transition
    )

    # After transition (coarse)
    rho_high = np.linspace(rho_c + 0.07, min(0.30, rho_c + 0.15), 5)

    # Combine and deduplicate
    rho_all = np.unique(np.concatenate([rho_low, rho_trans, rho_high]))
    return rho_all


# =============================================================================
# CANONICAL CONFIGURATION
# =============================================================================

CANONICAL_CONFIG = {
    'dimensions': [8, 16, 32, 64],
    'tau_values': [0.99, 0.95, 0.90],
    'n_trials': 200,
    'criteria': ['spectral', 'krylov', 'moment'],
    'seed_base': 10000,

    # Dimension-specific sampling
    'rho_points': {
        8: get_rho_points('canonical', 8),
        16: get_rho_points('canonical', 16),
        32: get_rho_points('canonical', 32),
        64: get_rho_points('canonical', 64),
    },
}


# =============================================================================
# GEO2 CONFIGURATION
# =============================================================================

GEO2_CONFIG = {
    'dimensions': [8, 16, 32],
    'tau_values': [0.99],  # Only τ=0.99 for GEO2
    'n_trials': 200,
    'criteria': ['spectral', 'krylov', 'moment'],
    'seed_base': 20000,

    # Lattice configurations
    'lattice_params': {
        8: {'nx': 1, 'ny': 3},   # 3 qubits = 2^3 = 8
        16: {'nx': 2, 'ny': 2},  # 4 qubits = 2^4 = 16
        32: {'nx': 1, 'ny': 5},  # 5 qubits = 2^5 = 32
    },

    # Dimension-specific sampling
    'rho_points': {
        8: get_rho_points('GEO2', 8),
        16: get_rho_points('GEO2', 16),
        32: get_rho_points('GEO2', 32),
    },
}


# =============================================================================
# GEO2 d=64 SPECIAL CONFIGURATION (long run)
# =============================================================================

GEO2_D64_CONFIG = {
    'dimensions': [64],
    'tau_values': [0.99],
    'n_trials': 100,  # Reduced for feasibility
    'criteria': ['spectral', 'moment'],  # Skip Krylov (uninformative)
    'seed_base': 30000,

    # 6 qubits = 2^6 = 64
    'lattice_params': {
        64: {'nx': 2, 'ny': 3},
    },

    # Conservative sampling
    'rho_points': {
        64: np.array([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]),
    },

    # Estimated runtime: 8 points × 100 trials × 20s/trial ≈ 4.5 hours
}


# =============================================================================
# PILOT CONFIGURATION (quick test)
# =============================================================================

PILOT_CONFIG = {
    'canonical': {
        'dimensions': [16, 32],
        'tau_values': [0.99],
        'n_trials': 50,
        'criteria': ['spectral', 'krylov', 'moment'],
        'seed_base': 40000,
        'rho_points': {
            16: np.linspace(0.02, 0.15, 10),
            32: np.linspace(0.02, 0.10, 10),
        },
    },
    'GEO2': {
        'dimensions': [16, 32],
        'tau_values': [0.99],
        'n_trials': 50,
        'criteria': ['spectral', 'krylov', 'moment'],
        'seed_base': 50000,
        'lattice_params': {
            16: {'nx': 2, 'ny': 2},
            32: {'nx': 1, 'ny': 5},
        },
        'rho_points': {
            16: np.linspace(0.01, 0.08, 10),
            32: np.linspace(0.005, 0.05, 10),
        },
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config(config_name: str) -> dict:
    """Get configuration by name."""
    configs = {
        'canonical': CANONICAL_CONFIG,
        'geo2': GEO2_CONFIG,
        'geo2_d64': GEO2_D64_CONFIG,
        'pilot': PILOT_CONFIG,
    }
    return configs.get(config_name.lower())


def estimate_runtime(config: dict, ensemble: str, time_per_trial_s: float = 1.0) -> float:
    """
    Estimate total runtime in hours.

    Args:
        config: Configuration dictionary
        ensemble: 'canonical' or 'GEO2'
        time_per_trial_s: Estimated seconds per trial

    Returns:
        Estimated runtime in hours
    """
    total_trials = 0

    for d in config['dimensions']:
        n_rho = len(config['rho_points'].get(d, []))
        n_tau = len(config['tau_values'])
        n_trials = config['n_trials']
        n_criteria = len(config['criteria'])

        total_trials += n_rho * n_tau * n_trials * n_criteria

    return total_trials * time_per_trial_s / 3600


def print_config_summary(config_name: str):
    """Print summary of a configuration."""
    config = get_config(config_name)
    if config is None:
        print(f"Unknown config: {config_name}")
        return

    print(f"\n{'='*60}")
    print(f"Configuration: {config_name.upper()}")
    print(f"{'='*60}")

    if 'canonical' in config:
        # Nested config (like PILOT_CONFIG)
        for ens, ens_config in config.items():
            print(f"\n  {ens}:")
            print(f"    Dimensions: {ens_config['dimensions']}")
            print(f"    Tau values: {ens_config['tau_values']}")
            print(f"    Trials: {ens_config['n_trials']}")
            print(f"    Criteria: {ens_config['criteria']}")
    else:
        # Flat config
        print(f"  Dimensions: {config['dimensions']}")
        print(f"  Tau values: {config['tau_values']}")
        print(f"  Trials: {config['n_trials']}")
        print(f"  Criteria: {config['criteria']}")

        for d in config['dimensions']:
            rho = config['rho_points'].get(d, [])
            print(f"  d={d}: {len(rho)} ρ points from {rho[0]:.4f} to {rho[-1]:.4f}")


if __name__ == '__main__':
    print("Publication Experiment Configurations")
    print("=" * 60)

    for name in ['canonical', 'geo2', 'geo2_d64', 'pilot']:
        print_config_summary(name)
