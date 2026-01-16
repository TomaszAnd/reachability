# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-03

### Added
- Initial release of reach package
- Time-free quantum reachability analysis using spectral overlap criterion
- Support for GOE (Gaussian Orthogonal Ensemble) and GUE (Gaussian Unitary Ensemble) random matrices
- Monte Carlo analysis functions for unreachability probability estimation
- Multiple optimization methods (L-BFGS-B, CG, Powell, TNC, SLSQP, Nelder-Mead)
- Publication-quality visualization tools with exact filename specifications
- Complete CLI with subcommands for all analysis types
- Deterministic seeding for full reproducibility
- Fast/full sampling modes for quick validation vs production runs
- Comprehensive documentation with mathematical equations
- Development tooling: pre-commit hooks, CI/CD, automated testing

### Mathematical Framework
- Parameterized Hamiltonian: H(λ) = Σᵢ λᵢ Hᵢ
- Spectral overlap: S(λ) = Σₙ |φₙ*(λ) ψₙ(λ)|
- Unreachability criterion: max_λ S(λ) < τ ⇒ unreachable
- Comparison with old moment-based (τ-free) criterion

### Features
- `monte_carlo_unreachability()`: P_unreach(d,K;τ) estimation
- `probability_vs_tau()`: Threshold sensitivity analysis
- `optimizer_Sstar_comparison()`: Optimizer validation
- `probability_vs_iterations()`: Convergence analysis
- `landscape_spectral_overlap()`: S(λ₁,λ₂) visualization
- `old_criterion_probabilities()`: Legacy criterion for comparison

### Repository Structure
- `reach/`: Core package with 7 modules (models, mathematics, optimize, analysis, viz, cli, settings)
- `scripts/`: Helper scripts for figure generation
- `tests/`: Smoke tests with minimal runtime
- `fig_summary/`: Output directory for publication figures
- Complete dev tooling: pyproject.toml, pre-commit, GitHub Actions

### Dependencies
- Python >= 3.10
- numpy >= 1.23
- scipy >= 1.9
- matplotlib >= 3.6
- qutip >= 4.7

[0.1.0]: https://github.com/<username>/reach/releases/tag/v0.1.0
