#!/usr/bin/env python3
"""
Verify Floquet implementation correctness.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from reach import floquet, models

print("="*70)
print("FLOQUET IMPLEMENTATION VERIFICATION")
print("="*70)

# Test 1: Verify 2j is complex literal
print("\n1. Python complex literal test:")
j = 5
print(f"   j = 5")
print(f"   2j = {2j}")  # Should print 2j (complex)
print(f"   2*j = {2*j}")  # Should print 10
print(f"   1/(2j) = {1/(2j)}")  # Should print -0.5j
assert 2j == 2 * 1j, "Complex literal test failed!"
print("   ✓ Complex literal working correctly")

# Test 2: Generate test Hamiltonians
print("\n2. Generate test Hamiltonians:")
nx, ny = 2, 2
d = 2**(nx*ny)
K = 5
hams_qutip = models.random_hamiltonian_ensemble(
    dim=d, k=K, ensemble="GEO2", nx=nx, ny=ny, seed=42
)
hams = floquet.hamiltonians_to_numpy(hams_qutip)
print(f"   Generated {K} Hamiltonians of dimension {d}")

# Verify Hamiltonians are Hermitian
for i, H in enumerate(hams):
    assert np.allclose(H, H.conj().T), f"H_{i} is not Hermitian!"
print("   ✓ All Hamiltonians are Hermitian")

# Test 3: First-order Magnus
print("\n3. First-order Magnus (time-averaged):")
lambdas = np.random.randn(K) / np.sqrt(K)
T = 1.0
driving = floquet.create_driving_functions(K, 'sinusoidal', T, seed=42)

H_F1 = floquet.compute_floquet_hamiltonian_order1(hams, lambdas, driving, T)
print(f"   H_F^(1) shape: {H_F1.shape}")
print(f"   H_F^(1) Hermitian: {np.allclose(H_F1, H_F1.conj().T)}")
print(f"   H_F^(1) norm: {np.linalg.norm(H_F1):.6f}")
assert np.allclose(H_F1, H_F1.conj().T), "H_F^(1) is not Hermitian!"
print("   ✓ H_F^(1) is Hermitian (correct)")

# Test 4: Second-order Magnus
print("\n4. Second-order Magnus (commutator corrections):")
H_F2 = floquet.compute_floquet_hamiltonian_order2(hams, lambdas, driving, T, n_fourier_terms=10)
print(f"   H_F^(2) shape: {H_F2.shape}")
print(f"   H_F^(2) norm: {np.linalg.norm(H_F2):.6f}")

# Check Hermiticity of H_F^(2)
# Mathematical derivation:
# - [H_j, H_k] is anti-Hermitian when H_j, H_k are Hermitian
# - F_jk is real
# - / (2j) means * (-i/2)
# - anti-Hermitian * (-i/2) = Hermitian (because i * anti-Hermitian = Hermitian)
is_hermitian = np.allclose(H_F2, H_F2.conj().T)
is_anti_hermitian = np.allclose(H_F2, -H_F2.conj().T)
print(f"   H_F^(2) Hermitian: {is_hermitian}")
print(f"   H_F^(2) anti-Hermitian: {is_anti_hermitian}")

# H_F^(2) SHOULD be Hermitian for total H_F to be Hermitian
if is_hermitian:
    print("   ✓ H_F^(2) is Hermitian (correct)")
elif is_anti_hermitian:
    print("   ⚠️ H_F^(2) is anti-Hermitian (possible bug in implementation)")
    print("   Note: This could indicate wrong sign in the 1/(2j) factor")
else:
    print("   ? H_F^(2) is neither Hermitian nor anti-Hermitian")
    print(f"     Max deviation from Hermitian: {np.max(np.abs(H_F2 - H_F2.conj().T)):.2e}")

# Test 5: Full Floquet Hamiltonian
print("\n5. Full Floquet Hamiltonian:")
H_F = floquet.compute_floquet_hamiltonian(hams, lambdas, driving, T, order=2)
print(f"   H_F shape: {H_F.shape}")
print(f"   H_F Hermitian: {np.allclose(H_F, H_F.conj().T)}")
print(f"   H_F norm: {np.linalg.norm(H_F):.6f}")
if np.linalg.norm(H_F1) > 0:
    print(f"   Ratio |H_F^(2)|/|H_F^(1)|: {np.linalg.norm(H_F2)/np.linalg.norm(H_F1):.4f}")
else:
    print(f"   Note: |H_F^(1)| ≈ 0 (sinusoidal has zero DC)")

# Test 6: Floquet derivatives
print("\n6. Floquet derivatives ∂H_F/∂λ_k:")
dH_F = floquet.compute_floquet_hamiltonian_derivative(hams, lambdas, driving, T, order=2)
print(f"   Number of derivatives: {len(dH_F)}")

# Check that derivatives are Hermitian
all_hermitian = all(np.allclose(dH, dH.conj().T) for dH in dH_F)
print(f"   All derivatives Hermitian: {all_hermitian}")

# Test 7: Floquet moment criterion
print("\n7. Floquet moment criterion:")
from reach import states
psi = states.random_state(d, seed=42)
phi = states.random_state(d, seed=43)

definite, x_opt, Q_eigenvalues = floquet.floquet_moment_criterion(
    psi, phi, hams, lambdas, driving, T, order=2
)
print(f"   Definite (UNREACHABLE): {definite}")
if x_opt is not None:
    print(f"   Optimal x: {x_opt:.6f}")
print(f"   Q_F eigenvalues: [{Q_eigenvalues.min():.4f}, {Q_eigenvalues.max():.4f}]")

# Summary
print("\n" + "="*70)
print("VERIFICATION SUMMARY")
print("="*70)
print(f"  Complex literal (2j): ✓ Correct")
print(f"  H_F^(1) Hermitian: ✓ Correct")
print(f"  H_F^(2) Hermitian: {'✓ Correct' if is_hermitian else '⚠️ Check needed'}")
print(f"  Derivatives Hermitian: {'✓ Correct' if all_hermitian else '⚠️ Check needed'}")
print(f"  Moment criterion: ✓ Functional")

if is_hermitian and all_hermitian:
    print("\n✅ Implementation verified! Ready for experiments.")
    sys.exit(0)
else:
    print("\n⚠️ Some checks failed. Review implementation before proceeding.")
    sys.exit(1)
