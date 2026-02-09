#!/usr/bin/env python3
"""
Generate publication-quality plots for Floquet vs Static moment criterion comparison.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patches as mpatches

# Load data
with open('results/scaling_static_20260107_160738.pkl', 'rb') as f:
    static = pickle.load(f)

with open('results/scaling_floquet_o2_20260108_040340.pkl', 'rb') as f:
    floquet = pickle.load(f)

# Extract data
d = 16
K_static = np.array(static['K_values'])
P_static = np.array(static['P_values'])
n_static = static['n_trials']

K_floquet = np.array(floquet['K_values'])
P_floquet = np.array(floquet['P_values'])
n_floquet = floquet['n_trials']

# Convert to ρ
rho_static = K_static / (d**2)
rho_floquet = K_floquet / (d**2)

# Fitted parameters
alpha_s = static['fit']['alpha']
A_s = static['fit']['A']
lambda_s = 1.0 / (alpha_s * d**2)

alpha_f = floquet['fit']['alpha']
A_f = floquet['fit']['A']
lambda_f = 1.0 / (alpha_f * d**2)

# Error bars (Wilson score interval, 68% CI)
def wilson_error(k, n, confidence=0.68):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return np.array([0]), np.array([0])
    k = np.atleast_1d(k)
    p = k / n
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * np.sqrt(p * (1-p) / n + z**2 / (4*n**2)) / denominator
    lower = np.maximum(0, center - margin)
    upper = np.minimum(1, center + margin)
    return center - lower, upper - center

k_static = (P_static * n_static).astype(int)
k_floquet = (P_floquet * n_floquet).astype(int)

err_static = wilson_error(k_static, n_static)
err_floquet = wilson_error(k_floquet, n_floquet)

# Generate fitted curves
rho_fit = np.linspace(0, 0.035, 200)
P_static_fit = A_s * np.exp(-alpha_s * rho_fit * d**2)
P_floquet_fit = A_f * np.exp(-alpha_f * rho_fit * d**2)

# ============================================================================
# PLOT 1: Main Comparison (Linear Scale)
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 7))

# Plot data with error bars
ax.errorbar(rho_static, P_static, yerr=err_static,
            fmt='o', markersize=10, capsize=5, capthick=2,
            color='#2E86AB', label='Static Moment', linewidth=2,
            elinewidth=2, zorder=3)

ax.errorbar(rho_floquet, P_floquet, yerr=err_floquet,
            fmt='s', markersize=10, capsize=5, capthick=2,
            color='#A23B72', label='Floquet Moment (O(2))', linewidth=2,
            elinewidth=2, zorder=3)

# Plot fitted curves
ax.plot(rho_fit, P_static_fit, '--', color='#2E86AB', alpha=0.6,
        linewidth=2.5, label=f'Static fit: λ={lambda_s:.4f}')
ax.plot(rho_fit, P_floquet_fit, '--', color='#A23B72', alpha=0.6,
        linewidth=2.5, label=f'Floquet fit: λ={lambda_f:.4f}')

# Styling
ax.set_xlabel(r'$\rho = K/d^2$', fontsize=16, fontweight='bold')
ax.set_ylabel(r'$P(\mathrm{unreachable})$', fontsize=16, fontweight='bold')
ax.set_title('Moment Criterion: Static vs Floquet-Magnus',
             fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=13, loc='upper right', framealpha=0.95)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim(-0.001, 0.035)
ax.set_ylim(-0.05, 0.55)

# Add text box with key results
textstr = '\n'.join([
    f'System: 4 qubits (d={d})',
    f'Trials: {n_static} (static), {n_floquet} (Floquet)',
    f'λ ratio: {lambda_f/lambda_s:.3f}',
    f'R²: {static["fit"]["R_squared"]:.3f} (static), {floquet["fit"]["R_squared"]:.3f} (Floquet)'
])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.40, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('plots/floquet_static_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/floquet_static_comparison.png")
plt.close()

# ============================================================================
# PLOT 2: Log Scale
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 7))

# Filter out zeros for log plot
mask_static = P_static > 0
mask_floquet = P_floquet > 0

ax.errorbar(rho_static[mask_static], P_static[mask_static],
            yerr=[err[mask_static] for err in err_static],
            fmt='o', markersize=10, capsize=5, capthick=2,
            color='#2E86AB', label='Static Moment', linewidth=2,
            elinewidth=2, zorder=3)

ax.errorbar(rho_floquet[mask_floquet], P_floquet[mask_floquet],
            yerr=[err[mask_floquet] for err in err_floquet],
            fmt='s', markersize=10, capsize=5, capthick=2,
            color='#A23B72', label='Floquet Moment (O(2))', linewidth=2,
            elinewidth=2, zorder=3)

# Fitted curves
rho_fit_masked = rho_fit[P_static_fit > 1e-4]
P_static_fit_masked = P_static_fit[P_static_fit > 1e-4]
P_floquet_fit_masked = P_floquet_fit[P_static_fit > 1e-4]

ax.plot(rho_fit_masked, P_static_fit_masked, '--', color='#2E86AB',
        alpha=0.6, linewidth=2.5, label=f'Static: exp(-ρ/{lambda_s:.4f})')
ax.plot(rho_fit_masked, P_floquet_fit_masked, '--', color='#A23B72',
        alpha=0.6, linewidth=2.5, label=f'Floquet: exp(-ρ/{lambda_f:.4f})')

ax.set_yscale('log')
ax.set_xlabel(r'$\rho = K/d^2$', fontsize=16, fontweight='bold')
ax.set_ylabel(r'$P(\mathrm{unreachable})$ (log scale)', fontsize=16, fontweight='bold')
ax.set_title('Exponential Decay: Static vs Floquet Criterion',
             fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=13, loc='upper right', framealpha=0.95)
ax.grid(alpha=0.3, linestyle='--', which='both')
ax.set_xlim(-0.001, 0.025)
ax.set_ylim(5e-3, 1.0)

plt.tight_layout()
plt.savefig('plots/floquet_static_logscale.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/floquet_static_logscale.png")
plt.close()

# ============================================================================
# PLOT 3: Improvement Ratio Bar Chart
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 7))

# Calculate improvements for overlapping K values
K_overlap = K_static[K_static <= max(K_floquet)]
improvements = []
k_values_plot = []

for K in K_overlap:
    idx_s = np.where(K_static == K)[0][0]
    idx_f = np.where(K_floquet == K)[0][0]

    p_s = P_static[idx_s]
    p_f = P_floquet[idx_f]

    if p_s > 0:
        improvement = 100 * (p_f - p_s) / p_s
    elif p_f > 0:
        improvement = float('inf')
    else:
        improvement = 0

    if improvement != float('inf'):
        improvements.append(improvement)
        k_values_plot.append(K)

# Create bar chart
colors = ['#27AE60' if imp > 0 else '#E74C3C' for imp in improvements]
bars = ax.bar(k_values_plot, improvements, color=colors, alpha=0.8,
              edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (k, imp) in enumerate(zip(k_values_plot, improvements)):
    ax.text(k, imp + 5, f'+{imp:.0f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('K (number of operators)', fontsize=16, fontweight='bold')
ax.set_ylabel('Improvement (%)', fontsize=16, fontweight='bold')
ax.set_title('Floquet vs Static: Relative Improvement',
             fontsize=18, fontweight='bold', pad=20)
ax.set_xticks(k_values_plot)
ax.grid(alpha=0.3, axis='y', linestyle='--')

# Add annotation for K=3 (best improvement)
idx_k3 = k_values_plot.index(3)
ax.annotate(f'Peak improvement:\n+{improvements[idx_k3]:.0f}%',
            xy=(3, improvements[idx_k3]), xytext=(3.8, improvements[idx_k3] + 50),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'),
            fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('plots/floquet_improvement_ratio.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/floquet_improvement_ratio.png")
plt.close()

# ============================================================================
# PLOT 4: Side-by-side comparison table visualization
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Create grouped bar chart
x = np.arange(len(K_overlap))
width = 0.35

bars1 = ax.bar(x - width/2, [P_static[np.where(K_static == k)[0][0]] for k in K_overlap],
               width, label='Static', color='#2E86AB', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, [P_floquet[np.where(K_floquet == k)[0][0]] for k in K_overlap],
               width, label='Floquet', color='#A23B72', alpha=0.8, edgecolor='black')

# Add value labels
for i, k in enumerate(K_overlap):
    p_s = P_static[np.where(K_static == k)[0][0]]
    p_f = P_floquet[np.where(K_floquet == k)[0][0]]

    ax.text(i - width/2, p_s + 0.01, f'{p_s:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(i + width/2, p_f + 0.01, f'{p_f:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('K (number of operators)', fontsize=16, fontweight='bold')
ax.set_ylabel(r'$P(\mathrm{unreachable})$', fontsize=16, fontweight='bold')
ax.set_title('Direct Comparison: Static vs Floquet at Each K',
             fontsize=18, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(K_overlap)
ax.legend(fontsize=14, loc='upper right')
ax.grid(alpha=0.3, axis='y', linestyle='--')
ax.set_ylim(0, 0.5)

plt.tight_layout()
plt.savefig('plots/floquet_static_bars.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plots/floquet_static_bars.png")
plt.close()

print("\n" + "="*70)
print("ALL PLOTS GENERATED SUCCESSFULLY")
print("="*70)
print("\nFiles created:")
print("  1. plots/floquet_static_comparison.png - Main comparison with fits")
print("  2. plots/floquet_static_logscale.png   - Log-scale decay curves")
print("  3. plots/floquet_improvement_ratio.png - Improvement bar chart")
print("  4. plots/floquet_static_bars.png       - Direct side-by-side comparison")
