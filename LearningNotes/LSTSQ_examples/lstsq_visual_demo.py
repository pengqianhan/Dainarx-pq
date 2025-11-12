"""
Visual demonstration of np.linalg.lstsq in NARX model
Shows the complete workflow with plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

print("Generating visual demonstration of least squares fitting...")

# ============================================================================
# Generate synthetic data
# ============================================================================
np.random.seed(42)

# True system: x[t] = 0.8*x[t-1] + 0.1*x[t-2] + 0.5*u[t] + 0.2
true_a1, true_a2, true_b, true_bias = 0.8, 0.1, 0.5, 0.2

T = 50
x = np.zeros(T)
u = np.sin(np.linspace(0, 4*np.pi, T)) * 0.2  # Sinusoidal input

x[0], x[1] = 1.0, 1.2

for t in range(2, T):
    x[t] = true_a1*x[t-1] + true_a2*x[t-2] + true_b*u[t] + true_bias
    x[t] += np.random.randn() * 0.02

# ============================================================================
# Construct least squares problem
# ============================================================================
order = 2
A = []
b = []

for t in range(order, T):
    feature = [x[t-1], x[t-2], u[t], 1.0]
    A.append(feature)
    b.append(x[t])

A = np.array(A)
b = np.array(b)

# ============================================================================
# Solve using np.linalg.lstsq (DEConfig.py line 136)
# ============================================================================
params = np.linalg.lstsq(A, b, rcond=None)[0]

predictions = A @ params
errors = np.abs(predictions - b)

print(f"\nTrue parameters:      [{true_a1}, {true_a2}, {true_b}, {true_bias}]")
print(f"Estimated parameters: [{params[0]:.4f}, {params[1]:.4f}, {params[2]:.4f}, {params[3]:.4f}]")
print(f"Max fitting error: {np.max(errors):.6f}")

# ============================================================================
# Create visualization
# ============================================================================
fig = plt.figure(figsize=(16, 12))

# Plot 1: Time series and input signal
ax1 = plt.subplot(3, 3, 1)
t_plot = np.arange(T)
ax1.plot(t_plot, x, 'b-', linewidth=2, label='State x[t]')
ax1.set_xlabel('Time step t', fontsize=10)
ax1.set_ylabel('x[t]', fontsize=10)
ax1.set_title('(1) Generated Time Series', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2 = plt.subplot(3, 3, 2)
ax2.plot(t_plot, u, 'g-', linewidth=2, label='Input u[t]')
ax2.set_xlabel('Time step t', fontsize=10)
ax2.set_ylabel('u[t]', fontsize=10)
ax2.set_title('(2) Input Signal', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Feature matrix visualization
ax3 = plt.subplot(3, 3, 3)
im = ax3.imshow(A[:20, :], aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax3.set_xlabel('Feature index', fontsize=10)
ax3.set_ylabel('Sample index', fontsize=10)
ax3.set_title('(3) Feature Matrix A (first 20 rows)', fontsize=11, fontweight='bold')
ax3.set_xticks([0, 1, 2, 3])
ax3.set_xticklabels(['x[t-1]', 'x[t-2]', 'u[t]', 'bias'], fontsize=8)
plt.colorbar(im, ax=ax3)

# Plot 4: Target vector
ax4 = plt.subplot(3, 3, 4)
ax4.plot(b[:20], 'ro-', markersize=6, linewidth=1.5)
ax4.set_xlabel('Sample index', fontsize=10)
ax4.set_ylabel('b (target x[t])', fontsize=10)
ax4.set_title('(4) Target Vector b (first 20 elements)', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Scatter plot of features
ax5 = plt.subplot(3, 3, 5)
ax5.scatter(A[:, 0], b, alpha=0.6, s=30, label='x[t] vs x[t-1]')
ax5.set_xlabel('x[t-1]', fontsize=10)
ax5.set_ylabel('x[t]', fontsize=10)
ax5.set_title('(5) Feature-Target Relationship', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()

# Plot 6: Parameter comparison
ax6 = plt.subplot(3, 3, 6)
param_names = ['a1', 'a2', 'b', 'bias']
x_pos = np.arange(len(param_names))
width = 0.35
ax6.bar(x_pos - width/2, [true_a1, true_a2, true_b, true_bias], 
        width, label='True', alpha=0.8, color='blue')
ax6.bar(x_pos + width/2, params, width, label='Estimated', alpha=0.8, color='orange')
ax6.set_xlabel('Parameters', fontsize=10)
ax6.set_ylabel('Value', fontsize=10)
ax6.set_title('(6) Parameter Comparison', fontsize=11, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(param_names)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# Plot 7: Predictions vs True values
ax7 = plt.subplot(3, 3, 7)
t_fit = np.arange(order, T)
ax7.plot(t_fit, b, 'o-', markersize=4, linewidth=1.5, label='True x[t]', alpha=0.7)
ax7.plot(t_fit, predictions, 's--', markersize=4, linewidth=1.5, label='Predicted x[t]', alpha=0.7)
ax7.set_xlabel('Time step t', fontsize=10)
ax7.set_ylabel('x[t]', fontsize=10)
ax7.set_title('(7) Fitting Results: True vs Predicted', fontsize=11, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Plot 8: Fitting errors
ax8 = plt.subplot(3, 3, 8)
ax8.plot(t_fit, errors, 'r-', linewidth=2)
ax8.axhline(y=np.mean(errors), color='b', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
ax8.axhline(y=np.max(errors), color='g', linestyle='--', 
            linewidth=2, label=f'Max: {np.max(errors):.4f}')
ax8.set_xlabel('Time step t', fontsize=10)
ax8.set_ylabel('Absolute error', fontsize=10)
ax8.set_title('(8) Fitting Errors |Ax - b|', fontsize=11, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9: Error histogram
ax9 = plt.subplot(3, 3, 9)
ax9.hist(errors, bins=20, alpha=0.7, color='purple', edgecolor='black')
ax9.axvline(x=np.mean(errors), color='r', linestyle='--', 
            linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
ax9.set_xlabel('Error magnitude', fontsize=10)
ax9.set_ylabel('Frequency', fontsize=10)
ax9.set_title('(9) Error Distribution', fontsize=11, fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3, axis='y')

plt.suptitle('np.linalg.lstsq in NARX Model (DEConfig.py:136)', 
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('result/lstsq_visual_demo.png', dpi=150, bbox_inches='tight')
print("Figure saved to: result/lstsq_visual_demo.png")

# ============================================================================
# Create a detailed workflow diagram
# ============================================================================
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Diagram 1: Data structure
ax = axes[0, 0]
ax.text(0.5, 0.9, 'Step 1: Data Structure', ha='center', va='top', 
        fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.1, 0.75, 'Time Series:', fontsize=11, fontweight='bold', transform=ax.transAxes)
ax.text(0.1, 0.65, f't:  0    1    2    3    4    5    ...', 
        fontsize=9, family='monospace', transform=ax.transAxes)
ax.text(0.1, 0.60, f'x: {x[0]:.2f} {x[1]:.2f} {x[2]:.2f} {x[3]:.2f} {x[4]:.2f} {x[5]:.2f} ...', 
        fontsize=9, family='monospace', transform=ax.transAxes)
ax.text(0.1, 0.55, f'u: {u[0]:.2f} {u[1]:.2f} {u[2]:.2f} {u[3]:.2f} {u[4]:.2f} {u[5]:.2f} ...', 
        fontsize=9, family='monospace', transform=ax.transAxes)

ax.text(0.1, 0.40, 'Sliding Window (order=2):', fontsize=11, fontweight='bold', transform=ax.transAxes)
ax.text(0.1, 0.30, 't=2: [x[1], x[0], u[2], 1] -> x[2]', 
        fontsize=9, family='monospace', transform=ax.transAxes, color='blue')
ax.text(0.1, 0.25, 't=3: [x[2], x[1], u[3], 1] -> x[3]', 
        fontsize=9, family='monospace', transform=ax.transAxes, color='blue')
ax.text(0.1, 0.20, 't=4: [x[3], x[2], u[4], 1] -> x[4]', 
        fontsize=9, family='monospace', transform=ax.transAxes, color='blue')
ax.text(0.1, 0.15, '...', fontsize=9, family='monospace', transform=ax.transAxes)
ax.axis('off')

# Diagram 2: Matrix construction
ax = axes[0, 1]
ax.text(0.5, 0.9, 'Step 2: Matrix Construction', ha='center', va='top', 
        fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.1, 0.75, 'Matrix A (features):', fontsize=11, fontweight='bold', transform=ax.transAxes)
for i in range(5):
    ax.text(0.1, 0.65-i*0.08, 
            f'[{A[i,0]:6.3f}, {A[i,1]:6.3f}, {A[i,2]:6.3f}, {A[i,3]:6.3f}]', 
            fontsize=8, family='monospace', transform=ax.transAxes)

ax.text(0.7, 0.75, 'Vector b:', fontsize=11, fontweight='bold', transform=ax.transAxes)
for i in range(5):
    ax.text(0.7, 0.65-i*0.08, f'[{b[i]:6.3f}]', 
            fontsize=8, family='monospace', transform=ax.transAxes)

ax.text(0.1, 0.20, f'Shape: A={A.shape}, b={b.shape}', 
        fontsize=10, transform=ax.transAxes, color='red')
ax.axis('off')

# Diagram 3: Least squares solution
ax = axes[1, 0]
ax.text(0.5, 0.9, 'Step 3: Least Squares Solution', ha='center', va='top', 
        fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.1, 0.75, 'Minimize: ||Ax - b||²', fontsize=12, 
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax.text(0.1, 0.60, 'Code:', fontsize=11, fontweight='bold', transform=ax.transAxes)
ax.text(0.1, 0.50, 'x = np.linalg.lstsq(A, b, rcond=None)[0]', 
        fontsize=10, family='monospace', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(0.1, 0.35, 'Solution:', fontsize=11, fontweight='bold', transform=ax.transAxes)
ax.text(0.1, 0.25, f'a1 (x[t-1]) = {params[0]:.6f}', fontsize=9, transform=ax.transAxes)
ax.text(0.1, 0.20, f'a2 (x[t-2]) = {params[1]:.6f}', fontsize=9, transform=ax.transAxes)
ax.text(0.1, 0.15, f'b  (u[t])   = {params[2]:.6f}', fontsize=9, transform=ax.transAxes)
ax.text(0.1, 0.10, f'bias        = {params[3]:.6f}', fontsize=9, transform=ax.transAxes)
ax.axis('off')

# Diagram 4: Model equation
ax = axes[1, 1]
ax.text(0.5, 0.9, 'Step 4: Fitted NARX Model', ha='center', va='top', 
        fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.70, 'x[t] = a1·x[t-1] + a2·x[t-2] + b·u[t] + bias', 
        fontsize=11, ha='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.text(0.5, 0.50, f'x[t] = {params[0]:.3f}·x[t-1] + {params[1]:.3f}·x[t-2]', 
        fontsize=10, ha='center', transform=ax.transAxes)
ax.text(0.5, 0.42, f'       + {params[2]:.3f}·u[t] + {params[3]:.3f}', 
        fontsize=10, ha='center', transform=ax.transAxes)
ax.text(0.5, 0.25, 'Quality Metrics:', fontsize=11, fontweight='bold', 
        ha='center', transform=ax.transAxes)
ax.text(0.5, 0.15, f'Max Error: {np.max(errors):.6f}', 
        fontsize=10, ha='center', transform=ax.transAxes)
ax.text(0.5, 0.08, f'Mean Error: {np.mean(errors):.6f}', 
        fontsize=10, ha='center', transform=ax.transAxes)
ax.axis('off')

plt.suptitle('NARX Least Squares Workflow (DEConfig.py)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('result/lstsq_workflow.png', dpi=150, bbox_inches='tight')
print("Workflow diagram saved to: result/lstsq_workflow.png")

print("\nVisualization complete!")
print("\nGenerated files:")
print("  1. result/lstsq_visual_demo.png - Complete visualization")
print("  2. result/lstsq_workflow.png - Step-by-step workflow")

