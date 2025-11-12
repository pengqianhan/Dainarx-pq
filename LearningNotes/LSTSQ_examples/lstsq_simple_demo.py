"""
Simple demonstration of np.linalg.lstsq in NARX model (DEConfig.py:136)
This shows exactly how the least squares fitting works step by step.
"""

import numpy as np

print("="*80)
print("DEMONSTRATION: np.linalg.lstsq in DEConfig.py line 136")
print("="*80)

# ============================================================================
# EXAMPLE 1: Simple 1D AR(2) Model
# ============================================================================
print("\n[EXAMPLE 1] Simple 1D AR(2) Model")
print("-"*80)

# True system: x[t] = 0.8*x[t-1] + 0.1*x[t-2] + 0.5*u[t] + 0.2
true_params = [0.8, 0.1, 0.5, 0.2]  # [a1, a2, b, bias]

# Generate synthetic data
np.random.seed(42)
T = 20
x = np.zeros(T)
u = np.random.randn(T) * 0.1

x[0], x[1] = 1.0, 1.2

for t in range(2, T):
    x[t] = 0.8*x[t-1] + 0.1*x[t-2] + 0.5*u[t] + 0.2
    x[t] += np.random.randn() * 0.01  # small noise

print(f"\nTrue parameters: a1={true_params[0]}, a2={true_params[1]}, " +
      f"b={true_params[2]}, bias={true_params[3]}")
print(f"Time series x: {x[:8]}")

# ============================================================================
# STEP 1: Construct matrix A and vector b
# ============================================================================
print("\n" + "-"*80)
print("STEP 1: Construct regression matrix A and target vector b")
print("-"*80)

order = 2
A = []
b = []

for t in range(order, T):
    # Feature vector: [x[t-1], x[t-2], u[t], 1]
    feature = [x[t-1], x[t-2], u[t], 1.0]
    A.append(feature)
    b.append(x[t])

A = np.array(A)
b = np.array(b)

print(f"\nMatrix A shape: {A.shape} (num_samples x num_features)")
print(f"Vector b shape: {b.shape}")
print("\nFirst 5 rows of matrix A:")
print("     x[t-1]    x[t-2]      u[t]     bias")
for i in range(min(5, len(A))):
    print(f"t={i+2}: [{A[i,0]:7.4f}, {A[i,1]:7.4f}, {A[i,2]:7.4f}, {A[i,3]:7.4f}]")

print(f"\nFirst 5 elements of vector b (target x[t]):")
print(b[:5])

# ============================================================================
# STEP 2: Solve using np.linalg.lstsq (THIS IS LINE 136 in DEConfig.py!)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Solve min ||Ax - b||^2 using np.linalg.lstsq")
print("="*80)

# THIS IS THE KEY LINE FROM DEConfig.py:136
params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

print(f"\nEstimated parameters:")
print(f"  a1 (coef of x[t-1]) = {params[0]:.6f}  (true: {true_params[0]})")
print(f"  a2 (coef of x[t-2]) = {params[1]:.6f}  (true: {true_params[1]})")
print(f"  b  (coef of u[t])   = {params[2]:.6f}  (true: {true_params[2]})")
print(f"  bias                = {params[3]:.6f}  (true: {true_params[3]})")

# Calculate fitting error (as done in DEConfig.py:138)
predictions = A @ params
errors = np.abs(predictions - b)
max_error = np.max(errors)

print(f"\nFitting error:")
print(f"  Max error: {max_error:.6f}")
print(f"  Mean error: {np.mean(errors):.6f}")

# ============================================================================
# EXAMPLE 2: Multi-variable system (like Duffing oscillator)
# ============================================================================
print("\n\n" + "="*80)
print("[EXAMPLE 2] 2D Nonlinear System (simulating Duffing oscillator)")
print("="*80)

# True system:
# x1[t] = 0.9*x1[t-1] + 0.5*x2[t-1] + 0.1
# x2[t] = -0.3*x1[t-1] + 0.8*x2[t-1] + 0.2*x1[t-1]^2 + 0.05

T = 30
x1 = np.zeros(T)
x2 = np.zeros(T)
u = np.zeros(T)

x1[0], x2[0] = 1.0, 0.5

for t in range(1, T):
    x1[t] = 0.9*x1[t-1] + 0.5*x2[t-1] + 0.1
    x2[t] = -0.3*x1[t-1] + 0.8*x2[t-1] + 0.2*x1[t-1]**2 + 0.05
    x1[t] += np.random.randn() * 0.005
    x2[t] += np.random.randn() * 0.005

print(f"\nGenerated time series:")
print(f"x1: {x1[:6]}")
print(f"x2: {x2[:6]}")

# ============================================================================
# Fit x1 (with nonlinear term x1^2)
# ============================================================================
print("\n" + "-"*80)
print("Fitting x1 with nonlinear term x1^2")
print("-"*80)

order = 1
A1 = []
b1 = []

for t in range(order, T):
    # Feature: [x1[t-1], x2[t-1], x1[t-1]^2, u[t], 1]
    feature = [x1[t-1], x2[t-1], x1[t-1]**2, u[t], 1.0]
    A1.append(feature)
    b1.append(x1[t])

A1 = np.array(A1)
b1 = np.array(b1)

print(f"\nMatrix A1 shape: {A1.shape}")
print("\nFirst 3 rows:")
print("     x1[t-1]   x2[t-1]   x1^2[t-1]   u[t]     bias")
for i in range(min(3, len(A1))):
    print(f"t={i+1}: [{A1[i,0]:7.4f}, {A1[i,1]:7.4f}, {A1[i,2]:7.4f}, " +
          f"{A1[i,3]:7.4f}, {A1[i,4]:7.4f}]")

# Solve for x1
params1 = np.linalg.lstsq(A1, b1, rcond=None)[0]

print(f"\nx1 fitted parameters:")
print(f"  coef(x1[t-1]) = {params1[0]:.6f}  (true: 0.9)")
print(f"  coef(x2[t-1]) = {params1[1]:.6f}  (true: 0.5)")
print(f"  coef(x1^2)    = {params1[2]:.6f}  (true: 0.0)")
print(f"  coef(u[t])    = {params1[3]:.6f}  (true: 0.0)")
print(f"  bias          = {params1[4]:.6f}  (true: 0.1)")

err1 = np.max(np.abs(A1 @ params1 - b1))
print(f"  Max error: {err1:.6f}")

# ============================================================================
# Fit x2 (with nonlinear term x1^2)
# ============================================================================
print("\n" + "-"*80)
print("Fitting x2 with nonlinear term x1^2")
print("-"*80)

A2 = []
b2 = []

for t in range(order, T):
    feature = [x1[t-1], x2[t-1], x1[t-1]**2, u[t], 1.0]
    A2.append(feature)
    b2.append(x2[t])

A2 = np.array(A2)
b2 = np.array(b2)

# Solve for x2
params2 = np.linalg.lstsq(A2, b2, rcond=None)[0]

print(f"\nx2 fitted parameters:")
print(f"  coef(x1[t-1]) = {params2[0]:.6f}  (true: -0.3)")
print(f"  coef(x2[t-1]) = {params2[1]:.6f}  (true: 0.8)")
print(f"  coef(x1^2)    = {params2[2]:.6f}  (true: 0.2)")
print(f"  coef(u[t])    = {params2[3]:.6f}  (true: 0.0)")
print(f"  bias          = {params2[4]:.6f}  (true: 0.05)")

err2 = np.max(np.abs(A2 @ params2 - b2))
print(f"  Max error: {err2:.6f}")

# ============================================================================
# EXAMPLE 3: Simulating DEConfig.py workflow
# ============================================================================
print("\n\n" + "="*80)
print("[EXAMPLE 3] Simulating DEConfig.py work_normal() function")
print("="*80)

def simulate_append_data(data, input_data, order):
    """
    Simulates DEConfig.py append_data() function
    
    Args:
        data: shape (n_vars, n_timesteps)
        input_data: shape (n_inputs, n_timesteps)
        order: model order
    
    Returns:
        matrix_list: list of feature matrices for each variable
        b_list: list of target vectors for each variable
    """
    n_vars = data.shape[0]
    n_timesteps = data.shape[1]
    
    matrix_list = [[] for _ in range(n_vars)]
    b_list = [[] for _ in range(n_vars)]
    
    for i in range(n_timesteps - order):
        # Extract lag terms (reversed: t-1, t-2, ..., t-order)
        if i == 0:
            this_line = data[:, (order - 1)::-1]
            this_line_input = input_data[:, order::-1]
        else:
            this_line = data[:, (order + i - 1):(i - 1):-1]
            this_line_input = input_data[:, (order + i):(i - 1):-1]
        
        # Build feature vector for each variable
        for var_idx in range(n_vars):
            feature = []
            
            # Add lag terms for this variable
            for lag in range(order):
                feature.append(this_line[var_idx, lag])
            
            # Add input term
            feature.append(this_line_input[0, 0])
            
            # Add bias
            feature.append(1.0)
            
            matrix_list[var_idx].append(feature)
            b_list[var_idx].append(data[var_idx, i + order])
    
    return matrix_list, b_list


def simulate_work_normal(data, input_data, order):
    """
    Simulates DEConfig.py work_normal() function
    """
    matrix_list, b_list = simulate_append_data(data, input_data, order)
    
    res = []
    err = []
    
    # Solve least squares for each variable
    for A, b in zip(matrix_list, b_list):
        A = np.array(A)
        b = np.array(b)
        
        # KEY OPERATION: least squares solve (line 136 in DEConfig.py)
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Calculate fitting error (line 138 in DEConfig.py)
        max_err = np.max(np.abs((A @ x) - b))
        
        res.append(x)
        err.append(max_err)
    
    return res, err


# Prepare test data
data = np.array([x1, x2])  # shape: (2, 30)
input_data = np.array([u])  # shape: (1, 30)

print(f"\nInput data shapes:")
print(f"  data: {data.shape} (n_vars x n_timesteps)")
print(f"  input_data: {input_data.shape}")

# Run fitting
order = 1
params_list, errors = simulate_work_normal(data, input_data, order)

print(f"\nFitting results:")
for i, (params, err) in enumerate(zip(params_list, errors)):
    print(f"\nVariable x{i+1}:")
    print(f"  Parameters: {params}")
    print(f"  Max fitting error: {err:.6f}")

# ============================================================================
# Summary
# ============================================================================
print("\n\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. np.linalg.lstsq(A, b, rcond=None)[0] solves:
   - Find optimal x that minimizes ||Ax - b||^2
   - This is standard least squares for overdetermined systems

2. In NARX model context:
   - Each row of A = feature vector at time t: [x[t-1], x[t-2], ..., u[t], 1]
   - Each element of b = target value x[t]
   - Solution x = [a1, a2, ..., b_input, bias]

3. DEConfig.py implementation:
   - append_data(): constructs matrix A and vector b
   - work_normal(): calls lstsq for each variable
   - get_items(): builds individual feature vectors (supports nonlinear terms)

4. Why this works:
   - Converts time series prediction to linear regression
   - Handles multi-variable, high-order, nonlinear systems
   - Fitting error indicates mode switching (key innovation)
""")

print("\nDemo completed successfully!")

