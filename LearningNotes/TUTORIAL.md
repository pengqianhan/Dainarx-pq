# Dainarx Tutorial: Derivative-Agnostic Inference of Nonlinear Hybrid Systems

This tutorial explains the Dainarx method for learning hybrid automata from time series data, combining the theoretical foundations from the paper with practical code examples.

## 1. Introduction and Problem Setup

### What is Dainarx?

Dainarx (Derivative-Agnostic inference of Nonlinear hybrid Automata with aRX models) is a method for learning hybrid automata from discrete-time input-output traces. Unlike traditional approaches that rely on derivative calculations and user-defined thresholds, Dainarx uses NARX (Nonlinear AutoRegressive with eXogenous inputs) model fitting throughout the entire pipeline.

### Key Innovation

Traditional methods use:
- **Derivative-based segmentation**: Detects mode switches by monitoring drastic changes in derivatives
- **Trace similarity clustering**: Groups segments based on visual similarity with user-defined thresholds

Dainarx uses:
- **NARX model fitting**: Unified, threshold-free approach for both segmentation and clustering

### Problem Formulation

**Input**: Set of discrete-time traces D = {ξⱼ}₁≤ⱼ≤M sampled from a hybrid system H
**Output**: Hybrid automaton Ĥ with NARX-modeled dynamics that approximates H

Each trace ξ is a sequence: ξ = ⟨(t₀,v̄₀), (t₀+Δt,v̄₁), ..., (t₀+ℓΔt,v̄ℓ)⟩ where v̄ᵢ ∈ ℝⁿ⁺ᵐ contains both output variables x̄ and input variables ū.

## 2. NARX Models - The Foundation

### NARX Model Definition

A k-th order NARX model represents the current output as a function of past outputs and current inputs:

```
x̄[τ] = Fq(x̄[τ-1], x̄[τ-2], ..., x̄[τ-k], ū[τ])
```

**Equation (3) in paper**: This can be expanded as:

```
x̄[τ] = Σᵢ₌₁ᵅ āᵢ ∘ fᵢ(x̄[τ-1],...,x̄[τ-k],ū[τ]) + Σᵢ₌₁ᵏ Bᵢ·x̄[τ-i] + Bₖ₊₁·ū[τ] + c̄
```

where:
- `fᵢ`: Nonlinear terms (e.g., x³[τ-1], sin(x[τ-2]))
- `āᵢ, c̄`: Vector coefficients
- `Bᵢ`: Matrix coefficients
- `∘`: Hadamard product

### Code Implementation

In `src/DEConfig.py`, the `FeatureExtractor` class constructs NARX features:

```python
def get_feature(self, data, input_data, start_idx):
    """Extract NARX features from data segment"""
    # data[start_idx:2] contains x[τ-1], x[τ-2], etc.
    # Linear terms: past outputs
    linear_features = data[start_idx-self.order:start_idx].flatten()
    
    # Input terms: current inputs
    input_features = input_data[start_idx]
    
    # Nonlinear terms: from other_items configuration
    nonlinear_features = self._compute_nonlinear_terms(data, input_data, start_idx)
    
    return np.concatenate([nonlinear_features, linear_features, input_features, [1]])
```

## 3. Dainarx Pipeline Overview

The method follows five main steps:

1. **Trace Segmentation**: Detect mode switching points
2. **Segment Clustering**: Group segments with same dynamics
3. **Mode Characterization**: Learn NARX model for each cluster
4. **Guard Learning**: Learn switching conditions using SVM
5. **Reset Learning**: Learn state updates during transitions

### Code Architecture

The main pipeline is implemented in `main.py:run()`:

```python
def run(data_list, input_data, config, evaluation):
    # Step 1: Trace Segmentation
    for data, input_val in zip(data_list, input_data):
        change_points = find_change_point(data, input_val, get_feature, w=config['window_size'])
        slice_curve(slice_data, data, input_val, change_points, get_feature)
    
    # Step 2: Segment Clustering  
    clustering(slice_data, config['self_loop'])
    
    # Step 3 & 4: Mode Characterization & Guard Learning
    adj = guard_learning(slice_data, get_feature, config)
    
    # Step 5: System Building (includes reset learning)
    sys = build_system(slice_data, adj, get_feature)
    
    return sys, slice_data
```

## 4. Step 1: Trace Segmentation

### Theoretical Foundation

**Property 9 (Changepoints)**: A set CP = {p₀, p₁, ..., pₛ} are changepoints w.r.t. trace ξ and template N if:
1. ξₚᵢ,ₚᵢ₊₁ is fittable by N for any 0 ≤ i < s
2. ξₚᵢ,₍ₚᵢ₊₁₎₊₁ is not fittable by N for any 0 ≤ i < s-1

**Key Insight**: A segment can be fitted by a single NARX model if and only if it comes from the same system mode. When dynamics change, the fitting fails.

### Implementation

In `src/ChangePoints.py:find_change_point()`:

```python
def find_change_point(data, input_data, feature_extractor, w=10):
    """Find changepoints using sliding window approach"""
    change_points = [0]
    left = 0
    data_len = data.shape[1]
    
    while left <= data_len - w:
        # Try to fit segment [left, left+w] with NARX model
        segment_data = data[:, left:left+w]
        segment_input = input_data[:, left:left+w]
        
        try:
            # Extract features and try NARX fitting
            features = feature_extractor.extract_features(segment_data, segment_input)
            coefficients = fit_narx_model(features)
            
            # If fitting succeeds, slide window
            left += 1
        except FittingError:
            # If fitting fails, we found a changepoint
            change_points.append(left + w - 1)
            left = left + w
            
    change_points.append(data_len)
    return change_points
```

### Algorithm 1: Sliding Window Segmentation

The paper provides **Algorithm 1** for segmentation:

```python
# Simplified version of the algorithm
CP = {0}
l = 0  # left bound of sliding window

while l <= length - w:
    if segment[l:l+w] is not fittable by NARX:
        CP = CP ∪ {l + w - 1}  # Add changepoint
        l = l + w  # Skip ahead
    else:
        l = l + 1  # Slide window
```

**Theorem 10**: This algorithm correctly identifies changepoints (under certain conditions).

## 5. Step 2: Segment Clustering

### Theoretical Foundation

Traditional methods cluster based on trace similarity (visual appearance). Dainarx uses **fittability**:

**Definition 11 (Mergeable)**: Segments S are mergeable w.r.t. template N if ∃N ∈ ⟨N⟩ such that N ⊨ S (N fits all segments in S).

**Definition 13 (Minimally Mergeable)**: More conservative - requires minimal order consistency.

### Implementation

In `src/Clustering.py:clustering()`:

```python
def clustering(slice_data, allow_self_loop=False):
    """Cluster segments based on NARX fittability"""
    clusters = []
    
    for slice_obj in slice_data:
        merged = False
        
        # Try to merge with existing clusters
        for cluster in clusters:
            if can_merge(cluster + [slice_obj]):
                cluster.append(slice_obj)
                merged = True
                break
                
        if not merged:
            # Create new cluster
            clusters.append([slice_obj])
    
    # Assign cluster IDs
    assign_cluster_ids(clusters)

def can_merge(segments):
    """Check if segments can be fitted by same NARX model"""
    try:
        # Extract features from all segments
        combined_features = []
        for segment in segments:
            combined_features.extend(segment.get_features())
            
        # Try to fit single NARX model
        coefficients = fit_narx_model(combined_features)
        return True
    except FittingError:
        return False
```

### Example from Paper

**Example 12**: Shows why mergeable criterion can be "aggressive":
- Segments ξ₁ and ξ₂ might come from different 1st-order modes (x' = 1, x' = 2)
- But they're mergeable under 2nd-order model (x'' = 0)
- Solution: Use "minimally mergeable" criterion

## 6. Step 3: Mode Characterization

### NARX Model Fitting via LLSQ

**Equation (5)** shows the optimization problem:
```
minimize ||O_{i,:} - Λ_{i,:} · D_i||_2  for i = 1,2,...,n
```

where:
- `O`: Observed value matrix
- `D_i`: Data matrix for i-th variable
- `Λ`: Coefficient matrix to be learned

### Implementation

In `src/BuildSystem.py:build_system()`:

```python
def build_system(slice_data, adjacency_matrix, feature_extractor):
    """Build hybrid automaton from clustered segments"""
    modes = {}
    
    # For each cluster, learn NARX model
    for cluster_id in get_unique_clusters(slice_data):
        cluster_segments = get_cluster_segments(slice_data, cluster_id)
        
        # Combine all segments in cluster
        combined_features = []
        combined_outputs = []
        
        for segment in cluster_segments:
            features = segment.get_features()
            outputs = segment.get_outputs()
            combined_features.extend(features)
            combined_outputs.extend(outputs)
        
        # Solve LLSQ problem (Equation 5)
        coefficients = solve_llsq(combined_features, combined_outputs)
        
        # Create NARX model for this mode
        modes[cluster_id] = NARXModel(coefficients, feature_extractor.nonlinear_terms)
    
    return HybridAutomaton(modes, adjacency_matrix)
```

### Linear Least Squares Solution

The LLSQ problem is solved using standard techniques:

```python
def solve_llsq(features, outputs):
    """Solve linear least squares: min ||Ax - b||_2"""
    A = np.array(features)  # Data matrix
    b = np.array(outputs)   # Output vector
    
    # Normal equation: x = (A^T A)^{-1} A^T b
    # Or use SVD for numerical stability
    coefficients, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return coefficients
```

## 7. Step 4: Guard Learning

### SVM-based Classification

Guard learning is formulated as binary classification: given state (x̄,ū), will transition from mode q to q' occur?

**Training Data Construction**:
- Positive samples (q,q')⁺: States where transition q→q' happens
- Negative samples (q,q')⁻: States where transition q→q' doesn't happen

### Implementation

In `src/GuardLearning.py:guard_learning()`:

```python
def guard_learning(slice_data, feature_extractor, config):
    """Learn guard conditions using SVM"""
    from sklearn.svm import SVC
    
    # Build training dataset
    transitions = extract_transitions(slice_data)
    adjacency_matrix = {}
    
    for (source_mode, target_mode), transition_points in transitions.items():
        # Positive samples: states just before transition
        positive_samples = []
        # Negative samples: states that don't lead to this transition
        negative_samples = []
        
        for point in transition_points:
            state_vector = point.get_state_vector()
            positive_samples.append(state_vector)
            
        # Collect negative samples from other transitions
        for other_transition in transitions:
            if other_transition != (source_mode, target_mode):
                for point in transitions[other_transition]:
                    negative_samples.append(point.get_state_vector())
        
        # Train SVM classifier
        X = positive_samples + negative_samples
        y = [1] * len(positive_samples) + [0] * len(negative_samples)
        
        svm = SVC(kernel=config['kernel'], C=config['svm_c'])
        svm.fit(X, y)
        
        adjacency_matrix[(source_mode, target_mode)] = svm
    
    return adjacency_matrix
```

### SVM Theory

The paper uses **kernel SVM** for nonlinear decision boundaries:
- Linear kernel: For linear guard conditions
- Polynomial/RBF kernels: For nonlinear guards (like x² ≥ 1.44 in Duffing oscillator)

## 8. Step 5: Reset Learning

### Challenge with High-Order Systems

For first-order systems: reset is simple linear transformation
For higher-order systems: derivatives x̄⁽¹⁾, x̄⁽²⁾, ... are not in the data

### NARX-based Reset Solution

**Equation (7)**: Reset function as NARX mapping:
```
r: ℝᵏˣⁿ⁺ᵐ → ⟨N⟩ᵏ
```

Instead of resetting derivatives directly, learn k NARX models that generate the next k outputs after transition.

### Implementation

In `src/Reset.py:learn_reset_function()`:

```python
def learn_reset_function(transition_points, feature_extractor, order):
    """Learn reset using NARX models"""
    reset_models = []
    
    # Learn k NARX models for post-transition behavior
    for i in range(order):
        # Collect training data: segments around transition points
        features = []
        outputs = []
        
        for transition in transition_points:
            # Get k+1 points around transition
            segment = transition.get_surrounding_segment(length=order+1)
            
            # Extract features for i-th step after transition
            feature_vector = feature_extractor.extract_features(
                segment.data, segment.input, transition.index + i
            )
            output_vector = segment.data[:, transition.index + i]
            
            features.append(feature_vector)
            outputs.append(output_vector)
        
        # Learn NARX model for this time step
        coefficients = solve_llsq(features, outputs)
        reset_models.append(NARXModel(coefficients, feature_extractor.nonlinear_terms))
    
    return reset_models
```

## 9. Example: Duffing Oscillator

Let's trace through the complete example from **Figure 2** in the paper.

### System Definition

The Duffing oscillator has two modes:
- q₁ (high damping): x⁽²⁾ = u - 0.5x⁽¹⁾ + x - 1.5x³
- q₂ (low damping): x⁽²⁾ = u - 0.2x⁽¹⁾ + x - 0.5x³

Guards:
- q₁ → q₂: x² ≥ 1.44
- q₂ → q₁: x² ≤ 0.64

Resets: x⁽¹⁾ := 0.95x⁽¹⁾

### Template NARX Model

Based on the ODE structure, we use template with nonlinear terms {x³[τ-1], x³[τ-2]}:

```json
{
  "other_items": "x[1] * x[1] * x[1]; x[2] * x[2] * x[2]"
}
```

### Dainarx Results

**Input**: 10 traces, 9 for training, 1 for testing
**Segmentation**: 155 segments detected
**Clustering**: 2 clusters (corresponding to 2 modes)
**Learned Models**:
- q₁: x[τ] = 2x[τ-1] - x[τ-2] - 1.5×10⁻⁶x³[τ-1] + 10⁻⁷u[τ]
- q₂: x[τ] = 2x[τ-1] - x[τ-2] - 5×10⁻⁷x³[τ-1] + 10⁻⁷u[τ]

**Accuracy**: Maximum deviation of 0.0003 from true trajectory!

### Code Configuration

In `automata/non_linear/duffing.json`:

```json
{
  "automaton": {
    "var": "x1",
    "input": "u", 
    "mode": [
      {
        "id": 1,
        "eq": "x1[2] = u - 0.5 * x1[1] + x1[0] - 1.5 * x1[0] * x1[0] * x1[0]"
      },
      {
        "id": 2, 
        "eq": "x1[2] = u - 0.2 * x1[1] + x1[0] - 0.5 * x1[0] * x1[0] * x1[0]"
      }
    ]
  },
  "config": {
    "order": 2,
    "other_items": "x[1] * x[1] * x[1]; x[2] * x[2] * x[2]",
    "need_reset": true
  }
}
```

## 10. Advantages of Dainarx

### 1. Threshold-Free Operation

**Traditional approaches** require user-tuned thresholds:
- Derivative threshold for segmentation
- Similarity threshold for clustering

**Dainarx** is parameter-free: Uses mathematical criterion (NARX fittability)

### 2. Unified Framework  

Same NARX fitting technique used throughout:
- Segmentation: Can segment be fitted by single NARX model?
- Clustering: Can multiple segments be fitted by single NARX model?
- Mode learning: Fit NARX model to clustered segments
- Reset learning: Use NARX models for post-transition dynamics

### 3. Handles Complex Systems

- **High-order dynamics**: Up to 4th order in experiments
- **Nonlinear dynamics**: Polynomial, rational, trigonometric
- **Nonlinear guards**: Using kernel SVM
- **Inputs and resets**: Full hybrid automaton features

## 11. Complexity Analysis

**Theorem**: Total worst-case time complexity is O(|D|³·n + |D|²·d²·n)

Where:
- |D|: Size of learning data
- n: Number of output variables  
- d: NARX model complexity (α + kn + m + 1)

**Practical complexity** is much lower: O(|D|·d²·n·(w + |SegD|))

### Breakdown by Step:
1. **Segmentation**: O(|D|·TN(w)) where w is window size
2. **Clustering**: O(TN(|D|)·|SegD|) in typical case  
3. **Mode characterization**: O(TN(|D|))
4. **Guard learning**: O(|D|³·n) worst case (SVM)
5. **Reset learning**: O(|D|·TN(k))

## 12. Limitations and Future Work

### Current Limitations

1. **Template dependency**: Requires user-provided nonlinear terms {fᵢ}
2. **Noise sensitivity**: Assumes noise-free data
3. **Approximation guarantee**: No formal bounds on inference quality

### Future Directions

1. **Automatic template learning**: Use ML to discover nonlinear terms
2. **Robust identification**: Handle noisy measurements
3. **PAC learning**: Provide formal guarantees (Probably Approximately Correct)

## 13. Experimental Results Summary

From **Tables 2-3** in the paper:

### Linear Systems (18 benchmarks):
- **Perfect changepoint detection**: HDTc = 0 in most cases
- **Superior accuracy**: Lowest Diffmax and Diffavg errors
- **Universal applicability**: Works on all benchmarks (vs. partial for baselines)

### Nonlinear Systems (8 benchmarks):
- **Only working method**: FaMoS (linear only), LearnHA (limited nonlinear support)
- **High accuracy**: HDTc ≤ 0.02, Diffavg ≤ 0.008 for all benchmarks
- **Complex dynamics**: Handles trigonometric, rational functions

### Efficiency:
- **1.5-6× faster** than baselines
- **Scalable**: Most problems solved in <5 seconds

## 14. Running Dainarx

### Basic Usage

```bash
# Run single test
python main.py

# Run all benchmarks  
python test_all.py
```

### Key Configuration Parameters

In JSON files under `automata/`:

```json
{
  "config": {
    "dt": 0.01,           // Sampling period
    "total_time": 10.0,   // Trace length
    "order": 3,           // NARX model order
    "window_size": 10,    // Segmentation window
    "other_items": "",    // Nonlinear terms specification
    "kernel": "linear",   // SVM kernel for guards
    "need_reset": false,  // Whether to learn resets
    "self_loop": false    // Allow self-transitions
  }
}
```

### Custom Nonlinear Terms

The `other_items` field specifies additional nonlinear terms:

```json
{
  "other_items": "x0, x2: x[1] * x_[?]; x[?] * x1[2]"
}
```

- `x0, x2:` - Apply to variables 0 and 2
- `x[1] * x_[?]` - Current var at lag 1 times other vars at any lag
- `x[?] * x1[2]` - Current var at any lag times variable 1 at lag 2

## Conclusion

Dainarx represents a significant advance in hybrid system identification by:

1. **Eliminating manual threshold tuning** through principled NARX-based criteria
2. **Providing unified framework** for all pipeline steps
3. **Handling complex nonlinear systems** beyond current method capabilities
4. **Delivering superior accuracy** with competitive efficiency

The method's theoretical foundation in NARX model fitting provides both practical benefits and opportunities for further theoretical development, making it a promising direction for automated hybrid system discovery.

**Reference**: Yu et al. "Derivative-Agnostic Inference of Nonlinear Hybrid Systems." ACM Trans. Embedd. Comput. Syst. 2025.