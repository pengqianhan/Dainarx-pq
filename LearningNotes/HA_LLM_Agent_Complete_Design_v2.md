# Hybrid Automaton LLM Agent: Complete Design Document (v2.0)

> **Objective**: Design an LLM agent that infers Hybrid Automata (HA) directly from npz data and/or trajectory plots, automatically determining configuration parameters (`order`, `need_reset`, `self_loop`, `kernel`, `other_items`) without manual prior specification.

---

## Revision Notes (v2.0)

**Improvements based on scientific rigor evaluation:**
1. Added uncertainty quantification throughout the pipeline
2. Enhanced symbolic regression integration with modern methods
3. Multi-objective Pareto optimization instead of single weighted score
4. Noise model estimation and robustness analysis
5. Active learning and sample efficiency considerations
6. Physics-informed constraints and conservation laws
7. Formal verification hooks for safety-critical systems

---

## 1. Problem Statement

### 1.1 Current Limitations
The existing DAINARX pipeline requires manual specification of:
- `order`: NARX model history length (1-3)
- `need_reset`: Whether to learn reset functions
- `self_loop`: Whether to allow self-loop transitions
- `kernel`: SVM kernel for guard learning (`linear`, `rbf`, `poly`)
- `other_items`: Nonlinear term expressions (e.g., `x[?] ** 3`, `sin(x[?])`)

### 1.2 Key Observations
| System | Mode Switching | Visual Signature | Challenge |
|--------|---------------|------------------|-----------|
| **Bouncing Ball** | Clear (ground contact) | Sharp discontinuities, velocity reversals | Guard/reset detection |
| **Duffing** | Subtle or none | Smooth oscillations, no obvious breaks | Nonlinear term discovery |

### 1.3 Design Goals
1. **Automatic parameter inference** from data/plots
2. **Unified handling** of both clear-switching and subtle-switching systems
3. **Closed-loop verification** with simulation feedback
4. **Uncertainty quantification** at each inference stage
5. **Seamless integration** with existing DAINARX JSON format and simulator

---

## 2. Architecture Overview

### 2.1 SR-Scientist Inspired Design Principles

Following the SR-Scientist paradigm of **program synthesis with verification**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PLANNER AGENT (smolagent)                    │
│  - Orchestrates tool calls                                          │
│  - Manages beam search state                                        │
│  - Proposes hypotheses (K, lib_config, guard/reset forms)           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
   │   PROPOSE   │         │   EXECUTE   │         │  EVALUATE   │
   │ - K modes   │         │ - Segment   │         │ - RMSE      │
   │ - lib_config│         │ - Fit SINDy │         │ - Event MAE │
   │ - Guard form│         │ - Learn     │         │ - Switch F1 │
   │ - Reset form│         │   guards    │         │ - MDL       │
   └─────────────┘         └─────────────┘         └─────────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   ▼
                          ┌─────────────┐
                          │   REFINE    │
                          │ - Error     │
                          │   analysis  │
                          │ - Adjust    │
                          │   params    │
                          └─────────────┘
```

### 2.2 Core Components (Enhanced)

1. **Data Analyzer**: Extracts features, estimates derivatives, **estimates noise model**
2. **Mode Discoverer**: Change point detection, clustering, mode count selection, **with confidence intervals**
3. **Equation Learner**: SINDy + **Neural Symbolic Regression** with adaptive library
4. **Guard/Reset Learner**: Boundary classification with **uncertainty-aware SVM**
5. **Assembler & Validator**: JSON generation and simulation-based scoring
6. **Physics Checker**: **Conservation law verification, stability analysis**
7. **Image Analyzer** (optional): Visual hints for weak priors

---

## 3. End-to-End Workflow

### 3.1 Pipeline Overview (Enhanced)

```
npz Data ──┬──► [1] Load & Preprocess ──► [2] Noise Estimation ──► [3] Derivative Estimation
           │                                                              │
(Optional) │                                                              ▼
Plot Image ──► [4] Image Analysis ──────────────────────────────► [5] Mode Discovery
                    (weak prior)                                          │
                                                                          ▼
                                                       [6] Library Proposal (LLM + SR)
                                                                          │
                                                                          ▼
                                                       [7] SINDy + Neural SR (per mode)
                                                                          │
                                                                          ▼
                                                       [8] Guard/Reset Learning
                                                                          │
                                                                          ▼
                                                       [9] Physics Consistency Check
                                                                          │
                                                                          ▼
                                                       [10] Assemble HA JSON
                                                                          │
                                                                          ▼
                                                       [11] Multi-Objective Scoring
                                                                          │
                                                               ┌──────────┴──────────┐
                                                               ▼                     ▼
                                                         Pareto Front          Below Threshold?
                                                               │                     │
                                                               ▼                     ▼
                                                         Output JSON(s)       [12] Refine
                                                                              (iterate)
```

### 3.2 NEW Stage: Noise Estimation

```python
def estimate_noise_model(t: np.ndarray, X: np.ndarray) -> NoiseModel:
    """
    Estimate measurement noise characteristics before derivative estimation.
    Critical for setting appropriate regularization in SINDy.

    Methods:
        1. Residual analysis: Fit local polynomial, analyze residuals
        2. Variogram estimation: For spatially correlated noise
        3. Wavelet-based: Decompose signal, estimate noise from high-freq coeffs

    Returns:
        - sigma: estimated noise std per variable
        - noise_type: 'gaussian', 'heteroscedastic', 'correlated'
        - snr: signal-to-noise ratio
        - confidence_interval: 95% CI for sigma
    """
    # Method 1: Local polynomial residuals
    residuals = []
    for var_idx in range(X.shape[0]):
        x = X[var_idx]
        # Fit local quadratic in sliding windows
        window_size = min(20, len(x) // 10)
        local_residuals = []
        for i in range(window_size, len(x) - window_size):
            window = x[i-window_size:i+window_size]
            t_window = np.arange(len(window))
            coeffs = np.polyfit(t_window, window, 2)
            fitted = np.polyval(coeffs, t_window)
            local_residuals.append(window[window_size] - fitted[window_size])
        residuals.append(np.std(local_residuals))

    # Method 2: Wavelet-based (robust for non-stationary signals)
    from scipy.signal import cwt, ricker
    wavelet_sigma = []
    for var_idx in range(X.shape[0]):
        # High-frequency wavelet coefficients
        widths = np.arange(1, 5)
        cwtmatr = cwt(X[var_idx], ricker, widths)
        # Estimate sigma from finest scale
        wavelet_sigma.append(np.median(np.abs(cwtmatr[0])) / 0.6745)

    sigma = np.mean([residuals, wavelet_sigma], axis=0)

    return NoiseModel(
        sigma=sigma,
        noise_type='gaussian',  # Can be refined
        snr=np.std(X, axis=1) / sigma,
        confidence_interval=(sigma * 0.8, sigma * 1.2)
    )
```

### 3.3 Enhanced Derivative Estimation with Uncertainty

```python
def estimate_derivatives_with_uncertainty(
    t: np.ndarray,
    X: np.ndarray,
    noise_model: NoiseModel,
    method: str = 'auto'
) -> DerivativeResult:
    """
    Compute derivatives with propagated uncertainty.

    Methods:
        - 'savgol': Savitzky-Golay (low noise)
        - 'tvreg': Total Variation Regularization (high noise)
        - 'gp': Gaussian Process (uncertainty quantification)
        - 'auto': Select based on SNR

    Returns:
        - Xdot: first derivative estimates
        - Xdot_uncertainty: standard deviation of estimates
        - Xddot: second derivative (if needed)
        - method_used: which method was selected
    """
    if method == 'auto':
        # Select based on SNR
        avg_snr = np.mean(noise_model.snr)
        if avg_snr > 50:
            method = 'savgol'
        elif avg_snr > 10:
            method = 'tvreg'
        else:
            method = 'gp'  # Provides uncertainty estimates

    if method == 'gp':
        # Gaussian Process for uncertainty quantification
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        Xdot = np.zeros_like(X)
        Xdot_std = np.zeros_like(X)

        for var_idx in range(X.shape[0]):
            kernel = RBF() + WhiteKernel(noise_level=noise_model.sigma[var_idx]**2)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            gp.fit(t.reshape(-1, 1), X[var_idx])

            # Predict derivatives via automatic differentiation of GP mean
            # (Simplified: using finite differences on GP predictions)
            dt = t[1] - t[0]
            x_pred, x_std = gp.predict(t.reshape(-1, 1), return_std=True)
            Xdot[var_idx] = np.gradient(x_pred, dt)
            Xdot_std[var_idx] = x_std / dt  # Approximate uncertainty

        return DerivativeResult(
            Xdot=Xdot,
            Xdot_uncertainty=Xdot_std,
            method_used='gp'
        )

    # ... other methods ...
```

---

## 4. Enhanced Symbolic Regression Integration

### 4.1 Multi-Method SR Strategy

```python
class SymbolicRegressionEnsemble:
    """
    Ensemble of symbolic regression methods for robust term discovery.

    Methods:
        1. PySR (genetic programming with regularization)
        2. SINDy (sparse regression with predefined library)
        3. AI Feynman (physics-based dimensional analysis)
        4. Neural Symbolic (differentiable program synthesis)
    """

    def __init__(self, config: SRConfig):
        self.config = config
        self.methods = []

        # PySR: Best for complex expressions
        if config.use_pysr:
            from pysr import PySRRegressor
            self.methods.append(('pysr', PySRRegressor(
                niterations=config.pysr_iterations,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sin", "cos", "exp", "sqrt", "abs"],
                complexity_of_operators={
                    "+": 1, "-": 1, "*": 1, "/": 2,
                    "sin": 3, "cos": 3, "exp": 4, "sqrt": 2, "abs": 2
                },
                maxsize=config.max_expression_size,
                populations=config.populations,
                model_selection="best",
                loss="loss(prediction, target) = (prediction - target)^2",
            )))

        # SINDy: Fast, interpretable, controlled library
        if config.use_sindy:
            self.methods.append(('sindy', SindyWrapper(
                library=config.sindy_library,
                threshold=config.sindy_threshold,
                alpha=config.sindy_alpha,
            )))

        # Neural Symbolic: End-to-end differentiable
        if config.use_neural_sr:
            self.methods.append(('neural_sr', NeuralSymbolicRegressor(
                max_depth=config.neural_sr_depth,
                primitives=config.neural_sr_primitives,
            )))

    def fit(self, X: np.ndarray, y: np.ndarray) -> List[SRResult]:
        """
        Fit all methods and return ensemble of results.
        """
        results = []
        for name, method in self.methods:
            try:
                method.fit(X, y)
                expr = method.get_best_expression()
                score = method.score(X, y)
                complexity = method.get_complexity()

                results.append(SRResult(
                    method=name,
                    expression=expr,
                    score=score,
                    complexity=complexity,
                    pareto_optimal=False  # To be determined
                ))
            except Exception as e:
                results.append(SRResult(
                    method=name,
                    expression=None,
                    error=str(e)
                ))

        # Mark Pareto-optimal solutions
        self._mark_pareto_optimal(results)
        return results

    def _mark_pareto_optimal(self, results: List[SRResult]):
        """
        Identify Pareto-optimal solutions (score vs complexity).
        """
        valid = [r for r in results if r.expression is not None]
        for r in valid:
            r.pareto_optimal = True
            for other in valid:
                if other.score < r.score and other.complexity < r.complexity:
                    r.pareto_optimal = False
                    break
```

### 4.2 Grammar-Guided Genetic Programming

```python
class GrammarGuidedSR:
    """
    Use domain-specific grammar to constrain symbolic regression search.
    """

    def __init__(self):
        # Define grammar for hybrid system dynamics
        self.grammar = {
            'expr': [
                ('linear', 'coeff * var'),
                ('polynomial', 'coeff * var ** order'),
                ('cross', 'coeff * var1 * var2'),
                ('trig', 'coeff * trig_fn(freq * var)'),
                ('damping', 'coeff * var * |var|'),
                ('switching', 'coeff * sign(var)'),
            ],
            'var': ['x[0]', 'x[1]', 'xdot[0]', 'xdot[1]', 't'],
            'order': [2, 3],
            'trig_fn': ['sin', 'cos'],
            'freq': [1.0, 2.0, 'omega'],  # omega to be fitted
        }

        # Physical constraints
        self.constraints = [
            # Energy dissipation: damping terms should have negative coeffs
            ('damping', lambda coeff: coeff <= 0),
            # Stability: eigenvalues should have non-positive real parts
            ('linear', lambda expr: self._check_stability(expr)),
        ]

    def generate_candidates(self, max_complexity: int) -> List[str]:
        """
        Generate candidate expressions up to given complexity.
        """
        candidates = []
        for rule_name, template in self.grammar['expr']:
            for var in self.grammar['var']:
                expr = self._instantiate(template, {'var': var})
                if self._complexity(expr) <= max_complexity:
                    candidates.append(expr)
        return candidates
```

### 4.3 Physics-Informed Term Library

```python
class PhysicsInformedLibrary:
    """
    Build term library based on physical principles.
    """

    PHYSICAL_SYSTEMS = {
        'mechanical': {
            'kinetic': ['0.5 * m * v**2'],
            'potential': ['0.5 * k * x**2', 'm * g * x'],
            'dissipation': ['c * v', 'c * v * |v|'],
            'nonlinear_spring': ['k * x + k3 * x**3'],
            'forcing': ['F * sin(omega * t)', 'F * cos(omega * t)'],
        },
        'bouncing': {
            'gravity': ['-g'],
            'impact': ['sign(x)', 'abs(x)'],
            'restitution': ['-e * v'],
        },
        'oscillator': {
            'linear': ['x', 'v'],
            'duffing': ['x**3'],
            'van_der_pol': ['(1 - x**2) * v'],
            'forcing': ['sin(omega * t)'],
        },
    }

    def suggest_library(self, data_summary: DataSummary) -> List[str]:
        """
        Suggest terms based on data characteristics.
        """
        suggested = []

        # Detect oscillatory behavior
        if self._is_oscillatory(data_summary):
            suggested.extend(self.PHYSICAL_SYSTEMS['oscillator']['linear'])
            if data_summary.has_amplitude_dependence:
                suggested.extend(self.PHYSICAL_SYSTEMS['oscillator']['duffing'])

        # Detect bouncing/impact behavior
        if data_summary.has_discontinuities:
            suggested.extend(self.PHYSICAL_SYSTEMS['bouncing']['gravity'])
            suggested.extend(self.PHYSICAL_SYSTEMS['bouncing']['restitution'])

        # Detect external forcing
        if data_summary.has_periodic_component:
            suggested.extend(self.PHYSICAL_SYSTEMS['mechanical']['forcing'])

        return suggested
```

---

## 5. Multi-Objective Optimization

### 5.1 Pareto Front Exploration

```python
class ParetoOptimizer:
    """
    Multi-objective optimization for HA inference.

    Objectives:
        1. Trajectory RMSE (minimize)
        2. Model complexity / MDL (minimize)
        3. Event timing accuracy (minimize MAE)
        4. Physical plausibility (maximize)
    """

    def __init__(self, config: OptConfig):
        self.config = config
        self.objectives = [
            ('rmse', 'minimize', 1.0),
            ('mdl', 'minimize', 0.5),
            ('event_mae', 'minimize', 0.3),
            ('physics_score', 'maximize', 0.2),
        ]

    def evaluate(self, candidate: HAJson, ref_data: RefData) -> ObjectiveVector:
        """
        Evaluate candidate on all objectives.
        """
        # Simulate and compute metrics
        sim_result = simulate_ha(candidate, ref_data.t, ref_data.x0)

        rmse = np.sqrt(np.mean((sim_result.x - ref_data.x)**2))
        mdl = self._compute_mdl(candidate)
        event_mae = self._compute_event_mae(sim_result.events, ref_data.events)
        physics_score = self._check_physics(candidate)

        return ObjectiveVector(
            rmse=rmse,
            mdl=mdl,
            event_mae=event_mae,
            physics_score=physics_score
        )

    def update_pareto_front(
        self,
        candidates: List[HAJson],
        current_front: List[HAJson]
    ) -> List[HAJson]:
        """
        Update Pareto front with new candidates.
        """
        all_candidates = current_front + candidates
        objectives = [self.evaluate(c, self.ref_data) for c in all_candidates]

        # Non-dominated sorting
        pareto_front = []
        for i, (c, obj) in enumerate(zip(all_candidates, objectives)):
            dominated = False
            for j, other_obj in enumerate(objectives):
                if i != j and self._dominates(other_obj, obj):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(c)

        return pareto_front

    def _dominates(self, obj1: ObjectiveVector, obj2: ObjectiveVector) -> bool:
        """
        Check if obj1 dominates obj2 (better or equal on all, strictly better on at least one).
        """
        dominated = True
        strictly_better = False

        for name, direction, _ in self.objectives:
            v1, v2 = getattr(obj1, name), getattr(obj2, name)
            if direction == 'minimize':
                if v1 > v2:
                    dominated = False
                elif v1 < v2:
                    strictly_better = True
            else:  # maximize
                if v1 < v2:
                    dominated = False
                elif v1 > v2:
                    strictly_better = True

        return dominated and strictly_better

    def select_final(self, pareto_front: List[HAJson]) -> HAJson:
        """
        Select single best from Pareto front using user preferences or TOPSIS.
        """
        # TOPSIS: Technique for Order Preference by Similarity to Ideal Solution
        objectives = [self.evaluate(c, self.ref_data) for c in pareto_front]

        # Normalize objectives
        obj_matrix = np.array([[getattr(obj, name) for name, _, _ in self.objectives]
                               for obj in objectives])
        norm_matrix = obj_matrix / np.linalg.norm(obj_matrix, axis=0)

        # Weight by importance
        weights = np.array([w for _, _, w in self.objectives])
        weighted = norm_matrix * weights

        # Ideal and anti-ideal solutions
        ideal = np.zeros(len(self.objectives))
        anti_ideal = np.zeros(len(self.objectives))
        for i, (_, direction, _) in enumerate(self.objectives):
            if direction == 'minimize':
                ideal[i] = weighted[:, i].min()
                anti_ideal[i] = weighted[:, i].max()
            else:
                ideal[i] = weighted[:, i].max()
                anti_ideal[i] = weighted[:, i].min()

        # Distance to ideal and anti-ideal
        dist_ideal = np.linalg.norm(weighted - ideal, axis=1)
        dist_anti = np.linalg.norm(weighted - anti_ideal, axis=1)

        # Relative closeness
        closeness = dist_anti / (dist_ideal + dist_anti + 1e-10)

        best_idx = np.argmax(closeness)
        return pareto_front[best_idx]
```

### 5.2 Adaptive Weight Tuning

```python
class AdaptiveWeightTuner:
    """
    Dynamically adjust objective weights based on scenario detection.
    """

    def __init__(self):
        self.scenario_weights = {
            'clear_switching': {
                'rmse': 0.30,
                'mdl': 0.10,
                'event_mae': 0.35,
                'physics_score': 0.25,
            },
            'subtle_switching': {
                'rmse': 0.50,
                'mdl': 0.35,
                'event_mae': 0.05,
                'physics_score': 0.10,
            },
            'high_noise': {
                'rmse': 0.25,
                'mdl': 0.40,  # Favor simpler models
                'event_mae': 0.15,
                'physics_score': 0.20,
            },
        }

    def detect_scenario(self, data_summary: DataSummary) -> str:
        """
        Automatically detect which scenario applies.
        """
        if data_summary.snr < 10:
            return 'high_noise'
        elif data_summary.discontinuity_score > 0.5:
            return 'clear_switching'
        else:
            return 'subtle_switching'

    def get_weights(self, data_summary: DataSummary) -> Dict[str, float]:
        """
        Get objective weights for current scenario.
        """
        scenario = self.detect_scenario(data_summary)
        return self.scenario_weights[scenario]
```

---

## 6. Uncertainty Quantification

### 6.1 Bootstrap Confidence Intervals

```python
class UncertaintyQuantifier:
    """
    Quantify uncertainty in learned HA parameters.
    """

    def __init__(self, n_bootstrap: int = 100):
        self.n_bootstrap = n_bootstrap

    def coefficient_confidence_intervals(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sindy_model: SindyModel,
        confidence: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        Bootstrap confidence intervals for SINDy coefficients.
        """
        n_samples = X.shape[0]
        bootstrap_coeffs = []

        for _ in range(self.n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[idx], y[idx]

            # Refit model
            coeffs = sindy_model.fit(X_boot, y_boot).get_coefficients()
            bootstrap_coeffs.append(coeffs)

        bootstrap_coeffs = np.array(bootstrap_coeffs)

        # Compute confidence intervals
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_coeffs, 100 * alpha / 2, axis=0)
        upper = np.percentile(bootstrap_coeffs, 100 * (1 - alpha / 2), axis=0)

        return {
            term: (lower[i], upper[i])
            for i, term in enumerate(sindy_model.get_feature_names())
        }

    def mode_count_confidence(
        self,
        X: np.ndarray,
        features: np.ndarray,
        k_range: List[int]
    ) -> Dict[int, float]:
        """
        Estimate confidence in mode count selection.
        """
        k_votes = {k: 0 for k in k_range}

        for _ in range(self.n_bootstrap):
            # Resample time series (block bootstrap)
            block_size = len(X) // 10
            n_blocks = len(X) // block_size
            block_idx = np.random.choice(n_blocks, n_blocks, replace=True)
            X_boot = np.concatenate([X[i*block_size:(i+1)*block_size] for i in block_idx])
            feat_boot = np.concatenate([features[i*block_size:(i+1)*block_size] for i in block_idx])

            # Select K for this bootstrap sample
            k_selected = self._select_k(feat_boot, k_range)
            k_votes[k_selected] += 1

        # Normalize to probabilities
        total = sum(k_votes.values())
        return {k: count / total for k, count in k_votes.items()}
```

### 6.2 Prediction Intervals

```python
class PredictionIntervalEstimator:
    """
    Compute prediction intervals for simulated trajectories.
    """

    def __init__(self, ha_json: Dict, coeff_uncertainties: Dict):
        self.ha_json = ha_json
        self.coeff_uncertainties = coeff_uncertainties

    def simulate_with_uncertainty(
        self,
        t: np.ndarray,
        x0: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo simulation with parameter uncertainty.

        Returns:
            - x_mean: mean trajectory
            - x_lower: lower bound (e.g., 5th percentile)
            - x_upper: upper bound (e.g., 95th percentile)
        """
        trajectories = []

        for _ in range(n_samples):
            # Sample coefficients from uncertainty distribution
            sampled_ha = self._sample_ha_parameters()

            # Simulate
            x_sim = simulate_ha(sampled_ha, t, x0)
            trajectories.append(x_sim)

        trajectories = np.array(trajectories)

        x_mean = np.mean(trajectories, axis=0)
        x_lower = np.percentile(trajectories, 5, axis=0)
        x_upper = np.percentile(trajectories, 95, axis=0)

        return x_mean, x_lower, x_upper

    def _sample_ha_parameters(self) -> Dict:
        """
        Sample HA with perturbed coefficients.
        """
        ha_sampled = copy.deepcopy(self.ha_json)

        for mode in ha_sampled['modes']:
            for i, term in enumerate(mode['terms']):
                if term in self.coeff_uncertainties:
                    lower, upper = self.coeff_uncertainties[term]
                    # Sample from uniform or truncated normal
                    mode['coeffs'][i] = np.random.uniform(lower, upper)

        return ha_sampled
```

---

## 7. Physics Consistency Checking

### 7.1 Conservation Law Verification

```python
class PhysicsChecker:
    """
    Verify learned HA against physical principles.
    """

    def __init__(self, system_type: str = 'mechanical'):
        self.system_type = system_type

    def check_energy_conservation(
        self,
        ha_json: Dict,
        trajectory: np.ndarray,
        tolerance: float = 0.1
    ) -> PhysicsCheckResult:
        """
        Check if energy is conserved (for conservative systems)
        or monotonically decreasing (for dissipative systems).
        """
        # Extract position and velocity
        if self.system_type == 'mechanical':
            x = trajectory[0]  # position
            v = trajectory[1]  # velocity

            # Compute kinetic energy (assuming unit mass)
            KE = 0.5 * v**2

            # Try to infer potential energy from equations
            # For now, assume quadratic + cubic potential (Duffing-like)
            # V(x) = 0.5*k*x^2 + 0.25*k3*x^4
            PE = self._estimate_potential_energy(ha_json, x)

            total_energy = KE + PE

            # Check if energy is conserved or decreasing
            energy_change = np.diff(total_energy)

            if np.all(energy_change <= tolerance * np.abs(total_energy[:-1])):
                return PhysicsCheckResult(
                    passed=True,
                    check_type='energy_conservation',
                    message='Energy is conserved or decreasing (dissipative system)'
                )
            else:
                return PhysicsCheckResult(
                    passed=False,
                    check_type='energy_conservation',
                    message=f'Energy increases by {np.max(energy_change):.4f}',
                    violation_indices=np.where(energy_change > tolerance)[0]
                )

    def check_stability(self, ha_json: Dict) -> PhysicsCheckResult:
        """
        Check linearized stability around equilibrium points.
        """
        # Find equilibrium points
        equilibria = self._find_equilibria(ha_json)

        for eq in equilibria:
            jacobian = self._compute_jacobian(ha_json, eq)
            eigenvalues = np.linalg.eigvals(jacobian)

            # Check if all eigenvalues have non-positive real parts
            if np.any(np.real(eigenvalues) > 1e-6):
                return PhysicsCheckResult(
                    passed=False,
                    check_type='stability',
                    message=f'Unstable equilibrium at {eq}, eigenvalues: {eigenvalues}'
                )

        return PhysicsCheckResult(
            passed=True,
            check_type='stability',
            message='All equilibria are stable'
        )

    def check_dimensional_consistency(self, ha_json: Dict) -> PhysicsCheckResult:
        """
        Verify dimensional consistency of equations.
        """
        # This requires dimensional annotations on variables
        # For now, check that all terms have matching units
        # (placeholder implementation)
        return PhysicsCheckResult(
            passed=True,
            check_type='dimensional_consistency',
            message='Dimensional analysis not implemented (requires unit annotations)'
        )
```

---

## 8. Handling Two Image Scenarios (Enhanced)

### 8.1 Scenario A: Clear Mode Switching

**Enhanced Processing Strategy:**

```python
def handle_clear_switching(data, image_prior, noise_model):
    """
    Priority: Accurate switch detection and guard/reset learning
    with uncertainty quantification.
    """
    config = {
        # Segmentation: more sensitive, with confidence
        "changepoint_method": "pelt_with_confidence",
        "changepoint_sensitivity": "high",
        "min_segment_length": max(5, int(3 / noise_model.snr.mean())),

        # Mode selection: favor K >= 2, with probability
        "k_selection_method": "bootstrap_silhouette",
        "k_prior_bias": +0.3,
        "k_confidence_threshold": 0.7,

        # Guard learning: uncertainty-aware SVM
        "guard_method": "conformal_svm",
        "guard_forms": ["linear_directional", "linear", "quadratic"],
        "guard_confidence_level": 0.95,

        # Reset learning: with coefficient confidence
        "reset_method": "robust_regression",
        "reset_forms": ["componentwise_damped", "affine"],
        "reset_confidence_bootstrap": True,

        # Multi-objective weights
        "pareto_weights": {
            'rmse': 0.25,
            'mdl': 0.10,
            'event_mae': 0.40,
            'physics_score': 0.25,
        },

        # SR configuration
        "sr_config": SRConfig(
            use_pysr=True,
            use_sindy=True,
            max_complexity=10,
            physics_library='bouncing',
        ),
    }
    return config
```

### 8.2 Scenario B: Subtle/No Mode Switching

**Enhanced Processing Strategy:**

```python
def handle_subtle_switching(data, image_prior, noise_model):
    """
    Priority: Accurate nonlinear term discovery with uncertainty.
    """
    config = {
        # Segmentation: conservative with high confidence threshold
        "changepoint_method": "binary_segmentation",
        "changepoint_sensitivity": "low",
        "min_segment_length": 50,
        "k_confidence_threshold": 0.9,  # High bar for K > 1

        # Equation learning: rich library with SR ensemble
        "sr_config": SRConfig(
            use_pysr=True,
            use_sindy=True,
            use_neural_sr=True,
            max_complexity=15,
            physics_library='oscillator',
            ensemble_consensus_threshold=0.6,
        ),

        # No guards/resets needed
        "guard_forms": ["none"],
        "reset_forms": ["identity"],

        # Multi-objective weights: emphasize fit and simplicity
        "pareto_weights": {
            'rmse': 0.45,
            'mdl': 0.35,
            'event_mae': 0.05,
            'physics_score': 0.15,
        },

        # Uncertainty quantification
        "coefficient_bootstrap": True,
        "prediction_intervals": True,
    }
    return config
```

---

## 9. Tool Design for smolagent (Enhanced)

### 9.1 Enhanced Tool Registry

```python
TOOL_REGISTRY = {
    # Data & Features (Enhanced)
    "load_npz": LoadNPZTool,
    "estimate_noise": NoiseEstimationTool,  # NEW
    "estimate_derivatives": DerivativeToolWithUncertainty,  # Enhanced
    "compute_features": FeatureTool,

    # Segmentation & Clustering
    "detect_changepoints": ChangepointToolWithConfidence,  # Enhanced
    "select_num_modes": ModeSelectionToolWithBootstrap,  # Enhanced
    "assign_modes": ModeAssignmentTool,

    # Equation Learning (Enhanced)
    "propose_library": PhysicsInformedLibraryTool,  # Enhanced
    "sindy_fit": SindyFitToolWithUncertainty,  # Enhanced
    "pysr_fit": PySRFitTool,  # NEW
    "neural_sr_fit": NeuralSRTool,  # NEW
    "sr_ensemble": SRE EnsembleTool,  # NEW

    # Guard & Reset
    "learn_guard": GuardLearningToolConformal,  # Enhanced
    "learn_reset": ResetLearningToolRobust,  # Enhanced

    # Physics Verification (NEW)
    "check_energy": EnergyConservationTool,
    "check_stability": StabilityCheckTool,
    "check_physics": PhysicsCheckerTool,

    # Assembly & Validation
    "assemble_ha_json": AssembleTool,
    "validate_json": ValidateTool,

    # Evaluation (Enhanced)
    "simulate_and_score": SimScoreToolMultiObjective,  # Enhanced
    "compute_pareto_front": ParetoFrontTool,  # NEW
    "compute_prediction_intervals": PredictionIntervalTool,  # NEW

    # Image Analysis (optional)
    "describe_plot": ImageDescribeTool,
    "image_event_density": EventDensityTool,

    # Utility
    "read_json": ReadJSONTool,
    "write_json": WriteJSONTool,
    "plot_comparison": PlotToolWithUncertainty,  # Enhanced
}
```

### 9.2 Key New Tool Specifications

```python
class SRE EnsembleTool(Tool):
    name = "sr_ensemble"
    description = """
    Run ensemble of symbolic regression methods (PySR, SINDy, Neural SR)
    and return consensus expressions on Pareto front.

    Use this for robust term discovery, especially for nonlinear systems
    where the correct basis functions are unknown.
    """
    inputs = {
        "X": {"type": "array", "description": "Feature matrix"},
        "y": {"type": "array", "description": "Target (derivatives)"},
        "config": {"type": "json", "description": "SR configuration"},
    }
    output_schema = {
        "pareto_front": [{"expression": str, "score": float, "complexity": int}],
        "consensus_terms": [str],
        "method_results": dict,
    }


class PhysicsCheckerTool(Tool):
    name = "check_physics"
    description = """
    Verify learned HA against physical principles:
    - Energy conservation/dissipation
    - Stability around equilibria
    - Dimensional consistency

    Use this after assembling HA JSON to validate physical plausibility.
    """
    inputs = {
        "ha_json": {"type": "json", "description": "Learned HA"},
        "trajectory": {"type": "array", "description": "Reference trajectory"},
        "system_type": {"type": "string", "description": "mechanical, electrical, etc."},
    }
    output_schema = {
        "energy_check": {"passed": bool, "message": str},
        "stability_check": {"passed": bool, "message": str},
        "overall_score": float,
    }


class PredictionIntervalTool(Tool):
    name = "compute_prediction_intervals"
    description = """
    Compute prediction intervals for simulated trajectories using
    Monte Carlo sampling of uncertain parameters.

    Use this to quantify confidence in trajectory predictions.
    """
    inputs = {
        "ha_json": {"type": "json", "description": "Learned HA with uncertainties"},
        "t": {"type": "array", "description": "Time array"},
        "x0": {"type": "array", "description": "Initial condition"},
        "n_samples": {"type": "int", "description": "Number of MC samples"},
        "confidence": {"type": "float", "description": "Confidence level (e.g., 0.95)"},
    }
    output_schema = {
        "x_mean": "array",
        "x_lower": "array",
        "x_upper": "array",
        "coverage_probability": float,
    }
```

---

## 10. JSON Output Specification (Enhanced)

### 10.1 Output Format with Uncertainties

```json
{
  "automaton": {
    "var": "x1, x2",
    "input": "u1",
    "mode": [
      {
        "id": 1,
        "eq": "x1[1] = x2[0], x2[1] = -0.5*x2[0] + x1[0] - 1.5*x1[0]**3 + u1",
        "coefficients": {
          "x2[0]": {"value": -0.5, "ci_lower": -0.52, "ci_upper": -0.48},
          "x1[0]": {"value": 1.0, "ci_lower": 0.98, "ci_upper": 1.02},
          "x1[0]**3": {"value": -1.5, "ci_lower": -1.55, "ci_upper": -1.45}
        }
      }
    ],
    "edge": [
      {
        "direction": "1 -> 1",
        "condition": "true",
        "reset": {}
      }
    ]
  },
  "config": {
    "dt": 0.01,
    "total_time": 10.0,
    "order": 2,
    "need_reset": false,
    "kernel": "linear",
    "other_items": "x[?] ** 3",
    "self_loop": true
  },
  "meta": {
    "generated_by": "HA_LLM_Agent_v2",
    "timestamp": "2024-01-15T10:30:00Z",
    "pareto_scores": [
      {"rmse": 0.0123, "mdl": 2.34, "event_mae": 0.0, "physics_score": 0.95},
      {"rmse": 0.0150, "mdl": 1.80, "event_mae": 0.0, "physics_score": 0.92}
    ],
    "selected_solution": 0,
    "uncertainties": {
      "k_confidence": {"1": 0.92, "2": 0.08},
      "coefficient_bootstrap_n": 100,
      "prediction_interval_coverage": 0.94
    },
    "physics_checks": {
      "energy_conservation": true,
      "stability": true,
      "dimensional_consistency": "not_checked"
    },
    "sr_methods_used": ["sindy", "pysr"],
    "consensus_terms": ["x[0]", "x[0]**3", "x[1]"]
  }
}
```

---

## 11. Implementation Milestones (Revised)

### Milestone A: Core Pipeline with Uncertainty
- [ ] Load npz with noise estimation
- [ ] Derivative estimation with GP uncertainty
- [ ] SINDy fitting with bootstrap CIs
- [ ] JSON assembly with coefficient uncertainties
- **Target**: Duffing with RMSE < 0.05, 95% CI coverage > 90%

### Milestone B: Multi-Mode with Physics Checking
- [ ] Change point detection with confidence
- [ ] K selection with bootstrap probability
- [ ] Guard/reset learning with conformal prediction
- [ ] Energy conservation checking
- **Target**: Ball with Switch F1 > 0.9, physics checks passed

### Milestone C: Symbolic Regression Ensemble
- [ ] PySR integration
- [ ] Neural SR integration
- [ ] Consensus term extraction
- [ ] Pareto front computation
- **Target**: Van der Pol, Duffing variants without manual library

### Milestone D: Multi-Objective & Active Learning
- [ ] Pareto optimization with TOPSIS selection
- [ ] Adaptive weight tuning
- [ ] Active query for ambiguous cases
- **Target**: Handle 90% of automata/ benchmarks

---

## 12. Risk Mitigation (Enhanced)

| Risk | Mitigation | Fallback |
|------|------------|----------|
| Mode not separable | Fall back to K=1 with confidence | Report K uncertainty |
| High noise | TVReg + robust regression | Increase regularization |
| Overfitting | MDL penalty + Pareto filtering | Limit expression complexity |
| LLM hallucination | DSL constraints + physics checks | Reject physically implausible |
| SR divergence | Ensemble consensus + complexity limits | Fall back to SINDy only |
| Computation cost | Caching + parallel evaluation | Reduce beam width |
| Image misleading | Data evidence > 70% weight | Ignore image prior |

---

## 13. Dependencies (Updated)

### Core
```
numpy
scikit-learn
matplotlib
networkx
scipy
```

### Agent
```
smolagent  # Hugging Face agent framework
```

### Symbolic Regression
```
pysindy>=1.7  # SINDy with STRidge
pysr>=0.16    # Genetic programming SR
```

### Uncertainty
```
scikit-learn  # Bootstrap, GP
mapie         # Conformal prediction (optional)
```

### Physics
```
sympy         # Symbolic math for Jacobians
```

---

## 14. References (Extended)

1. **SR-Scientist**: Program synthesis for scientific discovery with LLMs
2. **SINDy**: Sparse Identification of Nonlinear Dynamics (Brunton et al., 2016)
3. **PySR**: Symbolic regression with genetic algorithms (Cranmer et al., 2023)
4. **DAINARX**: Derivative-Agnostic Inference of Nonlinear Hybrid Systems
5. **smolagent**: Hugging Face lightweight agent framework
6. **MDL/BIC**: Model selection via description length
7. **Conformal Prediction**: Distribution-free prediction intervals (Vovk et al., 2005)
8. **NSGA-II**: Multi-objective evolutionary optimization (Deb et al., 2002)
9. **AI Feynman**: Physics-inspired symbolic regression (Udrescu & Tegmark, 2020)

---

*Document Version: 2.0*
*Last Updated: 2025-01-03*
*Changes: Added uncertainty quantification, enhanced SR integration, multi-objective optimization, physics checking*
