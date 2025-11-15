"""
Data Analyzer: Extract features from time series data for LLM analysis
"""

import numpy as np
from scipy import stats, signal
from typing import Dict, List, Any


class DataAnalyzer:
    """Extract statistical and dynamical features from time series data"""

    def __init__(self):
        pass

    def extract_features(self, data_list: List[np.ndarray],
                        input_list: List[np.ndarray] = None,
                        dt: float = 0.001) -> Dict[str, Any]:
        """
        Extract comprehensive features from time series data

        Args:
            data_list: List of state trajectories, each shape (var_num, time_steps)
            input_list: List of input trajectories (optional)
            dt: Sampling time step

        Returns:
            Dictionary of extracted features
        """
        # Concatenate all trajectories for global statistics
        all_data = np.concatenate(data_list, axis=1) if len(data_list) > 0 else data_list[0]

        features = {
            "dimension": all_data.shape[0],
            "num_trajectories": len(data_list),
            "avg_trajectory_length": np.mean([d.shape[1] for d in data_list]),
            "sample_rate": 1.0 / dt,
            "dt": dt,
            "duration": all_data.shape[1] * dt,

            # Statistical features
            "statistics": self._extract_statistics(all_data),

            # Spectral features
            "spectral": self._extract_spectral_features(all_data, dt),

            # Dynamics features
            "dynamics": self._extract_dynamics_features(data_list),

            # Transition features
            "transitions": self._extract_transition_features(data_list, dt),

            # Nonlinearity indicators
            "nonlinearity": self._detect_nonlinearity(all_data)
        }

        return features

    def _extract_statistics(self, data: np.ndarray) -> Dict[str, List[float]]:
        """Extract basic statistical features"""
        var_num = data.shape[0]

        return {
            "mean": data.mean(axis=1).tolist(),
            "std": data.std(axis=1).tolist(),
            "min": data.min(axis=1).tolist(),
            "max": data.max(axis=1).tolist(),
            "range": (data.max(axis=1) - data.min(axis=1)).tolist(),
            "skewness": [stats.skew(data[i]) for i in range(var_num)],
            "kurtosis": [stats.kurtosis(data[i]) for i in range(var_num)],
            "median": np.median(data, axis=1).tolist()
        }

    def _extract_spectral_features(self, data: np.ndarray, dt: float) -> Dict[str, Any]:
        """Extract frequency domain features"""
        var_num = data.shape[0]
        n = data.shape[1]

        spectral_features = {
            "dominant_frequencies": [],
            "power_spectrum_peaks": [],
            "frequency_bandwidth": []
        }

        for i in range(var_num):
            # Compute FFT
            fft_vals = np.fft.fft(data[i])
            fft_freq = np.fft.fftfreq(n, dt)

            # Only positive frequencies
            pos_mask = fft_freq > 0
            freqs = fft_freq[pos_mask]
            power = np.abs(fft_vals[pos_mask])**2

            # Find dominant frequency
            if len(power) > 0:
                dominant_idx = np.argmax(power)
                dominant_freq = freqs[dominant_idx]
                spectral_features["dominant_frequencies"].append(float(dominant_freq))

                # Find peaks in power spectrum
                peaks, _ = signal.find_peaks(power, height=np.max(power)*0.1)
                peak_freqs = freqs[peaks].tolist() if len(peaks) > 0 else []
                spectral_features["power_spectrum_peaks"].append(peak_freqs[:5])  # Top 5 peaks

                # Frequency bandwidth (frequency range containing 95% of power)
                cumsum_power = np.cumsum(power) / np.sum(power)
                f_low = freqs[np.argmax(cumsum_power >= 0.025)]
                f_high = freqs[np.argmax(cumsum_power >= 0.975)]
                spectral_features["frequency_bandwidth"].append(float(f_high - f_low))
            else:
                spectral_features["dominant_frequencies"].append(0.0)
                spectral_features["power_spectrum_peaks"].append([])
                spectral_features["frequency_bandwidth"].append(0.0)

        return spectral_features

    def _extract_dynamics_features(self, data_list: List[np.ndarray]) -> Dict[str, Any]:
        """Extract dynamical system features"""

        # Use first trajectory for analysis
        data = data_list[0] if len(data_list) > 0 else np.array([[]])

        if data.size == 0:
            return {"autocorrelation_decay": 0.0, "oscillatory_behavior": False}

        var_num = data.shape[0]

        dynamics = {
            "autocorrelation_decay": [],
            "oscillatory_behavior": False
        }

        # Autocorrelation analysis
        for i in range(var_num):
            acf = np.correlate(data[i] - data[i].mean(),
                              data[i] - data[i].mean(),
                              mode='full')
            acf = acf[len(acf)//2:]
            acf = acf / acf[0]  # Normalize

            # Find where autocorrelation drops below 0.5
            decay_idx = np.argmax(acf < 0.5) if np.any(acf < 0.5) else len(acf)
            dynamics["autocorrelation_decay"].append(int(decay_idx))

            # Check for oscillatory behavior (negative correlation after decay)
            if decay_idx < len(acf) and np.any(acf[decay_idx:] < -0.2):
                dynamics["oscillatory_behavior"] = True

        return dynamics

    def _extract_transition_features(self, data_list: List[np.ndarray],
                                    dt: float) -> Dict[str, Any]:
        """Estimate transition characteristics using simple derivative-based detection"""

        transition_features = {
            "rough_changepoint_count": 0,
            "avg_segment_length": 0.0,
            "has_sudden_jumps": False
        }

        # Analyze each trajectory
        all_segments = []
        total_changepoints = 0

        for data in data_list:
            if data.shape[1] < 10:
                continue

            # Simple change detection: large derivative changes
            derivatives = np.diff(data, axis=1)
            derivative_magnitude = np.linalg.norm(derivatives, axis=0)

            # Threshold: 3 * median
            threshold = 3 * np.median(derivative_magnitude)
            changepoints = np.where(derivative_magnitude > threshold)[0]

            # Filter close changepoints (merge within 5 time steps)
            if len(changepoints) > 0:
                filtered_cp = [changepoints[0]]
                for cp in changepoints[1:]:
                    if cp - filtered_cp[-1] > 5:
                        filtered_cp.append(cp)

                total_changepoints += len(filtered_cp)

                # Calculate segment lengths
                cp_with_boundaries = [0] + filtered_cp + [data.shape[1]]
                segments = np.diff(cp_with_boundaries)
                all_segments.extend(segments)
            else:
                all_segments.append(data.shape[1])

            # Check for sudden jumps (discontinuities)
            if np.any(derivative_magnitude > 10 * np.median(derivative_magnitude)):
                transition_features["has_sudden_jumps"] = True

        transition_features["rough_changepoint_count"] = total_changepoints
        transition_features["avg_segment_length"] = float(np.mean(all_segments) * dt) if all_segments else 0.0

        return transition_features

    def _detect_nonlinearity(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect nonlinearity indicators"""

        nonlinearity = {
            "likely_nonlinear": False,
            "suggested_terms": [],
            "reasoning": []
        }

        var_num = data.shape[0]

        for i in range(var_num):
            x = data[i]

            # Test 1: Non-Gaussian distribution (high kurtosis)
            kurt = stats.kurtosis(x)
            if abs(kurt) > 1.0:
                nonlinearity["likely_nonlinear"] = True
                nonlinearity["reasoning"].append(f"Variable {i}: High kurtosis ({kurt:.2f})")

            # Test 2: Amplitude-dependent frequency (nonlinear oscillator indicator)
            if var_num > 1:
                # Check if amplitude correlates with instantaneous frequency
                amplitude = np.abs(x)
                analytic_signal = signal.hilbert(x)
                instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                inst_freq = np.diff(instantaneous_phase)

                # Correlation between amplitude and frequency
                if len(inst_freq) > 10:
                    corr = np.corrcoef(amplitude[:-1], inst_freq)[0, 1]
                    if abs(corr) > 0.3:
                        nonlinearity["likely_nonlinear"] = True
                        nonlinearity["reasoning"].append(
                            f"Variable {i}: Amplitude-dependent frequency (corr={corr:.2f})"
                        )
                        # Suggest cubic term for typical Duffing-like behavior
                        if "x[?]**3" not in nonlinearity["suggested_terms"]:
                            nonlinearity["suggested_terms"].append("x[?]**3")

            # Test 3: Asymmetry in phase portrait (suggests odd nonlinearity)
            if var_num >= 2 and i < var_num - 1:
                x_next = data[i + 1]
                # Check symmetry: compare positive and negative regions
                pos_mask = x > np.median(x)
                neg_mask = x < np.median(x)

                if np.sum(pos_mask) > 10 and np.sum(neg_mask) > 10:
                    pos_std = np.std(x_next[pos_mask])
                    neg_std = np.std(x_next[neg_mask])
                    asymmetry = abs(pos_std - neg_std) / (pos_std + neg_std + 1e-10)

                    if asymmetry > 0.3:
                        nonlinearity["likely_nonlinear"] = True
                        nonlinearity["reasoning"].append(
                            f"Variables {i},{i+1}: Asymmetric phase portrait (asym={asymmetry:.2f})"
                        )
                        if "x[?]**2" not in nonlinearity["suggested_terms"]:
                            nonlinearity["suggested_terms"].append("x[?]**2")

        # Default suggestion if nonlinear but no specific terms identified
        if nonlinearity["likely_nonlinear"] and not nonlinearity["suggested_terms"]:
            nonlinearity["suggested_terms"] = ["x[?]**2", "x[?]**3"]

        return nonlinearity

    def format_for_llm(self, features: Dict[str, Any]) -> str:
        """Format features as a readable string for LLM prompt"""

        formatted = f"""
=== Time Series Data Analysis ===

Basic Information:
- Dimension: {features['dimension']}
- Number of trajectories: {features['num_trajectories']}
- Average trajectory length: {features['avg_trajectory_length']:.0f} samples
- Sampling rate: {features['sample_rate']:.1f} Hz (dt={features['dt']})
- Total duration: {features['duration']:.2f} seconds

Statistical Features:
- Mean: {features['statistics']['mean']}
- Std: {features['statistics']['std']}
- Range: {features['statistics']['range']}
- Skewness: {[f"{s:.2f}" for s in features['statistics']['skewness']]}
- Kurtosis: {[f"{k:.2f}" for k in features['statistics']['kurtosis']]}

Spectral Features:
- Dominant frequencies: {[f"{f:.2f} Hz" for f in features['spectral']['dominant_frequencies']]}
- Frequency bandwidth: {[f"{b:.2f} Hz" for b in features['spectral']['frequency_bandwidth']]}

Dynamics:
- Autocorrelation decay: {features['dynamics']['autocorrelation_decay']} samples
- Oscillatory behavior: {features['dynamics']['oscillatory_behavior']}

Transitions:
- Rough changepoint count: {features['transitions']['rough_changepoint_count']}
- Average segment length: {features['transitions']['avg_segment_length']:.3f} seconds
- Has sudden jumps: {features['transitions']['has_sudden_jumps']}

Nonlinearity Analysis:
- Likely nonlinear: {features['nonlinearity']['likely_nonlinear']}
- Suggested terms: {features['nonlinearity']['suggested_terms']}
- Reasoning: {'; '.join(features['nonlinearity']['reasoning'])}
"""
        return formatted.strip()
