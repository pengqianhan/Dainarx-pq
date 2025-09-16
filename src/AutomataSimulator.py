#!/usr/bin/env python3
"""
General Automata Simulation Framework

This module provides a comprehensive simulator for various types of automata,
including hybrid automata, finite state machines, and other dynamical systems.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import warnings


class AutomataLoader:
    """Handles loading different types of automata from JSON files."""

    @staticmethod
    def load_automata(json_path: str) -> Dict[str, Any]:
        """Load automata definition from JSON file."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Automata file not found: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        return data

    @staticmethod
    def detect_automata_type(data: Dict[str, Any]) -> str:
        """Detect the type of automata from the data structure."""
        if 'automaton' in data:
            automaton = data['automaton']

            # Check for hybrid automata characteristics
            if 'mode' in automaton and 'edge' in automaton:
                if 'var' in automaton:
                    return 'hybrid_automata'
                else:
                    return 'finite_state_machine'

            # Check for other types
            if 'states' in automaton:
                return 'generic_automata'

        # Default fallback
        return 'unknown'


class BaseAutomata(ABC):
    """Abstract base class for automata implementations."""

    def __init__(self, definition: Dict[str, Any]):
        self.definition = definition
        self.current_state = None
        self.state_history = []

    @abstractmethod
    def reset(self, initial_state: Dict[str, Any]) -> None:
        """Reset the automata to an initial state."""
        pass

    @abstractmethod
    def step(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute one step of the automata."""
        pass

    @abstractmethod
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the automata."""
        pass

    @abstractmethod
    def get_state_variables(self) -> List[str]:
        """Get list of state variables."""
        pass


class HybridAutomataAdapter(BaseAutomata):
    """Adapter for the existing HybridAutomata class."""

    def __init__(self, definition: Dict[str, Any]):
        super().__init__(definition)
        # Import here to avoid circular imports
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from HybridAutomata import HybridAutomata

        self.ha = HybridAutomata.from_json(definition)
        self.variable_names = definition.get('var', '').split(',')

    def reset(self, initial_state: Dict[str, Any]) -> None:
        """Reset to initial state."""
        self.ha.reset(initial_state)
        self.current_state = initial_state
        self.state_history = [initial_state]

    def step(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute one step."""
        dt = inputs.get('dt', 0.01) if inputs else 0.01
        next_values, prev_mode, switched = self.ha.next(dt)

        # Create state dictionary
        new_state = {'mode': self.ha.mode_state}
        for i, var_name in enumerate(self.variable_names):
            if i < len(next_values):
                new_state[var_name.strip()] = next_values[i]

        self.current_state = new_state
        self.state_history.append(new_state)

        return {
            'state': new_state,
            'switched': switched,
            'prev_mode': prev_mode,
            'dt': dt
        }

    def get_current_state(self) -> Dict[str, Any]:
        """Get current state."""
        return self.current_state

    def get_state_variables(self) -> List[str]:
        """Get state variables."""
        return [var.strip() for var in self.variable_names]


class Trajectory:
    """Represents a single trajectory from simulation."""

    def __init__(self, initial_state: Dict[str, Any]):
        self.initial_state = initial_state
        self.states = []
        self.times = []
        self.mode_history = []
        self.switches = []

    def add_step(self, time: float, state: Dict[str, Any], switched: bool = False):
        """Add a step to the trajectory."""
        self.times.append(time)
        self.states.append(state)
        self.mode_history.append(state.get('mode'))
        if switched:
            self.switches.append((time, state.get('mode')))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trajectory to pandas DataFrame."""
        data = {'time': self.times}

        # Add state variables
        for key in self.states[0].keys():
            if key != 'mode':
                data[key] = [state.get(key, np.nan) for state in self.states]

        # Add mode if it exists
        if any('mode' in state for state in self.states):
            data['mode'] = [state.get('mode') for state in self.states]

        return pd.DataFrame(data)

    def get_mode_changes(self) -> List[Tuple[float, Any, Any]]:
        """Get list of mode changes as (time, from_mode, to_mode)."""
        changes = []
        for i in range(1, len(self.mode_history)):
            if self.mode_history[i] != self.mode_history[i-1]:
                changes.append((self.times[i], self.mode_history[i-1], self.mode_history[i]))
        return changes


class AutomataSimulator:
    """
    General-purpose automata simulator that can handle different types of automata.

    Features:
    - Load automata from JSON files
    - Support multiple initial states
    - Configurable simulation parameters
    - Trajectory collection and analysis
    - Visualization capabilities
    - Export functionality
    """

    def __init__(self, automata_path: Optional[str] = None):
        self.automata_path = automata_path
        self.automata_data = None
        self.automata = None
        self.config = {}
        self.trajectories = []

        if automata_path:
            self.load_automata(automata_path)

    def load_automata(self, json_path: str) -> None:
        """Load automata from JSON file."""
        self.automata_path = json_path
        self.automata_data = AutomataLoader.load_automata(json_path)

        # Detect automata type
        automata_type = AutomataLoader.detect_automata_type(self.automata_data)

        # Create appropriate automata instance
        if automata_type == 'hybrid_automata':
            self.automata = HybridAutomataAdapter(self.automata_data['automaton'])
        else:
            raise ValueError(f"Unsupported automata type: {automata_type}")

        # Extract configuration
        self.config = self.automata_data.get('config', {})
        self.init_states = self.automata_data.get('init_state', [])

        print(f"Loaded {automata_type} from {json_path}")
        print(f"Configuration: {self.config}")
        print(f"Number of initial states: {len(self.init_states)}")

    def simulate_trajectory(self,
                          initial_state: Dict[str, Any],
                          dt: float = 0.01,
                          total_time: float = 10.0,
                          max_steps: Optional[int] = None,
                          verbose: bool = False) -> Trajectory:
        """
        Simulate a single trajectory from an initial state.

        Args:
            initial_state: Initial state dictionary
            dt: Time step size
            total_time: Total simulation time
            max_steps: Maximum number of steps (overrides total_time if set)
            verbose: Whether to print progress

        Returns:
            Trajectory object containing the simulation results
        """
        if self.automata is None:
            raise ValueError("No automata loaded. Call load_automata() first.")

        # Reset automata
        self.automata.reset(initial_state)

        # Create trajectory
        trajectory = Trajectory(initial_state)

        # Calculate number of steps
        if max_steps is None:
            steps = int(total_time / dt)
        else:
            steps = max_steps

        current_time = 0.0

        # Add initial state
        trajectory.add_step(current_time, initial_state)

        # Simulation loop
        for step in range(steps):
            # Execute step
            result = self.automata.step({'dt': dt})
            current_time += dt

            # Add to trajectory
            trajectory.add_step(current_time, result['state'], result['switched'])

            # Progress reporting
            if verbose and step % 1000 == 0 and step > 0:
                print(".3f")

        if verbose:
            print(".3f")

        return trajectory

    def simulate_all_trajectories(self,
                                dt: Optional[float] = None,
                                total_time: Optional[float] = None,
                                max_steps: Optional[int] = None,
                                verbose: bool = True) -> List[Trajectory]:
        """
        Simulate trajectories for all initial states.

        Args:
            dt: Time step size (uses config default if None)
            total_time: Total simulation time (uses config default if None)
            max_steps: Maximum steps per trajectory
            verbose: Whether to print progress

        Returns:
            List of Trajectory objects
        """
        if not self.init_states:
            raise ValueError("No initial states found in automata data")

        # Use defaults from config or provided values
        dt = dt or self.config.get('dt', 0.01)
        total_time = total_time or self.config.get('total_time', 10.0)

        trajectories = []

        for i, init_state in enumerate(self.init_states):
            if verbose:
                print(f"Simulating trajectory {i + 1}/{len(self.init_states)}")

            trajectory = self.simulate_trajectory(
                init_state, dt, total_time, max_steps, verbose=False
            )
            trajectories.append(trajectory)

        self.trajectories = trajectories

        if verbose:
            print(f"\nCompleted simulation of {len(trajectories)} trajectories")

        return trajectories

    def plot_trajectories(self,
                         trajectories: Optional[List[Trajectory]] = None,
                         variables: Optional[List[str]] = None,
                         save_path: Optional[str] = None,
                         show_mode_changes: bool = True,
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot simulation trajectories.

        Args:
            trajectories: Trajectories to plot (uses self.trajectories if None)
            variables: Variables to plot (all if None)
            save_path: Path to save plot
            show_mode_changes: Whether to highlight mode changes
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        if trajectories is None:
            trajectories = self.trajectories

        if not trajectories:
            raise ValueError("No trajectories to plot")

        fig, axes = plt.subplots(len(variables) if variables else 1,
                                1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # Get all state variables if not specified
        if variables is None:
            variables = self.automata.get_state_variables()

        for i, var in enumerate(variables):
            ax = axes[i] if i < len(axes) else axes[0]

            for j, traj in enumerate(trajectories):
                times = traj.times

                # Extract values, handling both list and scalar formats
                values = []
                for state in traj.states:
                    val = state.get(var, np.nan)
                    if isinstance(val, list):
                        values.append(val[0] if val else np.nan)
                    else:
                        values.append(val)

                # Plot trajectory
                label = f"Trajectory {j+1}" if len(trajectories) > 1 else var
                ax.plot(times, values, label=label, alpha=0.7)

                # Highlight mode changes
                if show_mode_changes and hasattr(traj, 'get_mode_changes'):
                    changes = traj.get_mode_changes()
                    for change_time, from_mode, to_mode in changes:
                        ax.axvline(x=change_time, color='red', linestyle='--',
                                 alpha=0.5, linewidth=1)

            ax.set_xlabel('Time')
            ax.set_ylabel(var)
            ax.set_title(f'{var} vs Time')
            ax.grid(True, alpha=0.3)

            if len(trajectories) > 1:
                ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        return fig

    def export_trajectories(self,
                          trajectories: Optional[List[Trajectory]] = None,
                          format: str = 'csv',
                          output_dir: str = 'simulation_results') -> str:
        """
        Export simulation results to various formats.

        Args:
            trajectories: Trajectories to export
            format: Export format ('csv', 'json', 'numpy')
            output_dir: Output directory

        Returns:
            Path to exported files
        """
        if trajectories is None:
            trajectories = self.trajectories

        if not trajectories:
            raise ValueError("No trajectories to export")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        if format == 'csv':
            for i, traj in enumerate(trajectories):
                df = traj.to_dataframe()
                filename = f"trajectory_{i+1}.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath, index=False)

        elif format == 'json':
            results = {
                'trajectories': [],
                'metadata': {
                    'automata_path': self.automata_path,
                    'config': self.config,
                    'num_trajectories': len(trajectories)
                }
            }

            for i, traj in enumerate(trajectories):
                traj_data = {
                    'id': i + 1,
                    'initial_state': traj.initial_state,
                    'states': traj.states,
                    'times': traj.times,
                    'mode_changes': traj.get_mode_changes() if hasattr(traj, 'get_mode_changes') else []
                }
                results['trajectories'].append(traj_data)

            filepath = os.path.join(output_dir, 'simulation_results.json')
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)

        elif format == 'numpy':
            for i, traj in enumerate(trajectories):
                # Convert to numpy arrays
                times = np.array(traj.times)
                states_array = np.array([[state.get(var, 0) for var in self.automata.get_state_variables()]
                                       for state in traj.states])

                filename = f"trajectory_{i+1}"
                np.savez(os.path.join(output_dir, filename),
                        times=times, states=states_array,
                        initial_state=traj.initial_state)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        print(f"Results exported to {output_dir} in {format} format")
        return output_dir

    def analyze_trajectories(self, trajectories: Optional[List[Trajectory]] = None) -> Dict[str, Any]:
        """
        Analyze simulation trajectories and return statistics.

        Args:
            trajectories: Trajectories to analyze

        Returns:
            Dictionary with analysis results
        """
        if trajectories is None:
            trajectories = self.trajectories

        if not trajectories:
            raise ValueError("No trajectories to analyze")

        analysis = {
            'num_trajectories': len(trajectories),
            'total_simulation_time': 0,
            'mode_changes': [],
            'variable_statistics': {}
        }

        # Analyze each trajectory
        for traj in trajectories:
            if traj.times:
                analysis['total_simulation_time'] = max(analysis['total_simulation_time'], traj.times[-1])

            # Count mode changes
            if hasattr(traj, 'get_mode_changes'):
                changes = traj.get_mode_changes()
                analysis['mode_changes'].extend(changes)

            # Calculate statistics for each variable
            for var in self.automata.get_state_variables():
                if var not in analysis['variable_statistics']:
                    analysis['variable_statistics'][var] = {
                        'min': float('inf'),
                        'max': float('-inf'),
                        'mean': 0,
                        'std': 0
                    }

                values = []
                for state in traj.states:
                    if var in state:
                        val = state.get(var, 0)
                        if isinstance(val, list):
                            values.append(val[0] if val else 0)
                        else:
                            values.append(val)

                if values:
                    analysis['variable_statistics'][var]['min'] = min(
                        analysis['variable_statistics'][var]['min'], min(values))
                    analysis['variable_statistics'][var]['max'] = max(
                        analysis['variable_statistics'][var]['max'], max(values))

        # Calculate overall statistics
        for var, stats in analysis['variable_statistics'].items():
            # This would require collecting all values across trajectories
            # For now, just report the range
            pass

        return analysis

    def get_simulation_summary(self) -> str:
        """Get a summary of the simulation setup and results."""
        if self.automata is None:
            return "No automata loaded"

        summary = ".3f"
        return summary


# Convenience functions for common use cases
def simulate_automata(json_path: str,
                     dt: Optional[float] = None,
                     total_time: Optional[float] = None,
                     plot: bool = True,
                     export: bool = False,
                     export_format: str = 'csv') -> AutomataSimulator:
    """
    Convenience function to simulate an automata with common settings.

    Args:
        json_path: Path to automata JSON file
        dt: Time step size
        total_time: Total simulation time
        plot: Whether to generate plots
        export: Whether to export results
        export_format: Export format

    Returns:
        AutomataSimulator instance with results
    """
    simulator = AutomataSimulator(json_path)

    # Use config defaults if not specified
    dt = dt or simulator.config.get('dt', 0.01)
    total_time = total_time or simulator.config.get('total_time', 10.0)

    # Run simulation
    trajectories = simulator.simulate_all_trajectories(dt=dt, total_time=total_time)

    # Plot if requested
    if plot:
        variables = simulator.automata.get_state_variables()
        simulator.plot_trajectories(variables=variables)

    # Export if requested
    if export:
        simulator.export_trajectories(format=export_format)

    return simulator


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Automata Simulator')
    parser.add_argument('automata_file', help='Path to automata JSON file')
    parser.add_argument('--dt', type=float, default=None, help='Time step size')
    parser.add_argument('--time', type=float, default=None, help='Total simulation time')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--export', action='store_true', help='Export results')
    parser.add_argument('--format', choices=['csv', 'json', 'numpy'], default='csv', help='Export format')

    args = parser.parse_args()

    try:
        simulator = simulate_automata(
            args.automata_file,
            dt=args.dt,
            total_time=args.time,
            plot=not args.no_plot,
            export=args.export,
            export_format=args.format
        )

        print(simulator.get_simulation_summary())

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
