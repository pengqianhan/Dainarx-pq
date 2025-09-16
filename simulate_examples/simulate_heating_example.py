#!/usr/bin/env python3
"""
Example: Simulating the Three-State Heating System Automata

This example demonstrates how to use the AutomataSimulator class
to simulate a three-state heating system with second-order dynamics.
"""

import sys
import os
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from AutomataSimulator import AutomataSimulator


def main():
    """Main simulation function for the heating system example."""

    # Path to the three-state heating system automata
    automata_path = os.path.join(os.path.dirname(__file__), '..', 'automata', 'FaMoS', 'three_state_ha.json')

    print("=== Three-State Heating System Simulation Example ===")
    print(f"Loading automata from: {automata_path}")

    # Create simulator and load automata
    simulator = AutomataSimulator(automata_path)

    # Display configuration
    print("\nConfiguration:")
    for key, value in simulator.config.items():
        print(f"  {key}: {value}")

    print(f"\nNumber of initial states: {len(simulator.init_states)}")
    print("Initial states:")
    for i, state in enumerate(simulator.init_states[:3]):  # Show first 3
        print(f"  State {i+1}: mode={state.get('mode')}, x1={state.get('x1', 'N/A')}")

    # Run simulation for all trajectories
    print("\nRunning simulation...")
    trajectories = simulator.simulate_all_trajectories(
        dt=0.01,        # Time step
        total_time=20.0, # Simulation time
        verbose=True
    )

    # Analyze the results
    print("\nAnalyzing trajectories...")
    analysis = simulator.analyze_trajectories(trajectories)

    print(f"Total simulation time: {analysis['total_simulation_time']:.2f} seconds")
    print(f"Number of mode changes across all trajectories: {len(analysis['mode_changes'])}")

    # Print mode change details
    if analysis['mode_changes']:
        print("\nMode Changes (first 10):")
        for i, (time, from_mode, to_mode) in enumerate(analysis['mode_changes'][:10]):
            print(".3f")

    # Print variable statistics
    print("\nVariable Statistics:")
    for var, stats in analysis['variable_statistics'].items():
        print(f"  {var}:")
        print(".3f")
        print(".3f")

    # Create plots
    print("\nGenerating plots...")

    # Plot temperature vs time
    fig1 = simulator.plot_trajectories(
        variables=['x1'],  # Temperature
        save_path='heating_temperature.png'
    )

    # Plot mode evolution over time
    plot_mode_evolution(trajectories, save_path='heating_modes.png')

    # Plot phase portrait (temperature vs time with mode coloring)
    plot_phase_portrait(trajectories, save_path='heating_phase.png')

    # Export results
    print("\nExporting results...")
    # simulator.export_trajectories(format='csv', output_dir='heating_simulation_results')
    # simulator.export_trajectories(format='json', output_dir='heating_simulation_results')

    # Show summary
    print("\n" + "="*50)
    print("SIMULATION SUMMARY")
    print("="*50)
    print(simulator.get_simulation_summary())

    # Display plots
    plt.show()

    print("\nExample completed successfully!")
    print("Check the generated plots and exported data in the current directory.")


def plot_mode_evolution(trajectories, save_path=None):
    """Plot mode evolution over time for all trajectories."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['blue', 'red', 'green']  # Colors for modes 1, 2, 3

    for i, traj in enumerate(trajectories):
        times = traj.times
        modes = traj.mode_history

        # Plot mode as step function
        ax.step(times, modes, where='post', label=f'Trajectory {i+1}',
               alpha=0.7, linewidth=2)

    ax.set_xlabel('Time')
    ax.set_ylabel('Mode')
    ax.set_title('Mode Evolution Over Time')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Mode 1', 'Mode 2', 'Mode 3'])
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mode evolution plot saved to {save_path}")

    return fig


def plot_phase_portrait(trajectories, save_path=None):
    """Plot phase portrait showing temperature evolution with mode coloring."""
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['blue', 'orange', 'green']  # Colors for modes 1, 2, 3

    for i, traj in enumerate(trajectories[:5]):  # Plot first 5 trajectories
        times = traj.times

        # Extract temperatures, handling both list and scalar formats
        temperatures = []
        for state in traj.states:
            val = state.get('x1', 0)
            if isinstance(val, list):
                temperatures.append(val[0] if val else 0)
            else:
                temperatures.append(val)

        modes = traj.mode_history

        # Plot segments with different colors for each mode
        start_idx = 0
        current_mode = modes[0]

        for j in range(1, len(modes)):
            if modes[j] != current_mode or j == len(modes) - 1:
                # Plot segment
                end_idx = j if modes[j] != current_mode else j + 1
                color_idx = current_mode - 1 if 0 <= current_mode - 1 < len(colors) else 0

                ax.plot(times[start_idx:end_idx], temperatures[start_idx:end_idx],
                       color=colors[color_idx], linewidth=2, alpha=0.8,
                       label=f'Traj {i+1}, Mode {current_mode}' if start_idx == 0 else "")

                start_idx = j
                current_mode = modes[j]

    # Add mode threshold lines
    ax.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Mode transitions')
    ax.axhline(y=20, color='red', linestyle='--', alpha=0.7)
    ax.axhline(y=25, color='red', linestyle='--', alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (x1)')
    ax.set_title('Temperature Evolution with Mode Coloring')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Phase portrait saved to {save_path}")

    return fig


def compare_initial_states():
    """Compare trajectories starting from different initial states."""

    print("=== Comparing Different Initial States ===")

    automata_path = os.path.join(os.path.dirname(__file__), '..', 'automata', 'FaMoS', 'three_state_ha.json')
    simulator = AutomataSimulator(automata_path)

    # Select a few representative initial states
    selected_states = simulator.init_states[:3]  # First 3 states

    trajectories = []
    for i, init_state in enumerate(selected_states):
        print(f"Simulating from initial state {i+1}: {init_state}")
        traj = simulator.simulate_trajectory(init_state, dt=0.01, total_time=15.0)
        trajectories.append(traj)

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    for i, traj in enumerate(trajectories):
        times = traj.times

        # Extract temperatures, handling both list and scalar formats
        temperatures = []
        for state in traj.states:
            val = state.get('x1', 0)
            if isinstance(val, list):
                temperatures.append(val[0] if val else 0)
            else:
                temperatures.append(val)

        modes = traj.mode_history

        # Temperature plot
        axes[0].plot(times, temperatures, label=f'Initial State {i+1}', linewidth=2)

        # Mode plot
        axes[1].step(times, modes, where='post', label=f'Initial State {i+1}', linewidth=2)

    axes[0].set_ylabel('Temperature (x1)')
    axes[0].set_title('Temperature Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Mode')
    axes[1].set_title('Mode Comparison')
    axes[1].set_yticks([1, 2, 3])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('heating_state_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("State comparison completed!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            from AutomataSimulator import simulate_automata
            automata_path = os.path.join(os.path.dirname(__file__), '..', 'automata', 'FaMoS', 'three_state_ha.json')
            simulate_automata(automata_path, dt=0.01, total_time=15.0, plot=True, export=True)
            print("Quick simulation completed!")
        elif sys.argv[1] == "--compare":
            compare_initial_states()
    else:
        main()
