#!/usr/bin/env python3
"""
Example: Simulating the Bouncing Ball Automata

This example demonstrates how to use the AutomataSimulator class
to simulate the classic bouncing ball hybrid automata.
"""

import sys
import os
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from AutomataSimulator import AutomataSimulator


def main():
    """Main simulation function for the bouncing ball example."""

    # Path to the bouncing ball automata
    automata_path = os.path.join(os.path.dirname(__file__), '..', 'automata', 'ATVA', 'ball.json')

    print("=== Bouncing Ball Simulation Example ===")
    print(f"Loading automata from: {automata_path}")

    # Create simulator and load automata
    simulator = AutomataSimulator(automata_path)

    # Display configuration
    print("\nConfiguration:")
    for key, value in simulator.config.items():
        print(f"  {key}: {value}")

    print(f"\nNumber of initial states: {len(simulator.init_states)}")

    # Run simulation for all trajectories
    print("\nRunning simulation...")
    trajectories = simulator.simulate_all_trajectories(
        dt=0.01,        # Time step
        total_time=25.0, # Extended time to see more bounces
        verbose=True
    )

    # Analyze the results
    print("\nAnalyzing trajectories...")
    analysis = simulator.analyze_trajectories(trajectories)

    print(f"Total simulation time: {analysis['total_simulation_time']:.2f} seconds")
    print(f"Number of mode changes across all trajectories: {len(analysis['mode_changes'])}")

    # Print variable statistics
    print("\nVariable Statistics:")
    for var, stats in analysis['variable_statistics'].items():
        print(f"  {var}:")
        print(".3f")
        print(".3f")

    # Create plots
    print("\nGenerating plots...")

    # Plot position vs time
    fig1 = simulator.plot_trajectories(
        variables=['x1'],  # Position
        save_path='ball_position.png'
    )

    # Plot velocity vs time
    fig2 = simulator.plot_trajectories(
        variables=['x2'],  # Velocity
        save_path='ball_velocity.png'
    )

    # Plot both position and velocity together
    fig3 = simulator.plot_trajectories(
        variables=['x1', 'x2'],
        figsize=(15, 8),
        save_path='ball_complete.png'
    )

    # Export results
    print("\nExporting results...")
    # simulator.export_trajectories(format='csv', output_dir='ball_simulation_results')
    # simulator.export_trajectories(format='json', output_dir='ball_simulation_results')

    # Show summary
    print("\n" + "="*50)
    print("SIMULATION SUMMARY")
    print("="*50)
    print(simulator.get_simulation_summary())

    # Display plots
    plt.show()

    print("\nExample completed successfully!")
    print("Check the generated plots and exported data in the current directory.")


def quick_simulation():
    """Quick simulation with minimal output."""

    print("=== Quick Bouncing Ball Simulation ===")

    # Use the convenience function for quick simulation
    from AutomataSimulator import simulate_automata

    automata_path = os.path.join(os.path.dirname(__file__), '..', 'automata', 'ATVA', 'ball.json')

    simulator = simulate_automata(
        automata_path,
        dt=0.01,
        total_time=15.0,
        plot=True,
        export=False,
        export_format='csv'
    )

    print("Quick simulation completed!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_simulation()
    else:
        main()
