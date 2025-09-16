# AutomataSimulator: General-Purpose Automata Simulation Framework

The `AutomataSimulator` is a comprehensive Python framework for simulating various types of automata, with particular focus on hybrid automata. It provides a unified interface for loading, simulating, analyzing, and visualizing different automata systems.

## Features

- **Universal Automata Support**: Load and simulate different types of automata from JSON files
- **Multiple Trajectory Simulation**: Run simulations from multiple initial states
- **Flexible Configuration**: Customizable time steps, simulation duration, and parameters
- **Rich Visualization**: Generate plots of trajectories, mode changes, and phase portraits
- **Data Export**: Export simulation results in CSV, JSON, and NumPy formats
- **Trajectory Analysis**: Built-in analysis tools for understanding simulation results
- **Extensible Architecture**: Easy to add support for new automata types

## Installation

The AutomataSimulator is part of the DAINARX project. Make sure you have the required dependencies:

```bash
pip install numpy matplotlib pandas
```

## Quick Start

### Basic Usage

```python
from AutomataSimulator import AutomataSimulator

# Create simulator and load automata
simulator = AutomataSimulator('automata/ATVA/ball.json')

# Run simulation
trajectories = simulator.simulate_all_trajectories(dt=0.01, total_time=10.0)

# Plot results
simulator.plot_trajectories()

# Export data
simulator.export_trajectories(format='csv')
```

### Convenience Function

For quick simulations, use the convenience function:

```python
from AutomataSimulator import simulate_automata

simulator = simulate_automata(
    'automata/ATVA/ball.json',
    dt=0.01,
    total_time=15.0,
    plot=True,
    export=True,
    export_format='csv'
)
```

## Supported Automata Types

### Hybrid Automata
The framework currently supports hybrid automata with the following structure:

```json
{
  "automaton": {
    "var": "x1, x2",
    "mode": [
      {
        "id": 1,
        "eq": "x1[1] = x2[0], x2[1] = -9.8"
      }
    ],
    "edge": [
      {
        "direction": "1 -> 1",
        "condition": "x1 <= 0",
        "reset": {
          "x1": [0],
          "x2": ["-0.9 * x2[0]"]
        }
      }
    ]
  },
  "init_state": [
    {"mode": 1, "x1": [1.2], "x2": [-2.0]}
  ],
  "config": {
    "dt": 0.01,
    "total_time": 10.0,
    "order": 1
  }
}
```

## API Reference

### AutomataSimulator Class

#### Constructor

```python
AutomataSimulator(automata_path=None)
```

- `automata_path` (str, optional): Path to automata JSON file

#### Methods

##### `load_automata(json_path)`

Load automata from JSON file.

- `json_path` (str): Path to the JSON file

##### `simulate_trajectory(initial_state, dt=0.01, total_time=10.0, max_steps=None, verbose=False)`

Simulate a single trajectory from an initial state.

- `initial_state` (dict): Initial state dictionary
- `dt` (float): Time step size
- `total_time` (float): Total simulation time
- `max_steps` (int, optional): Maximum number of steps
- `verbose` (bool): Print progress information

Returns: `Trajectory` object

##### `simulate_all_trajectories(dt=None, total_time=None, max_steps=None, verbose=True)`

Simulate trajectories for all initial states.

- `dt` (float, optional): Time step size (uses config default if None)
- `total_time` (float, optional): Total simulation time (uses config default if None)
- `max_steps` (int, optional): Maximum steps per trajectory
- `verbose` (bool): Print progress information

Returns: List of `Trajectory` objects

##### `plot_trajectories(trajectories=None, variables=None, save_path=None, show_mode_changes=True, figsize=(12, 8))`

Plot simulation trajectories.

- `trajectories` (list, optional): Trajectories to plot
- `variables` (list, optional): Variables to plot
- `save_path` (str, optional): Path to save plot
- `show_mode_changes` (bool): Highlight mode changes
- `figsize` (tuple): Figure size

Returns: matplotlib Figure object

##### `export_trajectories(trajectories=None, format='csv', output_dir='simulation_results')`

Export simulation results.

- `trajectories` (list, optional): Trajectories to export
- `format` (str): Export format ('csv', 'json', 'numpy')
- `output_dir` (str): Output directory

##### `analyze_trajectories(trajectories=None)`

Analyze simulation trajectories.

- `trajectories` (list, optional): Trajectories to analyze

Returns: Dictionary with analysis results

## Trajectory Class

Represents a single simulation trajectory.

### Methods

#### `to_dataframe()`

Convert trajectory to pandas DataFrame.

Returns: pandas DataFrame

#### `get_mode_changes()`

Get list of mode changes.

Returns: List of (time, from_mode, to_mode) tuples

## Examples

### Example 1: Bouncing Ball Simulation

```python
import sys
import os
sys.path.append('src')

from AutomataSimulator import AutomataSimulator

# Load and simulate bouncing ball
simulator = AutomataSimulator('automata/ATVA/ball.json')

# Run simulation with extended time to see multiple bounces
trajectories = simulator.simulate_all_trajectories(dt=0.01, total_time=25.0)

# Plot position and velocity
simulator.plot_trajectories(variables=['x1'])  # Position
simulator.plot_trajectories(variables=['x2'])  # Velocity

# Export results
simulator.export_trajectories(format='csv')
```

### Example 2: Three-State Heating System

```python
from AutomataSimulator import AutomataSimulator

# Load heating system automata
simulator = AutomataSimulator('automata/FaMoS/three_state_ha.json')

# Run simulation
trajectories = simulator.simulate_all_trajectories(dt=0.01, total_time=20.0)

# Analyze mode changes
analysis = simulator.analyze_trajectories()
print(f"Mode changes: {len(analysis['mode_changes'])}")

# Plot temperature evolution
simulator.plot_trajectories(variables=['x1'])
```

### Example 3: Custom Analysis and Visualization

```python
from AutomataSimulator import AutomataSimulator
import matplotlib.pyplot as plt

simulator = AutomataSimulator('automata/ATVA/ball.json')
trajectories = simulator.simulate_all_trajectories()

# Custom plotting
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

for traj in trajectories[:3]:  # Plot first 3 trajectories
    times = traj.times
    positions = [state['x1'] for state in traj.states]
    velocities = [state['x2'] for state in traj.states]

    axes[0].plot(times, positions, alpha=0.7)
    axes[1].plot(times, velocities, alpha=0.7)

axes[0].set_ylabel('Position (x1)')
axes[1].set_ylabel('Velocity (x2)')
axes[1].set_xlabel('Time')

plt.tight_layout()
plt.show()
```

## Command Line Usage

The AutomataSimulator can also be used from the command line:

```bash
# Basic simulation
python src/AutomataSimulator.py automata/ATVA/ball.json

# Custom parameters
python src/AutomataSimulator.py automata/ATVA/ball.json --dt 0.005 --time 20.0

# Export results without plotting
python src/AutomataSimulator.py automata/ATVA/ball.json --no-plot --export --format json
```

## Extending the Framework

### Adding New Automata Types

To add support for new automata types:

1. Create a new class inheriting from `BaseAutomata`
2. Implement the required abstract methods
3. Update `AutomataLoader.detect_automata_type()` to recognize the new type
4. Update `AutomataSimulator.load_automata()` to instantiate the new class

Example:

```python
class MyAutomataType(BaseAutomata):
    def reset(self, initial_state):
        # Implementation
        pass

    def step(self, inputs=None):
        # Implementation
        pass

    def get_current_state(self):
        # Implementation
        pass

    def get_state_variables(self):
        # Implementation
        pass
```

### Custom Visualization

The framework is designed to be extensible. You can create custom visualization functions:

```python
def custom_plotter(trajectories, **kwargs):
    # Your custom plotting logic
    pass

# Use with simulator
simulator.plot_trajectories = custom_plotter
```

## Configuration Options

### Automata JSON Configuration

```json
{
  "config": {
    "dt": 0.01,              // Time step size
    "total_time": 10.0,      // Total simulation time
    "order": 1,              // System order (for ODE systems)
    "need_reset": true,      // Whether resets are needed
    "self_loop": true        // Allow self-transitions
  }
}
```

### Simulation Parameters

- `dt`: Time step for numerical integration
- `total_time`: Total simulation duration
- `max_steps`: Alternative to total_time (maximum number of steps)
- `verbose`: Control progress reporting

## Performance Considerations

- For long simulations, consider using larger time steps
- Limit the number of trajectories for memory-intensive plotting
- Use `export_trajectories()` with 'numpy' format for large datasets
- Consider using `max_steps` instead of `total_time` for precise control

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the `src` directory is in your Python path
2. **File Not Found**: Check that automata JSON files exist at the specified path
3. **Memory Issues**: Reduce the number of trajectories or use smaller time steps
4. **Plotting Errors**: Ensure matplotlib is properly installed and configured

### Debug Mode

Enable verbose output for debugging:

```python
trajectories = simulator.simulate_all_trajectories(verbose=True)
```

## Dependencies

- numpy: Numerical computations
- matplotlib: Plotting and visualization
- pandas: Data manipulation and CSV export
- json: Built-in JSON support

## License

This framework is part of the DAINARX project. See the main project license for details.

## Contributing

To contribute to the AutomataSimulator:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:

1. Check the examples in the `examples/` directory
2. Review the API documentation
3. Create an issue in the project repository
4. Consult the DAINARX project documentation

---

The AutomataSimulator provides a powerful and flexible framework for automata simulation, combining ease of use with extensive customization options. Whether you're working with simple bouncing balls or complex multi-mode systems, this framework offers the tools you need for effective simulation and analysis.
