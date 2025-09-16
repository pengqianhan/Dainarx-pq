# Universal Automata Simulator (sim_all.py)

A flexible command-line tool for simulating hybrid automata from the DAINARX project.

## Features

- **Flexible Selection**: Choose specific automata, categories, or simulate all available ones
- **Interactive Mode**: User-friendly interface for selecting automata to simulate
- **Batch Processing**: Run multiple simulations with customizable parameters
- **Export Options**: Save results in CSV, JSON, or NumPy format
- **Visualization**: Automatic plotting of simulation results
- **Progress Tracking**: Real-time progress updates and detailed summaries

## Usage

### Basic Commands

```bash
# Interactive mode - choose automata interactively
python sim_all.py

# List all available automata
python sim_all.py --list

# Simulate all automata
python sim_all.py --all

# Simulate by category
python sim_all.py --category ATVA
python sim_all.py --category FaMoS

# Simulate specific automata
python sim_all.py --automata ball
python sim_all.py --automata three_state_ha

# Search for automata by name
python sim_all.py --search heating
```

### Advanced Options

```bash
# Custom simulation parameters
python sim_all.py --automata ball --dt 0.05 --time 20.0

# Export results without plotting
python sim_all.py --category FaMoS --export --format json --no-plot

# Quiet mode for batch processing
python sim_all.py --all --quiet --export

# Custom output directory
python sim_all.py --automata ball --export --output-dir my_results
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--all` | Simulate all available automata | - |
| `--category CAT` | Simulate all automata in category | - |
| `--automata NAME` | Simulate specific automata by name | - |
| `--search TERM` | Search for automata by name | - |
| `--list` | List all available automata | - |
| `--dt FLOAT` | Time step size | 0.01 |
| `--time FLOAT` | Total simulation time | 10.0 |
| `--no-plot` | Skip plotting (faster) | False |
| `--export` | Export simulation results | False |
| `--format FORMAT` | Export format (csv/json/numpy) | csv |
| `--output-dir DIR` | Output directory for results | auto-generated |
| `--verbose` | Verbose output | False |
| `--quiet` | Minimal output | False |

## Available Automata

The script automatically scans the `automata/` directory and organizes automata by category:

### ATVA (4 automata)
- ball - Bouncing ball with gravity and floor collision
- cell - Cellular automata model
- oci - OCI (Open Container Initiative) model
- tanks - Multi-tank system

### FaMoS (7 automata)
- buck_converter - DC-DC buck converter
- complex_tank - Complex tank system
- multi_room_heating - Multi-room heating system
- simple_heating_system - Basic heating system
- three_state_ha - Three-state hybrid automaton
- two_state_ha - Two-state hybrid automaton
- variable_heating_system - Variable heating system

### linear (7 automata)
- complex_underdamped_system - Complex underdamped system
- dc_motor_position_PID - DC motor with PID control
- linear_1 - Linear system 1
- loop - Looping system
- one_legged_jumper - One-legged jumper
- two_tank - Two-tank system
- underdamped_system - Basic underdamped system

### non_linear (8 automata)
- duffing - Duffing oscillator
- lander - Lunar lander
- lotkaVolterra - Lotka-Volterra predator-prey model
- oscillator - Nonlinear oscillator
- simple_non_linear - Simple nonlinear system
- simple_non_poly - Simple non-polynomial system
- spacecraft - Spacecraft dynamics
- sys_bio - Systems biology model

## Output

The script provides:
- **Real-time progress** during simulation
- **Analysis summary** with trajectory count, simulation time, and mode changes
- **Automatic plotting** of state variables over time
- **Export capabilities** in multiple formats
- **Success/failure tracking** with detailed error reporting

## Examples

### Example 1: Quick simulation of a specific automata
```bash
python sim_all.py --automata ball --dt 0.01 --time 15.0
```

### Example 2: Batch process all FaMoS automata
```bash
python sim_all.py --category FaMoS --no-plot --export --format csv
```

### Example 3: Search and simulate heating systems
```bash
python sim_all.py --search heating --time 20.0 --export
```

### Example 4: Full simulation with custom parameters
```bash
python sim_all.py --all --dt 0.05 --time 25.0 --export --format json --output-dir full_simulation_results
```

## Error Handling

The script includes comprehensive error handling:
- Invalid automata names are reported with suggestions
- Missing files are handled gracefully
- Simulation failures are tracked and reported
- Partial failures don't stop the entire batch

## Dependencies

- Python 3.7+
- NumPy
- Matplotlib
- Pandas
- AutomataSimulator (included in the project)

## Integration

This script integrates seamlessly with the existing DAINARX project structure and uses the `AutomataSimulator` class for all simulation tasks. It can be easily extended to support additional automata types or output formats.
