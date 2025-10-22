# Numpy Array Input Feature

## Overview

The ODE system now supports using numpy arrays as input values, in addition to the original string expression method. This enables simulation with pre-recorded input data loaded from `.npz` files.

## Usage

### Method 1: String Expression (Original, Backward Compatible)

```python
from src.HybridAutomata import HybridAutomata

sys = HybridAutomata.from_json(automaton_config)
init_state = {
    'mode': 1,
    'x': [4.0],
    'u': '0.5 * cos(1.2 * t)'  # String expression
}
sys.reset(init_state, dt=0.001)
```

### Method 2: Numpy Array (New Feature)

```python
import numpy as np
from src.HybridAutomata import HybridAutomata

# Load pre-recorded input data
npz_file = np.load('data_duffing/test_data0.npz')
input_array = npz_file['input']  # Shape: (1, num_steps) or (num_steps,)

sys = HybridAutomata.from_json(automaton_config)
init_state = {
    'mode': 1,
    'x': [4.0],
    'u': input_array  # Numpy array
}
sys.reset(init_state, dt=0.001)  # Must pass dt for array-based inputs
```

## Important Notes

### Array Indexing Convention

**The array indexing follows a specific convention**: `array[i]` corresponds to time `(i+1) * dt`, not `i * dt`.

This is because the original data collection loop works as follows:

```python
sys.reset(init_state, dt=dT)
idx = 0
while idx < num_steps:
    idx += 1
    state, mode, switched = sys.next(dT)    # Advances time to idx*dT
    input_val = sys.getInput()               # Gets input at time idx*dT
    input_data.append(input_val)             # Stores in array[idx-1]
```

Therefore:
- `array[0]` contains the input value at time `t = dt`
- `array[1]` contains the input value at time `t = 2*dt`
- `array[i]` contains the input value at time `t = (i+1)*dt`

### Creating Compatible Arrays

When creating input arrays manually (not from stored simulation data), ensure they follow this convention:

```python
dt = 0.001
# CORRECT: Times start at dt, not 0
times = np.arange(dt, total_time + dt, dt)
input_array = 0.5 * np.cos(1.2 * times)

# WRONG: Times start at 0 - will cause timing mismatch
times = np.arange(0, total_time, dt)
input_array = 0.5 * np.cos(1.2 * times)
```

### Required dt Parameter

When using array-based inputs, you **must** pass the `dt` parameter to `reset()`:

```python
sys.reset(init_state, dt=0.001)  # Required for arrays
```

This allows the system to correctly map continuous time values to discrete array indices.

## Supported Array Formats

The system accepts:

1. **2D arrays** with shape `(1, num_steps)` or `(num_steps, 1)`
2. **1D arrays** with shape `(num_steps,)`
3. **Python lists** (automatically converted to numpy arrays)

## Example: Full Workflow

```python
import numpy as np
import json
from src.HybridAutomata import HybridAutomata

# 1. Load automaton configuration
with open('automata/non_linear/duffing_simulate.json', 'r') as f:
    data = json.load(f)
    sys = HybridAutomata.from_json(data['automaton'])

# 2. Load pre-recorded input data
npz_file = np.load('data_duffing/test_data0.npz')
input_list = npz_file['input']      # Shape: (1, 10001)
state_data = npz_file['state']      # Shape: (1, 10001)

# 3. Set up initial state with array input
init_state_dict = {
    'mode': 1,
    'x': [state_data[0, 0]],
    'u': input_list  # Use numpy array
}

# 4. Run simulation
sys.reset(init_state_dict, dt=0.001)
results = []
for i in range(100):
    state, mode, switched = sys.next(0.001)
    results.append(state)
```

## Implementation Details

### Modified Files

1. **`src/ODE_System.py`**
   - Modified `analyticalInput()` to detect and handle numpy arrays
   - Added `dt` parameter to `reset()` method
   - Added `self.dt` attribute to store time step

2. **`src/HybridAutomata.py`**
   - Added `dt` parameter to `reset()` method to pass through to ODE systems

3. **`HA_simulation.py`**
   - Updated to use `input_list` directly from npz file
   - Passes `dt` parameter when calling `sys.reset()`

### Indexing Logic

The array indexing function maps continuous time `t` to array index:

```python
idx = int(round(t / dt - 1))
idx = max(0, min(idx, len(arr) - 1))  # Clamp to bounds
return arr[idx]
```

This correctly handles:
- Time offset (the `- 1` accounts for array values being post-integration)
- Intermediate times during RK4 integration (e.g., `t + 0.5*dt`)
- Boundary conditions (clamping to array bounds)

## Testing

Run the comprehensive test suite:

```bash
python test_array_input.py
```

This verifies:
- ✓ String expression inputs work (backward compatibility)
- ✓ 2D numpy array inputs work
- ✓ 1D numpy array inputs work
- ✓ All methods produce identical results

## Backward Compatibility

This feature is **fully backward compatible**. All existing code using string expressions will continue to work without modification. The array input feature is purely additive.

