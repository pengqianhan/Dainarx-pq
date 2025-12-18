import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import json

class HybridAutomaton:
    def __init__(self, json_data):
        """Initialize the hybrid automaton from JSON data"""
        self.automaton = json_data['automaton']
        self.init_states = json_data['init_state']
        self.config = json_data['config']
        
        # Parse configuration
        self.dt = self.config['dt']
        self.total_time = self.config['total_time']
        self.order = self.config['order']
        self.need_reset = self.config['need_reset']
        
        # Create time array
        self.t = np.arange(0, self.total_time, self.dt)
        
    def get_input(self, u_expr, t):
        """Evaluate the input function at time t"""
        return eval(u_expr, {'t': t, 'cos': np.cos, 'sin': np.sin, 'abs': abs})
    
    def mode_dynamics(self, mode_id, x, u):
        """
        Compute dynamics for a given mode
        x is a state vector: [x[0], x[1]] for second-order system
        Returns dx/dt = [x[1], x[2]] where x[2] is computed from mode equation
        """
        # Find the mode equation
        mode_eq = None
        for mode in self.automaton['mode']:
            if mode['id'] == mode_id:
                mode_eq = mode['eq']
                break
        
        if mode_eq is None:
            raise ValueError(f"Mode {mode_id} not found")
        
        # Parse the equation: "x[2] = ..."
        # Extract the right-hand side
        rhs = mode_eq.split('=')[1].strip()
        
        # Replace x[0], x[1] with actual values
        x_2 = eval(rhs, {'x': x, 'u': u, 'abs': abs, 'cos': np.cos, 'sin': np.sin})
        
        # Return [dx[0]/dt, dx[1]/dt] = [x[1], x[2]]
        return np.array([x[1], x_2])
    
    def check_guard(self, mode_id, x):
        """Check if any guard condition is satisfied for transitioning"""
        for edge in self.automaton['edge']:
            direction = edge['direction']
            source, target = map(int, direction.replace(' ', '').split('->'))
            
            if source == mode_id:
                condition = edge['condition']
                # Evaluate condition
                if eval(condition, {'x': x, 'abs': abs}):
                    return target, edge['reset']
        
        return None, None
    
    def apply_reset(self, x, reset_map):
        """Apply reset map to state"""
        if reset_map is None:
            return x
        
        x_new = x.copy()
        x_reset = reset_map['x']
        
        # Apply reset to each component
        for i, reset_expr in enumerate(x_reset):
            if reset_expr and reset_expr.strip():
                x_new[i] = eval(reset_expr, {'x': x})
        
        return x_new
    
    def simulate_trajectory(self, init_mode, init_x, u_expr):
        """Simulate a single trajectory"""
        # Initialize
        current_mode = init_mode
        x = np.array([init_x[0], 0.0])  # [position, velocity]
        
        # Storage
        trajectory = []
        modes = []
        times = []
        
        for i, t_val in enumerate(self.t):
            # Record current state
            trajectory.append(x.copy())
            modes.append(current_mode)
            times.append(t_val)
            
            # Get current input
            t = t_val  # for eval
            u = self.get_input(u_expr, t_val)
            
            # Check guard conditions
            new_mode, reset = self.check_guard(current_mode, x)
            
            if new_mode is not None and self.need_reset:
                # Mode transition occurred
                current_mode = new_mode
                x = self.apply_reset(x, reset)
            
            # Integrate dynamics for one time step
            dx = self.mode_dynamics(current_mode, x, u)
            x = x + dx * self.dt
        
        return np.array(trajectory), np.array(modes), np.array(times)
    
    def simulate_all(self):
        """Simulate all initial conditions"""
        all_trajectories = []
        all_modes = []
        
        for init in self.init_states:
            traj, modes, times = self.simulate_trajectory(
                init['mode'], 
                init['x'], 
                init['u']
            )
            all_trajectories.append(traj)
            all_modes.append(modes)
        
        return all_trajectories, all_modes, times
    
    def plot_results(self, all_trajectories, all_modes, times):
        """Plot simulation results"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot all trajectories - position
        ax1 = axes[0]
        for i, (traj, modes) in enumerate(zip(all_trajectories, all_modes)):
            ax1.plot(times, traj[:, 0], alpha=0.7, label=f'Init {i+1}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position x[0]')
        ax1.set_title('Position Trajectories')
        ax1.grid(True)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        # Plot all trajectories - velocity
        ax2 = axes[1]
        for i, (traj, modes) in enumerate(zip(all_trajectories, all_modes)):
            ax2.plot(times, traj[:, 1], alpha=0.7, label=f'Init {i+1}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity x[1]')
        ax2.set_title('Velocity Trajectories')
        ax2.grid(True)
        
        # Plot mode switching for first few trajectories
        ax3 = axes[2]
        for i, modes in enumerate(all_modes[:5]):  # Plot first 5
            ax3.plot(times, modes, alpha=0.7, label=f'Init {i+1}', marker='.')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Mode')
        ax3.set_title('Mode Switching (First 5 Trajectories)')
        ax3.set_yticks([1, 2])
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig('result/hybrid_automaton_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Phase portrait
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_trajectories)))
        for i, (traj, modes) in enumerate(zip(all_trajectories, all_modes)):
            # Color by mode
            mode1_mask = modes == 1
            mode2_mask = modes == 2
            
            if np.any(mode1_mask):
                ax.plot(traj[mode1_mask, 0], traj[mode1_mask, 1], 
                       'o', color=colors[i], alpha=0.3, markersize=1, label=f'Init {i+1} (Mode 1)' if i < 3 else '')
            if np.any(mode2_mask):
                ax.plot(traj[mode2_mask, 0], traj[mode2_mask, 1], 
                       's', color=colors[i], alpha=0.3, markersize=1, label=f'Init {i+1} (Mode 2)' if i < 3 else '')
        
        ax.set_xlabel('Position x[0]')
        ax.set_ylabel('Velocity x[1]')
        ax.set_title('Phase Portrait (Mode 1: circles, Mode 2: squares)')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('result/phase_portrait.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Plots saved successfully!")

def main():
    # Load JSON data
    json_data = {
        "automaton": {
            "var": "x",
            "input": "u",
            "mode": [
                {
                    "id": 1,
                    "eq": "x[2] = u - 0.5 * x[1] + x[0] - 1.5 * x[0] ** 3"
                },
                {
                    "id": 2,
                    "eq": "x[2] = u - 0.2 * x[1] + x[0] - 0.5 * x[0] ** 3"
                }
            ],
            "edge": [
                {
                    "direction": "1 -> 2",
                    "condition": "abs(x[0]) <= 0.8",
                    "reset": {
                        "x": ["", "x[1] * 0.95"]
                    }
                },
                {
                    "direction": "2 -> 1",
                    "condition": "abs(x[0]) >= 1.2",
                    "reset": {
                        "x": ["", "x[1] * 0.95"]
                    }
                }
            ]
        },
        "init_state": [
            {"mode": 1, "x": [4], "u": "0.5 * cos(1.2 * t)"},
            {"mode": 1, "x": [7.8], "u": "0.65 * cos(1.0 * t)"},
            {"mode": 1, "x": [7.4], "u": "0.7 * cos(1.3 * t)"},
            {"mode": 1, "x": [5.8], "u": "0.85 * cos(1.6 * t)"},
            {"mode": 1, "x": [6.4], "u": "0.9 * cos(1.0 * t)"},
            {"mode": 1, "x": [6.9], "u": "0.55 * cos(1.2 * t)"},
            {"mode": 1, "x": [4.8], "u": "0.75 * cos(1.4 * t)"},
            {"mode": 1, "x": [5.2], "u": "0.95 * cos(1.7 * t)"},
            {"mode": 1, "x": [6.5], "u": "0.8 * cos(1.1 * t)"},
            {"mode": 1, "x": [6.2], "u": "0.6 * cos(1.3 * t)"},
            {"mode": 1, "x": [5.3], "u": "0.7 * cos(1.2 * t)"},
            {"mode": 1, "x": [5], "u": "0.5 * cos(1.5 * t)"},
            {"mode": 1, "x": [6], "u": "0.8 * cos(1.5 * t)"},
            {"mode": 1, "x": [7], "u": "0.6 * cos(1.1 * t)"},
            {"mode": 1, "x": [8], "u": "0.75 * cos(0.9 * t)"}
        ],
        "config": {
            "dt": 0.001,
            "total_time": 10.0,
            "order": 2,
            "need_reset": True,
            "kernel": "rbf",
            "other_items": "x[?] ** 3"
        }
    }
    
    # Create and run simulator
    print("Initializing Hybrid Automaton Simulator...")
    ha = HybridAutomaton(json_data)
    
    print(f"Simulating {len(json_data['init_state'])} trajectories...")
    print(f"Time span: {ha.total_time}s with dt={ha.dt}s")
    
    all_trajectories, all_modes, times = ha.simulate_all()
    
    print(f"\nSimulation complete!")
    print(f"Total time steps: {len(times)}")
    
    # Analyze mode switches
    for i, modes in enumerate(all_modes):
        switches = np.sum(np.diff(modes) != 0)
        print(f"Trajectory {i+1}: {switches} mode switches")
    
    print("\nGenerating plots...")
    ha.plot_results(all_trajectories, all_modes, times)
    
    # Save trajectories to file
    np.savez('result/simulation_data.npz',
             trajectories=all_trajectories,
             modes=all_modes,
             times=times)
    
    print("\nData saved to result/simulation_data.npz")
    print("\nSimulation Summary:")
    print(f"  - Number of trajectories: {len(all_trajectories)}")
    print(f"  - Time range: 0 to {ha.total_time}s")
    print(f"  - Time step: {ha.dt}s")
    print(f"  - Number of modes: 2")
    print(f"  - Guard conditions: abs(x[0]) <= 0.8 (1->2), abs(x[0]) >= 1.2 (2->1)")

if __name__ == "__main__":
    main()
