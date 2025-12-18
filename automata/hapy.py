import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional
from scipy.integrate import solve_ivp


class Mode:
    """Represents a mode in the hybrid automaton"""
    
    def __init__(self, mode_id: int, dynamics: Callable):
        """
        Args:
            mode_id: Unique identifier for this mode
            dynamics: Function that computes x_dot = f(x, u, t)
                     For second-order system: returns [x[1], x[2]]
        """
        self.id = mode_id
        self.dynamics = dynamics
    
    def __repr__(self):
        return f"Mode({self.id})"


class Guard:
    """Represents a guard condition for mode transitions"""
    
    def __init__(self, condition: Callable):
        """
        Args:
            condition: Function that checks if guard is satisfied: condition(x, t) -> bool
        """
        self.condition = condition
    
    def is_satisfied(self, x: np.ndarray, t: float) -> bool:
        return self.condition(x, t)


class Reset:
    """Represents a reset map applied during transitions"""
    
    def __init__(self, reset_map: Callable):
        """
        Args:
            reset_map: Function that computes new state: x_new = reset_map(x)
        """
        self.reset_map = reset_map
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.reset_map(x)


class Edge:
    """Represents a transition edge between modes"""
    
    def __init__(self, source: Mode, target: Mode, guard: Guard, reset: Optional[Reset] = None):
        """
        Args:
            source: Source mode
            target: Target mode
            guard: Guard condition
            reset: Reset map (optional, identity if None)
        """
        self.source = source
        self.target = target
        self.guard = guard
        self.reset = reset if reset is not None else Reset(lambda x: x)
    
    def __repr__(self):
        return f"Edge({self.source.id} -> {self.target.id})"


class InitialState:
    """Represents an initial condition for simulation"""
    
    def __init__(self, mode: Mode, x0: np.ndarray, input_func: Callable):
        """
        Args:
            mode: Initial mode
            x0: Initial state vector
            input_func: Input function u(t)
        """
        self.mode = mode
        self.x0 = x0
        self.input_func = input_func


class HybridAutomaton:
    """Main hybrid automaton class"""
    
    def __init__(self, modes: List[Mode], edges: List[Edge]):
        """
        Args:
            modes: List of modes
            edges: List of transition edges
        """
        self.modes = {mode.id: mode for mode in modes}
        self.edges = edges
    
    def get_mode(self, mode_id: int) -> Mode:
        return self.modes[mode_id]
    
    def check_transitions(self, current_mode: Mode, x: np.ndarray, t: float) -> Optional[Tuple[Mode, Reset]]:
        """
        Check if any transition is enabled from current mode
        Returns: (target_mode, reset) if transition exists, None otherwise
        """
        for edge in self.edges:
            if edge.source.id == current_mode.id:
                if edge.guard.is_satisfied(x, t):
                    return edge.target, edge.reset
        return None


class HybridSimulator:
    """Simulator for hybrid automaton"""
    
    def __init__(self, automaton: HybridAutomaton, dt: float = 0.001, total_time: float = 10.0):
        """
        Args:
            automaton: HybridAutomaton to simulate
            dt: Time step
            total_time: Total simulation time
        """
        self.automaton = automaton
        self.dt = dt
        self.total_time = total_time
        self.t = np.arange(0, total_time, dt)
    
    def simulate(self, init_state: InitialState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate a single trajectory
        
        Returns:
            trajectory: Array of states over time
            modes: Array of mode IDs over time
            times: Time array
        """
        # Initialize
        current_mode = init_state.mode
        x = init_state.x0.copy()
        
        # Storage
        trajectory = []
        modes = []
        times = []
        
        for t_val in self.t:
            # Record current state
            trajectory.append(x.copy())
            modes.append(current_mode.id)
            times.append(t_val)
            
            # Get current input
            u = init_state.input_func(t_val)
            
            # Check for transitions
            transition = self.automaton.check_transitions(current_mode, x, t_val)
            if transition is not None:
                target_mode, reset = transition
                current_mode = target_mode
                x = reset.apply(x)
            
            # Integrate dynamics using RK45
            sol = solve_ivp(
                fun=lambda t, y: current_mode.dynamics(y, u, t),
                t_span=[t_val, t_val + self.dt],
                y0=x,
                method='RK45'
            )
            x = sol.y[:, -1]
        
        return np.array(trajectory), np.array(modes), np.array(times)
    
    def simulate_multiple(self, init_states: List[InitialState]) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """Simulate multiple initial conditions"""
        all_trajectories = []
        all_modes = []
        
        for init_state in init_states:
            traj, modes, times = self.simulate(init_state)
            all_trajectories.append(traj)
            all_modes.append(modes)
        
        return all_trajectories, all_modes, times
    
    def plot_results(self, all_trajectories: List[np.ndarray], all_modes: List[np.ndarray], times: np.ndarray):
        """Plot simulation results"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot position trajectories
        ax1 = axes[0]
        for i, traj in enumerate(all_trajectories):
            ax1.plot(times, traj[:, 0], alpha=0.7, label=f'Traj {i+1}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position x[0]')
        ax1.set_title('Position Trajectories')
        ax1.grid(True)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        # Plot velocity trajectories
        ax2 = axes[1]
        for i, traj in enumerate(all_trajectories):
            ax2.plot(times, traj[:, 1], alpha=0.7, label=f'Traj {i+1}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity x[1]')
        ax2.set_title('Velocity Trajectories')
        ax2.grid(True)
        
        # Plot mode switching
        ax3 = axes[2]
        for i, modes in enumerate(all_modes[:5]):
            ax3.plot(times, modes, alpha=0.7, label=f'Traj {i+1}', marker='.')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Mode')
        ax3.set_title('Mode Switching (First 5 Trajectories)')
        ax3.set_yticks([1, 2])
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig('result/class_based_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Phase portrait
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_trajectories)))
        for i, (traj, modes) in enumerate(zip(all_trajectories, all_modes)):
            mode1_mask = modes == 1
            mode2_mask = modes == 2
            
            if np.any(mode1_mask):
                ax.plot(traj[mode1_mask, 0], traj[mode1_mask, 1], 
                       'o', color=colors[i], alpha=0.3, markersize=1)
            if np.any(mode2_mask):
                ax.plot(traj[mode2_mask, 0], traj[mode2_mask, 1], 
                       's', color=colors[i], alpha=0.3, markersize=1)
        
        ax.set_xlabel('Position x[0]')
        ax.set_ylabel('Velocity x[1]')
        ax.set_title('Phase Portrait (Mode 1: circles, Mode 2: squares)')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('result/class_based_phase.png', dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# Define the specific hybrid automaton from your example
# =============================================================================

def create_duffing_automaton() -> HybridAutomaton:
    """
    Create the Duffing-like hybrid automaton with two modes
    
    Mode 1: ẍ = u - 0.5ẋ + x - 1.5x³
    Mode 2: ẍ = u - 0.2ẋ + x - 0.5x³
    
    Transitions:
    - 1 -> 2 when |x| <= 0.8
    - 2 -> 1 when |x| >= 1.2
    
    Reset: velocity *= 0.95
    """
    
    # Define Mode 1 dynamics
    def mode1_dynamics(x, u, t):
        """Mode 1: ẍ = u - 0.5ẋ + x - 1.5x³"""
        x_ddot = u - 0.5 * x[1] + x[0] - 1.5 * x[0]**3
        return np.array([x[1], x_ddot])
    
    # Define Mode 2 dynamics
    def mode2_dynamics(x, u, t):
        """Mode 2: ẍ = u - 0.2ẋ + x - 0.5x³"""
        x_ddot = u - 0.2 * x[1] + x[0] - 0.5 * x[0]**3
        return np.array([x[1], x_ddot])
    
    # Create modes
    mode1 = Mode(1, mode1_dynamics)
    mode2 = Mode(2, mode2_dynamics)
    
    # Define guards
    guard_1_to_2 = Guard(lambda x, t: abs(x[0]) <= 0.8)
    guard_2_to_1 = Guard(lambda x, t: abs(x[0]) >= 1.2)
    
    # Define reset map (velocity *= 0.95)
    def velocity_reset(x):
        x_new = x.copy()
        x_new[1] = x[1] * 0.95
        return x_new
    
    reset = Reset(velocity_reset)
    
    # Create edges
    edge1 = Edge(mode1, mode2, guard_1_to_2, reset)
    edge2 = Edge(mode2, mode1, guard_2_to_1, reset)
    
    # Create automaton
    automaton = HybridAutomaton([mode1, mode2], [edge1, edge2])
    
    return automaton


def create_initial_states(automaton: HybridAutomaton) -> List[InitialState]:
    """Create initial states for simulation"""
    mode1 = automaton.get_mode(1)
    
    initial_conditions = [
        (4.0, lambda t: 0.5 * np.cos(1.2 * t)),
        (7.8, lambda t: 0.65 * np.cos(1.0 * t)),
        (7.4, lambda t: 0.7 * np.cos(1.3 * t)),
        (5.8, lambda t: 0.85 * np.cos(1.6 * t)),
        (6.4, lambda t: 0.9 * np.cos(1.0 * t)),
        (6.9, lambda t: 0.55 * np.cos(1.2 * t)),
        (4.8, lambda t: 0.75 * np.cos(1.4 * t)),
        (5.2, lambda t: 0.95 * np.cos(1.7 * t)),
        (6.5, lambda t: 0.8 * np.cos(1.1 * t)),
        (6.2, lambda t: 0.6 * np.cos(1.3 * t)),
        (5.3, lambda t: 0.7 * np.cos(1.2 * t)),
        (5.0, lambda t: 0.5 * np.cos(1.5 * t)),
        (6.0, lambda t: 0.8 * np.cos(1.5 * t)),
        (7.0, lambda t: 0.6 * np.cos(1.1 * t)),
        (8.0, lambda t: 0.75 * np.cos(0.9 * t)),
    ]
    
    init_states = []
    for x0_val, u_func in initial_conditions:
        x0 = np.array([x0_val, 0.0])  # [position, velocity=0]
        init_states.append(InitialState(mode1, x0, u_func))
    
    return init_states


def main():
    print("Creating Hybrid Automaton using Python classes...")
    
    # Create the automaton
    automaton = create_duffing_automaton()
    print(f"Automaton created with {len(automaton.modes)} modes and {len(automaton.edges)} edges")
    
    # Create initial states
    init_states = create_initial_states(automaton)
    print(f"Created {len(init_states)} initial states")
    
    # Create simulator
    simulator = HybridSimulator(automaton, dt=0.001, total_time=10.0)
    print(f"\nSimulating {len(init_states)} trajectories...")
    print(f"Time span: {simulator.total_time}s with dt={simulator.dt}s")
    
    # Run simulation
    all_trajectories, all_modes, times = simulator.simulate_multiple(init_states)
    
    print(f"\nSimulation complete!")
    print(f"Total time steps: {len(times)}")
    
    # Analyze mode switches
    print("\nMode switching analysis:")
    for i, modes in enumerate(all_modes):
        switches = np.sum(np.diff(modes) != 0)
        print(f"  Trajectory {i+1}: {switches} mode switches")
    
    # Plot results
    print("\nGenerating plots...")
    simulator.plot_results(all_trajectories, all_modes, times)
    print("Plots saved!")
    
    # Save data
    np.savez('result/class_based_data.npz',
             trajectories=all_trajectories,
             modes=all_modes,
             times=times)
    print("\nData saved to result/class_based_data.npz")
    
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Automaton Structure:")
    print(f"  - Mode 1: ẍ = u - 0.5ẋ + x - 1.5x³")
    print(f"  - Mode 2: ẍ = u - 0.2ẋ + x - 0.5x³")
    print(f"\nTransition Guards:")
    print(f"  - 1 → 2: |x| ≤ 0.8")
    print(f"  - 2 → 1: |x| ≥ 1.2")
    print(f"\nReset Map:")
    print(f"  - velocity ← 0.95 × velocity")
    print(f"\nSimulation Parameters:")
    print(f"  - Number of trajectories: {len(init_states)}")
    print(f"  - Time range: 0 to {simulator.total_time}s")
    print(f"  - Time step: {simulator.dt}s")
    print("="*60)


if __name__ == "__main__":
    main()