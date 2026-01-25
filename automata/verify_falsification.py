import numpy as np
import matplotlib.pyplot as plt
from hapy import Mode, Guard, Reset, Edge, HybridAutomaton, HybridSimulator, InitialState, create_duffing_automaton

def check_safety_property(trajectory: np.ndarray, unsafe_threshold: float = 9.0) -> bool:
    """
    Check if the trajectory satisfies the safety property (|x| <= unsafe_threshold).
    Returns True if SAFE, False if UNSAFE.
    """
    # Check if any state in trajectory exceeds threshold
    max_displacement = np.max(np.abs(trajectory[:, 0]))
    if max_displacement > unsafe_threshold:
        return False
    return True

def run_falsification(num_samples: int = 100):
    print(f"Starting falsification with {num_samples} samples...")
    print("Safety Property: |x| <= 9.0")
    
    automaton = create_duffing_automaton()
    simulator = HybridSimulator(automaton, dt=0.01, total_time=20.0)
    
    unsafe_count = 0
    
    for i in range(num_samples):
        # Random initial condition
        # x0 in [4.0, 8.0], u_amp in [0.5, 1.0], freq in [0.8, 1.8]
        x0_val = np.random.uniform(4.0, 8.0)
        u_amp = np.random.uniform(0.5, 1.0)
        freq = np.random.uniform(0.8, 1.8)
        
        u_func = lambda t, a=u_amp, f=freq: a * np.cos(f * t)
        
        x0 = np.array([x0_val, 0.0])
        init_state = InitialState(automaton.get_mode(1), x0, u_func)
        
        traj, modes, times = simulator.simulate(init_state)
        
        is_safe = check_safety_property(traj)
        
        if not is_safe:
            print(f"VIOLATION FOUND at sample {i}!")
            print(f"  Initial: x0={x0_val:.2f}, u={u_amp:.2f}*cos({freq:.2f}t)")
            print(f"  Max x: {np.max(np.abs(traj[:, 0])):.2f}")
            unsafe_count += 1
            
            # Plot the violation
            plt.figure()
            plt.plot(times, traj[:, 0])
            plt.axhline(y=9.0, color='r', linestyle='--', label='Unsafe Upper')
            plt.axhline(y=-9.0, color='r', linestyle='--', label='Unsafe Lower')
            plt.title(f"Unsafe Trajectory (Max |x| = {np.max(np.abs(traj[:, 0])):.2f})")
            plt.xlabel("Time")
            plt.ylabel("x")
            plt.legend()
            plt.savefig(f"result/violation_{i}.png")
            plt.close()
            
    print("-" * 30)
    print(f"Falsification complete.")
    print(f"Total Samples: {num_samples}")
    print(f"Violations Found: {unsafe_count}")
    if unsafe_count == 0:
        print("Result: No counter-example found (System appears safe within sampled parameters)")
    else:
        print("Result: System is UNSAFE")

if __name__ == "__main__":
    # Ensure result directory exists
    import os
    os.makedirs("result", exist_ok=True)
    run_falsification()
