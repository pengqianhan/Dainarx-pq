import z3
import numpy as np
import matplotlib.pyplot as plt

def verify_with_z3(steps=20, dt=0.1, unsafe_bound=9.0):
    """
    Perform Bounded Model Checking (BMC) using Z3.
    We discretize the system using Euler method and check if an unsafe state is reachable within 'steps'.
    
    Args:
        steps: Number of discrete steps to check
        dt: Time step size
        unsafe_bound: The safety threshold for |x|
    """
    print(f"Setting up Z3 solver for {steps} steps with dt={dt}...")
    
    solver = z3.Solver()
    
    # Variables for each step
    # x[i]: position, y[i]: velocity, m[i]: mode (1 or 2)
    # u[i]: input (bounded non-determinism)
    x = [z3.Real(f'x_{i}') for i in range(steps + 1)]
    y = [z3.Real(f'y_{i}') for i in range(steps + 1)]
    m = [z3.Int(f'm_{i}') for i in range(steps + 1)]
    u = [z3.Real(f'u_{i}') for i in range(steps)] # Input at step i
    
    # 1. Initial Conditions
    # Mode 1, x in [4, 8], y = 0
    solver.add(m[0] == 1)
    solver.add(x[0] >= 4.0)
    solver.add(x[0] <= 8.0)
    solver.add(y[0] == 0.0)
    
    # 2. Transition Relation (Dynamics + Switching)
    for i in range(steps):
        # Input bounds: u in [-1, 1] (Conservative approximation of cosine input)
        solver.add(u[i] >= -1.0)
        solver.add(u[i] <= 1.0)
        
        # Current state values
        xi = x[i]
        yi = y[i]
        mi = m[i]
        ui = u[i]
        
        # Next state values
        xi_next = x[i+1]
        yi_next = y[i+1]
        mi_next = m[i+1]
        
        # --- Mode 1 Dynamics ---
        # dx = y
        # dy = u - 0.5*y + x - 1.5*x^3
        # Euler: x_next = x + dt*y
        #        y_next = y + dt*(...)
        dyn_m1_x = (xi_next == xi + dt * yi)
        dyn_m1_y = (yi_next == yi + dt * (ui - 0.5*yi + xi - 1.5*xi*xi*xi))
        
        # --- Mode 2 Dynamics ---
        # dy = u - 0.2*y + x - 0.5*x^3
        dyn_m2_x = (xi_next == xi + dt * yi)
        dyn_m2_y = (yi_next == yi + dt * (ui - 0.2*yi + xi - 0.5*xi*xi*xi))
        
        # --- Guards & Switching ---
        # Mode 1 -> 2 if |x| <= 0.8
        # Mode 2 -> 1 if |x| >= 1.2
        # Reset: y -> y * 0.95
        
        # Logic for Mode 1
        # If |x| <= 0.8, switch to 2, apply reset
        # Else, stay in 1, apply dynamics
        # Note: In discrete time, we check condition at step i
        
        abs_x = z3.If(xi >= 0, xi, -xi)
        
        # Transition from Mode 1
        # If guard (abs_x <= 0.8) is true:
        #   Next mode is 2
        #   x doesn't change (continuous) -> handled by dynamics or identity? 
        #   Usually hybrid automata resets happen instantly. 
        #   Here we model: Evolve continuous OR Jump. 
        #   For simplicity in BMC fixed-step: We check guard. If guard true, we jump AND evolve? 
        #   Let's model: If guard true, we take a discrete transition (time doesn't advance, or we combine).
        #   Simple Euler approach: Check guard at start of step.
        
        # Refined Transition Logic:
        # Case 1: m[i] == 1
        #   If (abs_x <= 0.8): Jump to 2. 
        #      m[i+1] = 2
        #      x[i+1] = x[i]
        #      y[i+1] = y[i] * 0.95
        #   Else: Flow in 1.
        #      m[i+1] = 1
        #      Apply Mode 1 Dynamics
        
        trans_m1_jump = z3.And(abs_x <= 0.8, 
                               mi_next == 2,
                               xi_next == xi,
                               yi_next == yi * 0.95)
                               
        trans_m1_flow = z3.And(abs_x > 0.8,
                               mi_next == 1,
                               dyn_m1_x,
                               dyn_m1_y)
                               
        logic_m1 = z3.Implies(mi == 1, z3.Or(trans_m1_jump, trans_m1_flow))
        
        # Case 2: m[i] == 2
        #   If (abs_x >= 1.2): Jump to 1.
        #      m[i+1] = 1
        #      x[i+1] = x[i]
        #      y[i+1] = y[i] * 0.95
        #   Else: Flow in 2.
        #      m[i+1] = 2
        #      Apply Mode 2 Dynamics
        
        trans_m2_jump = z3.And(abs_x >= 1.2,
                               mi_next == 1,
                               xi_next == xi,
                               yi_next == yi * 0.95)
                               
        trans_m2_flow = z3.And(abs_x < 1.2,
                               mi_next == 2,
                               dyn_m2_x,
                               dyn_m2_y)
                               
        logic_m2 = z3.Implies(mi == 2, z3.Or(trans_m2_jump, trans_m2_flow))
        
        solver.add(logic_m1)
        solver.add(logic_m2)
        
        # Constraint: Mode must be 1 or 2
        solver.add(z3.Or(mi == 1, mi == 2))

    # 3. Safety Property (Negated)
    # We want to find IF there exists a state where |x| > unsafe_bound
    # If SAT, then Unsafe. If UNSAT, then Safe (within k steps).
    unsafe_constraints = []
    for i in range(steps + 1):
        unsafe_constraints.append(x[i] > unsafe_bound)
        unsafe_constraints.append(x[i] < -unsafe_bound)
    
    solver.add(z3.Or(unsafe_constraints))
    
    # 4. Check
    print("Checking satisfiability...")
    result = solver.check()
    
    if result == z3.sat:
        print(f"VIOLATION FOUND! System is UNSAFE within {steps} steps.")
        model = solver.model()
        
        # Extract trajectory
        traj_x = [float(model[x[i]].as_decimal(5).replace('?','')) for i in range(steps+1)]
        traj_y = [float(model[y[i]].as_decimal(5).replace('?','')) for i in range(steps+1)]
        traj_m = [model[m[i]].as_long() for i in range(steps+1)]
        
        print("Counter-example trajectory:")
        for i in range(steps+1):
            print(f"Step {i}: Mode={traj_m[i]}, x={traj_x[i]:.4f}, y={traj_y[i]:.4f}")
            
        # Plot
        plt.figure()
        plt.plot(range(steps+1), traj_x, 'r-o', label='x (Counter-example)')
        plt.axhline(y=unsafe_bound, color='k', linestyle='--')
        plt.axhline(y=-unsafe_bound, color='k', linestyle='--')
        plt.title("Z3 Counter-Example")
        plt.xlabel("Step")
        plt.ylabel("x")
        plt.legend()
        plt.savefig('result/z3_violation.png')
        print("Violation plot saved to result/z3_violation.png")
        
    elif result == z3.unsat:
        print(f"UNSAT: No violation found within {steps} steps (dt={dt}).")
        print("System is SAFE for this bounded horizon and discretization.")
    else:
        print("UNKNOWN: Z3 could not solve the constraints.")

if __name__ == "__main__":
    # Run a small check
    # Note: Large steps with non-linear arithmetic can be slow
    verify_with_z3(steps=20, dt=0.05, unsafe_bound=9.0)
