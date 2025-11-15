"""
Duffing System Auto-Configuration Demo

This demo shows how to use LLM + Genetic Algorithm to automatically
configure Dainarx parameters for the Duffing oscillator system.

Usage:
    python demos/duffing_auto_config.py [--test] [--full-run]

Options:
    --test: Use gemini-flash-lite-latest for quick testing
    --full-run: Use gemini-flash-latest for production run
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from CreatData import creat_data
from src.utils import get_hash_code, check_data_update, save_hash_code
from src.Evaluation import Evaluation

# Import our LLM optimizer modules
from llm_config_optimizer.data_analyzer import DataAnalyzer
from llm_config_optimizer.llm_analyzer import LLMConfigAnalyzer
from llm_config_optimizer.genetic_optimizer import GeneticOptimizer
from llm_config_optimizer.fitness_evaluator import FitnessEvaluator


def load_duffing_data(json_path, data_path='data', force_regenerate=False):
    """Load or generate Duffing system data"""

    print("\n=== Loading Duffing Data ===")

    # Load JSON config
    with open(json_path, 'r') as f:
        json_file = json.load(f)

    # Get dt and total_time from config
    config = json_file.get('config', {})
    dt = config.get('dt', 0.001)
    total_time = config.get('total_time', 10.0)

    # Check if data needs regeneration
    hash_code = get_hash_code(json_file, config)
    need_create = force_regenerate or check_data_update(hash_code, data_path)

    if need_create:
        print("Generating data...")
        creat_data(json_path, data_path, dt, total_time)
        save_hash_code(hash_code, data_path)
        print("Data generated successfully")
    else:
        print("Using cached data")

    # Load data
    npz_path = os.path.join(data_path, f"{hash_code}.npz")
    data_npz = np.load(npz_path, allow_pickle=True)

    mode_list = data_npz['mode'].tolist()
    data_list = data_npz['data'].tolist()
    input_list = data_npz['input'].tolist()
    gt_list = data_npz.get('gt', [])

    print(f"Loaded {len(data_list)} trajectories")
    print(f"Data shape: {data_list[0].shape}")
    print(f"Input shape: {input_list[0].shape}")

    return {
        'json_file': json_file,
        'mode_list': mode_list,
        'data_list': data_list,
        'input_list': input_list,
        'gt_list': gt_list,
        'dt': dt,
        'total_time': total_time
    }


def run_dainarx_with_config(data_dict, config):
    """
    Run Dainarx with given configuration

    Returns:
        (sys, slice_data, evaluation)
    """
    from main import run

    # Create evaluation object
    evaluation = Evaluation("duffing_auto")

    # Prepare data (use last 6 for training as in main.py)
    test_num = 6
    data_train = data_dict['data_list'][test_num:]
    input_train = data_dict['input_list'][test_num:]

    # Submit ground truth if available
    if len(data_dict['gt_list']) > test_num:
        evaluation.submit(gt_chp=data_dict['gt_list'][test_num:])

    # Run Dainarx
    sys, slice_data = run(data_train, input_train, config, evaluation)

    return sys, slice_data, evaluation


def create_fitness_function(data_dict, has_ground_truth=True):
    """Create fitness function for genetic algorithm"""

    evaluator = FitnessEvaluator()

    def fitness_fn(config):
        """Evaluate configuration fitness"""
        try:
            # Run Dainarx with this config
            sys, slice_data, evaluation = run_dainarx_with_config(data_dict, config)

            # Evaluate fitness
            fitness, metrics = evaluator.evaluate(
                slice_data,
                evaluation,
                has_ground_truth=has_ground_truth
            )

            return fitness

        except Exception as e:
            print(f"Config evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return -1e6

    return fitness_fn


def main():
    """Main demo function"""

    # Parse arguments
    parser = argparse.ArgumentParser(description='Duffing Auto-Configuration Demo')
    parser.add_argument('--test', action='store_true',
                       help='Use gemini-flash-lite-latest for testing')
    parser.add_argument('--full-run', action='store_true',
                       help='Use gemini-flash-latest for production')
    parser.add_argument('--no-llm', action='store_true',
                       help='Skip LLM analysis, use only GA with random init')
    parser.add_argument('--pop-size', type=int, default=10,
                       help='GA population size (default: 10)')
    parser.add_argument('--generations', type=int, default=5,
                       help='GA generations (default: 5)')

    args = parser.parse_args()

    # Determine model
    if args.full_run:
        model_name = "models/gemini-flash-latest"
        print("Using production model: gemini-flash-latest")
    else:
        model_name = "models/gemini-flash-lite-latest"
        print("Using test model: gemini-flash-lite-latest")

    # Paths
    json_path = "./automata/non_linear/duffing.json"
    data_path = "data"

    # Load data
    data_dict = load_duffing_data(json_path, data_path)

    print("\n" + "="*60)
    print("STEP 1: Data Feature Extraction")
    print("="*60)

    # Extract features
    analyzer = DataAnalyzer()
    features = analyzer.extract_features(
        data_dict['data_list'],
        data_dict['input_list'],
        data_dict['dt']
    )

    features_text = analyzer.format_for_llm(features)
    print(features_text)

    # LLM Analysis
    llm_suggestions = None

    if not args.no_llm:
        print("\n" + "="*60)
        print("STEP 2: LLM Configuration Analysis")
        print("="*60)

        try:
            llm_analyzer = LLMConfigAnalyzer(model_name=model_name)
            llm_response = llm_analyzer.analyze_system(features_text)

            print(llm_analyzer.format_recommendations(llm_response))

            # Extract config ranges
            config_ranges = llm_analyzer.extract_config_ranges(llm_response)
            llm_suggestions = llm_response

        except Exception as e:
            print(f"\nWarning: LLM analysis failed: {e}")
            print("Continuing with fallback configuration ranges...\n")
            config_ranges = None

    else:
        print("\n" + "="*60)
        print("STEP 2: Skipping LLM Analysis (--no-llm flag)")
        print("="*60)
        config_ranges = None

    # Define configuration search space
    if config_ranges is None:
        print("\nUsing default configuration ranges...")
        config_ranges = {
            "order": [2, 3, 4],
            "other_items": ["", "x[?]**2", "x[?]**3", "x[?]**2; x[?]**3"],
            "window_size": [8, 10, 12, 15],
            "kernel": ["linear", "rbf"],
            "svm_c": [1e4, 1e5, 1e6],
            "class_weight": [1.0, 10.0, 30.0, 50.0],
            "self_loop": [False, True],
            "need_reset": [False, True]
        }

    # Add fixed parameters (from default config)
    fixed_config = {
        "dt": data_dict['dt'],
        "total_time": data_dict['total_time'],
        "minus": False,
        "need_bias": True,
        "clustering_method": "fit"
    }

    print("\n" + "="*60)
    print("STEP 3: Genetic Algorithm Optimization")
    print("="*60)

    print("\nConfiguration search space:")
    for param, options in config_ranges.items():
        print(f"  {param}: {options}")

    # Create fitness function
    fitness_fn = create_fitness_function(data_dict, has_ground_truth=True)

    # Wrap fitness function to merge with fixed config
    def wrapped_fitness_fn(config):
        full_config = {**fixed_config, **config}
        return fitness_fn(full_config)

    # Run genetic algorithm
    ga_optimizer = GeneticOptimizer(
        population_size=args.pop_size,
        generations=args.generations,
        mutation_rate=0.2,
        crossover_rate=0.7,
        elitism=2,
        random_seed=42
    )

    start_time = time.time()

    best_config, best_fitness = ga_optimizer.optimize(
        config_ranges=config_ranges,
        fitness_function=wrapped_fitness_fn,
        llm_suggestions=llm_suggestions,
        verbose=True
    )

    optimization_time = time.time() - start_time

    print("\n" + "="*60)
    print("STEP 4: Results Summary")
    print("="*60)

    print(f"\nOptimization completed in {optimization_time:.1f} seconds")
    print(f"Best fitness: {best_fitness:.4f}")

    print("\n=== Best Configuration ===")
    full_best_config = {**fixed_config, **best_config}
    for key, val in full_best_config.items():
        print(f"  {key}: {val}")

    # Compare with manual configuration
    print("\n=== Comparison with Manual Configuration ===")
    manual_config = data_dict['json_file'].get('config', {})

    print("\nManual config from JSON:")
    for key in best_config.keys():
        manual_val = manual_config.get(key, "N/A")
        auto_val = best_config[key]
        match = "✓" if manual_val == auto_val else "✗"
        print(f"  {match} {key}: manual={manual_val}, auto={auto_val}")

    # Save results
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)

    result_file = result_dir / "duffing_auto_config_result.json"
    with open(result_file, 'w') as f:
        json.dump({
            "best_config": full_best_config,
            "best_fitness": best_fitness,
            "optimization_time": optimization_time,
            "manual_config": manual_config,
            "llm_suggestions": llm_suggestions,
            "ga_history": ga_optimizer.history
        }, f, indent=2)

    print(f"\nResults saved to {result_file}")

    # Plot optimization history
    try:
        plot_file = result_dir / "duffing_ga_optimization.png"
        ga_optimizer.plot_history(save_path=str(plot_file))
    except Exception as e:
        print(f"Could not save plot: {e}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
