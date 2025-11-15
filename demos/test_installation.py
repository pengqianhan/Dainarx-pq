"""
Test installation and basic functionality

This script tests:
1. All required packages are installed
2. Data analyzer works
3. LLM analyzer can be initialized
4. Genetic optimizer works
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all required imports"""
    print("Testing imports...")

    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False

    try:
        import scipy
        print("  ✓ scipy")
    except ImportError as e:
        print(f"  ✗ scipy: {e}")
        return False

    try:
        from dotenv import load_dotenv
        print("  ✓ python-dotenv")
    except ImportError as e:
        print(f"  ✗ python-dotenv: {e}")
        return False

    try:
        import google.generativeai as genai
        print("  ✓ google-generativeai")
    except ImportError as e:
        print(f"  ✗ google-generativeai: {e}")
        print("\nInstall with: pip install google-generativeai")
        return False

    try:
        import matplotlib
        print("  ✓ matplotlib (optional)")
    except ImportError:
        print("  - matplotlib (optional, not installed)")

    return True


def test_data_analyzer():
    """Test data analyzer"""
    print("\nTesting DataAnalyzer...")

    try:
        from llm_config_optimizer.data_analyzer import DataAnalyzer
        import numpy as np

        analyzer = DataAnalyzer()

        # Create simple test data
        t = np.linspace(0, 10, 1000)
        data = np.sin(2 * np.pi * t).reshape(1, -1)
        data_list = [data]

        features = analyzer.extract_features(data_list, dt=0.01)

        assert 'dimension' in features
        assert 'statistics' in features
        assert 'spectral' in features

        print("  ✓ DataAnalyzer works")
        return True

    except Exception as e:
        print(f"  ✗ DataAnalyzer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_genetic_optimizer():
    """Test genetic optimizer"""
    print("\nTesting GeneticOptimizer...")

    try:
        from llm_config_optimizer.genetic_optimizer import GeneticOptimizer

        # Simple test fitness function
        def fitness_fn(config):
            # Maximize sum of values
            return sum([v for v in config.values() if isinstance(v, (int, float))])

        config_ranges = {
            "param1": [1, 2, 3],
            "param2": [10, 20, 30]
        }

        ga = GeneticOptimizer(
            population_size=5,
            generations=2,
            random_seed=42
        )

        best_config, best_fitness = ga.optimize(
            config_ranges,
            fitness_fn,
            verbose=False
        )

        assert best_config is not None
        assert best_fitness > 0

        print("  ✓ GeneticOptimizer works")
        return True

    except Exception as e:
        print(f"  ✗ GeneticOptimizer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_analyzer():
    """Test LLM analyzer initialization"""
    print("\nTesting LLMConfigAnalyzer...")

    try:
        from llm_config_optimizer.llm_analyzer import LLMConfigAnalyzer
        import os

        # Check if API key is set
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key or api_key == "your_gemini_api_key_here":
            print("  - LLMConfigAnalyzer: API key not set (expected)")
            print("    Set GEMINI_API_KEY in .env file to test LLM functionality")
            return True  # Not a failure, just not configured

        # Try to initialize
        try:
            analyzer = LLMConfigAnalyzer(model_name="models/gemini-flash-lite-latest")
            print("  ✓ LLMConfigAnalyzer initialized successfully")
            return True
        except ValueError as e:
            if "API" in str(e):
                print(f"  - LLMConfigAnalyzer: {e}")
                return True
            raise

    except Exception as e:
        print(f"  ✗ LLMConfigAnalyzer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Dainarx LLM Auto-Configuration Installation Test")
    print("="*60)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Only continue if imports work
    if results[0][1]:
        results.append(("DataAnalyzer", test_data_analyzer()))
        results.append(("GeneticOptimizer", test_genetic_optimizer()))
        results.append(("LLMConfigAnalyzer", test_llm_analyzer()))

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All tests passed! Installation is ready.")
        print("\nNext steps:")
        print("1. Set GEMINI_API_KEY in .env file")
        print("2. Run: python demos/duffing_auto_config.py --test")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
