"""
Genetic Algorithm Optimizer: Search for optimal Dainarx configurations
"""

import numpy as np
import random
import copy
from typing import Dict, List, Any, Tuple, Callable, Optional
import json


class GeneticOptimizer:
    """Genetic algorithm for optimizing Dainarx configuration parameters"""

    def __init__(self,
                 population_size: int = 20,
                 generations: int = 10,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.7,
                 elitism: int = 2,
                 tournament_size: int = 3,
                 random_seed: Optional[int] = None):
        """
        Initialize genetic optimizer

        Args:
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover
            elitism: Number of best individuals to preserve
            tournament_size: Tournament selection size
            random_seed: Random seed for reproducibility
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_size = tournament_size

        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # History tracking
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_configs': []
        }

    def optimize(self,
                 config_ranges: Dict[str, List[Any]],
                 fitness_function: Callable,
                 llm_suggestions: Optional[Dict] = None,
                 verbose: bool = True) -> Tuple[Dict, float]:
        """
        Run genetic algorithm optimization

        Args:
            config_ranges: Dictionary of parameter ranges
                Example: {"order": [2, 3, 4], "kernel": ["linear", "rbf"]}
            fitness_function: Function that takes config dict and returns fitness score
            llm_suggestions: Optional LLM suggestions to bias initialization
            verbose: Print progress

        Returns:
            (best_config, best_fitness)
        """
        self.config_ranges = config_ranges
        self.fitness_function = fitness_function

        if verbose:
            print("\n=== Starting Genetic Algorithm ===")
            print(f"Population size: {self.population_size}")
            print(f"Generations: {self.generations}")
            print(f"Mutation rate: {self.mutation_rate}")
            print(f"Config space size: ~{self._estimate_search_space()}")

        # Initialize population
        population = self._initialize_population(llm_suggestions)

        best_overall_config = None
        best_overall_fitness = -float('inf')

        # Evolution loop
        for gen in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for i, individual in enumerate(population):
                try:
                    fitness = self.fitness_function(individual)
                    fitness_scores.append(fitness)
                except Exception as e:
                    if verbose:
                        print(f"  Config {i} failed: {e}")
                    fitness_scores.append(-1e6)  # Penalize failed configs

            fitness_scores = np.array(fitness_scores)

            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            gen_best_config = population[gen_best_idx]

            if gen_best_fitness > best_overall_fitness:
                best_overall_fitness = gen_best_fitness
                best_overall_config = copy.deepcopy(gen_best_config)

            # Record history
            self.history['best_fitness'].append(float(gen_best_fitness))
            self.history['avg_fitness'].append(float(np.mean(fitness_scores)))
            self.history['best_configs'].append(copy.deepcopy(gen_best_config))

            if verbose:
                print(f"\nGeneration {gen + 1}/{self.generations}:")
                print(f"  Best fitness: {gen_best_fitness:.4f}")
                print(f"  Avg fitness: {np.mean(fitness_scores):.4f}")
                print(f"  Best config: {self._format_config_short(gen_best_config)}")

            # Check if last generation
            if gen == self.generations - 1:
                break

            # Selection
            parents = self._tournament_selection(population, fitness_scores)

            # Crossover and mutation
            offspring = self._create_offspring(parents)

            # Elitism: preserve best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elitism:]
            elite = [population[i] for i in elite_indices]

            # New population
            population = elite + offspring[:self.population_size - self.elitism]

        if verbose:
            print(f"\n=== Optimization Complete ===")
            print(f"Best fitness: {best_overall_fitness:.4f}")
            print(f"Best config:")
            self._print_config(best_overall_config)

        return best_overall_config, best_overall_fitness

    def _initialize_population(self, llm_suggestions: Optional[Dict] = None) -> List[Dict]:
        """
        Initialize population

        Strategy:
        - 50% based on LLM suggestions (if available)
        - 50% random sampling
        """
        population = []

        # Number of LLM-biased individuals
        num_llm_based = self.population_size // 2 if llm_suggestions else 0

        # Create LLM-biased individuals
        for _ in range(num_llm_based):
            individual = self._sample_config_biased(llm_suggestions)
            population.append(individual)

        # Create random individuals
        for _ in range(self.population_size - num_llm_based):
            individual = self._sample_config_random()
            population.append(individual)

        return population

    def _sample_config_random(self) -> Dict[str, Any]:
        """Sample a random configuration from ranges"""
        config = {}
        for param, options in self.config_ranges.items():
            config[param] = random.choice(options)
        return config

    def _sample_config_biased(self, llm_suggestions: Dict) -> Dict[str, Any]:
        """Sample configuration biased toward LLM suggestions"""
        config = {}

        recommendations = llm_suggestions.get('recommendations', {})

        for param, options in self.config_ranges.items():
            # Get LLM suggestion
            llm_value = recommendations.get(param, {}).get('value')

            if llm_value is None:
                # No suggestion, random sample
                config[param] = random.choice(options)
            elif isinstance(llm_value, list) and len(llm_value) > 0:
                # LLM provided multiple options
                # 80% chance to pick from LLM suggestions
                if random.random() < 0.8:
                    valid_llm_options = [v for v in llm_value if v in options]
                    if valid_llm_options:
                        config[param] = random.choice(valid_llm_options)
                    else:
                        config[param] = random.choice(options)
                else:
                    config[param] = random.choice(options)
            else:
                # Single LLM value
                if llm_value in options:
                    # 80% chance to use LLM value
                    if random.random() < 0.8:
                        config[param] = llm_value
                    else:
                        config[param] = random.choice(options)
                else:
                    config[param] = random.choice(options)

        return config

    def _tournament_selection(self, population: List[Dict],
                              fitness_scores: np.ndarray) -> List[Dict]:
        """Tournament selection"""
        parents = []

        for _ in range(self.population_size):
            # Select tournament participants
            tournament_idx = random.sample(range(len(population)), self.tournament_size)
            tournament_fitness = fitness_scores[tournament_idx]

            # Select winner
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            parents.append(copy.deepcopy(population[winner_idx]))

        return parents

    def _create_offspring(self, parents: List[Dict]) -> List[Dict]:
        """Create offspring through crossover and mutation"""
        offspring = []

        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            offspring.extend([child1, child2])

        return offspring

    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Single-point crossover"""
        child1 = {}
        child2 = {}

        params = list(self.config_ranges.keys())
        crossover_point = random.randint(1, len(params) - 1)

        for i, param in enumerate(params):
            if i < crossover_point:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]

        return child1, child2

    def _mutate(self, individual: Dict) -> Dict:
        """Mutate individual"""
        mutated = copy.deepcopy(individual)

        for param, options in self.config_ranges.items():
            if random.random() < self.mutation_rate:
                # Mutate this gene
                mutated[param] = random.choice(options)

        return mutated

    def _estimate_search_space(self) -> int:
        """Estimate total search space size"""
        size = 1
        for options in self.config_ranges.values():
            size *= len(options)
        return size

    def _format_config_short(self, config: Dict) -> str:
        """Format config as short string"""
        items = []
        for key, val in config.items():
            if isinstance(val, float):
                items.append(f"{key}={val:.1e}")
            else:
                items.append(f"{key}={val}")
        return ", ".join(items)

    def _print_config(self, config: Dict):
        """Print config in readable format"""
        for key, val in config.items():
            print(f"  {key}: {val}")

    def save_history(self, filepath: str):
        """Save optimization history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot_history(self, save_path: Optional[str] = None):
        """Plot optimization history"""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            generations = range(1, len(self.history['best_fitness']) + 1)

            ax.plot(generations, self.history['best_fitness'],
                   marker='o', label='Best Fitness', linewidth=2)
            ax.plot(generations, self.history['avg_fitness'],
                   marker='s', label='Average Fitness', linewidth=2, alpha=0.7)

            ax.set_xlabel('Generation', fontsize=12)
            ax.set_ylabel('Fitness', fontsize=12)
            ax.set_title('Genetic Algorithm Optimization Progress', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150)
                print(f"Plot saved to {save_path}")

            plt.show()

        except ImportError:
            print("matplotlib not available for plotting")
