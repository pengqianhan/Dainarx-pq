#!/usr/bin/env python3
"""
Universal Automata Simulator

This script allows you to simulate any or all of the available hybrid automata
from the automata directory. It provides a flexible interface for selecting
specific automata or running simulations on all available ones.

Usage:
    python sim_all.py                    # Interactive mode - choose automata
    python sim_all.py --all             # Simulate all automata
    python sim_all.py --category ATVA   # Simulate all ATVA automata
    python sim_all.py --automata ball   # Simulate specific automata by name
    python sim_all.py --list            # List all available automata
    python sim_all.py --help            # Show help

Examples:
    python sim_all.py --automata ball --dt 0.01 --time 20.0
    python sim_all.py --category FaMoS --export --format json
    python sim_all.py --all --no-plot --export
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from AutomataSimulator import AutomataSimulator
except ImportError as e:
    print(f"Error importing AutomataSimulator: {e}")
    print("Make sure you're running this from the simulate_examples directory")
    sys.exit(1)


class AutomataManager:
    """Manages the collection of available automata."""

    def __init__(self, automata_base_dir: str = None):
        if automata_base_dir is None:
            # Default to the automata directory relative to this script
            self.base_dir = Path(__file__).parent.parent / "automata"
        else:
            self.base_dir = Path(automata_base_dir)

        self.automata_map = self._scan_automata()

    def _scan_automata(self) -> Dict[str, Dict[str, Path]]:
        """Scan the automata directory and build a map of available automata."""
        automata_map = {}

        if not self.base_dir.exists():
            print(f"Warning: Automata directory not found: {self.base_dir}")
            return automata_map

        # Scan each category
        for category_dir in sorted(self.base_dir.iterdir()):
            if category_dir.is_dir():
                category_name = category_dir.name
                automata_map[category_name] = {}

                # Scan JSON files in this category
                for json_file in sorted(category_dir.glob("*.json")):
                    automata_name = json_file.stem
                    automata_map[category_name][automata_name] = json_file

        return automata_map

    def list_automata(self, category: Optional[str] = None) -> None:
        """List all available automata, optionally filtered by category."""
        if not self.automata_map:
            print("No automata found!")
            return

        if category:
            if category not in self.automata_map:
                print(f"Category '{category}' not found. Available categories:")
                for cat in sorted(self.automata_map.keys()):
                    print(f"  - {cat}")
                return

            print(f"\nAutomata in category '{category}':")
            for name in sorted(self.automata_map[category].keys()):
                print(f"  - {name}")
        else:
            print("\nAvailable automata by category:")
            total_count = 0
            for category_name, automata in sorted(self.automata_map.items()):
                print(f"\n{category_name}/ ({len(automata)} automata):")
                for name in sorted(automata.keys()):
                    print(f"  - {name}")
                total_count += len(automata)
            print(f"\nTotal: {total_count} automata")

    def get_automata_path(self, category: str, name: str) -> Optional[Path]:
        """Get the path to a specific automata."""
        if category in self.automata_map and name in self.automata_map[category]:
            return self.automata_map[category][name]
        return None

    def get_all_automata(self) -> List[Tuple[str, str, Path]]:
        """Get all available automata as (category, name, path) tuples."""
        all_automata = []
        for category, automata in self.automata_map.items():
            for name, path in automata.items():
                all_automata.append((category, name, path))
        return sorted(all_automata)

    def get_category_automata(self, category: str) -> List[Tuple[str, str, Path]]:
        """Get all automata in a specific category."""
        if category not in self.automata_map:
            return []
        return [(category, name, path) for name, path in self.automata_map[category].items()]

    def search_automata(self, search_term: str) -> List[Tuple[str, str, Path]]:
        """Search for automata by name (case-insensitive)."""
        results = []
        search_lower = search_term.lower()

        for category, automata in self.automata_map.items():
            for name, path in automata.items():
                if search_lower in name.lower():
                    results.append((category, name, path))

        return results


class SimulationRunner:
    """Handles running simulations for automata."""

    def __init__(self, manager: AutomataManager):
        self.manager = manager

    def simulate_automata(self,
                         category: str,
                         name: str,
                         dt: float = 0.01,
                         total_time: float = 10.0,
                         plot: bool = True,
                         export: bool = False,
                         export_format: str = 'csv',
                         output_dir: Optional[str] = None,
                         verbose: bool = True) -> bool:
        """
        Simulate a specific automata.

        Returns True if successful, False otherwise.
        """
        path = self.manager.get_automata_path(category, name)
        if not path:
            print(f"Error: Automata '{category}/{name}' not found!")
            return False

        if verbose:
            print(f"\n{'='*60}")
            print(f"Simulating: {category}/{name}")
            print(f"Path: {path}")
            print(f"{'='*60}")

        try:
            # Create simulator
            simulator = AutomataSimulator(str(path))

            if verbose:
                print(f"Configuration: {simulator.config}")
                print(f"Initial states: {len(simulator.init_states)}")

            # Run simulation
            if verbose:
                print(f"\nRunning simulation (dt={dt}, time={total_time})...")

            trajectories = simulator.simulate_all_trajectories(
                dt=dt,
                total_time=total_time,
                verbose=verbose
            )

            # Analyze results
            if verbose:
                analysis = simulator.analyze_trajectories(trajectories)
                print("\nAnalysis:")
                print(f"  - Trajectories: {analysis['num_trajectories']}")
                print(".2f")
                print(f"  - Mode changes: {len(analysis['mode_changes'])}")

            # Plot results
            if plot:
                if verbose:
                    print("\nGenerating plots...")

                try:
                    variables = simulator.automata.get_state_variables()
                    fig = simulator.plot_trajectories(variables=variables)
                    plt.show(block=False)
                    plt.pause(0.1)  # Allow time for plot to render
                except Exception as e:
                    print(f"Warning: Could not generate plots: {e}")

            # Export results
            if export:
                if verbose:
                    print("\nExporting results...")

                export_path = output_dir or f"{category}_{name}_results"
                try:
                    simulator.export_trajectories(
                        format=export_format,
                        output_dir=export_path
                    )
                    if verbose:
                        print(f"Results exported to: {export_path}")
                except Exception as e:
                    print(f"Warning: Could not export results: {e}")

            if verbose:
                print(f"\n✓ Successfully simulated {category}/{name}")

            return True

        except Exception as e:
            print(f"✗ Error simulating {category}/{name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return False

    def simulate_multiple(self,
                         automata_list: List[Tuple[str, str, Path]],
                         dt: float = 0.01,
                         total_time: float = 10.0,
                         plot: bool = True,
                         export: bool = False,
                         export_format: str = 'csv',
                         output_dir: Optional[str] = None,
                         verbose: bool = True) -> Tuple[int, int]:
        """
        Simulate multiple automata.

        Returns (successful_count, total_count).
        """
        if not automata_list:
            print("No automata to simulate!")
            return 0, 0

        successful = 0
        total = len(automata_list)

        if verbose:
            print(f"\nStarting simulation of {total} automata...")

        for i, (category, name, path) in enumerate(automata_list, 1):
            if verbose:
                print(f"\n[{i}/{total}] Processing {category}/{name}...")

            if self.simulate_automata(
                category=category,
                name=name,
                dt=dt,
                total_time=total_time,
                plot=plot,
                export=export,
                export_format=export_format,
                output_dir=output_dir,
                verbose=verbose
            ):
                successful += 1

        if verbose:
            print(f"\n{'='*60}")
            print(f"SIMULATION COMPLETE")
            print(f"{'='*60}")
            print(f"Successful: {successful}/{total}")
            if successful < total:
                print(f"Failed: {total - successful}")

        return successful, total


def interactive_selection(manager: AutomataManager) -> List[Tuple[str, str, Path]]:
    """Interactive mode for selecting automata to simulate."""
    print("\n=== Interactive Automata Selection ===")

    # Show categories
    manager.list_automata()

    selected_automata = []

    while True:
        print("\nOptions:")
        print("1. Select by category")
        print("2. Select specific automata")
        print("3. Search by name")
        print("4. Show all and simulate")
        print("5. Done with selection")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            category = input("Enter category name: ").strip()
            automata_in_category = manager.get_category_automata(category)
            if automata_in_category:
                print(f"\nFound {len(automata_in_category)} automata in {category}")
                for cat, name, path in automata_in_category:
                    print(f"  - {name}")
                if input("Add all to selection? (y/n): ").lower().startswith('y'):
                    selected_automata.extend(automata_in_category)
                    print(f"Added {len(automata_in_category)} automata from {category}")
            else:
                print(f"No automata found in category '{category}'")

        elif choice == "2":
            category = input("Enter category: ").strip()
            name = input("Enter automata name: ").strip()
            path = manager.get_automata_path(category, name)
            if path:
                selected_automata.append((category, name, path))
                print(f"Added {category}/{name}")
            else:
                print(f"Automata '{category}/{name}' not found")

        elif choice == "3":
            search_term = input("Enter search term: ").strip()
            results = manager.search_automata(search_term)
            if results:
                print(f"\nFound {len(results)} matching automata:")
                for i, (cat, name, path) in enumerate(results, 1):
                    print(f"{i}. {cat}/{name}")

                indices = input("Enter numbers to add (comma-separated, or 'all'): ").strip()
                if indices.lower() == 'all':
                    selected_automata.extend(results)
                    print(f"Added all {len(results)} automata")
                else:
                    try:
                        for idx in indices.split(','):
                            idx = int(idx.strip()) - 1
                            if 0 <= idx < len(results):
                                selected_automata.append(results[idx])
                                print(f"Added {results[idx][0]}/{results[idx][1]}")
                    except ValueError:
                        print("Invalid input")
            else:
                print("No matching automata found")

        elif choice == "4":
            all_automata = manager.get_all_automata()
            print(f"\nFound {len(all_automata)} total automata")
            if input("Simulate all? (y/n): ").lower().startswith('y'):
                selected_automata = all_automata
                break
            else:
                continue

        elif choice == "5":
            break

        else:
            print("Invalid choice")

        if selected_automata:
            print(f"\nCurrent selection ({len(selected_automata)} automata):")
            for cat, name, path in selected_automata[-5:]:  # Show last 5
                print(f"  - {cat}/{name}")
            if len(selected_automata) > 5:
                print(f"  ... and {len(selected_automata) - 5} more")

    return selected_automata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Universal Automata Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Interactive selection
  %(prog)s --all                    # Simulate all automata
  %(prog)s --category ATVA          # Simulate ATVA category
  %(prog)s --automata ball          # Simulate specific automata
  %(prog)s --search heating         # Search and simulate matching automata
  %(prog)s --list                   # List all available automata
        """
    )

    # Selection options
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', action='store_true', help='Simulate all automata')
    group.add_argument('--category', type=str, help='Simulate all automata in category')
    group.add_argument('--automata', type=str, help='Simulate specific automata by name')
    group.add_argument('--search', type=str, help='Search for automata by name')
    group.add_argument('--list', action='store_true', help='List all available automata')

    # Simulation parameters
    parser.add_argument('--dt', type=float, default=0.01, help='Time step size (default: 0.01)')
    parser.add_argument('--time', type=float, default=10.0, help='Total simulation time (default: 10.0)')

    # Output options
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--export', action='store_true', help='Export simulation results')
    parser.add_argument('--format', choices=['csv', 'json', 'numpy'], default='csv',
                       help='Export format (default: csv)')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')

    # Other options
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    args = parser.parse_args()

    # Set verbosity
    verbose = not args.quiet

    # Initialize manager
    try:
        manager = AutomataManager()
    except Exception as e:
        print(f"Error initializing automata manager: {e}")
        sys.exit(1)

    # Handle list command
    if args.list:
        manager.list_automata()
        return

    # Determine which automata to simulate
    selected_automata = []

    if args.all:
        selected_automata = manager.get_all_automata()
        if verbose:
            print(f"Selected all {len(selected_automata)} automata")
    elif args.category:
        selected_automata = manager.get_category_automata(args.category)
        if not selected_automata:
            print(f"No automata found in category '{args.category}'")
            manager.list_automata()
            sys.exit(1)
        if verbose:
            print(f"Selected {len(selected_automata)} automata from category '{args.category}'")
    elif args.automata:
        # Search for the automata in all categories
        results = manager.search_automata(args.automata)
        if not results:
            print(f"No automata found matching '{args.automata}'")
            manager.list_automata()
            sys.exit(1)
        elif len(results) == 1:
            selected_automata = results
            if verbose:
                cat, name, _ = results[0]
                print(f"Selected automata: {cat}/{name}")
        else:
            print(f"Multiple automata found matching '{args.automata}':")
            for i, (cat, name, _) in enumerate(results, 1):
                print(f"{i}. {cat}/{name}")
            choice = input("Enter number to select: ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(results):
                    selected_automata = [results[idx]]
                else:
                    print("Invalid selection")
                    sys.exit(1)
            except ValueError:
                print("Invalid input")
                sys.exit(1)
    elif args.search:
        results = manager.search_automata(args.search)
        if not results:
            print(f"No automata found matching '{args.search}'")
            sys.exit(1)
        selected_automata = results
        if verbose:
            print(f"Found {len(results)} automata matching '{args.search}'")
    else:
        # Interactive mode
        selected_automata = interactive_selection(manager)
        if not selected_automata:
            print("No automata selected. Exiting.")
            sys.exit(0)

    # Run simulations
    if selected_automata:
        runner = SimulationRunner(manager)

        successful, total = runner.simulate_multiple(
            selected_automata,
            dt=args.dt,
            total_time=args.time,
            plot=not args.no_plot,
            export=args.export,
            export_format=args.format,
            output_dir=args.output_dir,
            verbose=verbose
        )

        # Final summary
        if verbose:
            print(f"\n{'='*60}")
            print("FINAL SUMMARY")
            print(f"{'='*60}")
            print(f"Simulations completed: {successful}/{total}")
            print(f"Success rate: {(successful/total)*100:.1f}%")

            if successful < total:
                print("\nSome simulations failed. Check the error messages above.")

        # Keep plots open if they were generated
        if not args.no_plot and successful > 0:
            try:
                plt.show()
            except:
                pass

    else:
        print("No automata selected for simulation.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if '--verbose' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)
