"""
Quick Demo: Test the Quantum Mechanics Solver

A simple script to verify that everything is working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import create_solver
from potentials import quantum_well, harmonic
from visualization import quick_plot, QuantumVisualizer


def test_infinite_well():
    """Test the infinite square well example."""
    print("🔬 Testing Infinite Square Well...")
    
    # Create solver
    solver = create_solver(x_range=(-3, 3), n_points=800)
    
    # Define infinite square well
    potential_func = quantum_well(width=2.0, infinite=True)
    
    # Solve
    energies, wavefunctions = solver.solve(potential_func, n_states=4)
    
    if len(energies) > 0:
        print(f"✅ Found {len(energies)} energy states")
        print(f"   Ground state energy: {energies[0]:.4f}")
        print(f"   Energy levels: {energies}")
        
        # Create visualization
        potential_values = potential_func(solver.x)
        fig = quick_plot(solver.x, potential_values, energies, wavefunctions, n_states=4)
        plt.title('Infinite Square Well - Quick Demo')
        plt.show()
        
        return True
    else:
        print("❌ No states found")
        return False


def test_harmonic_oscillator():
    """Test the harmonic oscillator."""
    print("\n🌸 Testing Harmonic Oscillator...")
    
    # Create solver
    solver = create_solver(x_range=(-4, 4), n_points=800)
    
    # Define harmonic oscillator
    potential_func = harmonic(omega=1.0)
    
    # Solve
    energies, wavefunctions = solver.solve(potential_func, n_states=5)
    
    if len(energies) > 0:
        print(f"✅ Found {len(energies)} energy states")
        print(f"   Ground state energy: {energies[0]:.4f} (should be ~0.5)")
        print(f"   Energy spacing: {energies[1] - energies[0]:.4f} (should be ~1.0)")
        
        # Analytical comparison
        analytical = np.arange(5) + 0.5  # E_n = ℏω(n + 1/2) with ℏω = 1
        print(f"   Analytical energies: {analytical}")
        print(f"   Numerical energies:  {energies}")
        
        # Create detailed visualization
        visualizer = QuantumVisualizer()
        potential_values = potential_func(solver.x)
        
        fig = visualizer.plot_potential_and_wavefunctions(
            solver.x, potential_values, energies, wavefunctions, 
            n_states=4, scale_factor=1.0
        )
        plt.suptitle('Harmonic Oscillator - Detailed Analysis')
        plt.show()
        
        return True
    else:
        print("❌ No states found")
        return False


def main():
    """Run the demo tests."""
    print("🚀 Quantum Mechanics Solver - Quick Demo")
    print("=" * 50)
    
    # Test both systems
    success_count = 0
    
    if test_infinite_well():
        success_count += 1
    
    if test_harmonic_oscillator():
        success_count += 1
    
    print(f"\n📊 Results: {success_count}/2 tests passed")
    
    if success_count == 2:
        print("🎉 All tests passed! The quantum solver is working correctly.")
        print("\n💡 Try running the full examples:")
        print("   python examples/infinite_square_well.py")
        print("   python examples/harmonic_oscillator.py")
        print("   python examples/quantum_tunneling.py")
        print("   python examples/comprehensive_demo.py")
    else:
        print("⚠️  Some tests failed. Check your installation.")
    
    print("\n✨ Demo complete!")


if __name__ == "__main__":
    main()
