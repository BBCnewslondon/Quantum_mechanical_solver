"""
Example: Infinite Square Well

This example demonstrates solving the time-independent Schrödinger equation
for a particle in an infinite square well (particle in a box).

The analytical solution provides a perfect test case for our numerical methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import create_solver
from potentials import quantum_well
from visualization import quick_plot, QuantumVisualizer


def analytical_infinite_well_energies(n_max: int, width: float = 2.0, 
                                    mass: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    """
    Analytical energy levels for infinite square well.
    
    E_n = (n²π²ℏ²)/(2mL²) for n = 1, 2, 3, ...
    """
    n_values = np.arange(1, n_max + 1)
    return (n_values**2 * np.pi**2 * hbar**2) / (2 * mass * width**2)


def main():
    """Run the infinite square well example."""
    print("🔬 Quantum Mechanics: Infinite Square Well")
    print("=" * 50)
    
    # Parameters
    well_width = 2.0
    x_range = (-3, 3)
    n_points = 1000
    n_states = 6
    
    print(f"Well width: {well_width}")
    print(f"Spatial range: {x_range}")
    print(f"Grid points: {n_points}")
    print(f"States to compute: {n_states}")
    
    # Create solver
    solver = create_solver(x_range, n_points)
    
    # Define infinite square well potential
    potential_func = quantum_well(width=well_width, infinite=True)
    
    print("\n🧮 Solving Schrödinger equation...")
    
    # Solve the equation
    energies, wavefunctions = solver.solve(potential_func, n_states=n_states)
    
    if len(energies) == 0:
        print("❌ Failed to find solutions!")
        return
    
    print(f"✅ Found {len(energies)} energy states")
    
    # Get analytical solutions for comparison
    analytical_energies = analytical_infinite_well_energies(n_states, well_width)
    
    # Print results
    print("\n📊 Energy Levels Comparison:")
    print("State | Numerical | Analytical | Error")
    print("-" * 40)
    for i, (E_num, E_ana) in enumerate(zip(energies, analytical_energies)):
        error = abs(E_num - E_ana) / E_ana * 100
        print(f"  {i+1}   | {E_num:8.4f}  | {E_ana:9.4f}  | {error:.2e}%")
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    
    # Evaluate potential on grid for plotting
    potential_values = potential_func(solver.x)
    
    # Main plot
    visualizer = QuantumVisualizer(figsize=(14, 10))
    fig1 = visualizer.plot_potential_and_wavefunctions(
        solver.x, potential_values, energies, wavefunctions,
        n_states=n_states, scale_factor=0.5
    )
    plt.suptitle('Infinite Square Well: Energy Levels and Wavefunctions', fontsize=16)
    
    # Energy spectrum comparison
    fig2 = visualizer.plot_energy_spectrum(energies, analytical_energies)
    
    # Show quantum numbers and selection rules
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    # Plot probability densities with quantum number labels
    for i in range(min(4, n_states)):
        prob_density = np.abs(wavefunctions[:, i])**2
        ax.plot(solver.x, prob_density + i * 0.5, 
               label=f'n={i+1}, E={energies[i]:.3f}')
        ax.fill_between(solver.x, i * 0.5, prob_density + i * 0.5, alpha=0.3)
    
    ax.set_xlabel('Position x')
    ax.set_ylabel('Probability Density (offset)')
    ax.set_title('Probability Densities for Different Quantum States')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight well boundaries
    ax.axvline(-well_width/2, color='red', linestyle='--', alpha=0.7, label='Well boundaries')
    ax.axvline(well_width/2, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Create interactive plot
    print("🌐 Creating interactive plot...")
    interactive_fig = visualizer.create_interactive_plot(
        solver.x, potential_values, energies, wavefunctions
    )
    
    # Save interactive plot
    interactive_fig.write_html("infinite_square_well_interactive.html")
    print("💾 Interactive plot saved as 'infinite_square_well_interactive.html'")
    
    # Physical insights
    print("\n💡 Physical Insights:")
    print(f"• Ground state energy: {energies[0]:.4f}")
    print(f"• Energy spacing is proportional to n² - quantum nature!")
    print(f"• Zero-point energy ≠ 0 - uncertainty principle")
    print(f"• Wavefunctions have n-1 nodes - wave interference")
    
    # Calculate expectation values
    print("\n📏 Expectation Values:")
    for i in range(min(3, n_states)):
        psi = wavefunctions[:, i]
        
        # Position expectation value <x>
        x_exp = np.trapz(psi.conj() * solver.x * psi, solver.x)
        
        # Position variance <x²> - <x>²
        x2_exp = np.trapz(psi.conj() * solver.x**2 * psi, solver.x)
        x_var = x2_exp - x_exp**2
        
        print(f"State {i+1}: <x> = {x_exp.real:.6f}, Δx = {np.sqrt(x_var.real):.4f}")
    
    plt.show()
    
    print("\n✨ Analysis complete! Check the plots for detailed results.")


if __name__ == "__main__":
    main()
