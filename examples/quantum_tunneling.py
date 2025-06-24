"""
Example: Quantum Tunneling Simulation

This example demonstrates quantum tunneling through a potential barrier.
We calculate transmission coefficients and visualize the tunneling effect
for different particle energies.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import create_solver
from potentials import barrier
from visualization import QuantumVisualizer


def theoretical_transmission(energy: float, barrier_height: float, 
                           barrier_width: float, mass: float = 1.0, 
                           hbar: float = 1.0) -> float:
    """
    Theoretical transmission coefficient for rectangular barrier.
    
    Using the exact quantum mechanical formula for E < V₀.
    """
    if energy >= barrier_height:
        return 1.0  # Classical case
    
    # Wave number inside barrier
    kappa = np.sqrt(2 * mass * (barrier_height - energy)) / hbar
    
    # Transmission coefficient
    T = 1 / (1 + (barrier_height**2 * np.sinh(kappa * barrier_width)**2) / 
             (4 * energy * (barrier_height - energy)))
    
    return T


def main():
    """Run the quantum tunneling simulation."""
    print("🌊 Quantum Mechanics: Tunneling Through Potential Barrier")
    print("=" * 60)
    
    # Parameters
    barrier_width = 1.0
    barrier_height = 5.0
    x_range = (-8, 8)
    n_points = 1500
    
    print(f"Barrier width: {barrier_width}")
    print(f"Barrier height: {barrier_height}")
    print(f"Spatial range: {x_range}")
    print(f"Grid points: {n_points}")
    
    # Create solver
    solver = create_solver(x_range, n_points)
    
    # Define potential barrier
    potential_func = barrier(width=barrier_width, height=barrier_height)
    potential_values = potential_func(solver.x)
    
    # Find barrier boundaries
    barrier_indices = np.where(potential_values > 0.1)[0]
    if len(barrier_indices) > 0:
        barrier_start = solver.x[barrier_indices[0]]
        barrier_end = solver.x[barrier_indices[-1]]
    else:
        barrier_start = -barrier_width / 2
        barrier_end = barrier_width / 2
    
    print(f"Barrier region: [{barrier_start:.2f}, {barrier_end:.2f}]")
    
    # Test different energies for tunneling
    print("\n🔬 Calculating transmission coefficients...")
    
    test_energies = np.linspace(0.1, barrier_height * 1.5, 20)
    transmissions_numerical = []
    transmissions_theoretical = []
    
    for energy in test_energies:
        # Numerical calculation
        T_num = solver.transmission_coefficient(
            potential_func, energy, barrier_start, barrier_end
        )
        transmissions_numerical.append(T_num)
        
        # Theoretical calculation
        T_theo = theoretical_transmission(
            energy, barrier_height, barrier_width
        )
        transmissions_theoretical.append(T_theo)
        
        print(f"E = {energy:.2f}: T_num = {T_num:.4f}, T_theo = {T_theo:.4f}")
    
    # Solve for bound states (if any exist)
    print("\n🔍 Looking for bound states...")
    try:
        # For this barrier, there typically aren't bound states, but let's check
        energies, wavefunctions = solver.solve(potential_func, n_states=3)
        if len(energies) > 0:
            print(f"Found {len(energies)} bound states with energies: {energies}")
        else:
            print("No bound states found (as expected for a simple barrier)")
    except:
        print("No bound states found")
        energies, wavefunctions = np.array([]), np.array([])
    
    # Create comprehensive visualization
    print("\n📊 Creating visualizations...")
    
    visualizer = QuantumVisualizer(figsize=(15, 12))
    
    # Main tunneling plot
    fig1 = visualizer.plot_tunneling_analysis(
        solver.x, potential_values, test_energies[:8], 
        transmissions_numerical[:8], barrier_start, barrier_end
    )
    plt.suptitle('Quantum Tunneling Analysis', fontsize=16)
    
    # Detailed transmission comparison
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Transmission vs Energy
    ax1.plot(test_energies, transmissions_numerical, 'bo-', 
            label='Numerical (WKB)', markersize=6, linewidth=2)
    ax1.plot(test_energies, transmissions_theoretical, 'r--', 
            label='Theoretical', linewidth=2)
    ax1.axvline(barrier_height, color='gray', linestyle=':', 
               label=f'Barrier height = {barrier_height}')
    ax1.set_xlabel('Particle Energy')
    ax1.set_ylabel('Transmission Coefficient T')
    ax1.set_title('Quantum Tunneling: Transmission vs Energy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Log scale to show tunneling regime better
    ax2.semilogy(test_energies, transmissions_numerical, 'bo-', 
                label='Numerical (WKB)', markersize=6)
    ax2.semilogy(test_energies, transmissions_theoretical, 'r--', 
                label='Theoretical')
    ax2.axvline(barrier_height, color='gray', linestyle=':', 
               label=f'Barrier height = {barrier_height}')
    ax2.set_xlabel('Particle Energy')
    ax2.set_ylabel('Transmission Coefficient T (log scale)')
    ax2.set_title('Tunneling Regime (Logarithmic Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Analyze tunneling regimes
    print("\n💡 Tunneling Analysis:")
    
    # Classical regime (E > V₀)
    classical_mask = np.array(test_energies) > barrier_height
    if np.any(classical_mask):
        classical_T = np.array(transmissions_numerical)[classical_mask]
        print(f"• Classical regime (E > V₀): T ≈ {np.mean(classical_T):.3f}")
    
    # Tunneling regime (E < V₀)
    tunneling_mask = np.array(test_energies) < barrier_height
    if np.any(tunneling_mask):
        tunneling_energies = np.array(test_energies)[tunneling_mask]
        tunneling_T = np.array(transmissions_numerical)[tunneling_mask]
        
        print(f"• Tunneling regime (E < V₀):")
        print(f"  - Lowest energy: E = {tunneling_energies[0]:.2f}, T = {tunneling_T[0]:.2e}")
        print(f"  - Near barrier top: E = {tunneling_energies[-1]:.2f}, T = {tunneling_T[-1]:.3f}")
    
    # Calculate tunnel current ratio
    if len(tunneling_T) > 1:
        current_ratio = tunneling_T[-1] / tunneling_T[0] if tunneling_T[0] > 0 else np.inf
        print(f"  - Current enhancement factor: {current_ratio:.1e}")
    
    # Visualization of wave penetration
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Show potential and test energies
    sample_energies = [0.5, 2.0, 4.0, 6.0]  # Mix of tunneling and classical
    
    for i, E in enumerate(sample_energies):
        ax = axes[i]
        
        # Plot potential
        ax.plot(solver.x, potential_values, 'k-', linewidth=2, label='Potential')
        ax.axhline(E, color='red', linestyle='--', label=f'E = {E:.1f}')
        ax.axvspan(barrier_start, barrier_end, alpha=0.3, color='gray', 
                  label='Barrier')
        
        # Calculate and show transmission
        T = solver.transmission_coefficient(potential_func, E, barrier_start, barrier_end)
        ax.set_title(f'Energy = {E:.1f}, T = {T:.4f}')
        ax.set_xlabel('Position x')
        ax.set_ylabel('Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(barrier_height * 1.2, E * 1.2))
    
    plt.tight_layout()
    plt.suptitle('Wave Penetration at Different Energies', fontsize=14, y=1.02)
    
    # Physical insights
    print("\n🔬 Physical Insights:")
    print("• Quantum tunneling allows particles to pass through barriers")
    print("  even when classically forbidden (E < V₀)")
    print("• Transmission probability decreases exponentially with:")
    print("  - Barrier width (wider → less tunneling)")
    print("  - Barrier height (higher → less tunneling)")
    print("  - Particle mass (heavier → less tunneling)")
    print("• Applications: STM, tunnel diodes, alpha decay, etc.")
    
    # Save results
    np.savetxt('tunneling_results.csv', 
              np.column_stack([test_energies, transmissions_numerical, transmissions_theoretical]),
              header='Energy,Transmission_Numerical,Transmission_Theoretical',
              delimiter=',')
    print("\n💾 Results saved to 'tunneling_results.csv'")
    
    plt.show()
    
    print("\n✨ Tunneling analysis complete!")


if __name__ == "__main__":
    main()
