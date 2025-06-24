"""
Example: Harmonic Oscillator

This example solves the quantum harmonic oscillator and compares
numerical results with the well-known analytical solution.

The harmonic oscillator is fundamental in quantum mechanics and
provides an excellent test of our numerical methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import create_solver
from potentials import harmonic
from visualization import QuantumVisualizer


def analytical_harmonic_energies(n_max: int, omega: float = 1.0, 
                               hbar: float = 1.0) -> np.ndarray:
    """
    Analytical energy levels for quantum harmonic oscillator.
    
    E_n = ℏω(n + 1/2) for n = 0, 1, 2, ...
    """
    n_values = np.arange(n_max)
    return hbar * omega * (n_values + 0.5)


def hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    """
    Physicist's Hermite polynomials H_n(x) using recurrence relation.
    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        H_prev2 = np.ones_like(x)  # H_0
        H_prev1 = 2 * x            # H_1
        
        for i in range(2, n + 1):
            H_current = 2 * x * H_prev1 - 2 * (i - 1) * H_prev2
            H_prev2, H_prev1 = H_prev1, H_current
        
        return H_prev1


def analytical_harmonic_wavefunction(n: int, x: np.ndarray, omega: float = 1.0,
                                   mass: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    """
    Analytical wavefunction for quantum harmonic oscillator.
    
    ψ_n(x) = (mω/πℏ)^(1/4) * (1/√(2^n n!)) * H_n(√(mω/ℏ)x) * exp(-mωx²/2ℏ)
    """
    # Characteristic length scale
    x0 = np.sqrt(hbar / (mass * omega))
    
    # Dimensionless coordinate
    xi = x / x0
      # Normalization constant
    import math
    norm = (mass * omega / (np.pi * hbar))**(1/4) / np.sqrt(2**n * math.factorial(n))
    
    # Hermite polynomial
    H_n = hermite_polynomial(n, xi)
    
    # Gaussian envelope
    gaussian = np.exp(-xi**2 / 2)
    
    return norm * H_n * gaussian


def main():
    """Run the harmonic oscillator example."""
    print("🌸 Quantum Mechanics: Harmonic Oscillator")
    print("=" * 50)
    
    # Parameters
    omega = 1.0  # Angular frequency
    x_range = (-6, 6)
    n_points = 1500
    n_states = 8
    
    print(f"Angular frequency ω: {omega}")
    print(f"Spatial range: {x_range}")
    print(f"Grid points: {n_points}")
    print(f"States to compute: {n_states}")
    
    # Create solver
    solver = create_solver(x_range, n_points)
    
    # Define harmonic oscillator potential
    potential_func = harmonic(omega=omega)
    
    print("\n🧮 Solving Schrödinger equation...")
    
    # Solve the equation
    energies, wavefunctions = solver.solve(potential_func, n_states=n_states)
    
    if len(energies) == 0:
        print("❌ Failed to find solutions!")
        return
    
    print(f"✅ Found {len(energies)} energy states")
    
    # Get analytical solutions for comparison
    analytical_energies = analytical_harmonic_energies(n_states, omega)
    
    # Calculate analytical wavefunctions
    analytical_wavefunctions = np.zeros((len(solver.x), n_states))
    for i in range(n_states):
        analytical_wavefunctions[:, i] = analytical_harmonic_wavefunction(
            i, solver.x, omega
        )
    
    # Print energy comparison
    print("\n📊 Energy Levels Comparison:")
    print("State | Numerical | Analytical | Error    | ℏω(n+½)")
    print("-" * 55)
    for i, (E_num, E_ana) in enumerate(zip(energies, analytical_energies)):
        error = abs(E_num - E_ana) / E_ana * 100
        expected = omega * (i + 0.5)
        print(f"  {i}   | {E_num:8.4f}  | {E_ana:9.4f}  | {error:.2e}% | {expected:.4f}")
    
    # Evaluate potential on grid for plotting
    potential_values = potential_func(solver.x)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    
    visualizer = QuantumVisualizer(figsize=(15, 12))
    
    # Main plot with numerical results
    fig1 = visualizer.plot_potential_and_wavefunctions(
        solver.x, potential_values, energies, wavefunctions,
        n_states=min(6, n_states), scale_factor=1.0
    )
    plt.suptitle('Quantum Harmonic Oscillator: Numerical Solution', fontsize=16)
    
    # Comparison plot with analytical solutions
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    states_to_compare = [0, 1, 2, 3]  # Ground state and first few excited states
    
    for i, n in enumerate(states_to_compare):
        if n < len(energies):
            ax = axes[i]
            
            # Plot both numerical and analytical wavefunctions
            ax.plot(solver.x, wavefunctions[:, n], 'b-', linewidth=2, 
                   label=f'Numerical ψ_{n}')
            ax.plot(solver.x, analytical_wavefunctions[:, n], 'r--', linewidth=2,
                   label=f'Analytical ψ_{n}')
            
            # Also show probability densities
            ax.plot(solver.x, np.abs(wavefunctions[:, n])**2, 'b:', alpha=0.7,
                   label=f'|ψ_{n}|² (num)')
            ax.plot(solver.x, np.abs(analytical_wavefunctions[:, n])**2, 'r:', alpha=0.7,
                   label=f'|ψ_{n}|² (ana)')
            
            ax.set_title(f'State n = {n}, E = {energies[n]:.4f}')
            ax.set_xlabel('Position x')
            ax.set_ylabel('Wavefunction')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Numerical vs Analytical Wavefunctions', fontsize=14, y=1.02)
    
    # Energy spectrum plot
    fig3 = visualizer.plot_energy_spectrum(energies, analytical_energies)
    plt.title('Energy Spectrum: Harmonic Oscillator')
    
    # Classical turning points visualization
    fig4, ax = plt.subplots(figsize=(12, 8))
    
    # Plot potential
    ax.plot(solver.x, potential_values, 'k-', linewidth=3, label='V(x) = ½mω²x²')
    
    # Plot energy levels and classical turning points
    colors = plt.cm.viridis(np.linspace(0, 1, min(6, n_states)))
    
    for i in range(min(6, n_states)):
        E = energies[i]
        color = colors[i]
        
        # Energy level
        ax.axhline(E, color=color, linestyle='--', alpha=0.8, 
                  label=f'E_{i} = {E:.3f}')
        
        # Classical turning points: E = ½mω²x² → x = ±√(2E/mω²)
        x_turn = np.sqrt(2 * E / omega**2)  # Assuming m = 1
        ax.axvline(x_turn, color=color, linestyle=':', alpha=0.6)
        ax.axvline(-x_turn, color=color, linestyle=':', alpha=0.6)
        
        # Wavefunction (scaled)
        psi_scaled = 0.5 * wavefunctions[:, i] + E
        ax.plot(solver.x, psi_scaled, color=color, linewidth=2)
        
        # Highlight classically forbidden regions
        forbidden_x = solver.x[np.abs(solver.x) > x_turn]
        if len(forbidden_x) > 0:
            ax.axvspan(x_turn, solver.x[-1], alpha=0.1, color=color)
            ax.axvspan(solver.x[0], -x_turn, alpha=0.1, color=color)
    
    ax.set_xlabel('Position x')
    ax.set_ylabel('Energy')
    ax.set_title('Classical Turning Points and Quantum Penetration')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, energies[min(5, len(energies)-1)] * 1.1)
    
    plt.tight_layout()
    
    # Calculate and display physical properties
    print("\n💡 Physical Properties:")
    
    # Zero-point energy
    print(f"• Zero-point energy: E₀ = {energies[0]:.4f} = ½ℏω")
    print(f"  (Classical minimum would be 0)")
    
    # Energy spacing
    if len(energies) > 1:
        spacings = np.diff(energies)
        print(f"• Energy spacing: ΔE = {np.mean(spacings):.4f} ≈ ℏω = {omega:.4f}")
        print(f"  (Uniform spacing - characteristic of harmonic oscillator)")
    
    # Calculate expectation values and uncertainties
    print("\n📏 Quantum Uncertainties:")
    for i in range(min(3, n_states)):
        psi = wavefunctions[:, i]
        
        # Position moments
        x_exp = np.trapz(psi.conj() * solver.x * psi, solver.x).real
        x2_exp = np.trapz(psi.conj() * solver.x**2 * psi, solver.x).real
        delta_x = np.sqrt(x2_exp - x_exp**2)
        
        # For harmonic oscillator: Δx = √((n+½)/mω) in atomic units
        delta_x_theory = np.sqrt((i + 0.5) / omega)
        
        print(f"State {i}: <x> = {x_exp:.6f}, Δx = {delta_x:.4f} (theory: {delta_x_theory:.4f})")
    
    # Probability of tunneling into classically forbidden region
    print("\n🌊 Quantum Tunneling into Forbidden Regions:")
    for i in range(min(3, n_states)):
        E = energies[i]
        x_turn = np.sqrt(2 * E / omega**2)
        
        # Probability outside classical turning points
        psi = wavefunctions[:, i]
        prob_density = np.abs(psi)**2
        
        # Integrate probability in forbidden regions
        forbidden_mask = np.abs(solver.x) > x_turn
        prob_forbidden = np.trapz(prob_density[forbidden_mask], solver.x[forbidden_mask])
        
        print(f"State {i}: P(|x| > {x_turn:.3f}) = {prob_forbidden:.4f}")
    
    plt.show()
    
    print("\n✨ Harmonic oscillator analysis complete!")
    print("Key insights:")
    print("• Quantized energy levels with uniform spacing ℏω")
    print("• Non-zero ground state energy (zero-point motion)")
    print("• Gaussian-modulated oscillatory wavefunctions")
    print("• Quantum tunneling into classically forbidden regions")


if __name__ == "__main__":
    main()
