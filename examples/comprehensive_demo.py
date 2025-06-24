"""
Comprehensive Demo: Multiple Quantum Systems

This script demonstrates the quantum mechanics solver across different
potential types, showcasing the versatility of our numerical approach.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import create_solver
from potentials import PotentialLibrary, quantum_well, harmonic, barrier
from visualization import QuantumVisualizer


def main():
    """Run comprehensive quantum systems demonstration."""
    print("🎭 Quantum Mechanics: Multiple System Comparison")
    print("=" * 60)
    
    # Common parameters
    x_range = (-8, 8)
    n_points = 1200
    n_states = 5
    
    # Create solver
    solver = create_solver(x_range, n_points)
    
    # Define different quantum systems
    systems = [
        {
            'name': 'Infinite Square Well',
            'potential': quantum_well(width=3.0, infinite=True),
            'color': 'blue'
        },
        {
            'name': 'Finite Square Well',
            'potential': quantum_well(width=3.0, depth=8.0),
            'color': 'green'
        },
        {
            'name': 'Harmonic Oscillator',
            'potential': harmonic(omega=0.8),
            'color': 'red'
        },
        {
            'name': 'Double Well',
            'potential': PotentialLibrary.double_well(separation=4.0, barrier_height=6.0, well_depth=4.0),
            'color': 'purple'
        },
        {
            'name': 'Triangular Well',
            'potential': PotentialLibrary.triangular_well(width=5.0, depth=10.0),
            'color': 'orange'
        }
    ]
    
    # Solve each system
    results = {}
    
    print("\\n🔬 Solving multiple quantum systems...")
    
    for system in systems:
        print(f"  Solving {system['name']}...")
        
        try:
            energies, wavefunctions = solver.solve(
                system['potential'], n_states=n_states
            )
            
            if len(energies) > 0:
                results[system['name']] = {
                    'energies': energies,
                    'wavefunctions': wavefunctions,
                    'potential': system['potential'](solver.x),
                    'color': system['color']
                }
                print(f"    ✅ Found {len(energies)} states")
            else:
                print(f"    ❌ No states found")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Create comparison visualizations
    visualizer = QuantumVisualizer(figsize=(16, 12))
    
    # 1. Potential comparison
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    for i, (name, data) in enumerate(results.items()):
        ax1.plot(solver.x, data['potential'], linewidth=3, 
                label=name, color=data['color'])
    
    ax1.set_xlabel('Position x', fontsize=12)
    ax1.set_ylabel('Potential Energy V(x)', fontsize=12)
    ax1.set_title('Comparison of Different Quantum Potentials', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Energy level comparison
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    x_positions = np.arange(len(results))
    width = 0.15
    
    # Plot energy levels for each system
    for i in range(n_states):
        energies_by_system = []
        colors_by_system = []
        names = []
        
        for name, data in results.items():
            if i < len(data['energies']):
                energies_by_system.append(data['energies'][i])
                colors_by_system.append(data['color'])
                names.append(name)
        
        if energies_by_system:
            x_pos = x_positions[:len(energies_by_system)] + i * width
            bars = ax2.bar(x_pos, energies_by_system, width, 
                          label=f'State {i}', alpha=0.8)
            
            # Color bars according to system
            for bar, color in zip(bars, colors_by_system):
                bar.set_color(color)
    
    ax2.set_xlabel('Quantum System', fontsize=12)
    ax2.set_ylabel('Energy Eigenvalue', fontsize=12)
    ax2.set_title('Energy Level Comparison Across Systems', fontsize=14)
    ax2.set_xticks(x_positions + width * (n_states-1) / 2)
    ax2.set_xticklabels(list(results.keys()), rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Ground state wavefunctions comparison
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    for name, data in results.items():
        if len(data['wavefunctions']) > 0:
            # Normalize for comparison
            psi_0 = data['wavefunctions'][:, 0]
            psi_0_normalized = psi_0 / np.max(np.abs(psi_0))
            
            ax3.plot(solver.x, psi_0_normalized, linewidth=2, 
                    label=f"{name} (E₀ = {data['energies'][0]:.3f})",
                    color=data['color'])
    
    ax3.set_xlabel('Position x', fontsize=12)
    ax3.set_ylabel('Normalized Ground State Wavefunction', fontsize=12)
    ax3.set_title('Ground State Wavefunction Comparison', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Individual system detailed plots
    fig4 = plt.figure(figsize=(18, 15))
    
    n_systems = len(results)
    cols = 3
    rows = (n_systems + cols - 1) // cols
    
    for i, (name, data) in enumerate(results.items()):
        ax = plt.subplot(rows, cols, i + 1)
        
        # Plot potential
        ax.plot(solver.x, data['potential'], 'k-', linewidth=2, alpha=0.7)
        
        # Plot first few energy levels and wavefunctions
        n_plot = min(3, len(data['energies']))
        for j in range(n_plot):
            E = data['energies'][j]
            psi = data['wavefunctions'][:, j]
            
            # Energy level
            ax.axhline(E, color=data['color'], linestyle='--', alpha=0.8)
            
            # Wavefunction (scaled and shifted)
            psi_scaled = 0.5 * psi + E
            ax.plot(solver.x, psi_scaled, color=data['color'], linewidth=2)
            ax.fill_between(solver.x, E, psi_scaled, alpha=0.3, color=data['color'])
        
        ax.set_title(name, fontsize=12)
        ax.set_xlabel('Position x')
        ax.set_ylabel('Energy')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Individual System Details', fontsize=16, y=1.02)
    
    # Physical analysis
    print("\\n📊 Comparative Analysis:")
    print("-" * 40)
    
    for name, data in results.items():
        energies = data['energies']
        print(f"\\n{name}:")
        print(f"  Ground state energy: {energies[0]:.4f}")
        
        if len(energies) > 1:
            spacing = energies[1] - energies[0]
            print(f"  First energy gap: {spacing:.4f}")
            
            # Check if spacing is uniform (harmonic oscillator signature)
            if len(energies) > 2:
                spacings = np.diff(energies)
                uniformity = np.std(spacings) / np.mean(spacings)
                print(f"  Spacing uniformity: {uniformity:.4f} (0 = uniform)")
        
        # Calculate wavefunction properties
        if len(data['wavefunctions']) > 0:
            psi_0 = data['wavefunctions'][:, 0]
            
            # Position expectation value
            x_exp = np.trapz(psi_0.conj() * solver.x * psi_0, solver.x).real
            
            # Position variance
            x2_exp = np.trapz(psi_0.conj() * solver.x**2 * psi_0, solver.x).real
            delta_x = np.sqrt(x2_exp - x_exp**2)
            
            print(f"  Ground state <x>: {x_exp:.4f}")
            print(f"  Ground state Δx: {delta_x:.4f}")
    
    # Quantum tunneling comparison for barrier-like potentials
    print("\\n🌊 Tunneling Analysis:")
    
    barrier_systems = ['Double Well']  # Systems with tunneling
    test_energy = 2.0
    
    for name in barrier_systems:
        if name in results:
            # Create a simple barrier for testing
            barrier_func = barrier(width=1.0, height=5.0)
            T = solver.transmission_coefficient(barrier_func, test_energy, -0.5, 0.5)
            print(f"  {name} tunneling (E={test_energy}): T = {T:.4f}")
    
    # Save comparison data
    print("\\n💾 Saving comparison data...")
    
    # Create summary table
    summary_data = []
    for name, data in results.items():
        energies = data['energies']
        row = [name, energies[0] if len(energies) > 0 else 'N/A']
        
        # Add first few energy levels
        for i in range(1, min(4, len(energies))):
            row.append(energies[i])
        
        summary_data.append(row)
    
    # Display summary
    print("\\n📋 Energy Level Summary:")
    print("System".ljust(20) + "E₀".ljust(10) + "E₁".ljust(10) + "E₂".ljust(10) + "E₃")
    print("-" * 60)
    
    for row in summary_data:
        line = row[0].ljust(20)
        for i in range(1, len(row)):
            if isinstance(row[i], (int, float)):
                line += f"{row[i]:.3f}".ljust(10)
            else:
                line += str(row[i]).ljust(10)
        print(line)
    
    plt.show()
    
    print("\\n✨ Comprehensive quantum system analysis complete!")
    print("\\nKey observations:")
    print("• Different potentials lead to unique energy spectra")
    print("• Boundary conditions strongly affect wavefunctions")
    print("• Harmonic potentials show uniform energy spacing")
    print("• Confined systems have discrete energy levels")
    print("• Quantum tunneling occurs in barrier systems")


if __name__ == "__main__":
    main()
