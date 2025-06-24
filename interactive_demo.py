"""
Interactive Quantum Mechanics Demo

A Jupyter notebook-style demo that showcases the quantum solver capabilities.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import create_solver
from potentials import quantum_well, harmonic, barrier, PotentialLibrary
from visualization import QuantumVisualizer


def demo_quantum_systems():
    """Demonstrate multiple quantum systems."""
    print("🎭 Quantum Mechanics: Interactive Demo")
    print("=" * 60)
    
    # Create solver
    solver = create_solver(x_range=(-6, 6), n_points=1000)
    visualizer = QuantumVisualizer(figsize=(12, 8))
    
    # Define quantum systems to demonstrate
    systems = [
        {
            'name': 'Infinite Square Well',
            'potential': quantum_well(width=3.0, infinite=True),
            'description': 'Classic particle-in-a-box with infinite walls'
        },
        {
            'name': 'Harmonic Oscillator', 
            'potential': harmonic(omega=0.8),
            'description': 'Quadratic potential like a quantum spring'
        },
        {
            'name': 'Double Well',
            'potential': PotentialLibrary.double_well(separation=4.0, barrier_height=5.0),
            'description': 'Two wells separated by a barrier - tunneling dynamics'
        }
    ]
    
    results = {}
    
    # Solve each system
    for i, system in enumerate(systems):
        print(f"\n🔬 System {i+1}: {system['name']}")
        print(f"   {system['description']}")
        
        try:
            energies, wavefunctions = solver.solve(system['potential'], n_states=5)
            potential_values = system['potential'](solver.x)
            
            if len(energies) > 0:
                print(f"   ✅ Found {len(energies)} energy states")
                print(f"   📊 Ground state energy: E₀ = {energies[0]:.4f}")
                
                if len(energies) > 1:
                    print(f"   📈 First excitation energy: ΔE = {energies[1] - energies[0]:.4f}")
                
                # Store results
                results[system['name']] = {
                    'energies': energies,
                    'wavefunctions': wavefunctions,
                    'potential': potential_values,
                    'x': solver.x
                }
                
                # Save visualization
                fig = visualizer.plot_potential_and_wavefunctions(
                    solver.x, potential_values, energies, wavefunctions, 
                    n_states=min(4, len(energies)), scale_factor=0.8
                )
                
                filename = f"quantum_system_{i+1}_{system['name'].lower().replace(' ', '_')}.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   💾 Saved visualization: {filename}")
                
            else:
                print("   ❌ No bound states found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Comparative analysis
    print(f"\n📊 Comparative Analysis")
    print("-" * 40)
    
    if results:
        print("System                | Ground State | First Gap | States Found")
        print("-" * 65)
        
        for name, data in results.items():
            energies = data['energies']
            gap = energies[1] - energies[0] if len(energies) > 1 else 'N/A'
            gap_str = f"{gap:.4f}" if gap != 'N/A' else 'N/A'
            print(f"{name:<20} | {energies[0]:>11.4f} | {gap_str:>9} | {len(energies):>11}")
    
    # Quantum tunneling demonstration
    print(f"\n🌊 Quantum Tunneling Analysis")
    print("-" * 40)
    
    # Create a barrier for tunneling analysis
    barrier_func = barrier(width=1.5, height=4.0)
    test_energies = np.linspace(0.5, 6.0, 12)
    transmissions = []
    
    print("Energy | Transmission | Classical Prediction")
    print("-" * 45)
    
    for E in test_energies:
        T = solver.transmission_coefficient(barrier_func, E, -0.75, 0.75)
        classical = 1.0 if E > 4.0 else 0.0  # Classical over-barrier vs blocked
        transmissions.append(T)
        
        print(f"{E:>6.2f} | {T:>11.4f} | {classical:>17.1f}")
    
    # Save tunneling analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test_energies, transmissions, 'bo-', label='Quantum Tunneling')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Classical Limit')
    ax.axvline(x=4.0, color='gray', linestyle=':', label='Barrier Height')
    ax.set_xlabel('Particle Energy')
    ax.set_ylabel('Transmission Coefficient')
    ax.set_title('Quantum Tunneling vs Classical Physics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('quantum_tunneling_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n💾 Saved tunneling analysis: quantum_tunneling_analysis.png")
    
    # Physical insights
    print(f"\n💡 Key Quantum Mechanical Insights")
    print("-" * 40)
    print("• Energy quantization: Only discrete energy levels allowed")
    print("• Zero-point energy: Ground state energy > 0 (uncertainty principle)")
    print("• Wave-particle duality: Particles have wavelike properties")
    print("• Quantum tunneling: Particles can pass through barriers classically forbidden")
    print("• Boundary conditions: Shape of container affects allowed wavelengths")
    print("• Normalization: Total probability = 1 (∫|ψ|²dx = 1)")
    
    # Suggest next steps
    print(f"\n🚀 Next Steps")
    print("-" * 20)
    print("1. Run individual examples for detailed analysis:")
    print("   python examples/infinite_square_well.py")
    print("   python examples/quantum_tunneling.py")
    print("   python examples/comprehensive_demo.py")
    print("")
    print("2. Modify parameters in the code to explore:")
    print("   - Different potential shapes and depths")
    print("   - Varying grid resolution for convergence studies")
    print("   - Custom potential functions")
    print("")
    print("3. Extend the solver for:")
    print("   - Time-dependent problems")
    print("   - 2D/3D systems")
    print("   - Particle interactions")
    
    print(f"\n✨ Demo complete! Check the generated PNG files for visualizations.")


if __name__ == "__main__":
    demo_quantum_systems()
