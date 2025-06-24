"""
Unit tests for the quantum mechanics solver.

Tests numerical accuracy against known analytical solutions.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import create_solver
from potentials import quantum_well, harmonic


class TestSchrodingerSolver(unittest.TestCase):
    """Test cases for the Schrödinger equation solver."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.solver = create_solver(x_range=(-5, 5), n_points=1000)
        self.tolerance = 1e-2  # 1% tolerance for numerical accuracy
    
    def test_infinite_square_well(self):
        """Test infinite square well against analytical solution."""
        width = 2.0
        potential = quantum_well(width=width, infinite=True)
        
        # Solve numerically
        energies, wavefunctions = self.solver.solve(potential, n_states=4)
        
        # Analytical energies: E_n = n²π²/(2mL²) for n=1,2,3,4
        analytical_energies = np.array([1, 4, 9, 16]) * np.pi**2 / (2 * width**2)
        
        # Check energy accuracy
        for i, (E_num, E_ana) in enumerate(zip(energies, analytical_energies)):
            relative_error = abs(E_num - E_ana) / E_ana
            self.assertLess(relative_error, self.tolerance, 
                          f"Energy level {i+1} error too large: {relative_error}")
        
        # Check normalization
        for i in range(len(energies)):
            norm = np.trapz(np.abs(wavefunctions[:, i])**2, self.solver.x)
            self.assertAlmostEqual(norm, 1.0, places=2, 
                                 msg=f"Wavefunction {i} not properly normalized")
    
    def test_harmonic_oscillator(self):
        """Test harmonic oscillator against analytical solution."""
        omega = 1.0
        potential = harmonic(omega=omega)
        
        # Solve numerically
        energies, wavefunctions = self.solver.solve(potential, n_states=5)
        
        # Analytical energies: E_n = ℏω(n + 1/2) for n=0,1,2,3,4
        analytical_energies = omega * (np.arange(5) + 0.5)
        
        # Check energy accuracy
        for i, (E_num, E_ana) in enumerate(zip(energies, analytical_energies)):
            relative_error = abs(E_num - E_ana) / E_ana
            self.assertLess(relative_error, self.tolerance,
                          f"Harmonic oscillator energy {i} error too large: {relative_error}")
        
        # Check ground state properties
        psi_0 = wavefunctions[:, 0]
        
        # Ground state should be centered at x=0
        x_exp = np.trapz(psi_0.conj() * self.solver.x * psi_0, self.solver.x).real
        self.assertAlmostEqual(x_exp, 0.0, places=1, 
                             msg="Ground state not centered at origin")
    
    def test_finite_square_well(self):
        """Test finite square well has bound states."""
        width = 2.0
        depth = 10.0
        potential = quantum_well(width=width, depth=depth)
        
        # Solve numerically
        energies, wavefunctions = self.solver.solve(potential, n_states=3)
        
        # Should find at least one bound state
        self.assertGreater(len(energies), 0, "No bound states found in finite well")
        
        # All energies should be negative (bound states)
        for i, E in enumerate(energies):
            self.assertLess(E, 0, f"Energy level {i} is not bound (E >= 0)")
        
        # Energies should be ordered
        for i in range(len(energies) - 1):
            self.assertLess(energies[i], energies[i+1], 
                          "Energy levels not properly ordered")
    
    def test_normalization(self):
        """Test that wavefunctions are properly normalized."""
        potential = harmonic(omega=1.0)
        energies, wavefunctions = self.solver.solve(potential, n_states=3)
        
        for i in range(len(energies)):
            norm = np.trapz(np.abs(wavefunctions[:, i])**2, self.solver.x)
            self.assertAlmostEqual(norm, 1.0, places=2,
                                 msg=f"Wavefunction {i} normalization error")
    
    def test_orthogonality(self):
        """Test that wavefunctions are orthogonal."""
        potential = harmonic(omega=1.0)
        energies, wavefunctions = self.solver.solve(potential, n_states=4)
        
        # Check orthogonality between different states
        for i in range(len(energies)):
            for j in range(i+1, len(energies)):
                overlap = np.trapz(wavefunctions[:, i].conj() * wavefunctions[:, j], 
                                 self.solver.x)
                self.assertAlmostEqual(abs(overlap), 0.0, places=1,
                                     msg=f"States {i} and {j} not orthogonal")
    
    def test_solver_stability(self):
        """Test solver stability with different parameters."""
        # Test with different grid sizes
        for n_points in [500, 1000, 1500]:
            solver = create_solver(x_range=(-4, 4), n_points=n_points)
            potential = harmonic(omega=1.0)
            energies, _ = solver.solve(potential, n_states=2)
            
            # Ground state energy should converge to 0.5
            if len(energies) > 0:
                relative_error = abs(energies[0] - 0.5) / 0.5
                self.assertLess(relative_error, 0.05,
                              f"Ground state energy unstable with {n_points} points")


class TestPotentials(unittest.TestCase):
    """Test cases for potential functions."""
    
    def test_infinite_well_shape(self):
        """Test infinite square well potential shape."""
        x = np.linspace(-3, 3, 100)
        potential = quantum_well(width=2.0, infinite=True)
        V = potential(x)
        
        # Should be high outside the well
        outside_well = np.abs(x) > 1.0
        self.assertTrue(np.all(V[outside_well] > 1000), 
                       "Infinite well not high enough outside")
        
        # Should be zero inside the well
        inside_well = np.abs(x) <= 1.0
        self.assertTrue(np.all(V[inside_well] == 0), 
                       "Infinite well not zero inside")
    
    def test_harmonic_potential_shape(self):
        """Test harmonic oscillator potential shape."""
        x = np.linspace(-2, 2, 100)
        omega = 1.5
        potential = harmonic(omega=omega)
        V = potential(x)
        
        # Should be minimum at x=0
        min_idx = np.argmin(V)
        self.assertAlmostEqual(x[min_idx], 0.0, places=1,
                             msg="Harmonic potential minimum not at origin")
        
        # Should increase quadratically
        V_expected = 0.5 * omega**2 * x**2
        np.testing.assert_allclose(V, V_expected, rtol=1e-10,
                                  err_msg="Harmonic potential shape incorrect")


def run_accuracy_benchmark():
    """Run accuracy benchmark against analytical solutions."""
    print("🧪 Running Accuracy Benchmark")
    print("=" * 40)
    
    # Test infinite square well accuracy vs grid size
    widths = [2.0]
    grid_sizes = [500, 1000, 1500, 2000]
    
    print("Infinite Square Well Convergence:")
    print("Grid Size | Ground State Error | First Excited Error")
    print("-" * 50)
    
    for n_points in grid_sizes:
        solver = create_solver(x_range=(-3, 3), n_points=n_points)
        potential = quantum_well(width=2.0, infinite=True)
        energies, _ = solver.solve(potential, n_states=2)
        
        if len(energies) >= 2:
            # Analytical energies for L=2
            E_0_ana = np.pi**2 / 8  # n=1 state
            E_1_ana = 4 * np.pi**2 / 8  # n=2 state
            
            error_0 = abs(energies[0] - E_0_ana) / E_0_ana * 100
            error_1 = abs(energies[1] - E_1_ana) / E_1_ana * 100
            
            print(f"{n_points:>9} | {error_0:>17.3e}% | {error_1:>18.3e}%")
    
    print("\n✅ Benchmark complete")


if __name__ == "__main__":
    # Run accuracy benchmark
    run_accuracy_benchmark()
    
    # Run unit tests
    print("\n🧪 Running Unit Tests...")
    unittest.main(verbosity=2, exit=False)
