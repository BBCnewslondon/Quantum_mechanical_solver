"""
Core solver for the time-independent Schrödinger equation in 1D.

This module implements numerical methods to solve:
-ℏ²/2m ∇²ψ + V(x)ψ = Eψ

Using finite difference methods and eigenvalue solvers.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from typing import Tuple, Optional, Callable
import warnings


class SchrodingerSolver:
    """
    Numerical solver for the 1D time-independent Schrödinger equation.
    
    Uses finite difference method to discretize the Hamiltonian and 
    scipy's sparse eigenvalue solver to find energy levels and wavefunctions.
    """
    
    def __init__(self, x_min: float = -10.0, x_max: float = 10.0, 
                 n_points: int = 1000, mass: float = 1.0, hbar: float = 1.0):
        """
        Initialize the solver with spatial grid and physical constants.
        
        Parameters:
        -----------
        x_min, x_max : float
            Spatial domain boundaries
        n_points : int
            Number of grid points
        mass : float
            Particle mass (default: 1.0 in atomic units)
        hbar : float
            Reduced Planck constant (default: 1.0 in atomic units)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.n_points = n_points
        self.mass = mass
        self.hbar = hbar
        
        # Create spatial grid
        self.x = np.linspace(x_min, x_max, n_points)
        self.dx = (x_max - x_min) / (n_points - 1)
        
        # Kinetic energy coefficient
        self.kinetic_coeff = -hbar**2 / (2 * mass * self.dx**2)
        
    def _build_hamiltonian(self, potential: np.ndarray) -> np.ndarray:
        """
        Build the Hamiltonian matrix using finite difference method.
        
        Parameters:
        -----------
        potential : np.ndarray
            Potential energy values on the grid
            
        Returns:
        --------
        np.ndarray : Hamiltonian matrix
        """
        # Kinetic energy: second derivative using finite differences
        # T = -ℏ²/2m * d²/dx²
        kinetic_diag = -2 * self.kinetic_coeff * np.ones(self.n_points)
        kinetic_off = self.kinetic_coeff * np.ones(self.n_points - 1)
        
        # Build kinetic energy matrix
        kinetic_matrix = diags(
            [kinetic_off, kinetic_diag, kinetic_off],
            [-1, 0, 1],
            shape=(self.n_points, self.n_points),
            format='csr'
        )
        
        # Add potential energy (diagonal)
        potential_matrix = diags(potential, format='csr')
        
        # Total Hamiltonian
        hamiltonian = kinetic_matrix + potential_matrix
        
        return hamiltonian
    
    def solve(self, potential_func: Callable[[np.ndarray], np.ndarray], 
              n_states: int = 5, boundary_condition: str = 'hard_wall') -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the Schrödinger equation for given potential.
        
        Parameters:
        -----------
        potential_func : callable
            Function that takes x array and returns potential V(x)
        n_states : int
            Number of lowest energy states to find
        boundary_condition : str
            Type of boundary condition ('hard_wall', 'periodic')
            
        Returns:
        --------
        tuple : (energies, wavefunctions)
            energies : np.ndarray of shape (n_states,)
            wavefunctions : np.ndarray of shape (n_points, n_states)
        """
        # Evaluate potential on grid
        potential = potential_func(self.x)
        
        # Build Hamiltonian
        H = self._build_hamiltonian(potential)
        
        # Apply boundary conditions
        if boundary_condition == 'hard_wall':
            # Zero wavefunction at boundaries
            H = H[1:-1, 1:-1]  # Remove boundary points
            
        # Solve eigenvalue problem
        try:
            energies, wavefunctions = eigsh(H, k=n_states, which='SA')
        except Exception as e:
            warnings.warn(f"Eigenvalue solver failed: {e}")
            return np.array([]), np.array([])
        
        # Sort by energy
        idx = np.argsort(energies)
        energies = energies[idx]
        wavefunctions = wavefunctions[:, idx]
        
        # Restore boundary conditions and normalize
        if boundary_condition == 'hard_wall':
            # Add zero boundary points back
            full_wavefunctions = np.zeros((self.n_points, n_states))
            full_wavefunctions[1:-1, :] = wavefunctions
            wavefunctions = full_wavefunctions
        
        # Normalize wavefunctions
        for i in range(n_states):
            norm = np.trapz(np.abs(wavefunctions[:, i])**2, self.x)
            wavefunctions[:, i] /= np.sqrt(norm)
            
            # Ensure consistent phase (first non-zero element is positive)
            first_nonzero = np.where(np.abs(wavefunctions[:, i]) > 1e-10)[0]
            if len(first_nonzero) > 0:
                if wavefunctions[first_nonzero[0], i] < 0:
                    wavefunctions[:, i] *= -1
        
        return energies, wavefunctions
    
    def transmission_coefficient(self, potential_func: Callable[[np.ndarray], np.ndarray],
                               energy: float, x_barrier_start: float, x_barrier_end: float) -> float:
        """
        Calculate transmission coefficient for quantum tunneling.
        
        Parameters:
        -----------
        potential_func : callable
            Barrier potential function
        energy : float
            Particle energy
        x_barrier_start, x_barrier_end : float
            Barrier boundaries
            
        Returns:
        --------
        float : Transmission coefficient T
        """
        # This is a simplified calculation using WKB approximation
        # For more accurate results, use scattering matrix methods
        
        # Find barrier region
        barrier_mask = (self.x >= x_barrier_start) & (self.x <= x_barrier_end)
        x_barrier = self.x[barrier_mask]
        V_barrier = potential_func(x_barrier)
        
        # Classical turning points where E < V
        classically_forbidden = V_barrier > energy
        
        if not np.any(classically_forbidden):
            return 1.0  # No tunneling needed
        
        # WKB approximation: T ≈ exp(-2∫√(2m(V-E)/ℏ²)dx)
        integrand = np.sqrt(2 * self.mass * np.maximum(0, V_barrier - energy)) / self.hbar
        integral = np.trapz(integrand[classically_forbidden], 
                           x_barrier[classically_forbidden])
        
        transmission = np.exp(-2 * integral)
        
        return min(transmission, 1.0)


def create_solver(x_range: Tuple[float, float] = (-10, 10), 
                 n_points: int = 1000) -> SchrodingerSolver:
    """
    Convenience function to create a SchrodingerSolver instance.
    
    Parameters:
    -----------
    x_range : tuple
        (x_min, x_max) spatial domain
    n_points : int
        Number of grid points
        
    Returns:
    --------
    SchrodingerSolver instance
    """
    return SchrodingerSolver(x_range[0], x_range[1], n_points)
