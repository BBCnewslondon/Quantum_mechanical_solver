"""
Collection of potential energy functions for quantum mechanics simulations.

This module provides various 1D potentials commonly used in quantum mechanics:
- Infinite square well
- Finite square well  
- Harmonic oscillator
- Double well potential
- Potential barriers
- Coulomb potential
"""

import numpy as np
from typing import Callable, Union
import warnings


class PotentialLibrary:
    """Library of common quantum mechanical potentials."""
    
    @staticmethod
    def infinite_square_well(width: float = 2.0, center: float = 0.0, 
                           height: float = 1e6) -> Callable[[np.ndarray], np.ndarray]:
        """
        Infinite square well potential.
        
        Parameters:
        -----------
        width : float
            Well width
        center : float
            Well center position
        height : float
            Potential height outside well (approximating infinity)
            
        Returns:
        --------
        callable : Potential function V(x)
        """
        def potential(x):
            V = np.full_like(x, height)
            well_region = np.abs(x - center) <= width / 2
            V[well_region] = 0.0
            return V
        return potential
    
    @staticmethod
    def finite_square_well(width: float = 2.0, depth: float = 10.0, 
                          center: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
        """
        Finite square well potential.
        
        Parameters:
        -----------
        width : float
            Well width
        depth : float
            Well depth (positive value)
        center : float
            Well center position
            
        Returns:
        --------
        callable : Potential function V(x)
        """
        def potential(x):
            V = np.zeros_like(x)
            well_region = np.abs(x - center) <= width / 2
            V[~well_region] = 0.0
            V[well_region] = -depth
            return V
        return potential
    
    @staticmethod
    def harmonic_oscillator(omega: float = 1.0, mass: float = 1.0, 
                          center: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
        """
        Harmonic oscillator potential: V(x) = ½mω²(x-x₀)².
        
        Parameters:
        -----------
        omega : float
            Angular frequency
        mass : float
            Particle mass
        center : float
            Equilibrium position
            
        Returns:
        --------
        callable : Potential function V(x)
        """
        def potential(x):
            return 0.5 * mass * omega**2 * (x - center)**2
        return potential
    
    @staticmethod
    def double_well(separation: float = 4.0, barrier_height: float = 5.0,
                   well_depth: float = 3.0, center: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
        """
        Double well potential with central barrier.
        
        Parameters:
        -----------
        separation : float
            Distance between well centers
        barrier_height : float
            Central barrier height
        well_depth : float
            Individual well depths
        center : float
            System center position
            
        Returns:
        --------
        callable : Potential function V(x)
        """
        def potential(x):
            x_rel = x - center
            # Create two wells
            left_well = np.exp(-((x_rel + separation/2)**2) / 0.5)
            right_well = np.exp(-((x_rel - separation/2)**2) / 0.5)
            # Central barrier
            barrier = barrier_height * np.exp(-(x_rel**2) / 0.2)
            
            V = barrier - well_depth * (left_well + right_well)
            return V
        return potential
    
    @staticmethod
    def potential_barrier(width: float = 1.0, height: float = 5.0,
                         center: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
        """
        Rectangular potential barrier.
        
        Parameters:
        -----------
        width : float
            Barrier width
        height : float
            Barrier height
        center : float
            Barrier center position
            
        Returns:
        --------
        callable : Potential function V(x)
        """
        def potential(x):
            V = np.zeros_like(x)
            barrier_region = np.abs(x - center) <= width / 2
            V[barrier_region] = height
            return V
        return potential
    
    @staticmethod
    def triangular_well(width: float = 4.0, depth: float = 8.0,
                       center: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
        """
        Triangular potential well.
        
        Parameters:
        -----------
        width : float
            Well width
        depth : float
            Well depth at center
        center : float
            Well center position
            
        Returns:
        --------
        callable : Potential function V(x)
        """
        def potential(x):
            x_rel = x - center
            V = np.zeros_like(x)
            well_region = np.abs(x_rel) <= width / 2
            V[well_region] = -depth * (1 - 2 * np.abs(x_rel[well_region]) / width)
            return V
        return potential
    
    @staticmethod
    def coulomb_potential(charge: float = 1.0, center: float = 0.0,
                         cutoff: float = 0.1) -> Callable[[np.ndarray], np.ndarray]:
        """
        Coulomb potential with cutoff to avoid singularity.
        
        Parameters:
        -----------
        charge : float
            Charge strength
        center : float
            Charge position
        cutoff : float
            Minimum distance to avoid singularity
            
        Returns:
        --------
        callable : Potential function V(x)
        """
        def potential(x):
            r = np.abs(x - center)
            r = np.maximum(r, cutoff)  # Avoid singularity
            return -charge / r
        return potential
    
    @staticmethod
    def morse_potential(depth: float = 5.0, alpha: float = 1.0, 
                       equilibrium: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
        """
        Morse potential for molecular vibrations.
        
        Parameters:
        -----------
        depth : float
            Potential well depth
        alpha : float
            Potential width parameter
        equilibrium : float
            Equilibrium position
            
        Returns:
        --------
        callable : Potential function V(x)
        """
        def potential(x):
            r = x - equilibrium
            return depth * (1 - np.exp(-alpha * r))**2 - depth
        return potential
    
    @staticmethod
    def custom_polynomial(coefficients: list, center: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
        """
        Custom polynomial potential.
        
        Parameters:
        -----------
        coefficients : list
            Polynomial coefficients [c₀, c₁, c₂, ...] for c₀ + c₁x + c₂x² + ...
        center : float
            Expansion center
            
        Returns:
        --------
        callable : Potential function V(x)
        """
        def potential(x):
            x_rel = x - center
            V = np.zeros_like(x)
            for i, coeff in enumerate(coefficients):
                V += coeff * x_rel**i
            return V
        return potential


# Convenience functions for quick access
def quantum_well(width: float = 2.0, depth: float = 10.0, 
                infinite: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a quantum well potential.
    
    Parameters:
    -----------
    width : float
        Well width
    depth : float
        Well depth (ignored if infinite=True)
    infinite : bool
        Whether to create infinite or finite well
        
    Returns:
    --------
    callable : Potential function
    """
    if infinite:
        return PotentialLibrary.infinite_square_well(width=width)
    else:
        return PotentialLibrary.finite_square_well(width=width, depth=depth)


def harmonic(omega: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create harmonic oscillator potential.
    
    Parameters:
    -----------
    omega : float
        Angular frequency
        
    Returns:
    --------
    callable : Potential function
    """
    return PotentialLibrary.harmonic_oscillator(omega=omega)


def barrier(width: float = 1.0, height: float = 5.0) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create potential barrier for tunneling studies.
    
    Parameters:
    -----------
    width : float
        Barrier width
    height : float
        Barrier height
        
    Returns:
    --------
    callable : Potential function
    """
    return PotentialLibrary.potential_barrier(width=width, height=height)
