"""
Quantum Mechanics Solver

A comprehensive package for solving the time-independent Schrödinger equation
in one dimension with various potential energy functions.
"""

__version__ = "1.0.0"
__author__ = "Quantum Mechanics Solver Team"
__description__ = "Numerical solutions for 1D quantum mechanical systems"

# Import main modules for easy access
from core import SchrodingerSolver, create_solver
from potentials import PotentialLibrary, quantum_well, harmonic, barrier
from visualization import QuantumVisualizer, quick_plot

__all__ = [
    'SchrodingerSolver',
    'create_solver', 
    'PotentialLibrary',
    'quantum_well',
    'harmonic',
    'barrier',
    'QuantumVisualizer',
    'quick_plot'
]
