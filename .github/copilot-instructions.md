<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Quantum Mechanics Solver - Copilot Instructions

This project implements numerical solutions for the time-independent Schrödinger equation in one dimension.

## Project Context
- **Domain**: Computational quantum mechanics and physics simulation
- **Methods**: Finite difference methods, eigenvalue solvers, numerical integration
- **Visualization**: Interactive plots for wavefunctions, energy levels, and probability densities

## Code Guidelines
- Use NumPy for efficient numerical computations
- Implement physics calculations with proper units and dimensionless variables
- Follow quantum mechanics conventions (ψ for wavefunctions, E for energy, etc.)
- Include comprehensive docstrings with physics explanations
- Use type hints for better code clarity
- Optimize performance-critical sections with Numba when appropriate

## Physics Conventions
- Use atomic units (ℏ = m = e = 1) unless specified otherwise
- Normalize wavefunctions properly (∫|ψ|²dx = 1)
- Include proper boundary conditions for different potential types
- Validate energy eigenvalues and ensure physical consistency

## Structure
- `core/`: Core numerical solvers and algorithms
- `potentials/`: Different potential energy functions
- `visualization/`: Plotting and animation utilities
- `examples/`: Example simulations and tutorials
- `tests/`: Unit tests for numerical accuracy
