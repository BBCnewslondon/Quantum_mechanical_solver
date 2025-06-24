# Quantum Mechanics Solver

A comprehensive Python package for numerically solving the time-independent Schrödinger equation in one dimension. This project provides tools to study quantum mechanical systems including energy eigenvalues, wavefunctions, and quantum tunneling phenomena.

## 🌟 Features

- **Numerical Schrödinger Solver**: Finite difference method with sparse matrix eigenvalue solving
- **Multiple Potential Types**: Infinite/finite wells, harmonic oscillator, barriers, double wells, and more
- **Quantum Tunneling Analysis**: Transmission coefficient calculations and WKB approximation
- **Beautiful Visualizations**: Interactive plots using Matplotlib and Plotly
- **Analytical Comparisons**: Built-in analytical solutions for validation
- **Comprehensive Examples**: Ready-to-run demonstrations of quantum phenomena

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Quantum_mechanical_solver

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from core import create_solver
from potentials import quantum_well, harmonic
from visualization import quick_plot

# Create a solver
solver = create_solver(x_range=(-5, 5), n_points=1000)

# Define a quantum well potential
potential = quantum_well(width=2.0, depth=10.0)

# Solve the Schrödinger equation
energies, wavefunctions = solver.solve(potential, n_states=5)

# Visualize results
quick_plot(solver.x, potential(solver.x), energies, wavefunctions)
```

## 📊 Examples

### 1. Infinite Square Well
```bash
python examples/infinite_square_well.py
```
Demonstrates the classic "particle in a box" problem with analytical comparison.

### 2. Quantum Tunneling
```bash
python examples/quantum_tunneling.py
```
Explores tunneling through potential barriers with transmission coefficient analysis.

### 3. Harmonic Oscillator
```bash
python examples/harmonic_oscillator.py
```
Studies the quantum harmonic oscillator with Hermite polynomial wavefunctions.

### 4. Comprehensive Demo
```bash
python examples/comprehensive_demo.py
```
Compares multiple quantum systems in a single analysis.

## 🔬 Theory

### Time-Independent Schrödinger Equation

The solver numerically solves:

```
-ℏ²/2m ∇²ψ + V(x)ψ = Eψ
```

Using the finite difference method to discretize the second derivative:

```
d²ψ/dx² ≈ [ψ(x+Δx) - 2ψ(x) + ψ(x-Δx)] / Δx²
```

### Numerical Methods

- **Matrix Eigenvalue Problem**: Converts the differential equation to matrix form
- **Sparse Matrices**: Efficient handling of large grid sizes
- **Boundary Conditions**: Support for hard wall and periodic boundaries
- **Normalization**: Automatic wavefunction normalization
- **Convergence**: Grid-size independent results

## 📦 Project Structure

```
Quantum_mechanical_solver/
├── core/                   # Core numerical solvers
│   ├── __init__.py
│   └── schrodinger_solver.py
├── potentials/            # Potential energy functions
│   ├── __init__.py
│   └── potential_library.py
├── visualization/         # Plotting utilities
│   ├── __init__.py
│   └── quantum_plots.py
├── examples/             # Example simulations
│   ├── __init__.py
│   ├── infinite_square_well.py
│   ├── quantum_tunneling.py
│   ├── harmonic_oscillator.py
│   └── comprehensive_demo.py
├── tests/               # Unit tests
│   ├── __init__.py
│   └── test_solver.py
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## 🎯 Supported Potentials

- **Infinite Square Well**: `quantum_well(width, infinite=True)`
- **Finite Square Well**: `quantum_well(width, depth)`
- **Harmonic Oscillator**: `harmonic(omega)`
- **Potential Barrier**: `barrier(width, height)`
- **Double Well**: `PotentialLibrary.double_well(...)`
- **Triangular Well**: `PotentialLibrary.triangular_well(...)`
- **Morse Potential**: `PotentialLibrary.morse_potential(...)`
- **Custom Polynomials**: `PotentialLibrary.custom_polynomial(...)`

## 📈 Visualization Features

- Energy level diagrams
- Wavefunction plots with potential overlay
- Probability density distributions
- Interactive Plotly visualizations
- Quantum tunneling analysis plots
- Energy spectrum comparisons
- Classical turning point illustrations

## 🧪 Testing & Validation

Run the test suite to verify numerical accuracy:

```bash
python tests/test_solver.py
```

The tests compare numerical results against analytical solutions for:
- Infinite square well energies
- Harmonic oscillator eigenstates
- Wavefunction normalization
- State orthogonality

## 🔧 Physics Parameters

All calculations use atomic units by default (ℏ = m = e = 1), but the solver supports arbitrary unit systems:

```python
solver = SchrodingerSolver(
    x_min=-10, x_max=10, n_points=1000,
    mass=1.0,      # Particle mass
    hbar=1.0       # Reduced Planck constant
)
```

## 📚 Key Results & Insights

### Infinite Square Well
- Quantized energy levels: E_n ∝ n²
- Zero-point energy: E₀ > 0
- Wavefunction nodes: n-1 nodes for state n

### Harmonic Oscillator
- Uniform energy spacing: ΔE = ℏω
- Gaussian-modulated wavefunctions
- Classical turning points and quantum tunneling

### Quantum Tunneling
- Exponential transmission dependence on barrier width
- WKB approximation accuracy
- Energy-dependent tunneling probability

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional potential types
- Time-dependent Schrödinger equation
- 2D/3D extensions
- Advanced visualization features
- Performance optimizations

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with NumPy, SciPy, and Matplotlib
- Inspired by quantum mechanics textbooks and research
- Numerical methods from computational physics literature

## 📞 Support

For questions or issues:
1. Check the examples for usage patterns
2. Run tests to verify installation
3. Review the comprehensive documentation in code comments
4. Open an issue for bugs or feature requests

---

*Happy quantum computing! 🌌⚛️*
