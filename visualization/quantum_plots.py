"""
Visualization utilities for quantum mechanics simulations.

This module provides functions to create beautiful and informative plots of:
- Energy levels and wavefunctions
- Probability densities
- Potential energy diagrams
- Quantum tunneling animations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Tuple, Optional, List, Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Set up beautiful plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class QuantumVisualizer:
    """Main class for quantum mechanics visualizations."""
    
    def __init__(self, figsize: Tuple[float, float] = (12, 8)):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        self.figsize = figsize
        self.colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
    def plot_potential_and_wavefunctions(self, x: np.ndarray, potential: np.ndarray,
                                       energies: np.ndarray, wavefunctions: np.ndarray,
                                       n_states: Optional[int] = None,
                                       scale_factor: float = 1.0,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot potential energy curve with energy levels and wavefunctions.
        
        Parameters:
        -----------
        x : np.ndarray
            Spatial grid
        potential : np.ndarray
            Potential energy values
        energies : np.ndarray
            Energy eigenvalues
        wavefunctions : np.ndarray
            Wavefunction arrays
        n_states : int, optional
            Number of states to plot (if None, plot all)
        scale_factor : float
            Scale factor for wavefunction amplitudes
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        if n_states is None:
            n_states = len(energies)
        n_states = min(n_states, len(energies))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                      gridspec_kw={'height_ratios': [2, 1]})
        
        # Main plot: Potential + Wavefunctions
        ax1.plot(x, potential, 'k-', linewidth=2, label='Potential V(x)')
        
        # Plot energy levels and wavefunctions
        for i in range(n_states):
            color = self.colors[i % len(self.colors)]
            
            # Energy level line
            ax1.axhline(y=energies[i], color=color, linestyle='--', alpha=0.7)
            
            # Wavefunction shifted to energy level
            psi_scaled = scale_factor * wavefunctions[:, i] + energies[i]
            ax1.plot(x, psi_scaled, color=color, linewidth=2, 
                    label=f'ψ_{i}, E = {energies[i]:.3f}')
            
            # Fill under wavefunction for better visibility
            ax1.fill_between(x, energies[i], psi_scaled, alpha=0.3, color=color)
        
        ax1.set_xlabel('Position x')
        ax1.set_ylabel('Energy')
        ax1.set_title('Quantum States in Potential Well')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Probability densities
        for i in range(n_states):
            color = self.colors[i % len(self.colors)]
            prob_density = np.abs(wavefunctions[:, i])**2
            ax2.plot(x, prob_density, color=color, linewidth=2, 
                    label=f'|ψ_{i}|²')
        
        ax2.set_xlabel('Position x')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Probability Densities')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_energy_spectrum(self, energies: np.ndarray, 
                           analytical_energies: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot energy spectrum with comparison to analytical results if available.
        
        Parameters:
        -----------
        energies : np.ndarray
            Numerical energy eigenvalues
        analytical_energies : np.ndarray, optional
            Analytical energy values for comparison
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        n_states = len(energies)
        x_pos = np.arange(n_states)
        
        # Plot numerical energies
        bars1 = ax.bar(x_pos - 0.2, energies, 0.4, label='Numerical', 
                      color='steelblue', alpha=0.8)
        
        # Plot analytical energies if provided
        if analytical_energies is not None:
            bars2 = ax.bar(x_pos + 0.2, analytical_energies[:n_states], 0.4, 
                          label='Analytical', color='orange', alpha=0.8)
            
            # Add error bars
            errors = np.abs(energies - analytical_energies[:n_states])
            for i, (num, ana, err) in enumerate(zip(energies, analytical_energies[:n_states], errors)):
                ax.annotate(f'Δ={err:.1e}', (i, max(num, ana) + 0.1), 
                           ha='center', fontsize=8)
        
        ax.set_xlabel('Quantum State Number n')
        ax.set_ylabel('Energy')
        ax.set_title('Energy Spectrum')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'n={i}' for i in range(n_states)])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_tunneling_analysis(self, x: np.ndarray, potential: np.ndarray,
                              energies: List[float], transmissions: List[float],
                              barrier_start: float, barrier_end: float,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot quantum tunneling analysis.
        
        Parameters:
        -----------
        x : np.ndarray
            Spatial grid
        potential : np.ndarray
            Barrier potential
        energies : list
            Particle energies
        transmissions : list
            Transmission coefficients
        barrier_start, barrier_end : float
            Barrier boundaries
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure : The created figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # Top plot: Potential barrier with energy levels
        ax1.plot(x, potential, 'k-', linewidth=3, label='Potential Barrier')
        ax1.axvspan(barrier_start, barrier_end, alpha=0.3, color='red', 
                   label='Barrier Region')
        
        for i, E in enumerate(energies):
            ax1.axhline(y=E, color=self.colors[i % len(self.colors)], 
                       linestyle='--', label=f'E = {E:.2f}')
        
        ax1.set_xlabel('Position x')
        ax1.set_ylabel('Energy')
        ax1.set_title('Quantum Tunneling Through Potential Barrier')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Transmission coefficient vs energy
        ax2.plot(energies, transmissions, 'bo-', linewidth=2, markersize=8)
        ax2.set_xlabel('Particle Energy')
        ax2.set_ylabel('Transmission Coefficient T')
        ax2.set_title('Transmission Probability vs Energy')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Add annotations for specific points
        for i, (E, T) in enumerate(zip(energies, transmissions)):
            ax2.annotate(f'T={T:.3f}', (E, T), xytext=(10, 10), 
                        textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_plot(self, x: np.ndarray, potential: np.ndarray,
                              energies: np.ndarray, wavefunctions: np.ndarray) -> go.Figure:
        """
        Create interactive plot using Plotly.
        
        Parameters:
        -----------
        x : np.ndarray
            Spatial grid
        potential : np.ndarray
            Potential energy values
        energies : np.ndarray
            Energy eigenvalues
        wavefunctions : np.ndarray
            Wavefunction arrays
            
        Returns:
        --------
        plotly.graph_objects.Figure : Interactive figure
        """
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=['Potential and Wavefunctions', 'Probability Densities'],
                           vertical_spacing=0.12)
        
        # Add potential
        fig.add_trace(go.Scatter(x=x, y=potential, mode='lines', 
                               name='Potential V(x)', line=dict(color='black', width=3)),
                     row=1, col=1)
        
        # Add wavefunctions and probability densities
        for i in range(len(energies)):
            color = f'rgb({int(255*self.colors[i % len(self.colors)][0])}, ' + \
                   f'{int(255*self.colors[i % len(self.colors)][1])}, ' + \
                   f'{int(255*self.colors[i % len(self.colors)][2])})'
            
            # Energy level
            fig.add_hline(y=energies[i], line_dash="dash", line_color=color,
                         annotation_text=f"E_{i} = {energies[i]:.3f}", row=1, col=1)
            
            # Wavefunction
            psi_scaled = 2 * wavefunctions[:, i] + energies[i]
            fig.add_trace(go.Scatter(x=x, y=psi_scaled, mode='lines',
                                   name=f'ψ_{i}', line=dict(color=color)),
                         row=1, col=1)
            
            # Probability density
            prob_density = np.abs(wavefunctions[:, i])**2
            fig.add_trace(go.Scatter(x=x, y=prob_density, mode='lines',
                                   name=f'|ψ_{i}|²', line=dict(color=color)),
                         row=2, col=1)
        
        fig.update_layout(height=800, title_text="Quantum Mechanics Simulation")
        fig.update_xaxes(title_text="Position x", row=2, col=1)
        fig.update_yaxes(title_text="Energy", row=1, col=1)
        fig.update_yaxes(title_text="Probability Density", row=2, col=1)
        
        return fig


def quick_plot(x: np.ndarray, potential: np.ndarray, energies: np.ndarray, 
              wavefunctions: np.ndarray, n_states: int = 5) -> plt.Figure:
    """
    Quick plotting function for basic visualization.
    
    Parameters:
    -----------
    x : np.ndarray
        Spatial grid
    potential : np.ndarray
        Potential energy values
    energies : np.ndarray
        Energy eigenvalues
    wavefunctions : np.ndarray
        Wavefunction arrays
    n_states : int
        Number of states to plot
        
    Returns:
    --------
    plt.Figure : The created figure
    """
    visualizer = QuantumVisualizer()
    return visualizer.plot_potential_and_wavefunctions(
        x, potential, energies, wavefunctions, n_states
    )
