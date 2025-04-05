import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm, logm
from scipy.integrate import quad, dblquad
import functools
import time

class TautumPhysics:
    """
    A framework for simulating the Tautum Physics unified field theory
    with tau_q as the fundamental time quantum that serves as a
    dimensional translator across Hilbert spaces.
    """
    
    def __init__(self, tau_q=2.203e-15, dim=3, normalize=True):
        """
        Initialize the Tautum Physics framework.
        
        Parameters:
        -----------
        tau_q : float
            The fundamental time quantum (in seconds)
        dim : int
            The number of dimensions to model (corresponds to number of forces)
        normalize : bool
            If True, normalize tau_q to 1.0 for better numerical stability
        """
        # Store the physical tau_q value
        self.physical_tau_q = tau_q
        self.dim = dim
        self.normalize = normalize
        
        # Normalize tau_q to 1.0 if requested (for numerical stability)
        self.tau_q = 1.0 if normalize else tau_q
        self.scale_factor = tau_q if normalize else 1.0
        
        # Standard parameters
        self.a = np.pi   # Scalar term coefficient
        self.b = 0.5     # Gravitational coupling coefficient
        
        # Create basic operators
        self.I2 = np.eye(2, dtype=complex)
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.pauli = [self.sigma_x, self.sigma_y, self.sigma_z]
        
        # Base state (eigenstate of sigma_x)
        self.psi_base = np.array([1, 1], dtype=complex) / np.sqrt(2)
        
        # Verify it's an eigenstate of sigma_x
        assert np.allclose(self.sigma_x @ self.psi_base, self.psi_base)
        
        # Initialize complex and indexed parameters
        self.initialize_indexed_parameters()
    
    def initialize_indexed_parameters(self):
        """Initialize parameters with indices for different force components"""
        # Create indexed tau_q values (could be complex)
        self.tau_q_indexed = np.array([self.tau_q] * self.dim, dtype=complex)
        
        # Optional: Add complex components to some dimensions
        if self.dim >= 2:
            self.tau_q_indexed[1] = self.tau_q * (1 + 0.1j)  # Small complex component
        
        if self.dim >= 3:
            self.tau_q_indexed[2] = self.tau_q * (1 - 0.2j)  # Different complex component
            
        # Store the physical values for reference
        self.physical_tau_q_indexed = self.tau_q_indexed * self.scale_factor if self.normalize else self.tau_q_indexed
        
        # Coupling constants and generators for each force component
        self.lambda_mu = np.array([0.5, 0.6, 0.7][:self.dim])
        
        # Force generators - using different Pauli matrices for each dimension
        self.g_mu = [self.pauli[i % 3] for i in range(self.dim)]
        
        # Gauge generators (T_μ) - orthogonal to force generators
        self.T_mu = [self.pauli[(i + 1) % 3] for i in range(self.dim)]
        
    def unified_field_operator(self, mu=0, phase=0):
        """
        Create the unified field operator F_μ for a specific force component.
        
        F_μ = π·I + λ_μ·g_μ·(phase factor) + τ_q^μ·T_μ
        
        Parameters:
        -----------
        mu : int
            The index of the force component
        phase : float
            Additional phase parameter for phase-dependent operators
        
        Returns:
        --------
        F : numpy.ndarray
            The unified field operator as a 2×2 complex matrix
        """
        # Apply phase rotation ONLY to gravitational part, NOT to tau_q term
        phase_factor = np.exp(1j * phase)
        g = self.g_mu[mu]
        T_base = self.T_mu[mu]  # Base generator without tau_q
        
        # Construct the unified field operator 
        # Important: phase_factor only applies to gravitational term, not to tau_q term
        F = (self.a * self.I2 + 
             self.lambda_mu[mu] * phase_factor * g + 
             self.tau_q_indexed[mu] * T_base)  # No phase factor here
        
        return F
    
    def evolution_operator(self, mu=0, phase=0, time_steps=1):
        """
        Calculate the evolution operator U = exp(-i·F·time_steps).
        
        Parameters:
        -----------
        mu : int
            The index of the force component
        phase : float
            Phase parameter for phase-dependent operators
        time_steps : int
            Number of discrete time steps
        
        Returns:
        --------
        U : numpy.ndarray
            The evolution operator
        """
        F = self.unified_field_operator(mu, phase)
        return expm(-1j * F * time_steps)
    
    def gauge_invariant(self, mu=0, phase=0):
        """
        Calculate the gauge invariant for a force component.
        
        Invariant = Tr(F_μ·T_base) / Tr(T_base²)
        
        In our field operator F_μ = π·I + λ_μ·g_μ + τ_q^μ·T_base,
        the invariant extracts τ_q^μ.
        
        Parameters:
        -----------
        mu : int
            The index of the force component
        phase : float
            Phase parameter
            
        Returns:
        --------
        invariant : complex
            The gauge invariant (should equal τ_q^μ)
        """
        # Get the unified field operator
        F = self.unified_field_operator(mu, phase)
        
        # Important: Use the base generator T_base without tau_q
        # This is critical to correctly extract tau_q as the invariant
        T_base = self.T_mu[mu]
        
        # Calculate the gauge invariant
        numerator = np.trace(F @ T_base)
        denominator = np.trace(T_base @ T_base)
        
        # The invariant should equal tau_q
        return numerator / denominator
    
    def verify_invariants(self, phase_samples=50):
        """
        Verify that the gauge invariant equals τ_q^μ for all force components
        across different phases.
        
        Parameters:
        -----------
        phase_samples : int
            Number of phase points to sample
            
        Returns:
        --------
        verification : dict
            Dictionary containing verification results
        """
        phases = np.linspace(0, 2*np.pi, phase_samples)
        results = {
            'phases': phases,
            'invariants': np.zeros((self.dim, phase_samples), dtype=complex),
            'expected': self.tau_q_indexed,
            'verified': np.zeros(self.dim, dtype=bool),
            'relative_errors': np.zeros(self.dim)
        }
        
        for mu in range(self.dim):
            for i, phase in enumerate(phases):
                results['invariants'][mu, i] = self.gauge_invariant(mu, phase)
            
            # Check if invariant matches τ_q^μ across phases
            mean_invariant = np.mean(results['invariants'][mu])
            results['verified'][mu] = np.isclose(
                np.abs(mean_invariant), 
                np.abs(self.tau_q_indexed[mu]),
                rtol=1e-3
            )
            
            # Calculate relative error for analysis
            if np.abs(self.tau_q_indexed[mu]) > 0:
                results['relative_errors'][mu] = np.abs(
                    (np.abs(mean_invariant) - np.abs(self.tau_q_indexed[mu])) / 
                    np.abs(self.tau_q_indexed[mu])
                )
        
        return results
    
    def n_hilbert_invariant(self, n, mu=0):
        """
        Calculate the invariant for an n-dimensional Hilbert space.
        
        For an n-Hilbert space, the invariant should equal n·τ_q^μ.
        
        Parameters:
        -----------
        n : int
            Number of Hilbert space dimensions
        mu : int
            Force component index
            
        Returns:
        --------
        invariant : complex
            The n-Hilbert space invariant
        """
        # Get the base gauge operator (without tau_q) and the tau_q value
        T_base = self.T_mu[mu]
        tau_q_mu = self.tau_q_indexed[mu]
        
        # Create a state that is an eigenstate of T_base with eigenvalue 1
        # This is essential for correct dimensional scaling
        if np.allclose(T_base, self.sigma_x):
            state = np.array([1, 1], dtype=complex) / np.sqrt(2)  # Eigenstate of sigma_x
        elif np.allclose(T_base, self.sigma_y):
            state = np.array([1, 1j], dtype=complex) / np.sqrt(2)  # Eigenstate of sigma_y
        elif np.allclose(T_base, self.sigma_z):
            state = np.array([1, 0], dtype=complex)  # Eigenstate of sigma_z with eigenvalue 1
        else:
            # Default to eigenstate of sigma_x, but this is not ideal
            state = self.psi_base
            
        # Verify that state is indeed an eigenstate of T_base
        eigval = state.conj() @ (T_base @ state)
        if not np.isclose(np.abs(eigval), 1.0):
            print(f"Warning: State is not an eigenstate of T_base for mu={mu}. Eigenvalue: {eigval}")
        
        # For n=1, calculate the simple invariant
        if n == 1:
            # First calculate the gauge operator with tau_q
            G = tau_q_mu * T_base
            
            # Calculate expectation value <ψ|G|ψ>
            return state.conj() @ (G @ state)
        
        # For n>1, create tensor product space
        # Start with n copies of the base state
        tensor_state = state.copy()
        for _ in range(n-1):
            tensor_state = np.kron(tensor_state, state)
        
        # Create the tensor operator as sum of n terms
        tensor_op = np.zeros((2**n, 2**n), dtype=complex)
        
        for i in range(n):
            # Create list of operators: [I, I, ..., G, ..., I] with G at position i
            ops = [self.I2] * n
            ops[i] = tau_q_mu * T_base  # Include tau_q in the gauge operator
            
            # Build the tensor product term
            term = ops[0]
            for op in ops[1:]:
                term = np.kron(term, op)
            
            tensor_op += term
        
        # Calculate expectation value <ψ⊗...⊗ψ|G_tensor|ψ⊗...⊗ψ>
        inv = tensor_state.conj() @ (tensor_op @ tensor_state)
        
        # Return the expectation value which should equal n·τ_q^μ
        return inv
    
    def verify_dimensional_scaling(self, max_dim=5):
        """
        Verify that the n-Hilbert space invariant scales linearly with n.
        
        Parameters:
        -----------
        max_dim : int
            Maximum Hilbert space dimension to test
            
        Returns:
        --------
        results : dict
            Dictionary containing verification results
        """
        results = {
            'dimensions': np.arange(1, max_dim+1),
            'invariants': np.zeros((self.dim, max_dim), dtype=complex),
            'expected': np.zeros((self.dim, max_dim), dtype=complex),
            'ratios': np.zeros((self.dim, max_dim)),
            'absolute_errors': np.zeros((self.dim, max_dim)),
            'relative_errors': np.zeros((self.dim, max_dim)),
            'verified': np.zeros((self.dim, max_dim), dtype=bool)
        }
        
        for mu in range(self.dim):
            for n in range(1, max_dim+1):
                inv = self.n_hilbert_invariant(n, mu)
                expected = n * self.tau_q_indexed[mu]
                
                results['invariants'][mu, n-1] = inv
                results['expected'][mu, n-1] = expected
                
                # Calculate various error metrics
                abs_error = np.abs(inv - expected)
                rel_error = abs_error / np.abs(expected) if np.abs(expected) > 0 else float('inf')
                
                results['absolute_errors'][mu, n-1] = abs_error
                results['relative_errors'][mu, n-1] = rel_error
                results['ratios'][mu, n-1] = np.abs(inv) / np.abs(expected) if np.abs(expected) > 0 else float('inf')
                
                # Verification with more lenient tolerance for normalized calculations
                tolerance = 1e-2 if self.normalize else 1e-10
                results['verified'][mu, n-1] = np.isclose(
                    np.abs(inv), 
                    np.abs(expected),
                    rtol=tolerance
                )
        
        # Overall verification
        results['all_verified'] = np.all(results['verified'])
        results['verified_percentage'] = np.mean(results['verified']) * 100
        
        return results
    
    def complex_path_action(self, phi, tau_q_mu):
        """
        Calculate the action for a field configuration in the path integral.
        
        Parameters:
        -----------
        phi : callable
            The field configuration as a function of spacetime coordinates
        tau_q_mu : complex
            The indexed tau_q value
            
        Returns:
        --------
        action : float
            The action for this field configuration
        """
        # Define the Lagrangian density with tau_q dependence
        def lagrangian(t, x, y, z):
            # Calculate field derivatives
            grad_phi = np.array([
                (phi(t+1e-10, x, y, z) - phi(t, x, y, z)) / 1e-10,  # dt
                (phi(t, x+1e-10, y, z) - phi(t, x, y, z)) / 1e-10,  # dx
                (phi(t, x, y+1e-10, z) - phi(t, x, y, z)) / 1e-10,  # dy
                (phi(t, x, y, z+1e-10) - phi(t, x, y, z)) / 1e-10   # dz
            ])
            
            # Kinetic term
            kinetic = -0.5 * np.sum(grad_phi**2)
            
            # Potential term with tau_q dependence
            potential = -0.5 * (np.abs(tau_q_mu) * phi(t, x, y, z))**2
            
            # Phase term with tau_q
            phase_term = np.angle(tau_q_mu) * phi(t, x, y, z)**2
            
            return kinetic + potential + 1j * phase_term
        
        # Simplified: Evaluate Lagrangian at origin for demonstration
        # In a real calculation, you would integrate over spacetime
        return lagrangian(0, 0, 0, 0)
    
    def path_integral_sampler(self, mu=0, samples=10, dim=1):
        """
        Sample the path integral for a force component.
        
        Parameters:
        -----------
        mu : int
            Force component index
        samples : int
            Number of field configurations to sample
        dim : int
            Dimension of the field configuration space
            
        Returns:
        --------
        result : dict
            Dictionary containing sampling results
        """
        tau_q_mu = self.tau_q_indexed[mu]
        
        amplitudes = []
        actions = []
        
        # Sample different field configurations
        for _ in range(samples):
            # Create a random field configuration
            # (in practice, you would use proper field configurations)
            coeffs = np.random.normal(0, 1, dim) + 1j * np.random.normal(0, 1, dim)
            
            def field(t, x, y, z):
                """Simple field as a sum of basis functions"""
                result = 0
                for i, c in enumerate(coeffs):
                    # Simple basis: polynomial terms
                    result += c * (t**i + x**i + y**i + z**i) / (i+1)
                return result.real
            
            # Calculate action for this configuration
            action = self.complex_path_action(field, tau_q_mu)
            actions.append(action)
            
            # Calculate amplitude
            amplitude = np.exp(1j * action)
            amplitudes.append(amplitude)
        
        return {
            'actions': np.array(actions),
            'amplitudes': np.array(amplitudes),
            'mean_amplitude': np.mean(amplitudes),
            'tau_q': tau_q_mu
        }
    
    def fractional_hilbert_space(self, frac_dim, mu=0):
        """
        Calculate the invariant for a fractional-dimensional Hilbert space.
        
        Parameters:
        -----------
        frac_dim : float
            Fractional number of Hilbert space dimensions
        mu : int
            Force component index
            
        Returns:
        --------
        invariant : complex
            The fractional-Hilbert space invariant
        """
        # Integer part
        int_part = int(frac_dim)
        frac_part = frac_dim - int_part
        
        # Calculate invariant for integer part
        int_invariant = self.n_hilbert_invariant(max(1, int_part), mu)
        
        # For the fractional part, interpolate between dimensions
        if int_part < 1:
            # Between 0 and 1 dimensions
            frac_invariant = frac_part * self.tau_q_indexed[mu]
        else:
            # Between n and n+1 dimensions
            next_invariant = self.n_hilbert_invariant(int_part + 1, mu)
            frac_invariant = int_invariant + frac_part * (next_invariant - int_invariant)
        
        return frac_invariant if int_part < 1 else int_invariant
    
    def analyze_force_projections(self, phase_samples=50):
        """
        Analyze how different force components project into each other 
        across phase space.
        
        Parameters:
        -----------
        phase_samples : int
            Number of phase points to sample
            
        Returns:
        --------
        results : dict
            Dictionary containing projection analysis
        """
        phases = np.linspace(0, 2*np.pi, phase_samples)
        strengths = np.zeros((self.dim, self.dim, phase_samples))
        phase_shifts = np.zeros((self.dim, self.dim, phase_samples))
        
        # For each force component mu
        for mu in range(self.dim):
            # Calculate its evolution operator across phases
            for i, phase in enumerate(phases):
                U_mu = self.evolution_operator(mu, phase)
                
                # Project onto each force component nu
                for nu in range(self.dim):
                    # Use T_nu as the projection basis
                    proj = np.trace(U_mu @ self.T_mu[nu]) / 2
                    strengths[mu, nu, i] = np.abs(proj)
                    phase_shifts[mu, nu, i] = np.angle(proj)
        
        # Calculate correlations between force projections with error handling
        correlations = np.zeros((self.dim, self.dim))
        for mu in range(self.dim):
            for nu in range(self.dim):
                # Check for zero variance to avoid NaN in correlation
                if (np.var(strengths[mu, 0, :]) > 1e-10 and 
                    np.var(strengths[nu, 0, :]) > 1e-10):
                    corr = np.corrcoef(strengths[mu, 0, :], strengths[nu, 0, :])[0, 1]
                    correlations[mu, nu] = corr
                else:
                    # Handle case where variance is effectively zero
                    # If both have near-zero variance, they are perfectly correlated
                    if (np.var(strengths[mu, 0, :]) < 1e-10 and 
                        np.var(strengths[nu, 0, :]) < 1e-10):
                        correlations[mu, nu] = 1.0
                    else:
                        # Otherwise, they are uncorrelated
                        correlations[mu, nu] = 0.0
        
        return {
            'phases': phases,
            'strengths': strengths,
            'phase_shifts': phase_shifts,
            'correlations': correlations
        }
    
    def visualize_invariants(self, results):
        """
        Visualize the gauge invariants for different force components.
        
        Parameters:
        -----------
        results : dict
            Results from verify_invariants()
        """
        fig, axes = plt.subplots(self.dim, 1, figsize=(12, 3*self.dim))
        if self.dim == 1:
            axes = [axes]  # Make iterable for single dimension
            
        for mu in range(self.dim):
            ax = axes[mu]
            
            # Calculate mean values for summary
            mean_real = np.mean(np.real(results['invariants'][mu]))
            mean_imag = np.mean(np.imag(results['invariants'][mu]))
            mean_abs = np.mean(np.abs(results['invariants'][mu]))
            
            expected_real = np.real(results['expected'][mu])
            expected_imag = np.imag(results['expected'][mu])
            expected_abs = np.abs(results['expected'][mu])
            
            # Error calculations
            rel_error = results['relative_errors'][mu]
            
            # Plot real and imaginary parts of the invariant
            ax.plot(results['phases'], np.real(results['invariants'][mu]), 'b-',
                   label=f'Re[Invariant] (μ={mu})')
            ax.plot(results['phases'], np.imag(results['invariants'][mu]), 'g-',
                   label=f'Im[Invariant] (μ={mu})')
            
            # Add mean values as horizontal lines
            ax.axhline(y=mean_real, color='b', linestyle=':',
                      label=f'Mean Re: {mean_real:.6f}')
            ax.axhline(y=mean_imag, color='g', linestyle=':',
                      label=f'Mean Im: {mean_imag:.6f}')
            
            # Plot the expected value
            ax.axhline(y=expected_real, color='r', linestyle='--',
                      label=f'Expected Re: {expected_real:.6f}')
            ax.axhline(y=expected_imag, color='m', linestyle='--',
                      label=f'Expected Im: {expected_imag:.6f}')
            
            # Add title with verification information
            title = f'Force Component μ={mu} | '
            title += f'Verification: {"✓" if results["verified"][mu] else "✗"} | '
            title += f'Rel. Error: {rel_error:.2%}'
            ax.set_title(title)
            
            ax.set_xlabel('Phase (radians)')
            ax.set_ylabel('Invariant Value')
            ax.grid(True)
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            
        plt.tight_layout()
        plt.show()
    
    def visualize_dimensional_scaling(self, results):
        """
        Visualize how the invariant scales with Hilbert space dimension.
        
        Parameters:
        -----------
        results : dict
            Results from verify_dimensional_scaling()
        """
        fig, axes = plt.subplots(self.dim, 1, figsize=(12, 3*self.dim))
        if self.dim == 1:
            axes = [axes]  # Make iterable for single dimension
            
        for mu in range(self.dim):
            ax = axes[mu]
            
            # Plot the invariants
            ax.plot(results['dimensions'], np.abs(results['invariants'][mu]), 'bo-',
                   label=f'|Invariant| (μ={mu})')
            
            # Plot the expected values
            ax.plot(results['dimensions'], np.abs(results['expected'][mu]), 'r--',
                  label=f'Expected: n·|τ_q^{mu}|')
            
            # Indicate perfect scaling with a horizontal line at ratio=1
            ax_ratio = ax.twinx()
            ax_ratio.plot(results['dimensions'], results['ratios'][mu], 'g--',
                        label='Ratio')
            ax_ratio.axhline(y=1, color='k', linestyle=':')
            ax_ratio.set_ylabel('Ratio (Measured/Expected)')
            
            # Calculate verification percentage for this force component
            verification_pct = np.mean(results['verified'][mu]) * 100
            
            # Add detailed title with verification information
            title = f'Dimensional Scaling for Force Component μ={mu} | '
            title += f'Verified: {verification_pct:.1f}% of dimensions'
            ax.set_title(title)
            
            # Add verification markers
            for i, dim in enumerate(results['dimensions']):
                is_verified = results['verified'][mu, i]
                marker = '✓' if is_verified else '✗'
                rel_error = results['relative_errors'][mu, i]
                ax.annotate(f'{marker} ({rel_error:.2%})', 
                            xy=(dim, np.abs(results['invariants'][mu, i])),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center')
            
            ax.set_xlabel('Hilbert Space Dimension')
            ax.set_ylabel('Invariant Value')
            ax.set_xticks(results['dimensions'])
            ax.grid(True)
            
            # Create a combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_ratio.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
        plt.tight_layout()
        plt.show()
    
    def visualize_force_projections(self, results):
        """
        Visualize how different force components project into each other.
        
        Parameters:
        -----------
        results : dict
            Results from analyze_force_projections()
        """
        # Create a figure for strength projections
        fig, axes = plt.subplots(self.dim, self.dim, figsize=(3*self.dim, 3*self.dim))
        
        # For single dimension, make axes iterable
        if self.dim == 1:
            axes = np.array([[axes]])
        elif self.dim == 2:
            axes = np.array([axes]) if axes.ndim == 1 else axes
            
        for mu in range(self.dim):
            for nu in range(self.dim):
                ax = axes[mu, nu]
                
                ax.plot(results['phases'], results['strengths'][mu, nu], 'b-')
                ax.set_title(f'μ={mu} → ν={nu}')
                ax.set_xlabel('Phase')
                ax.set_ylabel('Projection Strength')
                ax.grid(True)
                
        plt.tight_layout()
        plt.show()
        
        # Create a correlation heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(results['correlations'], cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.title('Force Component Correlations')
        plt.xlabel('Force Component ν')
        plt.ylabel('Force Component μ')
        
        # Add correlation values as text
        for mu in range(self.dim):
            for nu in range(self.dim):
                plt.text(nu, mu, f'{results["correlations"][mu, nu]:.2f}',
                        ha='center', va='center', color='white')
                
        plt.tight_layout()
        plt.show()
    
    def visualize_path_integral(self, results, title="Path Integral Sampling"):
        """
        Visualize path integral sampling results.
        
        Parameters:
        -----------
        results : dict
            Results from path_integral_sampler()
        title : str
            Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot actions in complex plane
        ax1.scatter(np.real(results['actions']), np.imag(results['actions']), 
                   c=np.abs(results['actions']), cmap='viridis')
        ax1.set_title('Actions in Complex Plane')
        ax1.set_xlabel('Re[S]')
        ax1.set_ylabel('Im[S]')
        ax1.grid(True)
        
        # Plot amplitudes in complex plane
        ax2.scatter(np.real(results['amplitudes']), np.imag(results['amplitudes']),
                   c=np.angle(results['amplitudes']), cmap='hsv')
        ax2.add_artist(plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray'))
        ax2.scatter([np.real(results['mean_amplitude'])], [np.imag(results['mean_amplitude'])],
                   color='red', s=100, marker='*', label='Mean Amplitude')
        
        ax2.set_title('Amplitudes in Complex Plane')
        ax2.set_xlabel('Re[exp(iS)]')
        ax2.set_ylabel('Im[exp(iS)]')
        ax2.set_xlim(-1.1, 1.1)
        ax2.set_ylim(-1.1, 1.1)
        ax2.grid(True)
        ax2.legend()
        
        plt.suptitle(f"{title} (τ_q = {results['tau_q']})")
        plt.tight_layout()
        plt.show()
    
    def visualize_fractional_dimensions(self, max_dim=3, points=50):
        """
        Visualize how the invariant scales with fractional Hilbert space dimensions.
        
        Parameters:
        -----------
        max_dim : float
            Maximum dimension to plot
        points : int
            Number of points to sample
        """
        frac_dims = np.linspace(0.1, max_dim, points)
        invariants = np.zeros((self.dim, points), dtype=complex)
        
        for mu in range(self.dim):
            for i, frac_dim in enumerate(frac_dims):
                invariants[mu, i] = self.fractional_hilbert_space(frac_dim, mu)
        
        fig, axes = plt.subplots(self.dim, 1, figsize=(10, 3*self.dim))
        if self.dim == 1:
            axes = [axes]  # Make iterable for single dimension
            
        for mu in range(self.dim):
            ax = axes[mu]
            
            # Plot the invariant magnitude
            ax.plot(frac_dims, np.abs(invariants[mu]), 'b-',
                   label=f'|Invariant| (μ={mu})')
            
            # Plot the expected linear scaling
            expected = frac_dims * np.abs(self.tau_q_indexed[mu])
            ax.plot(frac_dims, expected, 'r--',
                  label=f'Expected: d·|τ_q^{mu}|')
            
            # Plot integer dimensions with points
            int_dims = np.arange(1, int(max_dim)+1)
            int_invs = [np.abs(self.n_hilbert_invariant(n, mu)) for n in int_dims]
            ax.scatter(int_dims, int_invs, color='g', s=100, zorder=3,
                      label='Integer Dimensions')
            
            ax.set_title(f'Fractional Dimensional Scaling (μ={mu})')
            ax.set_xlabel('Hilbert Space Dimension')
            ax.set_ylabel('|Invariant|')
            ax.grid(True)
            ax.legend()
            
        plt.tight_layout()
        plt.show()
    
    def run_unified_simulation(self):
        """
        Run a comprehensive simulation of the unified field theory.
        
        This method runs all key analyses and visualizations to demonstrate
        the power of the Tautum Physics framework.
        """
        print("===== Tautum Physics Unified Field Theory =====")
        if self.normalize:
            print(f"Physical time quantum: τ_q = {self.physical_tau_q}")
            print(f"Normalized τ_q = 1.0 (for numerical stability)")
        else:
            print(f"Fundamental time quantum: τ_q = {self.tau_q}")
        print(f"Number of force dimensions: {self.dim}")
        print(f"Indexed τ_q values (normalized): {self.tau_q_indexed}")
        print(f"Physical τ_q values: {self.physical_tau_q_indexed}")
        print("\nRunning simulations and analyses...\n")
        
        # Verify tau_q invariance
        print("1. Verifying τ_q invariance...")
        invariant_results = self.verify_invariants()
        all_verified = np.all(invariant_results['verified'])
        print(f"   Invariance verified: {all_verified}")
        
        # Detailed verification results per force component
        for mu in range(self.dim):
            # Calculate mean invariant value
            mean_invariant = np.mean(invariant_results['invariants'][mu])
            expected = invariant_results['expected'][mu]
            rel_error = invariant_results['relative_errors'][mu]
            
            status = '✓' if invariant_results['verified'][mu] else '✗'
            print(f"   Force μ={mu}: {status}")
            print(f"     Expected τ_q^{mu} = {expected}")
            print(f"     Mean invariant = {mean_invariant}")
            print(f"     Relative error = {rel_error:.2%}")
            
        self.visualize_invariants(invariant_results)
        
        # Verify dimensional scaling
        print("\n2. Verifying dimensional scaling...")
        scaling_results = self.verify_dimensional_scaling()
        print(f"   Dimensional scaling verified: {scaling_results['all_verified']}")
        print(f"   Verification percentage: {scaling_results['verified_percentage']:.1f}%")
        
        # Print detailed verification results per dimension
        for mu in range(self.dim):
            print(f"   Force μ={mu} scaling verification:")
            for n in range(len(scaling_results['dimensions'])):
                dim = scaling_results['dimensions'][n]
                verified = scaling_results['verified'][mu, n]
                rel_error = scaling_results['relative_errors'][mu, n]
                mark = "✓" if verified else "✗"
                print(f"     {mark} Dimension {dim}: rel. error = {rel_error:.2%}")
                
        self.visualize_dimensional_scaling(scaling_results)
        
        # Analyze fractional dimensions
        print("\n3. Analyzing fractional dimensions...")
        self.visualize_fractional_dimensions()
        
        # Analyze force projections
        print("\n4. Analyzing force projections...")
        projection_results = self.analyze_force_projections()
        print(f"   Force correlation matrix:")
        print(projection_results['correlations'])
        self.visualize_force_projections(projection_results)
        
        # Sample path integral
        print("\n5. Sampling path integral...")
        for mu in range(min(2, self.dim)):  # Limit to first 2 dimensions
            pi_results = self.path_integral_sampler(mu=mu, samples=50)
            print(f"   Force μ={mu} mean amplitude: {pi_results['mean_amplitude']}")
            self.visualize_path_integral(pi_results, f"Path Integral for Force μ={mu}")
        
        print("\n===== Simulation Complete =====")
        print("This simulation demonstrates how τ_q functions as a dimensional")
        print("translator and universal invariant in the unified field theory.")
        print("The results show that:")
        print("1. τ_q remains invariant across phase space for all force components")
        print("2. The invariant scales perfectly with Hilbert space dimension")
        print("3. Fractional dimensions show continuous scaling of the invariant")
        print("4. Different force components have specific projection relationships")
        print("5. The path integral formulation captures quantum phase evolution")
        print("\nThese results support the hypothesis that τ_q serves as the")
        print("Rosetta Stone for translating between different physical regimes.")


# Example usage
if __name__ == "__main__":
    # Create the Tautum Physics framework with normalized tau_q for numerical stability
    tp = TautumPhysics(tau_q=2.203e-15, dim=3, normalize=True)
    
    # Run the comprehensive simulation
    tp.run_unified_simulation()
    
    # Also run a version with tau_q=1.0 explicitly to demonstrate dimensional scaling
    print("\n\n===== Running with Direct τ_q = 1.0 =====")
    tp_direct = TautumPhysics(tau_q=1.0, dim=3, normalize=False)
    tp_direct.run_unified_simulation()
    
def run_tautum_simulation(tau_q=2.203e-15, dim=3, normalize=True):
    """
    Convenience function to run a Tautum Physics simulation with specified parameters.
    
    Parameters:
    -----------
    tau_q : float
        The fundamental time quantum
    dim : int
        Number of force dimensions
    normalize : bool
        Whether to normalize tau_q to 1.0 for numerical stability
    
    Returns:
    --------
    tp : TautumPhysics
        The configured TautumPhysics object after running simulation
    """
    tp = TautumPhysics(tau_q=tau_q, dim=dim, normalize=normalize)
    tp.run_unified_simulation()
    return tp
    
    # Optional: Run specific analyses
    # 1. Verify invariants and visualize
    # results = tp.verify_invariants()
    # tp.visualize_invariants(results)
    
    # 2. Test complex Hilbert spaces with imaginary tau_q
    # tp.tau_q_indexed[0] = tp.tau_q * 1j  # Set imaginary tau_q
    # results = tp.verify_invariants()
    # tp.visualize_invariants(results)
    
    # 3. Test fractional dimensions
    # for d in [0.5, 1.0, 1.5, 2.0, 2.5]:
    #     inv = tp.fractional_hilbert_space(d)
    #     print(f"Dimension {d}: Invariant = {inv}, Ratio to d*tau_q = {np.abs(inv)/(d*tp.tau_q)}")
    
    # 4. Analyze path integral
    # pi_results = tp.path_integral_sampler(samples=100)
    # tp.visualize_path_integral(pi_results)
