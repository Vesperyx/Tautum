import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

class UnifiedSchrodingerSimulator:
    """
    Simulates the Schrödinger equation simultaneously across three Hilbert spaces
    with tau_q as the coupling constant.
    """
    
    def __init__(self, tau_q=2.203e-15, n_points=100, x_max=10.0, potential_type='harmonic'):
        """
        Initialize the simulator.
        
        Parameters:
        -----------
        tau_q : float
            The fundamental time quantum (in seconds)
        n_points : int
            Number of spatial grid points
        x_max : float
            Maximum spatial extent
        potential_type : str
            Type of potential ('harmonic', 'well', or 'barrier')
        """
        self.tau_q = tau_q
        self.n_points = n_points
        self.x_max = x_max
        
        # Planck's constant (normalized)
        self.hbar = 1.0
        
        # Mass parameters (different for each force)
        self.m1 = 1.0    # EM space
        self.m2 = 10.0   # Weak space
        self.m3 = 100.0  # Strong space
        
        # Spatial grid
        self.x = np.linspace(-x_max, x_max, n_points)
        self.dx = self.x[1] - self.x[0]
        
        # Set up potentials
        self.potential_type = potential_type
        self.setup_potentials()
        
        # Set up initial wave functions
        self.setup_initial_states()
        
        # Set up Hamiltonian operators
        self.setup_hamiltonians()
        
        # Results storage
        self.times = None
        self.psi_history = None
    
    def setup_potentials(self):
        """Set up potential energy functions for each Hilbert space."""
        x = self.x
        
        if self.potential_type == 'harmonic':
            # Harmonic oscillator potentials (different frequencies)
            self.V1 = 0.5 * (x**2)           # EM space
            self.V2 = 0.3 * (x**2)           # Weak space
            self.V3 = 0.1 * (x**2)           # Strong space
        
        elif self.potential_type == 'well':
            # Potential wells (different widths)
            self.V1 = np.zeros_like(x)
            self.V1[np.abs(x) > 2.0] = 10.0  # EM space
            
            self.V2 = np.zeros_like(x)
            self.V2[np.abs(x) > 4.0] = 10.0  # Weak space
            
            self.V3 = np.zeros_like(x)
            self.V3[np.abs(x) > 6.0] = 10.0  # Strong space
        
        elif self.potential_type == 'barrier':
            # Potential barriers (different heights)
            self.V1 = np.zeros_like(x)
            self.V1[np.abs(x) < 0.5] = 5.0   # EM space
            
            self.V2 = np.zeros_like(x)
            self.V2[np.abs(x) < 1.0] = 3.0   # Weak space
            
            self.V3 = np.zeros_like(x)
            self.V3[np.abs(x) < 1.5] = 1.0   # Strong space
    
    def setup_initial_states(self):
        """Set up initial wave functions for each Hilbert space."""
        x = self.x
        
        # Gaussian wave packets (initially only in EM space)
        sigma = 0.5
        
        # EM space: Gaussian centered at x=-2
        self.psi1_0 = np.exp(-(x+2)**2 / (2*sigma**2))
        self.psi1_0 = self.psi1_0 / np.sqrt(np.sum(np.abs(self.psi1_0)**2) * self.dx)
        
        # Weak space: initially empty
        self.psi2_0 = np.zeros_like(x, dtype=complex)
        
        # Strong space: initially empty
        self.psi3_0 = np.zeros_like(x, dtype=complex)
        
        # Combined initial state
        self.psi_0 = np.concatenate([self.psi1_0, self.psi2_0, self.psi3_0])
    
    def setup_hamiltonians(self):
        """Set up Hamiltonian operators for each Hilbert space and coupling terms."""
        n = self.n_points
        dx = self.dx
        
        # Kinetic energy operator (second derivative)
        diag = np.ones(n) * (-2)
        off_diag = np.ones(n-1)
        D2 = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        D2 = D2 / (dx**2)
        
        # Individual Hamiltonians
        self.H1 = -0.5 * (self.hbar**2 / self.m1) * D2 + np.diag(self.V1)
        self.H2 = -0.5 * (self.hbar**2 / self.m2) * D2 + np.diag(self.V2)
        self.H3 = -0.5 * (self.hbar**2 / self.m3) * D2 + np.diag(self.V3)
        
        # Full Hamiltonian (block diagonal without coupling)
        self.H_uncoupled = np.zeros((3*n, 3*n), dtype=complex)
        self.H_uncoupled[:n, :n] = self.H1
        self.H_uncoupled[n:2*n, n:2*n] = self.H2
        self.H_uncoupled[2*n:, 2*n:] = self.H3
        
        # Coupling terms (tau_q scaled)
        self.H_coupling = np.zeros((3*n, 3*n), dtype=complex)
        
        # Create coupling matrix that connects corresponding positions between spaces
        coupling_strength = self.tau_q * 1e15  # Scale for numerical stability
        
        # Coupling from EM to Weak
        for i in range(n):
            self.H_coupling[i, n+i] = coupling_strength
            self.H_coupling[n+i, i] = coupling_strength  # Hermitian conjugate
        
        # Coupling from Weak to Strong
        for i in range(n):
            self.H_coupling[n+i, 2*n+i] = coupling_strength
            self.H_coupling[2*n+i, n+i] = coupling_strength  # Hermitian conjugate
        
        # Coupling from EM to Strong (weaker, second-order effect)
        for i in range(n):
            self.H_coupling[i, 2*n+i] = 0.1 * coupling_strength
            self.H_coupling[2*n+i, i] = 0.1 * coupling_strength  # Hermitian conjugate
        
        # Total Hamiltonian
        self.H_total = self.H_uncoupled + self.H_coupling
    
    def schrodinger_rhs(self, t, psi_flat):
        """
        Right-hand side of the Schrödinger equation for numerical integration.
        
        Parameters:
        -----------
        t : float
            Time
        psi_flat : array
            Flattened state vector (real and imaginary parts separated)
        
        Returns:
        --------
        dpsi_dt_flat : array
            Time derivative of psi (flattened)
        """
        # Reshape flat array to complex array
        n_total = 3 * self.n_points
        psi_real = psi_flat[:n_total]
        psi_imag = psi_flat[n_total:]
        psi = psi_real + 1j * psi_imag
        
        # Apply Hamiltonian
        dpsi_dt = -1j * (self.H_total @ psi) / self.hbar
        
        # Flatten the result
        return np.concatenate([dpsi_dt.real, dpsi_dt.imag])
    
    def simulate(self, t_max, n_steps=100):
        """
        Simulate the Schrödinger equation.
        
        Parameters:
        -----------
        t_max : float
            Maximum simulation time
        n_steps : int
            Number of time steps to save
        
        Returns:
        --------
        success : bool
            Whether the simulation completed successfully
        """
        # Time points
        self.times = np.linspace(0, t_max, n_steps)
        
        # Flatten initial state (separate real and imaginary parts)
        psi_0_flat = np.concatenate([self.psi_0.real, self.psi_0.imag])
        
        # Solve the Schrödinger equation
        solution = solve_ivp(
            self.schrodinger_rhs,
            t_span=(0, t_max),
            y0=psi_0_flat,
            t_eval=self.times,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        
        if not solution.success:
            print(f"Simulation failed: {solution.message}")
            return False
        
        # Reshape results back to complex form
        n_total = 3 * self.n_points
        self.psi_history = np.zeros((n_steps, n_total), dtype=complex)
        
        for i in range(n_steps):
            psi_real = solution.y[:n_total, i]
            psi_imag = solution.y[n_total:, i]
            self.psi_history[i] = psi_real + 1j * psi_imag
        
        return True
    
    def calculate_observables(self):
        """Calculate observable quantities from the simulation results."""
        if self.psi_history is None:
            print("No simulation results available.")
            return None
        
        n = self.n_points
        n_steps = len(self.times)
        
        # Probability densities
        prob1 = np.zeros((n_steps, n))
        prob2 = np.zeros((n_steps, n))
        prob3 = np.zeros((n_steps, n))
        
        # Total probabilities in each space
        total_prob1 = np.zeros(n_steps)
        total_prob2 = np.zeros(n_steps)
        total_prob3 = np.zeros(n_steps)
        
        # Calculate tau_q invariants
        invariant1 = np.zeros(n_steps)
        invariant2 = np.zeros(n_steps)
        invariant3 = np.zeros(n_steps)
        
        # Expectation values of position
        position1 = np.zeros(n_steps)
        position2 = np.zeros(n_steps)
        position3 = np.zeros(n_steps)
        
        for i in range(n_steps):
            # Extract wave functions
            psi1 = self.psi_history[i, :n]
            psi2 = self.psi_history[i, n:2*n]
            psi3 = self.psi_history[i, 2*n:]
            
            # Calculate probability densities
            prob1[i] = np.abs(psi1)**2
            prob2[i] = np.abs(psi2)**2
            prob3[i] = np.abs(psi3)**2
            
            # Calculate total probabilities
            total_prob1[i] = np.sum(prob1[i]) * self.dx
            total_prob2[i] = np.sum(prob2[i]) * self.dx
            total_prob3[i] = np.sum(prob3[i]) * self.dx
            
            # Calculate tau_q invariants (using position operator as a simple test)
            if total_prob1[i] > 1e-10:
                position1[i] = np.sum(self.x * prob1[i]) * self.dx / total_prob1[i]
                invariant1[i] = np.abs(np.sum(psi1.conj() * (self.x * psi1))) * self.tau_q
            
            if total_prob2[i] > 1e-10:
                position2[i] = np.sum(self.x * prob2[i]) * self.dx / total_prob2[i]
                invariant2[i] = np.abs(np.sum(psi2.conj() * (self.x * psi2))) * self.tau_q
            
            if total_prob3[i] > 1e-10:
                position3[i] = np.sum(self.x * prob3[i]) * self.dx / total_prob3[i]
                invariant3[i] = np.abs(np.sum(psi3.conj() * (self.x * psi3))) * self.tau_q
        
        return {
            'prob1': prob1,
            'prob2': prob2,
            'prob3': prob3,
            'total_prob1': total_prob1,
            'total_prob2': total_prob2,
            'total_prob3': total_prob3,
            'invariant1': invariant1,
            'invariant2': invariant2,
            'invariant3': invariant3,
            'position1': position1,
            'position2': position2,
            'position3': position3
        }
    
    def plot_results(self, observables=None):
        """
        Plot simulation results.
        
        Parameters:
        -----------
        observables : dict, optional
            Precomputed observables, if None, they will be calculated
        """
        if observables is None:
            observables = self.calculate_observables()
            if observables is None:
                return
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        plt.suptitle(f'Unified Schrödinger Equation Across Three Hilbert Spaces (τ_q = {self.tau_q:.2e})', fontsize=16)
        
        # Plot 1: Initial and final probability densities
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(self.x, observables['prob1'][0], 'r-', label='EM Space (t=0)')
        ax1.plot(self.x, observables['prob2'][0], 'g-', label='Weak Space (t=0)')
        ax1.plot(self.x, observables['prob3'][0], 'b-', label='Strong Space (t=0)')
        
        ax1.plot(self.x, observables['prob1'][-1], 'r--', label='EM Space (t=final)')
        ax1.plot(self.x, observables['prob2'][-1], 'g--', label='Weak Space (t=final)')
        ax1.plot(self.x, observables['prob3'][-1], 'b--', label='Strong Space (t=final)')
        
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Initial and Final Probability Densities')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Total probability in each space over time
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(self.times, observables['total_prob1'], 'r-', label='EM Space')
        ax2.plot(self.times, observables['total_prob2'], 'g-', label='Weak Space')
        ax2.plot(self.times, observables['total_prob3'], 'b-', label='Strong Space')
        ax2.plot(self.times, observables['total_prob1'] + observables['total_prob2'] + observables['total_prob3'], 
                'k-', label='Total')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Total Probability')
        ax2.set_title('Probability Conservation Across Spaces')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Tau_q invariants
        ax3 = plt.subplot(2, 2, 3)
        nonzero_mask1 = observables['invariant1'] > 0
        nonzero_mask2 = observables['invariant2'] > 0
        nonzero_mask3 = observables['invariant3'] > 0
        
        if np.any(nonzero_mask1):
            ax3.plot(self.times[nonzero_mask1], observables['invariant1'][nonzero_mask1], 'r-', label='EM Invariant')
        if np.any(nonzero_mask2):
            ax3.plot(self.times[nonzero_mask2], observables['invariant2'][nonzero_mask2], 'g-', label='Weak Invariant')
        if np.any(nonzero_mask3):
            ax3.plot(self.times[nonzero_mask3], observables['invariant3'][nonzero_mask3], 'b-', label='Strong Invariant')
        
        # Calculate sum of invariants where all three are non-zero
        all_nonzero = nonzero_mask1 & nonzero_mask2 & nonzero_mask3
        if np.any(all_nonzero):
            invariant_sum = (observables['invariant1'] + observables['invariant2'] + observables['invariant3'])[all_nonzero]
            ax3.plot(self.times[all_nonzero], invariant_sum, 'k-', label='Sum of Invariants')
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('τ_q Invariant')
        ax3.set_title('τ_q Invariants Across Spaces')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Position expectation values
        ax4 = plt.subplot(2, 2, 4)
        nonzero_mask1 = observables['total_prob1'] > 1e-3
        nonzero_mask2 = observables['total_prob2'] > 1e-3
        nonzero_mask3 = observables['total_prob3'] > 1e-3
        
        if np.any(nonzero_mask1):
            ax4.plot(self.times[nonzero_mask1], observables['position1'][nonzero_mask1], 'r-', label='EM Position')
        if np.any(nonzero_mask2):
            ax4.plot(self.times[nonzero_mask2], observables['position2'][nonzero_mask2], 'g-', label='Weak Position')
        if np.any(nonzero_mask3):
            ax4.plot(self.times[nonzero_mask3], observables['position3'][nonzero_mask3], 'b-', label='Strong Position')
        
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Position Expectation Value')
        ax4.set_title('Position Expectation Values')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    
    def create_animation(self, filename=None):
        """
        Create an animation of the wave functions.
        
        Parameters:
        -----------
        filename : str, optional
            If provided, save the animation to this file
        """
        if self.psi_history is None:
            print("No simulation results available.")
            return None
        
        # Calculate probability densities
        n = self.n_points
        prob1 = np.abs(self.psi_history[:, :n])**2
        prob2 = np.abs(self.psi_history[:, n:2*n])**2
        prob3 = np.abs(self.psi_history[:, 2*n:])**2
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Find maximum probability for consistent y-axis
        max_prob = np.max([np.max(prob1), np.max(prob2), np.max(prob3)])
        
        # Initial plots
        line1, = ax.plot([], [], 'r-', label='EM Space')
        line2, = ax.plot([], [], 'g-', label='Weak Space')
        line3, = ax.plot([], [], 'b-', label='Strong Space')
        
        # Potentials (scaled for visualization)
        pot_scale = max_prob / np.max([np.max(self.V1), np.max(self.V2), np.max(self.V3)])
        if np.isfinite(pot_scale) and pot_scale > 0:
            line_V1, = ax.plot(self.x, self.V1 * pot_scale * 0.5, 'r--', alpha=0.3)
            line_V2, = ax.plot(self.x, self.V2 * pot_scale * 0.5, 'g--', alpha=0.3)
            line_V3, = ax.plot(self.x, self.V3 * pot_scale * 0.5, 'b--', alpha=0.3)
        
        # Title with time indicator
        title = ax.set_title('')
        
        # Set up axes
        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(0, max_prob * 1.1)
        ax.set_xlabel('Position')
        ax.set_ylabel('Probability Density')
        ax.grid(True)
        ax.legend()
        
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            title.set_text('')
            return line1, line2, line3, title
        
        def animate(i):
            line1.set_data(self.x, prob1[i])
            line2.set_data(self.x, prob2[i])
            line3.set_data(self.x, prob3[i])
            title.set_text(f'Time: {self.times[i]:.2f}')
            return line1, line2, line3, title
        
        ani = FuncAnimation(fig, animate, frames=len(self.times),
                          init_func=init, blit=True, interval=50)
        
        plt.tight_layout()
        
        if filename:
            ani.save(filename, writer='pillow', fps=10)
        
        plt.show()
        
        return ani

def run_schrodinger_tests():
    """
    Run tests of the unified Schrödinger equation across three Hilbert spaces.
    """
    print("=== Unified Schrödinger Equation Tests ===")
    
    # Test 1: Standard tau_q value
    print("\nTest 1: Standard simulation with tau_q = 2.203e-15")
    sim1 = UnifiedSchrodingerSimulator(tau_q=2.203e-15, potential_type='harmonic')
    success = sim1.simulate(t_max=5.0, n_steps=200)
    
    if success:
        # Calculate and display observables
        obs1 = sim1.calculate_observables()
        
        # Calculate tau_q validation metrics
        final_prob1 = obs1['total_prob1'][-1]
        final_prob2 = obs1['total_prob2'][-1]
        final_prob3 = obs1['total_prob3'][-1]
        total_prob = final_prob1 + final_prob2 + final_prob3
        
        print(f"Final probability distribution:")
        print(f"  EM space: {final_prob1:.4f}")
        print(f"  Weak space: {final_prob2:.4f}")
        print(f"  Strong space: {final_prob3:.4f}")
        print(f"  Total: {total_prob:.4f}")
        
        # Check if probability spreads to other spaces
        if final_prob2 > 0.01 and final_prob3 > 0.01:
            print("VALIDATION PASSED: Probability spreads across Hilbert spaces")
        else:
            print("VALIDATION FAILED: Probability does not spread to all spaces")
            
        # Test 2: Different tau_q value
        print("\nTest 2: Different tau_q value (10x larger)")
        sim2 = UnifiedSchrodingerSimulator(tau_q=2.203e-14, potential_type='harmonic')
        success2 = sim2.simulate(t_max=5.0, n_steps=200)
        
        if success2:
            obs2 = sim2.calculate_observables()
            
            # Compare probability transfer rates
            final_prob2_test1 = obs1['total_prob2'][-1]
            final_prob2_test2 = obs2['total_prob2'][-1]
            
            print(f"Weak space probability (standard tau_q): {final_prob2_test1:.4f}")
            print(f"Weak space probability (10x tau_q): {final_prob2_test2:.4f}")
            
            # Verify that larger tau_q increases coupling
            if final_prob2_test2 > final_prob2_test1:
                print("VALIDATION PASSED: Larger tau_q increases cross-space coupling")
            else:
                print("VALIDATION FAILED: Larger tau_q does not increase coupling as expected")
        
        # Test 3: Different potential type
        print("\nTest 3: Different potential (barrier)")
        sim3 = UnifiedSchrodingerSimulator(tau_q=2.203e-15, potential_type='barrier')
        success3 = sim3.simulate(t_max=5.0, n_steps=200)
        
        if success3:
            obs3 = sim3.calculate_observables()
            
            # Compare final probability distributions
            print(f"Probability distribution with barrier potential:")
            print(f"  EM space: {obs3['total_prob1'][-1]:.4f}")
            print(f"  Weak space: {obs3['total_prob2'][-1]:.4f}")
            print(f"  Strong space: {obs3['total_prob3'][-1]:.4f}")
            
            # Test if behavior is different from harmonic case
            dist1 = abs(obs1['total_prob1'][-1] - obs3['total_prob1'][-1])
            dist2 = abs(obs1['total_prob2'][-1] - obs3['total_prob2'][-1])
            dist3 = abs(obs1['total_prob3'][-1] - obs3['total_prob3'][-1])
            
            if dist1 > 0.05 or dist2 > 0.05 or dist3 > 0.05:
                print("VALIDATION PASSED: Different potentials lead to different cross-space dynamics")
            else:
                print("VALIDATION FAILED: Different potentials show similar behavior")
        
        return {
            'sim1': sim1,
            'sim2': sim2 if success2 else None,
            'sim3': sim3 if success3 else None,
            'obs1': obs1,
            'obs2': obs2 if success2 else None,
            'obs3': obs3 if success3 else None
        }
    else:
        print("Simulation failed, cannot run tests.")
        return None

# Run the tests
test_results = run_schrodinger_tests()

# Plot results if successful
if test_results:
    # Plot standard simulation
    test_results['sim1'].plot_results(test_results['obs1'])
    
    # Compare different tau_q values
    if test_results['sim2']:
        # Create a comparison plot
        plt.figure(figsize=(12, 8))
        plt.suptitle('Effect of τ_q on Cross-Hilbert Space Transfer', fontsize=16)
        
        plt.subplot(2, 1, 1)
        plt.plot(test_results['sim1'].times, test_results['obs1']['total_prob1'], 'r-', label='EM (τ_q=2.203e-15)')
        plt.plot(test_results['sim1'].times, test_results['obs1']['total_prob2'], 'g-', label='Weak (τ_q=2.203e-15)')
        plt.plot(test_results['sim1'].times, test_results['obs1']['total_prob3'], 'b-', label='Strong (τ_q=2.203e-15)')
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.title('Standard τ_q Value')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(test_results['sim2'].times, test_results['obs2']['total_prob1'], 'r-', label='EM (τ_q=2.203e-14)')
        plt.plot(test_results['sim2'].times, test_results['obs2']['total_prob2'], 'g-', label='Weak (τ_q=2.203e-14)')
        plt.plot(test_results['sim2'].times, test_results['obs2']['total_prob3'], 'b-', label='Strong (τ_q=2.203e-14)')
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.title('10x τ_q Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
