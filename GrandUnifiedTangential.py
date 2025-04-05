import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import minimize

class GrandUnifiedEigenTangential:
    def __init__(self, tau_q=2.203e-15, dim=12):
        self.tau_q = tau_q
        self.dim = dim
        self.a = np.pi
        self.b = 0.5
        
        # Basic matrices
        self.I2 = np.eye(2, dtype=complex)
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Create the unified field operator
        self.F = self._create_unified_field_operator()
        self._compute_eigenstructure()
    
    def _create_unified_field_operator(self):
        return self.a * self.I2 + self.b * self.sigma_z + self.tau_q * self.sigma_x
    
    def _compute_eigenstructure(self):
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.F)
        self.eigenphases = np.angle(self.eigenvalues)
        self.tangent_phases = np.tan(self.eigenphases)
        self.H = self.F / self.tau_q
        self.H_eigenvalues, _ = np.linalg.eig(self.H)
        self.U = expm(-1j * self.F)
        self.U_eigenvalues, _ = np.linalg.eig(self.U)
        self.arccos_phases = np.arccos(np.real(self.U_eigenvalues))
    
    def find_grand_unified_tangential(self, particle_masses=None):
        if particle_masses is None:
            # Use standard particle masses in GeV/c²
            particle_masses = {
                'electron': 0.000511,
                'muon': 0.1057, 
                'tau': 1.777,
                'up': 0.0022,
                'down': 0.0047,
                'charm': 1.27,
                'strange': 0.095,
                'top': 172.5,
                'bottom': 4.18,
                'W': 80.4,
                'Z': 91.2,
                'Higgs': 125.0
            }
        
        N = len(particle_masses)
        particles = list(particle_masses.keys())
        masses = np.array(list(particle_masses.values()))
        log_masses = np.log10(masses)  # Work in log space for wide-ranging values
        
        def global_phase_objective(params):
            # First N-1 phases (last one is determined by 2π constraint)
            phases = np.append(params[:N-1], 2*np.pi - np.sum(params[:N-1]))
            
            # Calculate tangential relationships
            tangentials = np.tan(phases)
            
            # Calculate predicted masses using eigenphase-mass mapping
            predicted = np.zeros(N)
            for i in range(N):
                # Create superposition with proper phase relationships
                phase_sum = 0
                for j in range(N):
                    phase_diff = phases[i] - phases[j]
                    phase_sum += np.cos(phase_diff)
                
                predicted[i] = np.abs(phase_sum) * self.tau_q
            
            # Safe normalization to match mass range
            range_pred = np.max(predicted) - np.min(predicted)
            if range_pred > 1e-10:  # Prevent division by zero
                predicted = (predicted - np.min(predicted)) / range_pred
                predicted = predicted * (np.max(masses) - np.min(masses)) + np.min(masses)
            
            # Calculate error in log space
            log_predicted = np.log10(predicted + 1e-20)  # Avoid log(0)
            error = np.sum((log_predicted - log_masses)**2)
            
            # Add penalty for constraint violations
            penalty = 0
            if np.sum(phases) != 2*np.pi:
                penalty += 1000 * np.abs(np.sum(phases) - 2*np.pi)
            
            return error + penalty
        
        # Initial guess: evenly distributed phases
        initial_params = np.linspace(0, 2*np.pi, N+1)[:-2]
        
        # Bounds: all phases between 0 and 2π
        bounds = [(0, 2*np.pi)] * (N-1)
        
        # Optimize with multiple attempts from different starting points
        best_result = None
        best_value = float('inf')
        
        for attempt in range(5):
            if attempt > 0:
                # Random perturbation of initial guess
                initial_params = np.random.uniform(0, 2*np.pi, N-1)
                # Ensure they sum to less than 2π
                if np.sum(initial_params) > 2*np.pi:
                    initial_params = initial_params * (2*np.pi / (np.sum(initial_params) * 1.1))
            
            result = minimize(
                global_phase_objective, 
                initial_params, 
                bounds=bounds, 
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )
            
            if result.fun < best_value:
                best_value = result.fun
                best_result = result
        
        # Extract optimized phases
        opt_phases = np.append(best_result.x, 2*np.pi - np.sum(best_result.x))
        
        # Ensure exactly 2π sum (floating point precision issues)
        opt_phases[-1] = 2*np.pi - np.sum(opt_phases[:-1])
        
        # Calculate tangential relationships
        tangentials = np.tan(opt_phases)
        
        # Calculate pairwise tangential relationships
        tan_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                phase_diff = opt_phases[i] - opt_phases[j]
                tan_matrix[i, j] = np.tan(phase_diff)
        
        # Calculate the grand unified tangential
        # Use geometric mean to capture the fundamental relationship
        nonzero_tangentials = tangentials[np.isfinite(tangentials)]
        if len(nonzero_tangentials) > 0:
            # Take absolute value before geometric mean
            grand_tangential = np.exp(np.mean(np.log(np.abs(nonzero_tangentials) + 1e-10)))
        else:
            grand_tangential = np.nan
        
        return {
            'particles': particles,
            'optimized_phases': opt_phases,
            'tangentials': tangentials,
            'tangential_matrix': tan_matrix,
            'grand_unified_tangential': grand_tangential,
            'phase_sum': np.sum(opt_phases),
            'optimization_success': best_result.success
        }
    
    def visualize_grand_tangential(self, result):
        if result is None:
            print("No grand unified tangential result provided.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Phase distribution on unit circle
        angles = result['optimized_phases']
        particles = result['particles']
        ax1.set_aspect('equal')
        
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.2)
        
        # Plot particle phases
        for i, particle in enumerate(particles):
            x = np.cos(angles[i])
            y = np.sin(angles[i])
            ax1.plot([0, x], [0, y], 'b-', alpha=0.3)
            ax1.plot(x, y, 'ro')
            ax1.text(x*1.1, y*1.1, particle, fontsize=8, ha='center', va='center')
        
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Particle Phase Distribution (Sum: {result['phase_sum']:.6f})")
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        
        # Plot 2: Tangential relationships heatmap
        tan_matrix = np.abs(result['tangential_matrix'])
        # Apply log scaling with offset to handle zeros and infinities
        log_matrix = np.log10(tan_matrix + 1e-10)
        
        im = ax2.imshow(log_matrix, cmap='viridis')
        fig.colorbar(im, ax=ax2, label='log10(|tan(φi-φj)|)')
        
        # Properly set tick positions before labels
        ax2.set_xticks(np.arange(len(particles)))
        ax2.set_yticks(np.arange(len(particles)))
        ax2.set_xticklabels(particles, rotation=45, ha='right')
        ax2.set_yticklabels(particles)
        
        ax2.set_title(f"Tangential Relationships\nGrand Unified Tangential: {result['grand_unified_tangential']:.6f}")
        
        plt.tight_layout()
        plt.show()
        
        # Show eigenphase structure
        plt.figure(figsize=(10, 6))
        plt.subplot(121)
        plt.title("Eigenphase distribution")
        plt.plot(angles, 'bo-')
        plt.xlabel("Particle index")
        plt.ylabel("Phase (radians)")
        plt.grid(True)
        
        plt.subplot(122)
        plt.title("Tangential values")
        finite_tangentials = [t for t in result['tangentials'] if np.isfinite(t)]
        plt.plot(finite_tangentials, 'ro-')
        plt.axhline(y=result['grand_unified_tangential'], color='k', linestyle='--', 
                   label=f'GUT: {result["grand_unified_tangential"]:.6f}')
        plt.xlabel("Particle index")
        plt.ylabel("tan(φ)")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Run the solver
gut_solver = GrandUnifiedEigenTangential()
gut_result = gut_solver.find_grand_unified_tangential()
gut_solver.visualize_grand_tangential(gut_result)

print(f"Grand Unified Tangential: {gut_result['grand_unified_tangential']}")
print(f"Optimization success: {gut_result['optimization_success']}")
print(f"Phase sum: {gut_result['phase_sum']} (should be 2π={2*np.pi})")
print(f"\nParticle phases (radians):")
for particle, phase in zip(gut_result['particles'], gut_result['optimized_phases']):
    print(f"{particle}: {phase:.6f}")
