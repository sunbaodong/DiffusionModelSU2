"""
SU(2) HMC
=====================================
HMC for SU(2) two-dimensional pure gauge lattice field theory.

By B.-D. Sun.
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from tqdm import tqdm
from scipy.special import iv as bessel_i
from scipy.optimize import curve_fit
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class SU2:
    """
    SU(2) matrix operations using quaternion representation.
    
    A quaternion (a0, a1, a2, a3) with |a|^2 = 1 represents the SU(2) matrix:
    U = a0*I + i*(a1*sigma1 + a2*sigma2 + a3*sigma3)
    
    Storage format: (..., 4) tensor where last dimension is (a0, a1, a2, a3)
    """
    
    # Pauli matrices (stored as constants)
    # sigma1 = [[0, 1], [1, 0]]
    # sigma2 = [[0, -i], [i, 0]]
    # sigma3 = [[1, 0], [0, -1]]
    
    @staticmethod
    def identity(shape: Tuple, device: torch.device = None) -> torch.Tensor:
        """Create identity SU(2) elements (quaternion = (1, 0, 0, 0))."""
        q = torch.zeros(*shape, 4, device=device)
        q[..., 0] = 1.0
        return q
    
    @staticmethod
    def to_matrix(q: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to 2x2 complex matrix.
        
        U = [[a0 + i*a3,  a2 + i*a1],
             [-a2 + i*a1, a0 - i*a3]]
        
        Args:
            q: (..., 4) quaternion tensor
        Returns:
            U: (..., 2, 2) complex matrix tensor
        """
        a0, a1, a2, a3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        U = torch.zeros(*q.shape[:-1], 2, 2, dtype=torch.complex128, device=q.device)
        U[..., 0, 0] = a0 + 1j * a3
        U[..., 0, 1] = a2 + 1j * a1
        U[..., 1, 0] = -a2 + 1j * a1
        U[..., 1, 1] = a0 - 1j * a3
        
        return U
    
    @staticmethod
    def from_matrix(U: torch.Tensor) -> torch.Tensor:
        """
        Convert 2x2 complex matrix to quaternion.
        Uses the relations from the quaternion representation.
        
        Args:
            U: (..., 2, 2) complex matrix tensor
        Returns:
            q: (..., 4) quaternion tensor
        """
        q = torch.zeros(*U.shape[:-2], 4, device=U.device, dtype=torch.float64)
        
        # a0 = Re(U[0,0] + U[1,1]) / 2 = Re(U[0,0])  (since U[1,1] = conj(U[0,0]) for SU(2))
        # a3 = Im(U[0,0] + U[1,1]) / 2 = Im(U[0,0])
        # a1 = Im(U[0,1] + U[1,0]) / 2 = Im(U[0,1])  (since U[1,0] = -conj(U[0,1]))
        # a2 = Re(U[0,1] - U[1,0]) / 2 = Re(U[0,1])
        
        q[..., 0] = (U[..., 0, 0].real + U[..., 1, 1].real) / 2
        q[..., 1] = (U[..., 0, 1].imag + U[..., 1, 0].imag) / 2
        q[..., 2] = (U[..., 0, 1].real - U[..., 1, 0].real) / 2
        q[..., 3] = (U[..., 0, 0].imag - U[..., 1, 1].imag) / 2
        
        return q
    
    @staticmethod
    def multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Multiply two SU(2) elements: q_out = q1 * q2
        
        Quaternion multiplication formula:
        (a0, a) * (b0, b) = (a0*b0 - a·b, a0*b + b0*a + a×b)
        
        Args:
            q1, q2: (..., 4) quaternion tensors
        Returns:
            q_out: (..., 4) quaternion tensor
        """
        a0, a1, a2, a3 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        b0, b1, b2, b3 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        q_out = torch.zeros_like(q1)
        

        # CORRECT - uses -a×b for U = a₀I + i(a·σ) convention
        q_out[..., 0] = a0*b0 - a1*b1 - a2*b2 - a3*b3
        q_out[..., 1] = a0*b1 + a1*b0 - a2*b3 + a3*b2
        q_out[..., 2] = a0*b2 + a1*b3 + a2*b0 - a3*b1
        q_out[..., 3] = a0*b3 - a1*b2 + a2*b1 + a3*b0
        
        return q_out
    
    @staticmethod
    def dagger(q: torch.Tensor) -> torch.Tensor:
        """
        Hermitian conjugate of SU(2) element.
        For SU(2): U^† = U^{-1}, and in quaternion: (a0, -a1, -a2, -a3)
        
        Args:
            q: (..., 4) quaternion tensor
        Returns:
            q_dag: (..., 4) quaternion tensor
        """
        q_dag = q.clone()
        q_dag[..., 1:] = -q_dag[..., 1:]
        return q_dag
    
    @staticmethod
    def trace(q: torch.Tensor) -> torch.Tensor:
        """
        Compute trace of SU(2) matrix: Tr(U) = 2*a0
        
        Args:
            q: (..., 4) quaternion tensor
        Returns:
            tr: (...) real tensor
        """
        return 2.0 * q[..., 0]
    
    @staticmethod
    def normalize(q: torch.Tensor) -> torch.Tensor:
        """
        Project to SU(2) by normalizing quaternion to unit length.
        
        Args:
            q: (..., 4) quaternion tensor
        Returns:
            q_normalized: (..., 4) unit quaternion tensor
        """
        norm = torch.sqrt(torch.sum(q**2, dim=-1, keepdim=True))
        return q / (norm + 1e-15)
    
    @staticmethod
    def random(shape: Tuple, device: torch.device = None) -> torch.Tensor:
        """
        Generate random SU(2) elements uniformly distributed according to Haar measure.
        Method: Generate 4 Gaussian random numbers and normalize.
        
        Args:
            shape: Shape of the output (excluding the quaternion dimension)
            device: torch device
        Returns:
            q: (*shape, 4) random unit quaternion tensor
        """
        q = torch.randn(*shape, 4, device=device)
        return SU2.normalize(q)

class SU2Algebra:
    """
    Operations for su(2) Lie algebra elements.
    
    Algebra elements are stored as 3-component vectors (omega1, omega2, omega3)
    representing X = (i/2) * (omega1*sigma1 + omega2*sigma2 + omega3*sigma3)
    """
    
    @staticmethod
    def exp(omega: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from su(2) algebra to SU(2) group.
        
        exp(X) where X = (i/2) * omega . sigma
        
        Using: exp(i*n.sigma * theta/2) = cos(theta/2)*I + i*sin(theta/2)*(n.sigma)
        where |omega| = theta and n = omega/|omega|
        
        Result in quaternion: (cos(|omega|/2), sin(|omega|/2) * omega / |omega|)
        
        Args:
            omega: (..., 3) algebra coefficients
        Returns:
            q: (..., 4) quaternion (SU(2) group element)
        """
        # |omega| / 2
        norm = torch.sqrt(torch.sum(omega**2, dim=-1, keepdim=True))
        half_norm = norm / 2.0
        
        # Handle small norm case to avoid division by zero
        # For small x: sin(x)/x ≈ 1 - x^2/6
        small_mask = (norm < 1e-7).squeeze(-1)
        
        q = torch.zeros(*omega.shape[:-1], 4, device=omega.device, dtype=omega.dtype)
        
        # cos(|omega|/2)
        q[..., 0] = torch.cos(half_norm).squeeze(-1)
        
        # sin(|omega|/2) / (|omega|/2) * (omega/2)
        # = sin(|omega|/2) * omega / |omega|
        sinc_factor = torch.where(
            norm > 1e-7,
            torch.sin(half_norm) / norm,
            0.5 * torch.ones_like(norm)  # lim_{x->0} sin(x/2)/x = 1/2
        )
        
        q[..., 1:] = omega * sinc_factor
        
        return q
    
    @staticmethod
    def random_momentum(shape: Tuple, device: torch.device = None) -> torch.Tensor:
        """
        Generate random momentum from su(2) with Gaussian distribution.
        
        The HMC kinetic term is (1/2) * Tr(pi^2) = (1/2) * sum_a omega_a^2
        So we draw omega_a from unit Gaussians.
        
        Args:
            shape: Shape of output (excluding algebra dimension)
            device: torch device
        Returns:
            omega: (*shape, 3) Gaussian random algebra elements
        """
        return torch.randn(*shape, 3, device=device)
    
    @staticmethod
    def from_matrix(X: torch.Tensor) -> torch.Tensor:
        """
        Extract algebra coefficients from traceless anti-Hermitian matrix.
        
        X = (i/2) * (omega1*sigma1 + omega2*sigma2 + omega3*sigma3)
        
        Args:
            X: (..., 2, 2) complex matrix
        Returns:
            omega: (..., 3) algebra coefficients
        """
        omega = torch.zeros(*X.shape[:-2], 3, device=X.device, dtype=torch.float64)
        
        # omega1 = 2 * Im(X[0,1])
        # omega2 = 2 * Re(X[0,1]) 
        # omega3 = 2 * Im(X[0,0])
        omega[..., 0] = 2.0 * X[..., 0, 1].imag
        omega[..., 1] = 2.0 * X[..., 0, 1].real
        omega[..., 2] = 2.0 * X[..., 0, 0].imag
        
        return omega
    
    @staticmethod
    def to_matrix(omega: torch.Tensor) -> torch.Tensor:
        """
        Convert algebra coefficients to traceless anti-Hermitian matrix.
        
        X = (i/2) * (omega1*sigma1 + omega2*sigma2 + omega3*sigma3)
          = [[i*omega3/2,      (omega2 + i*omega1)/2],
             [(-omega2 + i*omega1)/2,    -i*omega3/2]]
        
        Args:
            omega: (..., 3) algebra coefficients
        Returns:
            X: (..., 2, 2) complex matrix
        """
        X = torch.zeros(*omega.shape[:-1], 2, 2, dtype=torch.complex128, device=omega.device)
        
        o1, o2, o3 = omega[..., 0], omega[..., 1], omega[..., 2]
        
        X[..., 0, 0] = 0.5j * o3
        X[..., 0, 1] = 0.5 * (o2 + 1j * o1)
        X[..., 1, 0] = 0.5 * (-o2 + 1j * o1)
        X[..., 1, 1] = -0.5j * o3
        
        return X
    
    @staticmethod
    def traceless_antiherm_proj(W: torch.Tensor) -> torch.Tensor:
        """
        Project a matrix onto traceless anti-Hermitian matrices.
        
        W_TA = (W - W^dag) / 2 - Tr(W - W^dag) / 4 * I
        
        This is used to compute the force in HMC.
        
        Args:
            W: (..., 2, 2) complex matrix
        Returns:
            W_TA: (..., 2, 2) traceless anti-Hermitian matrix
        """
        W_dag = W.conj().transpose(-2, -1)
        anti_herm = (W - W_dag) / 2.0
        
        # Subtract trace part
        trace = anti_herm[..., 0, 0] + anti_herm[..., 1, 1]
        trace_part = trace.unsqueeze(-1).unsqueeze(-1) / 2.0
        
        I = torch.eye(2, dtype=torch.complex128, device=W.device)
        W_TA = anti_herm - trace_part * I
        
        return W_TA

class LatticeGaugeField2D:
    """
    2D SU(2) lattice gauge field with periodic boundary conditions.
    
    Storage: (Lx, Lt, 2, 4) tensor
    - Lx, Lt: spatial and temporal extent
    - 2: number of directions (mu = 0 for x, mu = 1 for t)
    - 4: quaternion components
    """
    
    def __init__(self, Lx: int, Lt: int, device: torch.device = None):
        self.Lx = Lx
        self.Lt = Lt
        self.device = device
        
        # Initialize to identity (cold start)
        self.U = SU2.identity((Lx, Lt, 2), device=device)
    
    def hot_start(self):
        """Initialize with random SU(2) elements."""
        self.U = SU2.random((self.Lx, self.Lt, 2), device=self.device)
    
    def cold_start(self):
        """Initialize with identity elements."""
        self.U = SU2.identity((self.Lx, self.Lt, 2), device=self.device)
    
    def get_link(self, x: int, t: int, mu: int) -> torch.Tensor:
        """Get U_mu(x, t) with periodic boundary conditions."""
        return self.U[x % self.Lx, t % self.Lt, mu]
    
    def set_link(self, x: int, t: int, mu: int, q: torch.Tensor):
        """Set U_mu(x, t) with periodic boundary conditions."""
        self.U[x % self.Lx, t % self.Lt, mu] = q
    
    def shift(self, direction: int, shift: int = 1) -> torch.Tensor:
        """
        Shift the entire field in given direction.
        Returns U_mu(x + shift * hat{direction})
        
        Args:
            direction: 0 for x, 1 for t
            shift: number of sites to shift (can be negative)
        Returns:
            Shifted field tensor
        """
        return torch.roll(self.U, shifts=-shift, dims=direction)
    
    def clone(self) -> 'LatticeGaugeField2D':
        """Create a deep copy of the gauge field."""
        new_field = LatticeGaugeField2D(self.Lx, self.Lt, self.device)
        new_field.U = self.U.clone()
        return new_field
    
    def copy_from(self, other: 'LatticeGaugeField2D'):
        """Copy data from another gauge field."""
        self.U = other.U.clone()

class WilsonAction2D:
    """
    Wilson gauge action for 2D SU(2) lattice gauge theory.
    
    S = -(beta/2) * sum_x Re Tr[P_{01}(x)]
    
    where P_{01}(x) = U_0(x) * U_1(x+0) * U_0^dag(x+1) * U_1^dag(x)
    """
    
    def __init__(self, beta: float):
        self.beta = beta
    
    def compute_plaquette(self, field: LatticeGaugeField2D) -> torch.Tensor:
        """
        Compute all plaquettes P_{01}(x) for all sites.
        
        P_{01}(x) = U_0(x) * U_1(x+0) * U_0^dag(x+1) * U_1^dag(x)
        
        Returns:
            plaq: (Lx, Lt, 4) tensor of plaquette quaternions
        """
        U = field.U
        
        # U_0(x)
        U0 = U[:, :, 0, :]
        
        # U_1(x + hat{0}) - shift in x direction
        U1_shifted_x = torch.roll(U[:, :, 1, :], shifts=-1, dims=0)
        
        # U_0^dag(x + hat{1}) - shift in t direction, then dagger
        U0_shifted_t = torch.roll(U[:, :, 0, :], shifts=-1, dims=1)
        U0_shifted_t_dag = SU2.dagger(U0_shifted_t)
        
        # U_1^dag(x)
        U1 = U[:, :, 1, :]
        U1_dag = SU2.dagger(U1)
        
        # P = U_0 * U_1(x+0) * U_0^dag(x+1) * U_1^dag(x)
        temp1 = SU2.multiply(U0, U1_shifted_x)
        temp2 = SU2.multiply(temp1, U0_shifted_t_dag)
        plaq = SU2.multiply(temp2, U1_dag)
        
        return plaq
    
    def compute_action(self, field: LatticeGaugeField2D) -> torch.Tensor:
        """
        Compute the Wilson action.
        
        S = -(beta/2) * sum_x Re Tr[P(x)]
          = -(beta/2) * sum_x 2 * a0(x)  [since Tr(U) = 2*a0]
          = -beta * sum_x a0(x)
        
        Returns:
            action: scalar tensor
        """
        plaq = self.compute_plaquette(field)
        # Re Tr[P] = 2 * a0
        re_tr_plaq = SU2.trace(plaq)  # This is 2*a0
        action = -0.5 * self.beta * torch.sum(re_tr_plaq)
        return action
    
    def compute_average_plaquette(self, field: LatticeGaugeField2D) -> torch.Tensor:
        """
        Compute average plaquette: <(1/2) Re Tr[P]> = <a0>
        
        Returns:
            avg_plaq: scalar in range [-1, 1]
        """
        plaq = self.compute_plaquette(field)
        avg_plaq = torch.mean(plaq[..., 0])  # Average of a0
        return avg_plaq
    
    def compute_staple(self, field: LatticeGaugeField2D, mu: int) -> torch.Tensor:
        """
        Compute the staple for direction mu at all sites.
        
        In 2D, for each link U_mu(x), there are 2 staples:
        - Forward: U_nu(x+mu) * U_mu^dag(x+nu) * U_nu^dag(x)
        - Backward: U_nu^dag(x+mu-nu) * U_mu^dag(x-nu) * U_nu(x-nu)
        
        Wait, let me reconsider. The staple Sigma_mu(x) is the sum of the 
        "staple" parts of all plaquettes containing U_mu(x).
        
        For link U_mu(x), the plaquettes containing it are:
        1. P_{mu,nu}(x) = U_mu(x) * U_nu(x+mu) * U_mu^dag(x+nu) * U_nu^dag(x)
           Staple contribution: U_nu(x+mu) * U_mu^dag(x+nu) * U_nu^dag(x)
           Wait, this is wrong. The staple is what remains when U_mu(x) is removed.
           
        Let me be more careful. The action contribution from link U_mu(x) is:
        S_link = -(beta/2) Re Tr[U_mu(x) * Sigma_mu^dag(x)]
        
        where Sigma_mu(x) is the sum of staples.
        
        In 2D with mu=0 and nu=1:
        Sigma_0(x) = U_1(x+0) * U_0^dag(x+1) * U_1^dag(x) +   [forward in nu=1]
                     U_1^dag(x+0-1) * U_0^dag(x-1) * U_1(x-1)  [backward in nu=1]
        
        Args:
            field: gauge field
            mu: direction of the link
        Returns:
            staple: (Lx, Lt, 4) quaternion tensor
        """
        U = field.U
        nu = 1 - mu  # The other direction in 2D
        
        # Forward staple: U_nu(x+mu) * U_mu^dag(x+nu) * U_nu^dag(x)
        # U_nu(x+mu)
        U_nu_shifted_mu = torch.roll(U[:, :, nu, :], shifts=-1, dims=mu)
        # U_mu(x+nu)
        U_mu_shifted_nu = torch.roll(U[:, :, mu, :], shifts=-1, dims=nu)
        U_mu_shifted_nu_dag = SU2.dagger(U_mu_shifted_nu)
        # U_nu(x)
        U_nu = U[:, :, nu, :]
        U_nu_dag = SU2.dagger(U_nu)
        
        # Forward: U_nu(x+mu) * U_mu^dag(x+nu) * U_nu^dag(x)
        temp = SU2.multiply(U_nu_shifted_mu, U_mu_shifted_nu_dag)
        forward_staple = SU2.multiply(temp, U_nu_dag)
        
        # Backward staple: U_nu^dag(x+mu-nu) * U_mu^dag(x-nu) * U_nu(x-nu)
        # U_nu(x+mu-nu) = shift U_nu by +1 in mu, -1 in nu (or +1 in mu then +1 in nu for shift=-1)
        U_nu_shift_mu_minus_nu = torch.roll(torch.roll(U[:, :, nu, :], shifts=-1, dims=mu), shifts=1, dims=nu)
        U_nu_shift_mu_minus_nu_dag = SU2.dagger(U_nu_shift_mu_minus_nu)
        
        # U_mu(x-nu)
        U_mu_shift_minus_nu = torch.roll(U[:, :, mu, :], shifts=1, dims=nu)
        U_mu_shift_minus_nu_dag = SU2.dagger(U_mu_shift_minus_nu)
        
        # U_nu(x-nu)
        U_nu_shift_minus_nu = torch.roll(U[:, :, nu, :], shifts=1, dims=nu)
        
        # Backward: U_nu^dag(x+mu-nu) * U_mu^dag(x-nu) * U_nu(x-nu)
        temp = SU2.multiply(U_nu_shift_mu_minus_nu_dag, U_mu_shift_minus_nu_dag)
        backward_staple = SU2.multiply(temp, U_nu_shift_minus_nu)
        
        # Total staple is sum (need to add quaternions properly)
        # For quaternions representing SU(2), we just add component-wise
        # (the result is not SU(2) but that's OK for the staple)
        staple = forward_staple + backward_staple
        
        return staple

class HMC2D:
    """
    Hybrid Monte Carlo for 2D SU(2) gauge theory.
    
    Uses leapfrog integrator for molecular dynamics.
    """
    
    def __init__(self, action: WilsonAction2D, tau: float, n_steps: int):
        """
        Args:
            action: Wilson action object
            tau: trajectory length
            n_steps: number of leapfrog steps
        """
        self.action = action
        self.tau = tau
        self.n_steps = n_steps
        self.epsilon = tau / n_steps
    
    def compute_kinetic_energy(self, pi: torch.Tensor) -> torch.Tensor:
        """
        Compute kinetic energy: K = (1/2) sum_{x,mu} Tr(pi^2)
                                  = (1/2) sum_{x,mu,a} omega_a^2
        
        Args:
            pi: (Lx, Lt, 2, 3) momentum tensor (algebra coefficients)
        Returns:
            K: scalar kinetic energy
        """
        return 0.5 * torch.sum(pi**2)
    
    def compute_hamiltonian(self, field: LatticeGaugeField2D, pi: torch.Tensor) -> torch.Tensor:
        """
        Compute total Hamiltonian H = K + S.
        
        Args:
            field: gauge field configuration
            pi: momentum tensor
        Returns:
            H: scalar Hamiltonian
        """
        K = self.compute_kinetic_energy(pi)
        S = self.action.compute_action(field)
        return K + S
    
    def compute_force(self, field: LatticeGaugeField2D) -> torch.Tensor:
        """
        Compute the HMC force F_mu(x) = (beta/4) * [U_mu(x) * Sigma_mu^dag(x)]_TA
        
        Returns algebra coefficients (omega1, omega2, omega3) for each link.
        
        Args:
            field: gauge field
        Returns:
            force: (Lx, Lt, 2, 3) tensor of algebra coefficients
        """
        Lx, Lt = field.Lx, field.Lt
        force = torch.zeros(Lx, Lt, 2, 3, device=field.device)
        
        for mu in range(2):
            # Compute staple
            staple = self.action.compute_staple(field, mu)  # (Lx, Lt, 4)
            
            # Convert to matrix form for projection
            U_mu = SU2.to_matrix(field.U[:, :, mu, :])  # (Lx, Lt, 2, 2)
            staple_mat = SU2.to_matrix(staple)  # (Lx, Lt, 2, 2)
            # BUG FIX: compute_staple returns Σ^†, not Σ
            # So staple_mat is already Σ^† - no need to take dagger!
            # Old buggy code: staple_dag = staple_mat.conj().transpose(-2, -1)
            
            # U * Sigma^dag
            # U * Σ^† (staple_mat is already Σ^†)
            U_Sigma_dag = torch.matmul(U_mu, staple_mat)
            
            # Traceless anti-Hermitian projection
            Z = SU2Algebra.traceless_antiherm_proj(U_Sigma_dag)
            
            # Extract algebra coefficients
            omega = SU2Algebra.from_matrix(Z)  # (Lx, Lt, 3)
            

            # The derivative of the action wrt U gives:
            # dS/dU = -(beta/2) * Sigma^dag
            # The force on pi is -dS/dU projected onto the algebra.
            # F_mu(x) = -[dS_link / dU_mu(x)]_TA
            #         = (beta/2) * [U * Sigma^dag]_TA ... but this needs more care.
            # computeZ returns -tap_proj(U * Sigma^dag)
            # computeForce multiplies by 0.5 * beta_over_N
            # For SU(2): beta_over_2 = beta/2, factor = 0.5 * beta/2 = beta/4
            # So F = (beta/4) * (-tap_proj(U * Sigma^dag)) = -(beta/4) * tap_proj(U * Sigma^dag)
            
            force[:, :, mu, :] = - (self.action.beta / 4.0) * omega
        
        return force
    
    def update_momentum(self, pi: torch.Tensor, field: LatticeGaugeField2D, 
                        epsilon: float) -> torch.Tensor:
        """
        Update momentum: pi -> pi + epsilon * F
        
        Args:
            pi: current momentum
            field: current gauge field
            epsilon: step size
        Returns:
            updated momentum
        """
        force = self.compute_force(field)
        return pi + epsilon * force
    
    def update_field(self, field: LatticeGaugeField2D, pi: torch.Tensor, 
                     epsilon: float):
        """
        Update gauge field: U -> exp(epsilon * pi) * U
        
        Args:
            field: gauge field to update (modified in place)
            pi: momentum
            epsilon: step size
        """
        # exp(epsilon * pi) where pi is in the algebra
        exp_pi = SU2Algebra.exp(epsilon * pi)  # (Lx, Lt, 2, 4)
        
        # U_new = exp(epsilon * pi) * U
        field.U = SU2.multiply(exp_pi, field.U)
    
    def leapfrog(self, field: LatticeGaugeField2D, pi: torch.Tensor) -> Tuple[LatticeGaugeField2D, torch.Tensor]:
        """
        Perform leapfrog integration for one trajectory.
        
        Leapfrog scheme:
        pi(eps/2) = pi(0) + (eps/2) * F(U(0))
        for i = 1 to n_steps-1:
            U(i*eps) = exp(eps * pi) * U((i-1)*eps)
            pi((i+1/2)*eps) = pi((i-1/2)*eps) + eps * F(U(i*eps))
        U(tau) = exp(eps * pi) * U((n-1)*eps)
        pi(tau) = pi((n-1/2)*eps) + (eps/2) * F(U(tau))
        
        Args:
            field: initial gauge field (will be modified)
            pi: initial momentum (will be modified)
        Returns:
            field, pi: final configuration
        """
        eps = self.epsilon
        
        # Half step for momentum
        pi = self.update_momentum(pi, field, eps / 2)
        
        # Full steps
        for i in range(self.n_steps - 1):
            self.update_field(field, pi, eps)
            pi = self.update_momentum(pi, field, eps)
        
        # Final field update
        self.update_field(field, pi, eps)
        
        # Final half step for momentum
        pi = self.update_momentum(pi, field, eps / 2)
        
        return field, pi
    
    def step(self, field: LatticeGaugeField2D) -> Tuple[bool, float]:
        """
        Perform one HMC step.
        
        Args:
            field: gauge field (modified in place if accepted)
        Returns:
            accepted: whether the proposal was accepted
            delta_H: change in Hamiltonian
        """
        # Save initial configuration
        field_old = field.clone()
        
        # Sample momentum
        pi = SU2Algebra.random_momentum((field.Lx, field.Lt, 2), device=field.device)
        
        # Compute initial Hamiltonian
        H_old = self.compute_hamiltonian(field, pi)
        
        # Leapfrog integration
        field, pi = self.leapfrog(field, pi)
        
        # Compute final Hamiltonian
        H_new = self.compute_hamiltonian(field, pi)
        
        # Metropolis accept/reject
        delta_H = H_new - H_old
        accept_prob = torch.exp(-delta_H).clamp(max=1.0)
        
        if torch.rand(1, device=field.device) < accept_prob:
            accepted = True
        else:
            # Reject: restore old configuration
            field.copy_from(field_old)
            accepted = False
        
        return accepted, delta_H.item()

def exact_plaquette_2d_su2(beta: float) -> float:
    """
    Exact average plaquette for 2D SU(2) gauge theory.
    
    <(1/2) Re Tr[P]> = I_2(beta) / I_1(beta)
    
    where I_n is the modified Bessel function of the first kind.
    """
    return bessel_i(2, beta) / bessel_i(1, beta)

def run_hmc_simulation(L: int, beta: float, n_therm: int, n_traj: int, 
                       tau: float = 1.0, n_steps: int = 10,
                       device: torch.device = None, verbose: bool = True):
    """
    Run HMC simulation and return measurements.
    
    Args:
        L: lattice size (L x L)
        beta: inverse coupling
        n_therm: number of thermalization trajectories
        n_traj: number of measurement trajectories
        tau: trajectory length
        n_steps: number of leapfrog steps
        device: torch device
        verbose: print progress
    
    Returns:
        plaquettes: list of average plaquette values
        accept_rate: acceptance rate
    """
    # Initialize
    field = LatticeGaugeField2D(L, L, device=device)
    field.hot_start()  # Start from random configuration
    
    action = WilsonAction2D(beta)
    hmc = HMC2D(action, tau, n_steps)
    
    # Thermalization
    if verbose:
        print(f"Thermalizing ({n_therm} trajectories)...")
    for i in range(n_therm):
        hmc.step(field)
    
    # Production
    plaquettes = []
    n_accept = 0
    
    if verbose:
        print(f"Production ({n_traj} trajectories)...")
        iterator = tqdm(range(n_traj))
    else:
        iterator = range(n_traj)
    
    for i in iterator:
        accepted, delta_H = hmc.step(field)
        if accepted:
            n_accept += 1
        
        # Measure
        plaq = action.compute_average_plaquette(field).item()
        plaquettes.append(plaq)
    
    accept_rate = n_accept / n_traj
    
    return plaquettes, accept_rate
def compute_single_plaquette(field, x, t):
    """
    Compute plaquette at site (x,t) - helper function.
    
    P(x,t) = (1/2) Re Tr[U_0(x,t) U_1(x+1,t) U_0†(x,t+1) U_1†(x,t)]
    
    Returns: scalar value in [0, 1]
    """
    Lx, Lt = field.Lx, field.Lt
    
    # Get the four links
    U0 = field.U[x, t, 0, :]  # U_0(x,t) - quaternion (4,)
    U1_right = field.U[(x+1)%Lx, t, 1, :]  # U_1(x+1,t)
    U0_up = field.U[x, (t+1)%Lt, 0, :]  # U_0(x,t+1)
    U1 = field.U[x, t, 1, :]  # U_1(x,t)
    
    # Reshape to (1,4) for SU2 operations
    U0 = U0.unsqueeze(0)
    U1_right = U1_right.unsqueeze(0)
    U0_up = U0_up.unsqueeze(0)
    U1 = U1.unsqueeze(0)
    
    # Product: U_0(x,t) * U_1(x+1,t) * U_0†(x,t+1) * U_1†(x,t)
    P = SU2.multiply(U0, U1_right)
    P = SU2.multiply(P, SU2.dagger(U0_up))
    P = SU2.multiply(P, SU2.dagger(U1))
    
    # Convert to matrix and trace
    P_matrix = SU2.to_matrix(P)  # (1, 2, 2) -> take [0]
    trace = torch.trace(P_matrix[0]).real
    
    return (trace / 2.0).item()  # 1/N_c for SU(2)

def compute_wilson_loop(field, R, T):
    """
    Compute rectangular Wilson loop of size R×T.
    
    Path (starting from origin, periodic BC):
    1. Go RIGHT R steps in x-direction (use U_0 links)
    2. Go FORWARD T steps in t-direction (use U_1 links)
    3. Go LEFT R steps back (use U_0† links)
    4. Go BACK T steps (use U_1† links)
    
    W(R,T) = (1/N_c) <Tr[closed loop]>
    
    Args:
        field: LatticeGaugeField2D object
        R: spatial extent (lattice units)
        T: temporal extent (lattice units)
    
    Returns:
        W_avg: average Wilson loop value (scalar)
    """
    Lx, Lt = field.Lx, field.Lt
    
    W_values = []
    
    # Average over all starting positions
    for x_start in range(Lx):
        for t_start in range(Lt):
            
            # Initialize path product as identity
            U_path = SU2.identity((1,), device=field.U.device)
            
            # STEP 1: Right R steps
            # Links are at position, so we need U_0(x, t_start) for x = x_start, x_start+1, ..., x_start+R-1
            for step in range(R):
                x = (x_start + step) % Lx
                t = t_start
                U = field.U[x, t, 0, :].unsqueeze(0)  # U_0(x, t_start)
                U_path = SU2.multiply(U_path, U)
            
            # STEP 2: Forward T steps
            # Now at position (x_start + R, t_start), move forward in t
            for step in range(T):
                x = (x_start + R) % Lx
                t = (t_start + step) % Lt
                U = field.U[x, t, 1, :].unsqueeze(0)  # U_1(x, t)
                U_path = SU2.multiply(U_path, U)
            
            # STEP 3: Left R steps (backwards in x)
            # Now at position (x_start + R, t_start + T), move left
            for step in range(R):
                x = (x_start + R - 1 - step) % Lx  # Go backwards
                t = (t_start + T) % Lt
                U = field.U[x, t, 0, :].unsqueeze(0)  # U_0(x, t)
                U_dag = SU2.dagger(U)  # We're going backwards
                U_path = SU2.multiply(U_path, U_dag)
            
            # STEP 4: Back T steps (backwards in t)
            # Now at position (x_start, t_start + T), move backwards in t
            for step in range(T):
                x = x_start
                t = (t_start + T - 1 - step) % Lt  # Go backwards
                U = field.U[x, t, 1, :].unsqueeze(0)  # U_1(x, t)
                U_dag = SU2.dagger(U)  # We're going backwards
                U_path = SU2.multiply(U_path, U_dag)
            
            # Compute trace of closed loop
            U_matrix = SU2.to_matrix(U_path)  # Shape: (1, 2, 2)
            trace = torch.trace(U_matrix[0]).real  # Extract 2x2 matrix and trace it
            W = (trace / 2.0).item()  # 1/N_c for SU(2)
            W_values.append(W)
    
    # Average over all starting positions
    W_avg = np.mean(W_values)
    
    return W_avg

# Extract string tension from Wilson loops
def extract_string_tension(W_dict):
    """
    From W(R,T) ≈ exp(-V(R)·T), extract static potential V(R)
    Then fit V(R) = σ·R + const to extract string tension σ
    """
    
    # For each R, fit over T to extract V(R)
    R_values = sorted(set([R for R, T in W_dict.keys()]))
    V_values = []
    
    for R in R_values:
        # Get W(R,T) for this R at various T
        T_vals = sorted([T for (r, T) in W_dict.keys() if r == R])
        W_vals = np.array([W_dict[(R, T)] for T in T_vals])
        
        # Take log: ln W(R,T) = -(V(R) + const)·T
        # So V(R) is the slope
        if np.all(W_vals > 0):
            log_W = -np.log(W_vals)
            T_array = np.array(T_vals)
            
            # Linear fit: log_W = V(R)·T_array + ...
            # Slope is V(R)
            V_R = np.polyfit(T_array, log_W, 1)[0]
            V_values.append(V_R)
        else:
            V_values.append(np.nan)
    
    # Now fit V(R) = σ·R + const
    R_fit = np.array(R_values[:len(V_values)])
    V_fit = np.array(V_values)
    
    # Remove NaNs
    mask = ~np.isnan(V_fit)
    R_fit = R_fit[mask]
    V_fit = V_fit[mask]
    
    if len(R_fit) > 1:
        popt = np.polyfit(R_fit, V_fit, 1)
        sigma = popt[0]  # Slope
        return sigma, R_values, V_values, R_fit, popt
    else:
        return None, R_values, V_values, None, None

def compute_all_plaquettes(field):
    """
    Compute plaquette at every lattice site.
    
    Returns:
        P: tensor of shape (Lx, Lt) with plaquette values
    """
    Lx, Lt = field.Lx, field.Lt
    P = torch.zeros(Lx, Lt, device=field.U.device)
    
    for x in range(Lx):
        for t in range(Lt):
            P[x, t] = compute_single_plaquette(field, x, t)
    
    return P

def compute_spatial_correlator(field):
    """
    Compute spatial correlation of plaquettes:
    C(r) = <P(x)·P(x+r)> - <P(x)>²
    
    Averaged over temporal direction.
    
    Returns:
        corr: array of length Lx/2 with correlation function
    """
    Lx, Lt = field.Lx, field.Lt
    
    # Compute all plaquettes
    P = compute_all_plaquettes(field)
    
    # Average over time
    P_x = P.mean(dim=1)  # Shape: (Lx,)
    P_avg = P_x.mean().item()
    
    # Compute correlation for each spatial distance
    r_max = Lx // 2
    corr = torch.zeros(r_max, device=field.U.device)
    
    for r in range(r_max):
        # C(r) = <P(x) P(x+r)> - <P>²
        prod = torch.zeros(1, device=field.U.device)
        
        for x in range(Lx):
            x_r = (x + r) % Lx
            for t in range(Lt):
                prod += (P[x, t] - P_avg) * (P[x_r, t] - P_avg)
        
        corr[r] = prod / (Lx * Lt)
    
    return corr.cpu().numpy()

def extract_correlation_length(corr):
    """
    Fit correlation to C(r) = C₀ exp(-r/ξ) and extract ξ.
    
    Args:
        corr: array of correlation values
    
    Returns:
        xi: correlation length, or None if fit fails
    """
    r = np.arange(len(corr))
    
    def exp_decay(r, C0, xi):
        return np.abs(C0) * np.exp(-r / xi)
    
    try:
        # Fit using first half of data (to avoid noise at large r)
        r_fit = r[1:min(15, len(r))]  # Skip r=0, use up to r=15
        c_fit = np.abs(corr[1:min(15, len(corr))])
        
        popt, _ = curve_fit(exp_decay, r_fit, c_fit,
                           p0=[corr[1] if len(corr) > 1 else 0.1, 2.0],
                           maxfev=5000)
        xi = popt[1]
        return xi
    except Exception as e:
        print(f"Fit failed: {e}")
        return None

def run_finite_volume_study(beta, L_values=[8, 12, 16, 20, 24], n_configs=500):
    """
    Run HMC simulations at multiple lattice sizes and measure average plaquette.
    
    Args:
        beta: coupling parameter
        L_values: list of lattice sizes to simulate
        n_configs: number of MD trajectories after thermalization
    
    Returns:
        results: dict with keys L containing {plaq, err, acc}
        exact: exact value from I_2(beta)/I_1(beta)
    """
    
    results = {}
    
    for L in L_values:
        print(f"\nRunning L={L}×{L}, β={beta}...")
        
        # Initialize
        field = LatticeGaugeField2D(L, L, device=device)
        field.hot_start()
        action = WilsonAction2D(beta)
        hmc = HMC2D(action, tau=1.0, n_steps=20)
        
        # Thermalize
        print(f"  Thermalizing ({300} trajectories)...")
        for _ in tqdm(range(300), leave=False):
            hmc.step(field)
        
        # Production
        print(f"  Production ({n_configs} trajectories)...")
        plaquettes = []
        n_accept = 0
        
        for _ in tqdm(range(n_configs), leave=False):
            accepted, _ = hmc.step(field)
            if accepted:
                n_accept += 1
            
            # Measure plaquette
            plaq = action.compute_average_plaquette(field).item()
            plaquettes.append(plaq)
        
        # Statistics
        plaq_mean = np.mean(plaquettes)
        plaq_std = np.std(plaquettes)
        plaq_err = plaq_std / np.sqrt(len(plaquettes))
        acc_rate = n_accept / n_configs
        
        results[L] = {
            'plaq': plaq_mean,
            'err': plaq_err,
            'std': plaq_std,
            'acc': acc_rate,
            'n_configs': len(plaquettes)
        }
        
        print(f"  <P> = {plaq_mean:.6f} ± {plaq_err:.6f}")
        print(f"  Acceptance: {acc_rate:.1%}")
    
    # Exact result
    exact = bessel_i(2, beta) / bessel_i(1, beta)
    
    return results, exact
def extrapolate_to_infinite_volume(results_fv, exact_val, beta):
    """
    Fit <P>_L = <P>_∞ + a/L + b/L² and extract P_∞
    """
    
    L_array = np.array(sorted(results_fv.keys()), dtype=float)
    plaq_array = np.array([results_fv[int(L)]['plaq'] for L in L_array])
    err_array = np.array([results_fv[int(L)]['err'] for L in L_array])
    
    # Fit: P_L = P_inf + a/L + b/L²
    def finite_vol(L, P_inf, a, b):
        return P_inf + a/L + b/(L**2)
    
    try:
        popt, pcov = curve_fit(finite_vol, L_array, plaq_array,
                              p0=[exact_val, 0.01, 0.01],
                              sigma=err_array, absolute_sigma=True,
                              maxfev=5000)
        
        P_inf = popt[0]
        P_inf_err = np.sqrt(pcov[0, 0])
        
        return P_inf, P_inf_err, popt, pcov
    except Exception as e:
        print(f"Extrapolation failed: {e}")
        return None, None, None, None

def finite_vol(L, P_inf, a, b):
    return P_inf + a/L + b/(L**2)


def compute_topological_charge(field: LatticeGaugeField2D) -> float:
    """
    Compute the topological charge Q of a 2D SU(2) gauge field configuration.
    
    For 2D SU(2) gauge theory, Q ∈ ℤ₂ = {0, 1}, representing two topological sectors.
    
    CORRECT FORMULA:
    Q = (1/4π) Σ_plaquettes θ_plaquette
    
    where the plaquette angle is:
    θ = 2 * arctan2(||a_vector||, a₀)
    
    with a₀, a₁, a₂, a₃ the quaternion components of the plaquette.
    
    The relation: Tr(U_plaq) = 2cos(θ/2) = 2a₀ confirms the angle extraction.
    
    Args:
        field: LatticeGaugeField2D object
    
    Returns:
        Q: topological charge in ℤ₂ (either 0 or 1)
    """
    action = WilsonAction2D(beta=1.0)
    plaq_quats = action.compute_plaquette(field)
    
    # Extract quaternion components
    a0 = plaq_quats[..., 0]
    a1 = plaq_quats[..., 1]
    a2 = plaq_quats[..., 2]
    a3 = plaq_quats[..., 3]
    
    # Compute vector part norm: ||a_vector|| = sqrt(a1² + a2² + a3²)
    a_vec_norm = torch.sqrt(a1**2 + a2**2 + a3**2)
    
    # Extract rotation angle: θ = 2 * arctan2(||a_vector||, a₀)
    # This is correct because Tr(U) = 2*a₀ = 2*cos(θ/2)
    theta = 2.0 * torch.atan2(a_vec_norm, a0)
    # Wrap to [0, 2π) to avoid branch cut issues
    theta = theta % (2 * np.pi)
    
    # Sum all plaquette angles
    sum_theta = torch.sum(theta)
    
    # Apply the correct normalization: Q = (1/4π) * Σθ
    Q_raw = sum_theta / (4.0 * np.pi)
    
    # Reduce to fundamental domain [0, 1) to extract ℤ₂ charge
    Q_winding = Q_raw % 1.0
    
    # Classify in ℤ₂:
    # Q = 0 if winding ∈ [0, 0.5) (even sector)
    # Q = 1 if winding ∈ [0.5, 1) (odd sector)
    Q = 1 if Q_winding >= 0.5 else 0
    
    return float(Q)


def apply_gauge_transformation(U_field, omega_field):
    """
    Apply a local gauge transformation to a gauge field.
    
    Mathematical formula:
    U'_μ(x) = Ω(x) · U_μ(x) · Ω†(x + μ̂)
    
    Args:
        U_field: gauge field of shape (Lx, Lt, 2, 4) with quaternion components
        omega_field: gauge transformation elements of shape (Lx, Lt, 4)
    
    Returns:
        U_prime: transformed gauge field of shape (Lx, Lt, 2, 4)
    """
    Lx, Lt = U_field.shape[0], U_field.shape[1]
    U_prime = torch.zeros_like(U_field)
    
    # Direction 0 (spatial x-direction)
    # U'_0(x,t) = Ω(x,t) · U_0(x,t) · Ω†(x+1,t)
    omega_x = omega_field                        # Ω(x,t)
    U_0 = U_field[:, :, 0, :]                    # U_0(x,t)
    omega_x_plus = torch.roll(omega_field, shifts=-1, dims=0)  # Ω(x+1,t)
    omega_x_plus_dag = omega_x_plus.clone()
    omega_x_plus_dag[..., 1:] = -omega_x_plus_dag[..., 1:]  # Conjugate
    
    # First multiply: Ω(x,t) · U_0(x,t)
    temp = quaternion_multiply(omega_x, U_0)
    # Then multiply: (...) · Ω†(x+1,t)
    U_prime[:, :, 0, :] = quaternion_multiply(temp, omega_x_plus_dag)
    
    # Direction 1 (temporal t-direction)  
    # U'_1(x,t) = Ω(x,t) · U_1(x,t) · Ω†(x,t+1)
    U_1 = U_field[:, :, 1, :]                    # U_1(x,t)
    omega_t_plus = torch.roll(omega_field, shifts=-1, dims=1)  # Ω(x,t+1)
    omega_t_plus_dag = omega_t_plus.clone()
    omega_t_plus_dag[..., 1:] = -omega_t_plus_dag[..., 1:]  # Conjugate
    
    # First multiply: Ω(x,t) · U_1(x,t)
    temp = quaternion_multiply(omega_x, U_1)
    # Then multiply: (...) · Ω†(x,t+1)
    U_prime[:, :, 1, :] = quaternion_multiply(temp, omega_t_plus_dag)
    
    return U_prime


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions: q_out = q1 * q2
    
    Quaternion representation: (a₀, a₁, a₂, a₃) representing
    a₀·I + i·(a₁·σ₁ + a₂·σ₂ + a₃·σ₃)
    
    Args:
        q1, q2: tensors of shape (..., 4) with quaternion components
    
    Returns:
        q_out: product quaternion of shape (..., 4)
    """
    a0, a1, a2, a3 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    b0, b1, b2, b3 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    q_out = torch.zeros_like(q1)
    
    # Standard quaternion multiplication formula
    q_out[..., 0] = a0*b0 - a1*b1 - a2*b2 - a3*b3
    q_out[..., 1] = a0*b1 + a1*b0 - a2*b3 + a3*b2
    q_out[..., 2] = a0*b2 + a1*b3 + a2*b0 - a3*b1
    q_out[..., 3] = a0*b3 - a1*b2 + a2*b1 + a3*b0
    
    return q_out


def generate_random_gauge_transformation(Lx, Lt, device):
    """
    Generate a random gauge transformation (random SU(2) element at each site).
    
    Args:
        Lx, Lt: lattice dimensions
        device: torch device
    
    Returns:
        omega: tensor of shape (Lx, Lt, 4) with random normalized quaternions
    """
    # Generate random Gaussian quaternions
    omega = torch.randn(Lx, Lt, 4, device=device)
    # Normalize to unit length (SU(2) ≅ S^3)
    omega = omega / torch.sqrt((omega**2).sum(dim=-1, keepdim=True) + 1e-8)
    return omega


# Verify gauge invariance: plaquette should remain unchanged

def compute_plaquette_quaternion(U_field, x, t, Lx, Lt):
    """
    Compute plaquette at position (x, t).
    P(x,t) = U_0(x,t) · U_1(x+1,t) · U_0†(x,t+1) · U_1†(x,t)
    
    Returns the quaternion representation and its trace.
    """
    U0 = U_field[x, t, 0, :]
    U1_right = U_field[(x+1)%Lx, t, 1, :]
    U0_up = U_field[x, (t+1)%Lt, 0, :]
    U1 = U_field[x, t, 1, :]
    
    # Add batch dimension for quaternion multiply
    U0 = U0.unsqueeze(0)
    U1_right = U1_right.unsqueeze(0)
    U0_up = U0_up.unsqueeze(0)
    U1 = U1.unsqueeze(0)
    
    P = quaternion_multiply(U0, U1_right)
    U0_dag = U0_up.clone()
    U0_dag[..., 1:] = -U0_dag[..., 1:]
    P = quaternion_multiply(P, U0_dag)
    U1_dag = U1.clone()
    U1_dag[..., 1:] = -U1_dag[..., 1:]
    P = quaternion_multiply(P, U1_dag)
    
    # Trace = 2 * a0 for quaternion (a0, a1, a2, a3)
    trace = 2.0 * P[0, 0]
    
    return P[0], trace
