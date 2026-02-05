"""
SU(2) Diffusion Model
=====================================
A diffusion model for SU(2) two-dimensional pure gauge lattice field theory.

By B.-D. Sun.
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import pi
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from tqdm import tqdm
from scipy.special import iv as bessel_i
from scipy.optimize import curve_fit
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from su2_hmc import *



# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_configurations(path):
    """Load configurations from .pt file."""
    print(f"Loading configurations from: {path}")
    configs = torch.load(path)
    print(f"  Shape: {configs.shape}")
    print(f"  dtype: {configs.dtype}")
    return configs


def reshape_for_diffusion(configs_tensor):
    """
    Reshape from (N, Lx, Lt, 2, 4) to (N, 8, Lx, Lt) for diffusion model.
    
    Channel mapping:
      - Channels 0-3: quaternion (a0, a1, a2, a3) for direction 0
      - Channels 4-7: quaternion (a0, a1, a2, a3) for direction 1
    """
    N, Lx, Lt, dirs, quat = configs_tensor.shape
    # (N, Lx, Lt, 2, 4) -> (N, Lx, Lt, 8) -> (N, 8, Lx, Lt)
    reshaped = configs_tensor.view(N, Lx, Lt, dirs * quat)
    reshaped = reshaped.permute(0, 3, 1, 2)
    return reshaped.double()


def reshape_from_diffusion(config_tensor, Lx, Lt):
    """
    Reshape from (8, Lx, Lt) back to (Lx, Lt, 2, 4) for physics computations.
    """
    # (8, Lx, Lt) -> (Lx, Lt, 8) -> (Lx, Lt, 2, 4)
    reshaped = config_tensor.permute(1, 2, 0)
    reshaped = reshaped.view(Lx, Lt, 2, 4)
    return reshaped

def compute_plaquette_for_generated(generated_configs, Lx, Lt, beta, device='cpu'):
    """
    Compute average plaquette for a batch of generated configurations.
    
    Args:
        generated_configs: tensor of shape (N, 8, Lx, Lt)
        Lx, Lt: lattice dimensions
        beta: coupling for action object
        device: computation device
    
    Returns:
        list of plaquette values
    """
    action_obj = WilsonAction2D(beta=beta)
    temp_field = LatticeGaugeField2D(Lx, Lt, device=device)
    
    plaquettes = []
    for i in range(generated_configs.shape[0]):
        config = generated_configs[i].to(device)
        temp_field.U = reshape_from_diffusion(config, Lx, Lt)
        plaq = action_obj.compute_average_plaquette(temp_field).item()
        plaquettes.append(plaq)
    
    return plaquettes

def validate_model(diffusion_model, Lx, Lt, beta, n_samples, steps, device):
    """
    Generate samples and compute physics validation metrics.
    
    Returns:
        dict with validation results
    """
    diffusion_model.unet.eval()
    
    with torch.no_grad():
        # Generate samples
        generated = diffusion_model.sample(
            num_samples=n_samples,
            Lx=Lx,
            Lt=Lt,
            steps=steps,
            beta_target=beta
        )
        
        # Compute plaquettes
        plaquettes = compute_plaquette_for_generated(
            generated.cpu(), Lx, Lt, beta, device='cpu'
        )
    
    diffusion_model.unet.train()
    
    # Statistics
    plaq_mean = np.mean(plaquettes)
    plaq_std = np.std(plaquettes)
    plaq_err = plaq_std / np.sqrt(len(plaquettes))
    exact_plaq = bessel_i(2, beta) / bessel_i(1, beta)
    
    return {
        'plaq_mean': plaq_mean,
        'plaq_std': plaq_std,
        'plaq_err': plaq_err,
        'exact_plaq': exact_plaq,
        'sigma_diff': abs(plaq_mean - exact_plaq) / plaq_err if plaq_err > 0 else float('inf'),
        'plaquettes': plaquettes
    }

def verify_training_data(configs_tensor, beta, n_check=5):
    """
    Verify that training data has correct physics before training.
    """
    print("\nVerifying training data quality...")
    
    Lx, Lt = configs_tensor.shape[1], configs_tensor.shape[2]
    action_obj = WilsonAction2D(beta=beta)
    temp_field = LatticeGaugeField2D(Lx, Lt, device='cpu')
    
    exact_plaq = bessel_i(2, beta) / bessel_i(1, beta)
    
    # Check random configurations
    indices = np.random.choice(configs_tensor.shape[0], size=n_check, replace=False)
    plaquettes = []
    
    for idx in indices:
        temp_field.U = configs_tensor[idx]
        plaq = action_obj.compute_average_plaquette(temp_field).item()
        plaquettes.append(plaq)
        print(f"  Config {idx}: ⟨P⟩ = {plaq:.6f}")
    
    mean_plaq = np.mean(plaquettes)
    print(f"\n  Sample mean ⟨P⟩: {mean_plaq:.6f}")
    print(f"  Exact ⟨P⟩:       {exact_plaq:.6f}")
    
    # Check quaternion norms
    norms = torch.sqrt((configs_tensor[0]**2).sum(dim=-1))
    print(f"\n  Quaternion norms (config 0):")
    print(f"    Mean: {norms.mean().item():.8f}")
    print(f"    Std:  {norms.std().item():.8f}")
    print(f"    Min:  {norms.min().item():.8f}")
    print(f"    Max:  {norms.max().item():.8f}")
    
    if abs(mean_plaq - exact_plaq) > 0.1:
        print("\n  ⚠ WARNING: Training data plaquette differs significantly from exact!")
    else:
        print("\n  ✓ Training data looks correct!")
    
    return mean_plaq





class DiffusionModel:
    """
    Complete diffusion model for SU(2) gauge fields.
    
    Implements training and sampling procedures.
    """
    
    def __init__(self, unet, noise_schedule, device, learning_rate=1e-4, beta_training=None):
        """
        Args:
            unet: DiffusionUNet model
            noise_schedule: NoiseSchedule object
            device: torch device
            learning_rate: optimizer learning rate
            beta_training: the coupling β₀ at which training data was generated
                          THIS IS CRUCIAL FOR PHYSICS-CONDITIONING!
        """
        self.unet = unet.to(device)
        self.noise_schedule = noise_schedule
        self.device = device
        if beta_training is None:
            raise ValueError(
                "beta_training must be specified! "
                "This is the coupling at which your HMC training data was generated. "
                "For example, if you generated configs at β=2.0, set beta_training=2.0"
            )
        self.beta_training = beta_training
        print(f"[PhysicsConditionedDiffusionModel] Training coupling β₀ = {beta_training}")
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        self.loss_history = []
    
    def add_noise(self, x0, t):
        """
        Add noise to clean sample according to schedule.
        
        x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        
        Args:
            x0: clean sample (batch, channels, height, width)
            t: timestep (batch,)
        
        Returns:
            x_t: noisy sample (batch, channels, height, width)
            epsilon: the noise that was added (batch, channels, height, width)
        """
        # Sample random noise
        epsilon = torch.randn_like(x0)
        
        # Get noise levels
        alpha_bar = self.noise_schedule.get_alpha_bar(t, self.device)
        
        # Reshape for broadcasting: (batch,) -> (batch, 1, 1, 1)
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        
        # Create noisy sample
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_1_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        
        x_t = sqrt_alpha_bar * x0 + sqrt_1_minus_alpha_bar * epsilon
        
        return x_t, epsilon
    
    def train_step(self, batch):
        """
        Single training step.
        
        Args:
            batch: tensor of shape (batch_size, channels, height, width)
        
        Returns:
            loss: scalar loss value
        """
        batch_size = batch.shape[0]
        
        # Sample random timesteps for this batch
        t = torch.randint(0, self.noise_schedule.T, (batch_size,), device=self.device)
        
        # Add noise
        x_t, epsilon = self.add_noise(batch, t)
        
        # Predict noise with U-Net
        epsilon_pred = self.unet(x_t, t)
        
        # Compute MSE loss
        loss = F.mse_loss(epsilon_pred, epsilon)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Args:
            dataloader: PyTorch DataLoader
        
        Returns:
            avg_loss: average loss over epoch
        """
        self.unet.train()
        total_loss = 0.0
        
        for batch in dataloader:
            batch = batch[0].to(self.device)
            loss = self.train_step(batch)
            total_loss += loss
        
        avg_loss = total_loss / len(dataloader)
        self.loss_history.append(avg_loss)
        self.scheduler.step()
        
        return avg_loss
    '''
    @torch.no_grad()
    def sample(self, num_samples, Lx=8, Lt=8, steps=100, beta_target=None):
        """
        Generate new gauge field configurations using reverse diffusion.
    
        Args:
            num_samples: number of configurations to generate
            Lx, Lt: lattice dimensions
            steps: number of denoising steps (< T for fast sampling)
            beta_target: TARGET coupling β for generation
                        - If None, uses β₀ (same as training)
                        - If different from β₀, applies physics-conditioning
        
        Returns:
            samples: tensor of shape (num_samples, 8, Lx, Lt) with normalized quaternions
        """
        self.unet.eval()
        # Determine the coupling ratio
        if beta_target is None:
            beta_target = self.beta_training  # Generate at training coupling
        
        # =====================================================================
        # THE PHYSICS-CONDITIONING RATIO
        # =====================================================================
        beta_ratio = beta_target / self.beta_training
        
        #print(f"[Sampling] β_target = {beta_target}, β_training = {self.beta_training}")
        #print(f"[Sampling] Physics-conditioning ratio: β/β₀ = {beta_ratio:.4f}")
        
        #if abs(beta_ratio - 1.0) > 0.01:
        #    print(f"[Sampling] NOTE: Generating at DIFFERENT coupling than training!")
        # Start with pure noise
        x = torch.randn(num_samples, 8, Lx, Lt, device=self.device, dtype=torch.float64)
        
        # Create timestep sequence (from T-1 down to 0)
        # Use integer steps properly spaced
        timesteps = torch.linspace(self.noise_schedule.T - 1, 0, steps).long()
    
        for i, t_current in enumerate(timesteps):
            # Current timestep for all samples
            t = torch.full((num_samples,), t_current.item(), dtype=torch.long, device=self.device)
        
            # Predict noise
            epsilon_pred = self.unet(x, t)

            # =================================================================
            # PHYSICS-CONDITIONING: RESCALE THE PREDICTED NOISE
            # =================================================================
            # This is the KEY modification!
            # The network learned s_θ ≈ -β₀ ∇S̃
            # We want s ≈ -β ∇S̃
            # So we multiply by β/β₀
            epsilon_pred = beta_ratio * epsilon_pred
            # =================================================================

            # Get alpha_bar for current timestep
            alpha_bar_t = self.noise_schedule.get_alpha_bar(t, self.device).view(-1, 1, 1, 1)
            
            # Get alpha_bar for previous timestep (t-1), clamped to valid range
            t_prev = max(0, t_current.item() - 1)
            alpha_bar_t_prev = self.noise_schedule.get_alpha_bar(
                torch.full((num_samples,), t_prev, dtype=torch.long, device=self.device),
                self.device
            ).view(-1, 1, 1, 1)
        
            # Compute alpha_t = alpha_bar_t / alpha_bar_{t-1}
            # Add small epsilon to prevent division by zero
            alpha_t = alpha_bar_t / (alpha_bar_t_prev + 1e-8)
            alpha_t = torch.clamp(alpha_t, min=1e-8, max=1.0)
        
            # Compute beta_t = 1 - alpha_t
            beta_t = 1.0 - alpha_t
        
            # Compute the mean of p(x_{t-1} | x_t)
            # Avoid division by zero in sqrt(1 - alpha_bar_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))
            
            # x_0 prediction (optional, for debugging)
            # x0_pred = (x - sqrt_one_minus_alpha_bar * epsilon_pred) / torch.sqrt(alpha_bar_t + 1e-8)
        
            # Mean of the reverse process
            mean = (1.0 / torch.sqrt(alpha_t + 1e-8)) * (
                x - (beta_t / sqrt_one_minus_alpha_bar) * epsilon_pred
            )
        
            # Add noise (except at the last step)
            if i < len(timesteps) - 1:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean + sigma * noise
            else:
                x = mean
        
            # Clamp to prevent extreme values from accumulating
            x = torch.clamp(x, min=-10.0, max=10.0)
    
        # Project back to SU(2) manifold: normalize quaternions
        x_normalized = self._normalize_quaternions(x)
        
        return x_normalized
        '''
    
    @torch.no_grad()
    def sample(self, num_samples, Lx=8, Lt=8, steps=100, beta_target=None, eta=0.0):
        """
        Generate new gauge field configurations using DDIM reverse diffusion.
    
        DDIM (Denoising Diffusion Implicit Models) is used because:
        1. It works correctly with subsampled timesteps (steps < T)
        2. It's numerically stable for large step sizes
        3. The eta parameter controls stochasticity (eta=0: deterministic, eta=1: DDPM-like)

        Args:
            num_samples: number of configurations to generate
            Lx, Lt: lattice dimensions
            steps: number of denoising steps (can be much less than T)
            beta_target: TARGET coupling β for physics-conditioning
            eta: stochasticity parameter (0 = deterministic DDIM, 1 = fully stochastic)
        
        Returns:
            samples: tensor of shape (num_samples, 8, Lx, Lt) with normalized quaternions
        """
        self.unet.eval()
        network_strength = 0.15 if beta_target > 4 else 0.5
    
        # Physics-conditioning ratio
        if beta_target is None:
            beta_target = self.beta_training
        beta_ratio = beta_target / self.beta_training
    
        # Start with pure noise
        x = predict_starting_noise(beta_target) * torch.randn(num_samples, 8, Lx, Lt, device=self.device, dtype=torch.float64) / 2.0
    
        # Create timestep sequence (from T-1 down to 0)
        timesteps = torch.linspace(self.noise_schedule.T - 1, 0, steps).long()

        for i, t_current in enumerate(timesteps):
            t = torch.full((num_samples,), t_current.item(), dtype=torch.long, device=self.device)
            net_pred = self.unet(x, t)
            epsilon_pred = (1.0 - network_strength) * 1.0 + network_strength * net_pred * beta_ratio
            # Predict noise
            #epsilon_pred = #self.unet(x, t)
            # Physics-conditioning: rescale predicted noise
            #epsilon_pred = 1.0#beta_ratio * epsilon_pred
            # Get α̅_t for current timestep
            alpha_bar_t = self.noise_schedule.get_alpha_bar(t, self.device).view(-1, 1, 1, 1)
        
            # Get α̅_{t_next} for next timestep in our sequence
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1].item()
            else:
                t_next = 0
        
            alpha_bar_t_next = self.noise_schedule.get_alpha_bar(
                torch.full((num_samples,), t_next, dtype=torch.long, device=self.device),
                self.device
            ).view(-1, 1, 1, 1)
        
            # =================================================================
            # DDIM SAMPLING (works correctly for any step size)
            # =================================================================
            # 
            # Why DDIM instead of DDPM?
            # --------------------------
            # DDPM formula: μ = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε)
            #   where α_t = ᾱ_t / ᾱ_{t-1}
            # 
            # Problem: When steps are subsampled (e.g., t=999 → t=900),
            #   α_t = ᾱ_999 / ᾱ_900 can be very small (e.g., 0.01)
            #   Then 1/√α_t ≈ 10, which amplifies noise and causes instability
            #
            # DDIM formula: x_{t_next} = √ᾱ_{t_next} * x₀_pred + √(1-ᾱ_{t_next}) * ε_pred
            #   This uses √(ᾱ_{t_next}/ᾱ_t) as a MULTIPLIER, not divisor
            #   Much more stable for large step sizes!
            # =================================================================
        
            # Step 1: Predict x_0 from x_t and predicted noise
            # From forward process: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
            # Rearranging: x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
            x0_pred = (x - sqrt_one_minus_alpha_bar_t * epsilon_pred) / sqrt_alpha_bar_t
        
            # Clamp x0 prediction to prevent extreme values
            x0_pred = torch.clamp(x0_pred, -5.0, 5.0)
        
            # Step 2: Compute x_{t_next}
            sqrt_alpha_bar_t_next = torch.sqrt(alpha_bar_t_next)
            sqrt_one_minus_alpha_bar_t_next = torch.sqrt(1.0 - alpha_bar_t_next)
        
            if i < len(timesteps) - 1:
                # =============================================================
                # DDIM with controllable stochasticity (eta parameter)
                # =============================================================
                # sigma_t controls how much fresh noise to add
                # eta = 0: deterministic (reuse epsilon_pred direction)
                # eta = 1: maximum stochasticity (like DDPM)
                # =============================================================
            
                # Compute sigma for stochastic DDIM
                # σ_t = η * √((1-ᾱ_{t_next})/(1-ᾱ_t)) * √(1 - ᾱ_t/ᾱ_{t_next})
                if eta > 0:
                    sigma_t = eta * torch.sqrt(
                        (1.0 - alpha_bar_t_next) / (1.0 - alpha_bar_t + 1e-8)
                    ) * torch.sqrt(
                        1.0 - alpha_bar_t / (alpha_bar_t_next + 1e-8)
                    )
                    sigma_t = torch.clamp(sigma_t, min=0.0)
                else:
                    sigma_t = 0.0
            
                # Direction pointing to x_t (the "predicted noise" direction)
                # In deterministic DDIM, we reuse epsilon_pred
                # The coefficient for epsilon_pred is adjusted when eta > 0
                if eta > 0 and isinstance(sigma_t, torch.Tensor):
                    # Adjusted coefficient to account for added noise
                    coef_epsilon = torch.sqrt(1.0 - alpha_bar_t_next - sigma_t**2)
                else:
                    coef_epsilon = sqrt_one_minus_alpha_bar_t_next
                
                # DDIM update
                x = sqrt_alpha_bar_t_next * x0_pred + coef_epsilon * epsilon_pred
            
                # Add stochastic noise if eta > 0
                if eta > 0:
                    noise = torch.randn_like(x)
                    x = x + sigma_t * noise
            else:
                # Final step: just return x_0 prediction
                x = x0_pred
        
            # Clamp to prevent numerical issues
            x = torch.clamp(x, -10.0, 10.0)

        # Project back to SU(2) manifold
        x_normalized = self._normalize_quaternions(x)
    
        return x_normalized
    
    @staticmethod
    def _normalize_quaternions(x):
        """
        Project sampled values back to SU(2) by normalizing quaternions.
        
        For each of the 2 directions, the 4 quaternion components are normalized.
        
        Args:
            x: tensor of shape (batch, 8, Lx, Lt)
                where channels 0-3 are first quaternion, 4-7 are second
        
        Returns:
            x_norm: normalized tensor (batch, 8, Lx, Lt)
        """
        batch_size, channels, Lx, Lt = x.shape
        
        # Reshape to separate quaternions: (batch, 2, 4, Lx, Lt)
        x_reshaped = x.view(batch_size, 2, 4, Lx, Lt)
        
        # Normalize each quaternion
        norms = torch.sqrt((x_reshaped ** 2).sum(dim=2, keepdim=True) + 1e-8)
        x_normalized = x_reshaped / norms
        
        # Reshape back to (batch, 8, Lx, Lt)
        x_normalized = x_normalized.view(batch_size, channels, Lx, Lt)
        
        return x_normalized




class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion timesteps.
    
    Maps timestep t ∈ [0, T] to a high-dimensional vector that captures
    frequency information at multiple scales.
    
    Formula:
    PE(t, 2i) = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: timestep tensor of shape (batch_size,)
        
        Returns:
            embedding: sinusoidal embedding of shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Create frequency bands
        # freq_i = 1 / 10000^(2i/d)
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # Apply to timesteps
        emb = t[:, None] * emb[None, :]
        
        # Concatenate sin and cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        return emb


class DiffusionUNet(nn.Module):
    """
    U-Net architecture for SU(2) diffusion model.
    
    Designed for 2D lattice gauge field with:
    - Input channels: 8 (4 quaternion components × 2 directions)
    - Periodic boundary conditions (periodic padding)
    - Time embedding for diffusion timestep
    - Skip connections for preserving information
    
    For 8×8 input: only 2 pooling levels (8→4→2→4→8)
    """
    
    def __init__(self, in_channels=8, out_channels=8, time_emb_dim=128, 
                 base_channels=32, max_channels=256):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        
        # ===== Time Embedding =====
        self.time_embedding = SinusoidalPositionalEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # ===== Encoder (Downsampling) =====
        # Level 1: 8×8 → 8×8
        self.enc1 = ConvResidualBlock(in_channels, base_channels, time_emb_dim)
        # Level 2: 4×4 → 4×4 (after pooling)
        self.enc2 = ConvResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        
        # ===== Bottleneck ===== (at 2×2 after another pool)
        self.bottleneck = ConvResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        
        # ===== Decoder (Upsampling) =====
        # Level 2: receives bottleneck (upsampled to 4×4) + enc2 skip (4×4)
        self.dec2 = ConvResidualBlock(base_channels * 4, base_channels, time_emb_dim)
        # Level 1: receives dec2 (upsampled to 8×8) + enc1 skip (8×8)
        self.dec1 = ConvResidualBlock(base_channels * 2, base_channels, time_emb_dim)
        
        # ===== Final Output Layer =====
        self.final = nn.Sequential(
            nn.GroupNorm(4, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular')
        )
    
    def forward(self, x, t):
        """
        Forward pass of U-Net.
        
        Args:
            x: input tensor of shape (batch_size, 8, Lx, Lt)
            t: timestep tensor of shape (batch_size,)
        
        Returns:
            output: noise prediction of shape (batch_size, 8, Lx, Lt)
        """
        # Encode time
        time_emb = self.time_embedding(t)
        time_emb = self.time_mlp(time_emb)  # (batch, time_emb_dim)
        
        # Encoder path
        # Level 1: 8×8
        enc1 = self.enc1(x, time_emb)  # (batch, base_channels, 8, 8)
        
        # Level 2: 4×4
        enc2 = self.enc2(F.avg_pool2d(enc1, 2), time_emb)  # (batch, base_channels*2, 4, 4)
        
        # Bottleneck: 2×2
        bottleneck = self.bottleneck(F.avg_pool2d(enc2, 2), time_emb)  # (batch, base_channels*2, 2, 2)
        
        # Decoder path
        # Upsample bottleneck 2×2 → 4×4, concat with enc2
        dec2 = F.interpolate(bottleneck, scale_factor=2, mode='nearest')  # (batch, base_channels*2, 4, 4)
        dec2 = torch.cat([dec2, enc2], dim=1)  # (batch, base_channels*4, 4, 4)
        dec2 = self.dec2(dec2, time_emb)  # (batch, base_channels, 4, 4)
        
        # Upsample dec2 4×4 → 8×8, concat with enc1
        dec1 = F.interpolate(dec2, scale_factor=2, mode='nearest')  # (batch, base_channels, 8, 8)
        dec1 = torch.cat([dec1, enc1], dim=1)  # (batch, base_channels*2, 8, 8)
        dec1 = self.dec1(dec1, time_emb)  # (batch, base_channels, 8, 8)
        
        # Final output layer
        out = self.final(dec1)
        
        return out


class ConvResidualBlock(nn.Module):
    """
    Convolutional residual block with time embedding conditioning.
    
    Block structure:
    1. GroupNorm + GELU activation
    2. Conv2d (periodic padding for lattice BC)
    3. Add time embedding
    4. GroupNorm + GELU activation
    5. Conv2d (periodic padding)
    6. Residual connection (if same # channels)
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(4, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               padding=1, padding_mode='circular')
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, padding_mode='circular')
        
        # Residual connection (only if channels match)
        self.use_res = (in_channels == out_channels)
        if not self.use_res:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x, time_emb):
        """
        Args:
            x: input tensor (batch, in_channels, height, width)
            time_emb: time embedding (batch, time_emb_dim)
        
        Returns:
            out: output tensor (batch, out_channels, height, width)
        """
        # First conv block
        h = self.norm1(x)
        h = F.gelu(h)
        h = self.conv1(h)
        
        # Add time embedding (reshape to broadcast across spatial dims)
        time_emb_proj = self.time_proj(time_emb)  # (batch, out_channels)
        time_emb_proj = time_emb_proj[:, :, None, None]  # (batch, out_channels, 1, 1)
        h = h + time_emb_proj
        
        # Second conv block
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.conv2(h)
        
        # Residual connection
        if self.use_res:
            h = h + x
        else:
            h = h + self.skip(x)
        
        return h


class NoiseSchedule:
    """
    Noise schedule for diffusion process.
    
    Controls how much signal vs noise at each timestep:
    x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
    
    We use a cosine schedule which performs well empirically.
    """
    
    def __init__(self, T=1000, s=0.008):
        """
        Args:
            T: number of diffusion steps
            s: small offset to prevent α from being 1
        """
        self.T = T
        
        # Cosine schedule
        t = torch.arange(T + 1, dtype=torch.float32)
        f_t = torch.cos((pi / 2) * (t / T + s) / (1 + s)) ** 2
        
        # ᾱ_t: cumulative product of alphas
        self.alpha_bar = f_t / f_t[0]
        
        # Register for easy access
        self.register_buffer('_alpha_bar', self.alpha_bar)
    
    def get_alpha_bar(self, t, device):
        """
        Get ᾱ_t for given timesteps.
        
        Args:
            t: timestep tensor (batch_size,) with values in [0, T]
            device: torch device
        
        Returns:
            alpha_bar: ᾱ_t values
        """
        # Clamp to valid range
        t = torch.clamp(t, 0, self.T)
        return self.alpha_bar[t].to(device)
    
    def get_sigmas(self, t, device):
        """
        Get noise levels √(1 - ᾱ_t) for given timesteps.
        """
        alpha_bar = self.get_alpha_bar(t, device)
        return torch.sqrt(1.0 - alpha_bar)

    def register_buffer(self, name, tensor):
        """Simple buffer registration (for CPU use)"""
        setattr(self, name, tensor)

def compute_config_statistics(configs_tensor):
    """
    Compute simple statistics on quaternion field.
    
    Args:
        configs_tensor: (N, 8, Lx, Lt) or (N, Lx, Lt, 2, 4)
    
    Returns:
        Dictionary with statistics
    """
    if len(configs_tensor.shape) == 5:
        # Reshape to (N, 8, Lx, Lt)
        N, Lx, Lt, dirs, quats = configs_tensor.shape
        configs = configs_tensor.view(N, Lx, Lt, dirs * quats).permute(0, 3, 1, 2)
    else:
        configs = configs_tensor
    
    # Statistics
    mean_val = configs.mean()
    std_val = configs.std()
    
    # First quaternion component (a0) which relates to plaquette
    a0_components = configs[:, ::4, :, :]  # Channels 0, 4 (first component of each quat)
    a0_mean = a0_components.mean()
    
    return {
        'mean': mean_val.item(),
        'std': std_val.item(),
        'a0_mean': a0_mean.item()
    }

def predict_starting_noise(beta):
    """
    Predict optimal starting_noise for a given beta.
    
    Improved formula: exp-decay + saturating rise
      sigma_0(beta) = A*exp(-B*beta) + C*(1 - exp(-D*max(0, beta - beta_knee)))
    
    Physical interpretation:
      - Decay term: strong coupling needs more noise (disordered configs)
      - Rise term: far from training beta_0=2.0, model needs more noise
      - Floor at C~0.94: sigma_0 plateaus (just like <P> plateaus)
    
    Fitted to: beta = [1.0, 2.0, 3.0, 3.5, 4.0] (calibration)
               beta = 7.0 (validation estimate)
    Model: trained at beta_0=2.0, 8x8 lattice, 150 sampling steps.
    """
    A = 2.364299
    B = 0.542192
    C = 0.940399
    D = 0.160694
    beta_knee = 3.0
    
    decay = A * np.exp(-B * beta)
    rise = C * (1 - np.exp(-D * max(0, beta - beta_knee)))
    return decay + rise

def predict_starting_noise_old(beta):
    """
    Predict optimal starting_noise for a given β.
    
    Based on quadratic log-polynomial fit to experimental data:
      σ₀(β) = exp(0.0817·β² − 0.829·β + 1.065)
    
    Model trained at β₀ = 2.0, lattice = 8×8, sampling_steps = 150.
    Validated for β ∈ [1.0, 4.0].
    
    Parameters
    ----------
    beta : float
        Target coupling constant.
    
    Returns
    -------
    float
        Predicted optimal starting_noise.
    """
    ln_sn = 0.081687 * beta**2 + (-0.829109) * beta + 1.065357
    return np.exp(ln_sn)
