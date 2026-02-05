"""
SU(2) Diffusion Model Training Script
=====================================
Trains a diffusion model on HMC-generated gauge field configurations.
Uses classes directly from DiffusionModel.py and su2_hmc.py.

By B.-D. Sun.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "Source")))

import torch
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
from scipy.special import iv as bessel_i

# Import from your source files
from DiffusionModel import DiffusionModel, DiffusionUNet, NoiseSchedule
from su2_hmc import LatticeGaugeField2D, WilsonAction2D, compute_wilson_loop


# =============================================================================
# TRAINING PARAMETERS
# =============================================================================
PARAMS = {
    # Data
    'data_path': 'su2_beta2.0_L16_16_XXXXXXXX_1/configurations.pt',  # UPDATE THIS PATH
    'beta_training': 2.0,          # Coupling at which data was generated
    
    # Lattice
    'Lx': 16,
    'Lt': 16,
    
    # Training
    'batch_size': 64,              # Batch size (adjust based on GPU memory)
    'num_epochs': 100,             # Number of training epochs
    'learning_rate': 1e-4,         # Adam learning rate
    
    # Model architecture
    'time_emb_dim': 128,           # Time embedding dimension
    'base_channels': 64,           # Base channels in U-Net
    'max_channels': 256,           # Max channels in U-Net
    
    # Noise schedule
    'T': 1000,                     # Number of diffusion steps
    'noise_schedule_s': 0.008,     # Cosine schedule offset
    
    # Validation
    'validate_every': 10,          # Validate every N epochs
    'n_validation_samples': 50,    # Samples to generate for validation
    'validation_steps': 100,       # Denoising steps for validation
    
    # Output
    'save_every': 20,              # Save checkpoint every N epochs
    'output_dir': None,            # Will be set automatically
    
    # Reproducibility
    'seed': 42,
}


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train():
    """Main training loop."""
    
    # =========================================================================
    # SETUP
    # =========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(PARAMS['seed'])
    np.random.seed(PARAMS['seed'])
    
    print(f"\n{'='*70}")
    print(f"  SU(2) DIFFUSION MODEL TRAINING")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Lattice: {PARAMS['Lx']} × {PARAMS['Lt']}")
    print(f"  β (training): {PARAMS['beta_training']}")
    print(f"{'='*70}\n")
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    configs_tensor = load_configurations(PARAMS['data_path'])
    
    # Verify data
    verify_training_data(configs_tensor, PARAMS['beta_training'])
    
    # Reshape for diffusion model
    print("\nReshaping data for diffusion model...")
    configs_reshaped = reshape_for_diffusion(configs_tensor)
    print(f"  Reshaped: {configs_reshaped.shape}")
    print(f"  (N_configs, channels, Lx, Lt)")
    
    N_train = configs_reshaped.shape[0]
    
    # =========================================================================
    # CREATE DATALOADER
    # =========================================================================
    print(f"\nCreating DataLoader...")
    train_dataset = torch.utils.data.TensorDataset(configs_reshaped)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=PARAMS['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    print(f"  Batch size: {PARAMS['batch_size']}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Total samples: {N_train}")
    
    # =========================================================================
    # INITIALIZE MODEL
    # =========================================================================
    print(f"\nInitializing diffusion model...")
    
    # Noise schedule
    noise_schedule = NoiseSchedule(T=PARAMS['T'], s=PARAMS['noise_schedule_s'])
    
    # U-Net
    unet = DiffusionUNet(
        in_channels=8,
        out_channels=8,
        time_emb_dim=PARAMS['time_emb_dim'],
        base_channels=PARAMS['base_channels'],
        max_channels=PARAMS['max_channels']
    )
    
    # Diffusion model
    diffusion_model = DiffusionModel(
        unet=unet,
        noise_schedule=noise_schedule,
        device=device,
        learning_rate=PARAMS['learning_rate'],
        beta_training=PARAMS['beta_training']
    )
    
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"  U-Net parameters: {total_params:,}")
    print(f"  Samples per parameter: {N_train / total_params:.2f}")
    
    # =========================================================================
    # OUTPUT DIRECTORY
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"diffusion_training_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    PARAMS['output_dir'] = output_dir
    print(f"\nOutput directory: {output_dir}")
    
    # Save parameters
    with open(os.path.join(output_dir, "params.json"), 'w') as f:
        json.dump(PARAMS, f, indent=2)
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  TRAINING")
    print(f"{'='*70}")
    print(f"  Epochs: {PARAMS['num_epochs']}")
    print(f"  Validate every: {PARAMS['validate_every']} epochs")
    print(f"  Save every: {PARAMS['save_every']} epochs")
    print(f"{'='*70}\n")
    
    epoch_losses = []
    validation_history = []
    best_loss = float('inf')
    
    for epoch in range(PARAMS['num_epochs']):
        # Train one epoch
        avg_loss = diffusion_model.train_epoch(train_loader)
        epoch_losses.append(avg_loss)
        
        # Progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{PARAMS['num_epochs']} | Loss: {avg_loss:.6f}")
        
        # Validation
        if (epoch + 1) % PARAMS['validate_every'] == 0:
            print(f"  → Validating...", end=" ", flush=True)
            val_results = validate_model(
                diffusion_model,
                PARAMS['Lx'],
                PARAMS['Lt'],
                PARAMS['beta_training'],
                PARAMS['n_validation_samples'],
                PARAMS['validation_steps'],
                device
            )
            validation_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                **{k: v for k, v in val_results.items() if k != 'plaquettes'}
            })
            
            status = "✓" if val_results['sigma_diff'] < 3 else ("⚠" if val_results['sigma_diff'] < 5 else "✗")
            print(f"⟨P⟩ = {val_results['plaq_mean']:.5f} ± {val_results['plaq_err']:.5f} "
                  f"(exact: {val_results['exact_plaq']:.5f}, {val_results['sigma_diff']:.1f}σ) {status}")
        
        # Save checkpoint
        if (epoch + 1) % PARAMS['save_every'] == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': diffusion_model.unet.state_dict(),
                'optimizer_state_dict': diffusion_model.optimizer.state_dict(),
                'loss': avg_loss,
                'epoch_losses': epoch_losses,
            }, checkpoint_path)
            print(f"  → Checkpoint saved: {checkpoint_path}")
        
        # Track best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(output_dir, "best_model.pt")
            torch.save(diffusion_model.unet.state_dict(), best_path)
    
    # =========================================================================
    # FINAL SAVE
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")
    
    # Save final model
    final_path = os.path.join(output_dir, "final_model.pt")
    torch.save(diffusion_model.unet.state_dict(), final_path)
    print(f"  Final model saved: {final_path}")
    
    # Save training history
    history = {
        'epoch_losses': epoch_losses,
        'validation_history': validation_history,
        'best_loss': best_loss,
        'final_loss': epoch_losses[-1],
    }
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved: {history_path}")
    
    # Final validation
    print(f"\n  Final validation with more samples...")
    final_val = validate_model(
        diffusion_model,
        PARAMS['Lx'],
        PARAMS['Lt'],
        PARAMS['beta_training'],
        n_samples=200,
        steps=200,
        device=device
    )
    
    print(f"\n  FINAL RESULTS:")
    print(f"    Best loss:     {best_loss:.6f}")
    print(f"    Final loss:    {epoch_losses[-1]:.6f}")
    print(f"    Generated ⟨P⟩: {final_val['plaq_mean']:.6f} ± {final_val['plaq_err']:.6f}")
    print(f"    Exact ⟨P⟩:     {final_val['exact_plaq']:.6f}")
    print(f"    Deviation:     {final_val['sigma_diff']:.2f}σ")
    
    if final_val['sigma_diff'] < 3:
        print(f"\n  ✓ SUCCESS: Model has learned the correct physics!")
    elif final_val['sigma_diff'] < 5:
        print(f"\n  ⚠ PARTIAL SUCCESS: Model is learning, consider more training.")
    else:
        print(f"\n  ✗ Model needs more training or debugging.")
    
    print(f"\n{'='*70}")
    print(f"  All outputs saved to: {output_dir}/")
    print(f"{'='*70}\n")
    
    return diffusion_model, epoch_losses, validation_history


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # UPDATE THIS PATH before running!
    # PARAMS['data_path'] = 'path/to/your/configurations.pt'
    
    if 'XXXXXXXX' in PARAMS['data_path']:
        print("\n" + "="*70)
        print("  ERROR: Please update PARAMS['data_path'] with your actual data path!")
        print("  Edit line 32 of this script.")
        print("="*70 + "\n")
        sys.exit(1)
    
    train()
