import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm

# Assuming data_ingestion and model are in the same src directory
from data_ingestion import get_dataloaders, load_all_datasets
from model import CyberbullyShieldFusion

def train_model(train_loader, val_loader, device):
    """
    Main training loop implementing high-efficiency constraints:
    - Mixed Precision (torch.cuda.amp)
    - Gradient Accumulation
    - Small Batch Sizes
    """
    model = CyberbullyShieldFusion().to(device)
    
    # 4GB VRAM Optimization Constants
    # Using torch.utils.checkpoint on text_model can save VRAM if needed:
    # model.text_model.gradient_checkpointing_enable()
    
    EPOCHS = 10
    UNFREEZE_EPOCH = 3  # Unfreeze specific base layers after epoch 3
    ACCUMULATION_STEPS = 8  # Simulates batch_size of 32 (if batch_size is 4)
    PATIENCE = 3
    
    # Optimizer & Scheduler
    # Start high for the frozen head before tuning layers
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=1e-2)
    total_steps = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    # Early Stopping Tracking
    best_val_loss = float('inf')
    early_stop_counter = 0
    os.makedirs('checkpoints', exist_ok=True)
    best_model_path = 'checkpoints/best_shield_model.pth'

    print(f"Starting Training for {EPOCHS} Epochs...")
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # --- GRADUAL UNFREEZING LOGIC ---
        if epoch == UNFREEZE_EPOCH:
            print("\n>>> PHASE 2: Gradual Unfreezing Initiated! Unfreezing top base layers... <<<")
            model.unfreeze_top_layers()
            # Recreate optimizer with lower learning rate to prevent catastrophic forgetting
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6, weight_decay=1e-2)
            remaining_steps = len(train_loader) * (EPOCHS - UNFREEZE_EPOCH) // ACCUMULATION_STEPS
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(0.1 * remaining_steps), num_training_steps=remaining_steps
            )
            
        model.train()
        total_train_loss = 0
        
        # Wrapped train_loader with tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for step, batch in enumerate(train_pbar):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_tensor = batch['image_tensor'].to(device)
            labels = batch['label'].to(device)
            
            # 1. Mixed Precision Forward Pass
            with autocast('cuda'):
                logits = model(input_ids, attention_mask, image_tensor)
                # Scale loss by accumulation steps
                loss = criterion(logits, labels) / ACCUMULATION_STEPS
            
            # 2. Accumulate Gradients
            scaler.scale(loss).backward()
            total_train_loss += loss.item() * ACCUMULATION_STEPS
            
            # 3. Step Optimizer Every N steps
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                # Gradient Clipping is also helpful for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
            # Update the progress bar postfix with the current running loss
            train_pbar.set_postfix({'loss': f"{(total_train_loss / (step + 1)):.4f}"})
                
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        total_val_loss = 0
        print("\nRunning Validation...")
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                image_tensor = batch['image_tensor'].to(device)
                labels = batch['label'].to(device)
                
                with autocast('cuda'):
                    logits = model(input_ids, attention_mask, image_tensor)
                    val_loss = criterion(logits, labels)
                    total_val_loss += val_loss.item()
                    
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # --- EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            # Save state dict
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved best model to {best_model_path}")
        else:
            early_stop_counter += 1
            print(f"--> No improvement. Early stopping counter: {early_stop_counter}/{PATIENCE}")
            if early_stop_counter >= PATIENCE:
                print("Early Stopping Triggered! Halting training.")
                break

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize DataLoaders
    print("Loading unifying dataset...")
    datasets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Datasets'))
    train_df, val_df = load_all_datasets(datasets_dir)
    
    # Enforcing small batch sizes due to constraints
    train_loader, val_loader = get_dataloaders(train_df, val_df, batch_size=4)
    
    # Run the training loop
    train_model(train_loader, val_loader, device)
