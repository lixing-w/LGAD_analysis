import shutil
import random
import os

import seaborn as sns
from scipy.special import softmax
import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader 
import matplotlib.pylab as plt
import numpy as np
import shap

from tqdm import tqdm 
from datetime import datetime

from model import EnvToLatent, Decoder
from dataset import AggregateLatentDataset, AggregateIVDatasetForAutoEncoder
from utils import DATABASE_DIR, disable_top_and_right_bounds, load_model_from_pth

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
        
ptwise_se = nn.MSELoss(reduction="none")
sigmoid = nn.Sigmoid()

def reconstruction_criterion(output, target, seq_len, max_seq_len, device):
    """
    Reconstruction loss function (adapted from train_autoencoder.py)
    Only focuses on curve reconstruction, no metrics regression
    """
    # 1. find loss of reconstruction
    current = output[:,[0],:seq_len]
    target = target[:,:,:seq_len]
    weight = torch.ones_like(current)
    weight[:,:,int(seq_len*0.8):] *= 2
    mse = ptwise_se(current, target)
    mse *= weight 
    mse = torch.mean(mse)
    
    current_diff = torch.diff(current)
    target_diff = torch.diff(target)
    weight = weight[:,:,:-1]
    mse_1deriv = ptwise_se(current_diff, target_diff)
    mse_1deriv *= weight 
    mse_1deriv = torch.mean(mse_1deriv)
    
    mse_2deriv = torch.mean(torch.abs(torch.diff(current, n=2)[:,:,40:int(seq_len*0.9)]))
    
    bias_loss = torch.mean(torch.abs(torch.mean(current - target, dim=2)))
    
    l1_loss = torch.mean(torch.abs(current - target)) + torch.mean(torch.abs(current_diff - target_diff))
    
    # curve reconstruction loss
    curve_loss = mse + mse_1deriv + mse_2deriv + l1_loss * 1.7 + bias_loss * 1.1
    
    # 2. find loss of end_prob mask 
    true_mask = torch.zeros((1, max_seq_len)).to(device)
    true_mask[:,:seq_len] = 1 
    mask_loss = torch.mean(torch.square(sigmoid(output[:,1,:]) - true_mask))
    
    return curve_loss + mask_loss * 0.5

def criterion(p_latent, t_latent):
    # loss for latent 
    return torch.mean(torch.square(p_latent - t_latent))

def train_end_to_end(autoencoder_model_path: str, reconstruction_weight: float = 1.0, latent_weight: float = 1.0, is_conditional: bool = False):
    """
    Trains an MLP that maps environmental variables to latent vectors using end-to-end training.
    The MLP-predicted latents are fed through a frozen decoder to minimize reconstruction loss
    and also compared against ground truth latents from the encoder.
    
    Parameters
    ----------
    autoencoder_model_path : str
        The relative path to autoencoder that provides the decoder and encoder.
    reconstruction_weight : float, optional
        Weight for the reconstruction loss. Defaults to 1.0.
    latent_weight : float, optional
        Weight for the latent space loss. Defaults to 1.0.
    is_conditional : bool, optional
        Whether to use the conditional autoencoder architecture. Defaults to False.
    """
    start_time = datetime.now()
    train_dir = f"./env_to_latent_model/{DATABASE_DIR.split(os.sep)[-1]}-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}-e2e"
    os.makedirs(train_dir)
    
    # back-up the model.py
    shutil.copy("./model.py", f"{train_dir}/model-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}.py")
    # back-up the train_env_to_latent.py
    shutil.copy("./train_env_to_latent.py", f"{train_dir}/train_env_to_latent-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}.py")
    
    # Save training configuration
    with open(f"{train_dir}/config.txt", "w") as f:
        f.write(f"reconstruction_weight: {reconstruction_weight}\n")
        f.write(f"latent_weight: {latent_weight}\n")
        f.write(f"autoencoder_model: {autoencoder_model_path}\n")
    
    config = {
        'lr': 0.0005,        # Learning rate
        'batch_size': 1,     # Single video per batch
        'num_epochs': 300,   # Number of full passes over data
    }
    
    print(f"Using device: {device}")
    print(f"Reconstruction weight: {reconstruction_weight}, Latent weight: {latent_weight}")
    
    # Use AggregateLatentDataset to get both IV sequences and ground truth latents
    dataset = AggregateLatentDataset(DATABASE_DIR, autoencoder_model_path, is_conditional=is_conditional)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Load frozen decoder from autoencoder
    if not is_conditional:
        decoder = load_model_from_pth(autoencoder_model_path, "Decoder", dataset.max_seq_len, device)
    else:
        decoder = load_model_from_pth(autoencoder_model_path, "ConditionalDecoder", dataset.max_seq_len, device)
    decoder.eval()
    # Freeze decoder parameters
    for param in decoder.parameters():
        param.requires_grad = False
    print("Decoder loaded and frozen")
    
    # Initialize MLP model 
    mlp_model = EnvToLatent().to(device)
    total_params = sum(p.numel() for p in mlp_model.parameters())
    trainable_params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
    print(f"MLP Total parameters: {total_params}")
    print(f"MLP Trainable parameters: {trainable_params}")
    
    optimizer = optim.Adam(mlp_model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-8)
    
    mlp_model.train()
    
    min_epoch_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        epoch_reconstruction_loss = 0
        epoch_latent_loss = 0
        
        for temp, date, iv_seq, humi, ramp_type, dura, seq_len, bd_v, sensor_num, sensor_name, gt_latent in tqdm(train_loader):
            # Prepare inputs - handle inf values
            if humi == float('inf'): humi = torch.tensor([0.0])
            if dura == float('inf'): dura = torch.tensor([0.0])
            if ramp_type == float('inf'): ramp_type = torch.tensor([0.0])
            if temp == float('inf'): temp = torch.tensor([25.0])

            iv_seq = iv_seq.to(device)
            target_current = iv_seq[:,[1],:].to(device)  # Only current channel for reconstruction
            gt_latent = gt_latent.to(device)  # Ground truth latent from encoder
            metrics = torch.stack([temp, date, humi, ramp_type, dura, sensor_num], dim=1).float().to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: MLP predicts latent, decoder reconstructs from latent
            predicted_latent = mlp_model(metrics)
            if not is_conditional:
                reconstructed = decoder(predicted_latent) # usual decoder
            else:
                reconstructed = decoder(predicted_latent, metrics) # Conditional decoder

            # Calculate reconstruction loss
            reconstruction_loss = reconstruction_criterion(reconstructed, target_current, seq_len, dataset.max_seq_len, device)
            
            # Calculate latent space loss (MSE between predicted and ground truth latents)
            latent_loss = criterion(predicted_latent, gt_latent)
            
            # Combine losses with weights
            total_loss = reconstruction_weight * reconstruction_loss + latent_weight * latent_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp_model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_reconstruction_loss += reconstruction_loss.item()
            epoch_latent_loss += latent_loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_reconstruction_loss = epoch_reconstruction_loss / len(train_loader)
        avg_latent_loss = epoch_latent_loss / len(train_loader)
        
        scheduler.step(avg_loss)
        if avg_loss < min_epoch_loss: # save model
            min_epoch_loss = avg_loss
            torch.save(mlp_model.state_dict(), f"{train_dir}/e{epoch}_l{avg_loss:.3f}.pth")
            
        print(f"Epoch {epoch}, Total Loss: {avg_loss:.4f}, Recon Loss: {avg_reconstruction_loss:.4f}, "
              f"Latent Loss: {avg_latent_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}")

def train(autoencoder_model_path: str):
    """
    Trains an MLP that maps environmental variables to corresponding 
    latent vectors generated by autoencoder
    
    Parameters
    ----------
    autoencoder_model_path : str
        The relative path to autoencoder that generates latents.
    """
    start_time = datetime.now()
    train_dir = f"./env_to_latent_model/{DATABASE_DIR.split(os.sep)[-1]}-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(train_dir)
    
    # back-up the model.py
    shutil.copy("./model.py", f"{train_dir}/model-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}.py")
    # back-up the train_env_to_latent.py
    shutil.copy("./train_env_to_latent.py", f"{train_dir}/train_env_to_latent-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}.py")
    config = {
        'lr': 0.0005,        # Learning rate
        'batch_size': 1,    # Single video per batch
        'num_epochs': 300,   # Number of full passes over data
    }
    
    print(f"Using device: {device}")
    dataset = AggregateLatentDataset(DATABASE_DIR, autoencoder_model_path)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # initialize model 
    model = EnvToLatent().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-8)
    
    model.train() 
    
    min_epoch_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        
        for temp, date, iv_curve, humi, ramp_type, dura, seq_len, bd_v, sensor_num, sensor_name, t_latent in tqdm(train_loader):
            t_latent = t_latent.to(device)
            metrics = torch.stack([temp, date, humi, ramp_type, dura, sensor_num], dim=1).float().to(device)
            optimizer.zero_grad()
            
            p_latent = model(metrics)
            # print(output.shape, iv_curve.shape)
            loss = criterion(p_latent, t_latent)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        if avg_loss < min_epoch_loss: # save model
            min_epoch_loss = avg_loss
            torch.save(model.state_dict(), f"{train_dir}/e{epoch}_l{avg_loss:.3g}.pth")
            
        print(f"Epoch {epoch}, Loss: {avg_loss:.4g}, lr: {optimizer.param_groups[0]['lr']}")

def explain(model_path: str, autoencoder_model_path: str):
    """
    Run SHAP analysis on the MLP that predicts latent. SHAP values indicate 
    whether the presence of an input increases/reduces model's output. 
    Generates a heatmap showing **relative** SHAP magnitude. Takes some time 
    to run.
    
    Parameters
    ----------
    model_path : str
        The relative path to the MLP model.
    autoencoder_model_path : str
        The relative path to the autoencoder that generates the latents.
    """
    
    print(f"Using device: {device}")
    
    dataset = AggregateLatentDataset(DATABASE_DIR, autoencoder_model_path)
    
    model = EnvToLatent().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Computing background..")
    background_idx = set(random.sample(range(len(dataset)), k=300))
    selected_metrics = set([0,1,3,4,5,8]) # temp, date, humi, ramp_type, dura, sensor_num
    background = [torch.tensor([dataset[i][j] for j in selected_metrics]) for i in background_idx]
    background = torch.stack(background, dim=0).float().to(device)
    e = shap.GradientExplainer(model, background)
    
    print(f"Computing inputs to explain..")
    input_to_explain = [torch.tensor([dataset[i][j] for j in selected_metrics]) for i in range(len(dataset)) if i not in background_idx]
    input_to_explain = torch.stack(input_to_explain, dim=0).float().to(device)
    shap_values = e.shap_values(input_to_explain) 
    # print(shap_values.shape) # (num_samples, 6, 18)
    
    print(f"Plotting..")
    mean_abs_shap_per_output = np.mean(np.abs(shap_values), axis=0)
    contribution_per_input = softmax(mean_abs_shap_per_output, axis=0)
    input_names = ["Temperature", "Date", "Humidity", "Ramp Type", "Duration", "Sensor Number"]
    plt.figure(figsize=(16, 6))
    sns.heatmap(contribution_per_input, annot=True, cmap=plt.get_cmap("RdBu").reversed(), 
                yticklabels=[input_names[i] for i in range(6)], 
                xticklabels=[f"Dim {j}" for j in range(18)],
                fmt=".2f")
    plt.xlabel("Latent Space Dimension")
    plt.ylabel("Input Feature")
    plt.title("Relative SHAP Magnitude Per Input")
    plt.tight_layout()
    plt.show()
    
    


if __name__ == "__main__":
    # train("autoencoder_model/ivcvscans-2025-07-30-06-51-58/e97_l12.517.pth")
    # explain("env_to_latent_model/ivcvscans-2025-08-12-02-23-15/e133_l0.124.pth", 
    #         "autoencoder_model/ivcvscans-2025-08-12-01-59-55/e97_l21.182.pth")
    # train("autoencoder_model/ivcvscans-2025-08-12-03-24-52/e125_l15.918.pth")

    # End-to-end training with different weight combinations
    # You can experiment with different weight ratios:
    # train_end_to_end("autoencoder_model/ivcvscans-2025-08-12-03-24-52/e125_l15.918.pth", reconstruction_weight=1.0, latent_weight=1.0)  # Equal weighting
    # train_end_to_end("autoencoder_model/ivcvscans-2025-08-12-03-24-52/e125_l15.918.pth", reconstruction_weight=2.0, latent_weight=1.0)  # More emphasis on reconstruction
    # train_end_to_end("autoencoder_model/ivcvscans-2025-08-12-03-24-52/e125_l15.918.pth", reconstruction_weight=1.0, latent_weight=2.0)  # More emphasis on latent matching

    train_end_to_end("conditional_autoencoder_model/ivcvscans-2025-10-13-09-31-32/e105_l0.300.pth", reconstruction_weight=2.0, latent_weight=1.0, is_conditional=True)  # More emphasis on reconstruction