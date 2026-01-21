import shutil
import random
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # workaround

import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader 
import matplotlib.pylab as plt
import numpy as np

from tqdm import tqdm 
from datetime import datetime

from model import AutoEncoder, ConditionalAutoEncoder
from dataset import AggregateIVDatasetForAutoEncoder
from utils import Sensor, DATABASE_DIR, disable_top_and_right_bounds

# torch.autograd.set_detect_anomaly(True)

ptwise_se = nn.MSELoss(reduction="none")
sigmoid = nn.Sigmoid()
def criterion(output, target, seq_len, pred_metrics, target_metrics, max_seq_len, device):
    """
    Loss function
    """
    # 1. first find loss of reconstruction
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
    
    mse_2deriv = torch.mean(torch.abs(torch.diff(current, n=2)[:,:,40:int(seq_len*0.9)])) # [] #[:,:,30:120]
    
    bias_loss = torch.mean(torch.abs(torch.mean(current - target, dim=2)))
    
    l1_loss = torch.mean(torch.abs(current - target)) + torch.mean(torch.abs(current_diff - target_diff))
    # so the loss of reconstruction is:
    curve_loss = mse + mse_1deriv + mse_2deriv + l1_loss *1.7 + bias_loss * 1.1 #+ smooth_diff * 0.1 + smooth_coeff * 0.025 # 0.025
    
    # 2. then find loss of metrics prediction
    temp_loss = 0 if target_metrics[:,0] == float('inf') else torch.square(pred_metrics[:,0] - target_metrics[:,0])
    date_sensor_num_loss = torch.mean(ptwise_se(pred_metrics[:,[1,6]], target_metrics[:,[1,6]]))
    humi_loss = 0 if target_metrics[:,2] == float('inf') else torch.square(pred_metrics[:,2] - target_metrics[:,2])
    ramp_type_loss = 0 if target_metrics[:,3] == float('inf') else torch.square(pred_metrics[:,3] - target_metrics[:,3])
    dura_loss = 0 if target_metrics[:,4] == float('inf') else torch.square(pred_metrics[:,4] - target_metrics[:,4])
    bdv_loss = torch.square(pred_metrics[:,5] - target_metrics[:,5])
    metrics_loss = temp_loss+date_sensor_num_loss+humi_loss+ramp_type_loss+dura_loss+bdv_loss
    
    # 3. find loss of end_prob mask 
    true_mask = torch.zeros((1, max_seq_len)).to(device)
    true_mask[:,:seq_len] = 1 
    mask_loss = torch.mean(torch.square(sigmoid(output[:,1,:]) - true_mask))
    
    return curve_loss + metrics_loss * 0.2 + mask_loss * 0.5

def aggregate_train():
    """ 
    Trains an Autoencoder that learns to extract features from all IV curves 
    in DATABASE_DIR.
    
    Notes
    -----
    The generated models are saved to ./autoencoder_model/{database}-{timestamp}
    And models.py and train_autoencoder.py (this file) are backed up there as well.
    """
    start_time = datetime.now()
    train_dir = f"./autoencoder_model/{DATABASE_DIR.split(os.sep)[-1]}-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(train_dir)
    
    # back-up the model.py
    shutil.copy("./model.py", f"{train_dir}/model-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}.py")
    # back-up the train_autoencoder.py
    shutil.copy("./train_autoencoder.py", f"{train_dir}/train_autoencoder-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}.py")
    config = {
        'lr': 0.0005,        # Learning rate
        'batch_size': 1,    # Single video per batch
        'num_epochs': 1000,   # Number of full passes over data
        'l1_lambda': 1e-5,   # L1 weight decay lambda
    }
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    dataset = AggregateIVDatasetForAutoEncoder(DATABASE_DIR, mode="full")
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # initialize model 
    model = AutoEncoder(dataset.max_seq_len).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    model.train() # set to training mode
    
    min_epoch_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        
        for temp, date, iv_curve, humi, ramp_type, dura, seq_len, bd_v, sensor_num, sensor_name in tqdm(train_loader):
            i_curve = iv_curve[:,[1],:].to(device)
            t_metrics = torch.stack([temp, date, humi, ramp_type, dura, bd_v, sensor_num], dim=1).float().to(device)
            # iv_curve = iv_curve.to(device)
            optimizer.zero_grad()
            
            recons, p_metrics = model(i_curve)
            # print(output.shape, iv_curve.shape)
            loss = criterion(recons, i_curve, seq_len, p_metrics, t_metrics, dataset.max_seq_len, device)
            
            # Add L1 weight decay
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss += config['l1_lambda'] * l1_penalty
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        if avg_loss < min_epoch_loss: # save model
            min_epoch_loss = avg_loss
            torch.save(model.state_dict(), f"{train_dir}/e{epoch}_l{avg_loss:.3f}.pth")
            
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}")
   
def aggregate_run(model_path: str):
    """
    Evaluates the model's outputs on all IV data in DATABASE_DIR.
    
    Parameters
    ----------
    model_path : str
        The relative path to the model.
    
    Notes
    -----
    This function displays several figures comparing the original scans 
    and the reconstructed scans (model outputs).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    dataset = AggregateIVDatasetForAutoEncoder(DATABASE_DIR)
    
    model = AutoEncoder(dataset.max_seq_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # give some examples of iv_curve and reconstruction pairs 
    number_of_examples = 20
    idxs = random.sample(range(len(dataset)), number_of_examples)
    avg_rmse = 0
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            sample, seq_len = dataset[idx]
            
            sample = sample.unsqueeze(0).to(device) # add batch dim
            i_sample = sample[:,[1],:] # only take current
            i_recons, _ = model(i_sample)
            
            plt.figure(figsize=(10, 8))
            orig = sample.squeeze().cpu().detach().numpy()[:,:seq_len].transpose()
            volts = orig[:,0]
            curr = orig[:,1]
            plt.plot(volts, curr, color="black", label="Original")
            curr = i_recons[:,0,:seq_len].squeeze().cpu().detach().numpy().transpose()
            plt.plot(volts, curr, color="pink", label="Reconstructed")
            rmse = np.sqrt(np.mean(np.square(orig[:,1] - curr)))
            avg_rmse += rmse
            plt.xlabel("Reverse-bias Voltage (V)")
            plt.ylabel(f"log(Pad Current (A))")
            plt.title(rf"IV Scan Evaluations for Database {DATABASE_DIR.split(os.sep)[-1]} ({i} of {len(idxs)}); RMSE {rmse:.3g}")
            disable_top_and_right_bounds(plt)
            plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
            plt.legend()
            plt.tight_layout()
            head, file_name = os.path.split(model_path)
            plt.savefig(f"{head}/{file_name.split('_')[0]}_vals_{i}.png")
            plt.show()
            plt.close()
    avg_rmse /= number_of_examples 
    print(f"Average RMSE {avg_rmse}")
    


def conditional_criterion(output, target, seq_len, max_seq_len, device):
    """
    Loss function for conditional autoencoder (only reconstruction loss, no metrics regression)
    """
    # 1. first find loss of reconstruction
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
    # so the loss of reconstruction is:
    curve_loss = mse + mse_1deriv + mse_2deriv + l1_loss *1.7 + bias_loss * 1.1
    
    # 2. find loss of end_prob mask 
    true_mask = torch.zeros((1, max_seq_len)).to(device)
    true_mask[:,:seq_len] = 1 
    mask_loss = torch.mean(torch.square(sigmoid(output[:,1,:]) - true_mask))
    
    return curve_loss + mask_loss * 0.5

def conditional_aggregate_train():
    """ 
    Trains a Conditional Autoencoder that learns to extract features from all IV curves 
    in DATABASE_DIR. The decoder is conditioned on environmental parameters.
    
    Notes
    -----
    The generated models are saved to ./conditional_autoencoder_model/{database}-{timestamp}
    And models.py and train_autoencoder.py (this file) are backed up there as well.
    """
    start_time = datetime.now()
    train_dir = f"./conditional_autoencoder_model/{DATABASE_DIR.split(os.sep)[-1]}-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(train_dir)
    
    # back-up the model.py
    shutil.copy("./model.py", f"{train_dir}/model-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}.py")
    # back-up the train_autoencoder.py
    shutil.copy("./train_autoencoder.py", f"{train_dir}/train_autoencoder-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}.py")
    config = {
        'lr': 5e-4,        # Learning rate
        'batch_size': 1,    # Batch size
        'num_epochs': 1000,   # Number of full passes over data
        'l1_lambda': 1e-5,   # L1 weight decay lambda
    }
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    dataset = AggregateIVDatasetForAutoEncoder(DATABASE_DIR, mode="full", data_config_path="./ml_data_config.txt")
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # initialize conditional autoencoder model 
    model = ConditionalAutoEncoder(dataset.max_seq_len).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Latent dimension: {model.latent_dim}")
    print(f"Decoder input dimension: {model.decoder_input_dim}")
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    model.train() # set to training mode
    
    min_epoch_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0

        for temp, date, iv_curve, humi, ramp_type, dura, seq_len, bd_v, sensor_num, sensor_name, sensor_thickness, sensor_type in tqdm(train_loader):
            # Prepare inputs - handle inf values for parameters using tensor operations
            # Replace inf values with appropriate defaults
            humi = torch.where(torch.isinf(humi), torch.tensor(0.0), humi)
            dura = torch.where(torch.isinf(dura), torch.tensor(0.0), dura)  
            ramp_type = torch.where(torch.isinf(ramp_type), torch.tensor(0.0), ramp_type)
            temp = torch.where(torch.isinf(temp), torch.tensor(25.0), temp)
            sensor_thickness = torch.where(torch.isinf(sensor_thickness), torch.tensor(0.0), sensor_thickness)
            
            i_curve = iv_curve[:,[1],:].to(device)  # Only current channel
            # params = torch.stack([temp, date, humi, ramp_type, dura, sensor_num], dim=1).float().to(device)
            params = torch.stack([temp, humi, ramp_type, sensor_num, sensor_thickness, sensor_type], dim=1).float().to(device)
            # Debug: Check input parameters for NaN/inf
            if torch.isnan(params).any() or torch.isinf(params).any():
                print(f"WARNING: NaN/inf in input parameters after cleaning")
                print(f"Params shape: {params.shape}")
                continue
                
            optimizer.zero_grad()
            
            recons = model(i_curve, params)
            # Calculate reconstruction loss (no metrics loss for conditional autoencoder)
            loss = conditional_criterion(recons, i_curve, seq_len, dataset.max_seq_len, device)
            
            # Add L1 weight decay
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss += config['l1_lambda'] * l1_penalty
                
            # loss = criterion(recons, i_curve, seq_len, p_params, params, dataset.max_seq_len, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        if avg_loss < min_epoch_loss: # save model
            min_epoch_loss = avg_loss
            torch.save(model.state_dict(), f"{train_dir}/e{epoch}_l{avg_loss:.3f}.pth")
            
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}")

def conditional_aggregate_run(model_path: str):
    """
    Evaluates the conditional autoencoder model's outputs on all IV data in DATABASE_DIR.
    
    Parameters
    ----------
    model_path : str
        The relative path to the model.
    
    Notes
    -----
    This function displays several figures comparing the original scans 
    and the reconstructed scans (model outputs).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    dataset = AggregateIVDatasetForAutoEncoder(DATABASE_DIR, mode="full", data_config_path="./ml_data_config.txt")
    
    model = ConditionalAutoEncoder(dataset.max_seq_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # give some examples of iv_curve and reconstruction pairs 
    number_of_examples = 20
    idxs = random.sample(range(len(dataset)), number_of_examples)
    avg_rmse = 0
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            # sample, seq_len, params = dataset[idx]
            temp, date, iv_curve, humi, ramp_type, dura, seq_len, bd_v, sensor_num, sensor_name, sensor_thickness, sensor_type = dataset[idx]
            # prepare params
            sample = iv_curve
            if sensor_thickness == float('inf'): sensor_thickness = 0.0
            if humi == float('inf'): humi = 0.0
            if dura == float('inf'): dura = 0.0
            if ramp_type == float('inf'): ramp_type = 0.0
            if temp == float('inf'): temp = 25.0
            # params = torch.tensor([temp, date, humi, ramp_type, dura, sensor_num])
            params = torch.tensor([temp, humi, ramp_type, sensor_num, sensor_thickness, sensor_type])
            
            sample = sample.unsqueeze(0).to(device) # add batch dim
            params = params.unsqueeze(0).float().to(device)
            i_sample = sample[:,[1],:] # only take current
            i_recons = model(i_sample, params)
            # check for NaN/inf in outputs
            if torch.isnan(i_recons).any() or torch.isinf(i_recons).any():
                print(f"WARNING: NaN/inf in model outputs for index {idx}")
                continue
            
            plt.figure(figsize=(10, 8))
            orig = sample.squeeze().cpu().detach().numpy()[:,:seq_len].transpose()
            volts = orig[:,0]
            curr = orig[:,1]
            plt.plot(volts, curr, color="black", label="Original")
            curr = i_recons[:,0,:seq_len].squeeze().cpu().detach().numpy().transpose()
            plt.plot(volts, curr, color="pink", label="Reconstructed")
            rmse = np.sqrt(np.mean(np.square(orig[:,1] - curr)))
            avg_rmse += rmse
            plt.xlabel("Reverse-bias Voltage (V)")
            plt.ylabel(f"log(Pad Current (A))")
            plt.title(rf"Conditional Autoencoder IV Scan Evaluations for Database {DATABASE_DIR.split(os.sep)[-1]} ({i} of {len(idxs)}); RMSE {rmse:.3g}")
            disable_top_and_right_bounds(plt)
            plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
            plt.legend()
            plt.tight_layout()
            head, file_name = os.path.split(model_path)
            plt.savefig(f"{head}/{file_name.split('_')[0]}_vals_{i}.png")
            plt.show()
            plt.close()
    avg_rmse /= number_of_examples
    print(f"Average RMSE {avg_rmse}")
    
if __name__ == '__main__':
    # train("DC_W3058")
    # run("autoencoder_model/ivcvscans-DC_W3058-2025-07-28-00:10:19/e292_l0.018.pth")
    # aggregate_train()
    # aggregate_run("autoencoder_model/ivcvscans-2025-08-12-01:59:55/e97_l21.182.pth")
    conditional_aggregate_train()
    # conditional_aggregate_run("conditional_autoencoder_model/ivcvscans-2025-10-13-14-25-08/e97_l0.269.pth") # latent 16 + params 6
    # conditional_aggregate_run("conditional_autoencoder_model/ivcvscans-2025-10-28-10-30-59/e95_l0.211.pth") # latent 8 + params 6
    # conditional_aggregate_run("conditional_autoencoder_model/ivcvscans-2025-10-28-11-02-38/e97_l0.305.pth") # latent 4 + params 6