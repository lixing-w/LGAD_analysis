import shutil
import random
import os

import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader 
import matplotlib.pylab as plt
import numpy as np

from tqdm import tqdm 
from datetime import datetime

from model import AutoEncoder
from dataset import SingleIVDatasetForAutoEncoder, AggregateIVDatasetForAutoEncoder
from utils import Sensor, DATABASE_DIR, disable_top_and_right_bounds

# torch.autograd.set_detect_anomaly(True)

ptwise_se = nn.MSELoss(reduction="none")
def criterion(output, target, seq_len):
    """
    Loss function
    """
    output = output[:,:,:seq_len]
    target = target[:,:,:seq_len]
    weight = torch.ones_like(output)
    weight[:,:,int(seq_len*0.8):] *= 2
    mse = ptwise_se(output, target)
    mse *= weight 
    mse = torch.mean(mse)
    
    output_diff = torch.diff(output)
    target_diff = torch.diff(target)
    weight = weight[:,:,:-1]
    mse_1deriv = ptwise_se(output_diff, target_diff)
    mse_1deriv *= weight 
    mse_1deriv = torch.mean(mse_1deriv)
    
    mse_2deriv = torch.mean(torch.abs(torch.diff(output, n=2)[:,:,40:int(seq_len*0.9)])) # [] #[:,:,30:120]
    
    bias_loss = torch.mean(torch.abs(torch.mean(output - target, dim=2)))
    return mse + mse_1deriv + mse_2deriv * 0.7 + bias_loss #+ smooth_diff * 0.1 + smooth_coeff * 0.025 # 0.025
    
def train(sensor_name: str):
    """ 
    Trains an Autoencoder that learns to extract features from all IV curves 
    of a given sensor.
    
    Parameters
    ----------
    sensor_name : str
        Name of the sensor (in DATABASE_DIR).
    
    Notes
    -----
    The generated models are saved to ./autoencoder_model/{database}-{sensor_name}-{timestamp}
    And models.py and train_autoencoder.py (this file) are backed up there as well.
    """
    start_time = datetime.now()
    train_dir = f"./autoencoder_model/{DATABASE_DIR.split(os.sep)[-1]}-{sensor_name}-{start_time.strftime("%Y-%m-%d-%H:%M:%S")}"
    os.makedirs(train_dir)
    
    # back-up the model.py
    shutil.copy("./model.py", f"{train_dir}/model-{start_time.strftime("%Y-%m-%d-%H:%M:%S")}.py")
    # back-up train_autoencoder.py
    shutil.copy("./train_autoencoder.py", f"{train_dir}/train_autoencoder-{start_time.strftime("%Y-%m-%d-%H:%M:%S")}.py")
    config = {
        'lr': 0.005,        # Learning rate
        'batch_size': 1,    # Single video per batch
        'num_epochs': 300,   # Number of full passes over data
    }
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    dataset = SingleIVDatasetForAutoEncoder(DATABASE_DIR, sensor_name)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # initialize model 
    model = AutoEncoder(dataset.max_seq_len).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20, min_lr=1e-8)
    
    model.train() # set to training mode
    
    min_epoch_loss = float('inf')
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        
        for iv_curve, seq_len in tqdm(train_loader):
            i_curve = iv_curve[:,[1],:].to(device)
            # iv_curve = iv_curve.to(device)
            optimizer.zero_grad()
            
            output = model(i_curve)
            # print(output.shape, iv_curve.shape)
            loss = criterion(output, i_curve, seq_len)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        if avg_loss < min_epoch_loss: # save model
            min_epoch_loss = avg_loss
            torch.save(model.state_dict(), f"{train_dir}/e{epoch}_l{avg_loss:.3f}.pth")
            
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}")
        
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
    train_dir = f"./autoencoder_model/{DATABASE_DIR.split(os.sep)[-1]}-{start_time.strftime("%Y-%m-%d-%H:%M:%S")}"
    os.makedirs(train_dir)
    
    # back-up the model.py
    shutil.copy("./model.py", f"{train_dir}/model-{start_time.strftime("%Y-%m-%d-%H:%M:%S")}.py")
    # back-up the train_autoencoder.py
    shutil.copy("./train_autoencoder.py", f"{train_dir}/train_autoencoder-{start_time.strftime("%Y-%m-%d-%H:%M:%S")}.py")
    config = {
        'lr': 0.005,        # Learning rate
        'batch_size': 1,    # Single video per batch
        'num_epochs': 300,   # Number of full passes over data
    }
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    dataset = AggregateIVDatasetForAutoEncoder(DATABASE_DIR)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # initialize model 
    model = AutoEncoder(dataset.max_seq_len).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=11, min_lr=1e-8)
    
    model.train() # set to training mode
    
    min_epoch_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        
        for iv_curve, seq_len in tqdm(train_loader):
            i_curve = iv_curve[:,[1],:].to(device)
            # iv_curve = iv_curve.to(device)
            optimizer.zero_grad()
            
            output = model(i_curve)
            # print(output.shape, iv_curve.shape)
            loss = criterion(output, i_curve, seq_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        if avg_loss < min_epoch_loss: # save model
            min_epoch_loss = avg_loss
            torch.save(model.state_dict(), f"{train_dir}/e{epoch}_l{avg_loss:.3f}.pth")
            
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}")
    
def run(model_path: str):
    """
    Evaluates the model's outputs on IV data of the sensor trained on.
    
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
    
    sensor_name = model_path.split(os.sep)[-2].split('-', 1)[1][:-20]
    
    dataset = SingleIVDatasetForAutoEncoder(DATABASE_DIR, sensor_name)
    
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
            i_recons = model(i_sample)
            
            plt.figure(figsize=(10, 8))
            orig = sample.squeeze().cpu().detach().numpy()[:,:seq_len].transpose()
            volts = orig[:,0]
            curr = orig[:,1]
            plt.plot(volts, curr, color="black", label="Original")
            curr = i_recons.squeeze().cpu().detach().numpy()[:seq_len].transpose()
            plt.plot(volts, curr, color="pink", label="Reconstructed")
            rmse = np.sqrt(np.mean(np.square(orig[:,1] - curr)))
            avg_rmse += rmse
            plt.xlabel("Reverse-bias Voltage (V)")
            plt.ylabel(f"log(Pad Current (A))")
            plt.title(rf"IV Scan Evaluations for Sensor {sensor_name} ({i} of {len(idxs)}); RMSE {rmse:.3g}")
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
            i_recons = model(i_sample)
            
            plt.figure(figsize=(10, 8))
            orig = sample.squeeze().cpu().detach().numpy()[:,:seq_len].transpose()
            volts = orig[:,0]
            curr = orig[:,1]
            plt.plot(volts, curr, color="black", label="Original")
            curr = i_recons.squeeze().cpu().detach().numpy()[:seq_len].transpose()
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
    
if __name__ == '__main__':
    # train("DC_W3058")
    # run("autoencoder_model/ivcvscans-DC_W3058-2025-07-28-00:10:19/e292_l0.018.pth")
    # aggregate_train()
    aggregate_run("autoencoder_model/ivcvscans-2025-07-28-20:20:03/e275_l0.010.pth")