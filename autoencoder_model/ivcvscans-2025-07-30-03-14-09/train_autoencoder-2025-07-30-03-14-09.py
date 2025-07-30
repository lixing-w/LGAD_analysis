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
def criterion(output, target, seq_len, pred_metrics, target_metrics):
    """
    Loss function
    """
    # 1. first find loss of reconstruction
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
    # so the loss of reconstruction is:
    curve_loss = mse + mse_1deriv + mse_2deriv * 0.7 + bias_loss #+ smooth_diff * 0.1 + smooth_coeff * 0.025 # 0.025
    
    # 2. then find loss of metrics prediction
    temp_loss = 0 if target_metrics[:,0] == float('inf') else torch.square(pred_metrics[:,0] - target_metrics[:,0])
    date_sensor_num_loss = torch.mean(ptwise_se(pred_metrics[:,[1,5]], target_metrics[:,[1,5]]))
    humi_loss = 0 if target_metrics[:,2] == float('inf') else torch.square(pred_metrics[:,2] - target_metrics[:,2])
    ramp_type_loss = 0 if target_metrics[:,3] == float('inf') else torch.square(pred_metrics[:,3] - target_metrics[:,3])
    dura_loss = 0 if target_metrics[:,4] == float('inf') else torch.square(pred_metrics[:,4] - target_metrics[:,4])
    
    metrics_loss = temp_loss+date_sensor_num_loss+humi_loss+ramp_type_loss+dura_loss
    return curve_loss + metrics_loss * 0.2

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
    train_dir = f"./autoencoder_model/{DATABASE_DIR.split(os.sep)[-1]}-{start_time.strftime('%Y-%m-%d-%H:%M:%S')}"
    os.makedirs(train_dir)
    
    # back-up the model.py
    shutil.copy("./model.py", f"{train_dir}/model-{start_time.strftime('%Y-%m-%d-%H:%M:%S')}.py")
    # back-up the train_autoencoder.py
    shutil.copy("./train_autoencoder.py", f"{train_dir}/train_autoencoder-{start_time.strftime('%Y-%m-%d-%H:%M:%S')}.py")
    config = {
        'lr': 0.0005,        # Learning rate
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
    
    dataset = AggregateIVDatasetForAutoEncoder(DATABASE_DIR, mode="full")
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # initialize model 
    model = AutoEncoder(dataset.max_seq_len).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-8)
    
    model.train() # set to training mode
    
    min_epoch_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        
        for temp, date, iv_curve, humi, ramp_type, dura, seq_len, sensor_num, sensor_name in tqdm(train_loader):
            i_curve = iv_curve[:,[1],:].to(device)
            t_metrics = torch.stack([temp, date, humi, ramp_type, dura, sensor_num], dim=1).float().to(device)
            # iv_curve = iv_curve.to(device)
            optimizer.zero_grad()
            
            recons, p_metrics = model(i_curve)
            # print(output.shape, iv_curve.shape)
            loss = criterion(recons, i_curve, seq_len, p_metrics, t_metrics)
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
    aggregate_train()
    # aggregate_run("autoencoder_model/ivcvscans-2025-07-30-03:01:20/e150_l21.231.pth")