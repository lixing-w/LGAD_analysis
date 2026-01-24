import torch 
import numpy as np
import torch.nn as nn
from analyze import find_breakdown
import torch.optim as optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader 
import matplotlib.pylab as plt
from tqdm import tqdm 
from datetime import datetime
import os
from model import RNNModel, MLPModel 
from dataset import IVDataset
from utils import is_close, temperature_to_color, linear, linear_fit, disable_top_and_right_bounds, DATABASE_DIR




def loss_fxn(output, target):
    # make last 20 percent more heavily weighted
    n = output.shape[0]
    weights = torch.ones_like(target)
    weights[int(n*0.8):] *= 3
    mse_loss = torch.mean((output - target) ** 2 * weights)
    
    output_diff = torch.diff(output)
    target_diff = torch.diff(target)
    diff_mse_loss = torch.mean((output_diff - target_diff) ** 2 * weights[1:])
    
    output_2nd_diff = torch.diff(output, n=2)
    first_half_mean_2nd_diff = torch.abs(torch.mean(output_2nd_diff[:n//2]))
    
    return mse_loss + first_half_mean_2nd_diff * 2 + diff_mse_loss * 0.7
    
def train(sensor_name, start_model_path=None):
    start_time = datetime.now()
    train_dir = f"./model/{sensor_name}-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(train_dir)
    config = {
        'lr': 0.005,        # Learning rate
        'batch_size': 1,    # Single video per batch
        'num_epochs': 240,   # Number of full passes over data
    }
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    dataset = IVDataset(DATABASE_DIR, sensor_name)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # initialize model
    model = MLPModel().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    if start_model_path is not None:
        model.load_state_dict(torch.load(start_model_path, map_location=device))
        print("Loaded base model.")
    criterion = loss_fxn
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4, min_lr=1e-7)
    
    min_epoch_loss = float('inf')
    for epoch in range(config['num_epochs']):
        model.train() # set to train mode 
        epoch_loss = 0
        
        # loop over each sample in dataset 
        viewed = False
        for v_temp_seq, iv_seq in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            v_temp_seq = v_temp_seq.to(device) # shape: (B, N, 2)
            iv_seq = iv_seq.to(device) # shape: (B, N, 2)
            
            optimizer.zero_grad()
            output = model(v_temp_seq)
            
            if (epoch+1) % 80 == 0 and not viewed:
                cpu_iv_seq = iv_seq.cpu().detach().numpy()
                cpu_output = output.cpu().detach().numpy()
                print(cpu_output)
                plt.figure(figsize=(10, 8))
                plt.plot(cpu_iv_seq[0,:,0], cpu_iv_seq[0,:,1], color="black", label="Ground Truth")
                plt.plot(cpu_iv_seq[0,:,0], cpu_output[0,:,0], color="pink", label="Predicted")
                plt.legend()
                plt.show()
                plt.close()
                viewed = True
                
            loss = criterion(output.squeeze(), iv_seq[0,:,1])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < min_epoch_loss:
            torch.save(model.state_dict(), f"{train_dir}/e{epoch}_l{avg_loss:.3f}.pth")
            min_epoch_loss = avg_loss
            
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, lr: {optimizer.param_groups[0]['lr']}")

def run(model_path: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    model = MLPModel().to(device)
    # torch.load(model_path, model.state_dict())
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    sensor_name = model_path.split(os.sep)[-2][:-20]
    # a) create a sensor scan prediction across temperature
    
    all_temp_bdv_std = []
    plt.figure(figsize=(15, 12))
    for temp in range(-60, 120, 4):
        volt_grid = np.arange(0, 100 + (temp + 60)*1.2, 0.5)
        temperature_seq = np.ones_like(volt_grid)
        temperature_seq *= temp
        v_temp_seq = np.stack([volt_grid, temperature_seq], axis=-1)
        v_temp_seq = torch.from_numpy(v_temp_seq).float().to(device)
        
        input_seq = v_temp_seq.unsqueeze(0) # add batch dim
        pred_curr_log10 = model(input_seq)
        pred_curr_log10 = pred_curr_log10.squeeze()
        
        plt.plot(volt_grid, pred_curr_log10.cpu().detach(), label=rf"Prediction at {temp:.1f}$^\circ$C", color=temperature_to_color(temp))

        lines, std = find_breakdown(volt_grid, pred_curr_log10.cpu().detach().numpy(), 25, None, 0.5, False)
        if lines is not None:
            all_temp_bdv_std.append([temp, lines[0, 2], std])
        else:
            # something went wrong! did not find satisfying lines
            pass
        
        
    plt.xlabel("Reverse-bias Voltage (V)")
    plt.ylabel(f"log(Pad Current (A))")
    plt.title(rf"IV Scan Predictions for Sensor {sensor_name}")
    disable_top_and_right_bounds(plt)
    plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    head, file_name = os.path.split(model_path)
    plt.savefig(f"{head}/{file_name.split('_')[0]}_scan_preds.png")
    plt.close()
    
    # a2) run ransac on the predicted scans and fit a line
    all_temp_bdv_std = np.array(all_temp_bdv_std)
    popt, perr, _, r2 = linear_fit(all_temp_bdv_std[:,0], all_temp_bdv_std[:,1], [1,150], sigmas=all_temp_bdv_std[:,2])
    slope_err = perr[0]
    plt.figure(figsize=(10, 6))
    plt.errorbar(all_temp_bdv_std[:,0], all_temp_bdv_std[:,1], yerr=all_temp_bdv_std[:,2], fmt='o', capsize=5, label=f'RANSAC on Predicted Scan')
    plt.plot(all_temp_bdv_std[:,0], linear(all_temp_bdv_std[:,0], popt[0], popt[1]), label=f'r2 value: {r2:0.2f} Best Fit: y = {popt[0]:.2f}x + { popt[1]:.2f}', linestyle='--')
    disable_top_and_right_bounds(plt)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Breakdown Voltage (V)")
    plt.title(f"Breakdown Voltage vs. Temperature for Sensor {sensor_name}")
    disable_top_and_right_bounds(plt)
    plt.legend()
    plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{head}/{file_name.split('_')[0]}_bdv_preds.png")
    plt.close()
    
    # b) plot what model says with training data
    dataset = IVDataset(DATABASE_DIR, sensor_name)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    plt.figure(figsize=(15, 12))
    seen = set()
    for v_temp_seq, iv_seq in train_loader:
        v_temp_seq = v_temp_seq.to(device)
        output = model(v_temp_seq)
        
        volt_grid = v_temp_seq.cpu().detach().squeeze()[:,0].numpy()
        output = output.cpu().detach().squeeze().numpy()
        temp = float(v_temp_seq[0,0,1])
        label = rf"Prediction at {temp:.1f}$^\circ$C"
        if label not in seen:
            seen.add(label)
            plt.plot(volt_grid, output, label=label, color=temperature_to_color(temp))
        else:
            plt.plot(volt_grid, output, color=temperature_to_color(temp))
    plt.xlabel("Reverse-bias Voltage (V)")
    plt.ylabel(f"log(Pad Current (A))")
    plt.title(rf"IV Scan Evaluations for Sensor {sensor_name}")
    disable_top_and_right_bounds(plt)
    plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    head, file_name = os.path.split(model_path)
    plt.savefig(f"{head}/{file_name.split('_')[0]}_vals.png")
    plt.close()

    # c) for a specific temp, plot all training data, and model output
    plt.figure(figsize=(15, 12))
    target_temp = -10
    label = rf"Training Data at {target_temp:.1f}$^\circ$C"
    for v_temp_seq, iv_seq in train_loader:
        if not is_close(v_temp_seq[0,0,1], target_temp, 0.2, cmp_abs=False):
            continue 
        volt_grid = v_temp_seq.detach().squeeze()[:,0].numpy()
        plt.plot(iv_seq[0,:,0], iv_seq[0,:,1], label=label, color=temperature_to_color(target_temp))
        label = None 
    for v_temp_seq, iv_seq in train_loader:
        if is_close(v_temp_seq[0,0,1], target_temp, 0.2, cmp_abs=False):
            v_temp_seq = v_temp_seq.to(device)
            output = model(v_temp_seq)
            volt_grid = v_temp_seq.cpu().detach().squeeze()[:,0].numpy()
            output = output.cpu().detach().squeeze().numpy()
            plt.plot(iv_seq[0,:,0], output, label=rf"Prediction at {target_temp:.1f}$^\circ$C", linewidth=5, color="black")
            break 
    plt.xlabel("Reverse-bias Voltage (V)")
    plt.ylabel(f"log(Pad Current (A))")
    plt.title(rf"IV Scan Evaluations for Sensor {sensor_name} at {target_temp:.1f}$^\circ$C")
    disable_top_and_right_bounds(plt)
    plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    head, file_name = os.path.split(model_path)
    plt.savefig(f"{head}/{file_name.split('_')[0]}_comp_{target_temp:.1f}.png")
    plt.close()
    
        
        
if __name__ == "__main__":
    # train("BNL_LGAD_W3045_B", start_model_path=None)
    run("model/BNL_LGAD_W3045-2025-07-23-17:00:24/e229_l1.075.pth")