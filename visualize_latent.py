from datetime import datetime
import random 

import numpy as np
import shap
import torch
from torch.utils.data import DataLoader 
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm

# we have more than 20 sensors! need to expand color cycle
colors = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) 

from model import AutoEncoder, Encoder
from dataset import AggregateIVDatasetForAutoEncoder
from utils import Sensor, DATABASE_DIR


def plot_latent(model_path: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
        
    dataset = AggregateIVDatasetForAutoEncoder(DATABASE_DIR, mode="full")
    
    model = Encoder(dataset.max_seq_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # loop thru dataset and get latent
    all_latent = []
    all_temp = []
    all_humi = []
    all_date = []
    all_ramp = []
    all_dura = []
    all_sensor_num = []
    all_sensor_name = []
    for temp, date, iv_seq, humi, ramp_type, dura, seq_len, sensor_num, sensor_name in dataset:
        i_seq = iv_seq[[1],:].unsqueeze(0).to(device)
        latent = model(i_seq)
        all_latent.append(latent.cpu().detach().squeeze())
        all_temp.append(temp if temp is not None else float('nan'))
        all_humi.append(humi if humi is not None else float('nan'))
        all_dura.append(dura if dura is not None else float('nan'))
        all_date.append(date if date is not None else float('nan'))
        all_ramp.append(ramp_type if ramp_type is not None else float('nan'))
        all_sensor_name.append(sensor_name)
        all_sensor_num.append(sensor_num)
    
    all_latent = np.array(all_latent)
    print(f"Visualizing {all_latent.shape[0]} latents of {all_latent.shape[1]} dimensions.")
    
    params = ["Temperature (C)", "Humidity (%)", "Ramp Type", "Duration (s)", "Date", "Sensor Number"]
    value_lsts = [all_temp, all_humi, all_ramp, all_dura, all_date, all_sensor_num]
    value_lsts = [np.array(lst) for lst in value_lsts]

    dim_1 = 0
    dim_2 = 6
    for param, value_lst in zip(params, value_lsts):
        plt.figure(figsize=(12,10))
        if param == "Ramp Type":
            ramp_map = {-1: "Down", float('inf'): "NA", 1: "Up"}
            for ramp_type in np.unique(value_lst):
                mask = (value_lst == ramp_type)
                plt.scatter(all_latent[mask][:,dim_1], all_latent[mask][:,dim_2], label=ramp_map[ramp_type])
            plt.legend()

        elif param == "Date":
            # convert ordinal to date string
            labels = [datetime.fromordinal(int(dataset.z_score_to_date_ordinal(d))).strftime("%Y-%m-%d") for d in value_lst]
            uniq_dates = sorted(set(labels))
            # re-index dates
            label_to_int = {l: i for i, l in enumerate(uniq_dates)} 
            numeric_vals = [label_to_int[l] for l in labels]
            sc = plt.scatter(all_latent[:,dim_1], all_latent[:,dim_2], c=numeric_vals, cmap="viridis")
            # set colorbar
            cbar = plt.colorbar(sc, ticks=range(len(uniq_dates)))
            cbar.ax.set_yticklabels(uniq_dates)
            cbar.set_label(param)

        elif param == "Sensor Number":
            for sensor_num in np.unique(value_lst):
                mask = (value_lst == sensor_num)
                plt.scatter(all_latent[mask][:,dim_1], all_latent[mask][:,dim_2], label=dataset.sensor_number_to_name[sensor_num])
            plt.legend()

        else: # all other continuous types
            sc = plt.scatter(all_latent[:,dim_1], all_latent[:,dim_2], c=value_lst, cmap="rainbow")
            plt.colorbar(sc, label=param)

        plt.title(f"Latent Space Dim {dim_2} vs {dim_1}")
        plt.xlabel(f"Latent Dim {dim_1}")
        plt.ylabel(f"Latent Dim {dim_2}")
        plt.tight_layout()
        plt.show()

def explain_latent(model_path: str, dim: int, num_samples: int=100, ):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    dataset = AggregateIVDatasetForAutoEncoder(DATABASE_DIR, mode="compact")

    model = Encoder(dataset.max_seq_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # select a set of background examples to take an expectation over
    background_idx = set(random.sample(range(len(dataset)), k=300))
    background = [dataset[i][0][[1],:] for i in background_idx] # take i curve only
    background = torch.stack(background, dim=0).to(device).float()
    e = shap.DeepExplainer(model, background)
    
    
    input_to_explain = [dataset[i] for i in range(len(dataset)) if i not in background_idx]
    i_curves_to_explain = [out[0][[1],:] for out in input_to_explain]
    seq_lens = [out[1] for out in input_to_explain]
    i_curves_to_explain = torch.stack(i_curves_to_explain, dim=0).to(device).float()
    
    # explain output of the encoder
    
    shap_values = e.shap_values(i_curves_to_explain[:num_samples]) # shape (num_samples, 1, 400, 16)

    volt_grid = np.arange(0, 400, 1)
    norm = TwoSlopeNorm(vmin=-shap_values.max()/10, vcenter=0, vmax=shap_values.max()/10)
    plt.figure(figsize=(14,10))
    for i in range(num_samples):
        plt.scatter(volt_grid[:seq_lens[i]], i_curves_to_explain[i].cpu().detach().squeeze()[:seq_lens[i]] , c=shap_values[i,:,:,dim].squeeze()[:seq_lens[i]], cmap=plt.get_cmap("RdBu").reversed(),norm=norm)
    plt.title(f"SHAP Values on Dim {dim} on Training Data")
    plt.xlabel("Reverse Bias Voltage (V)")
    plt.ylabel("log(Pad Current (A))")
    plt.colorbar(label=f"SHAP Values on Dim {dim}")
    plt.tight_layout()
    # plt.savefig(f"dim{dim}.png")
    plt.show()
    plt.close()
    
    # sample = 0
    # for dim in range(16):
    #     plt.figure(figsize=(14,10))
    #     sc = plt.plt.scatter(volt_grid[:seq_lens[sample]], i_curves_to_explain[sample].cpu().detach().squeeze()[:seq_lens[sample]] , c=shap_values[i,:,:,dim].squeeze()[:seq_lens[sample]], cmap=plt.get_cmap("RdBu").reversed(),norm=norm)
    #     plt.colorbar(sc, label=f"SHAP Values on Dim {dim}")
    #     plt.show()


if __name__ == '__main__':
    model_path = "autoencoder_model/ivcvscans-2025-07-29-22:58:43/e211_l23.311.pth"
    plot_latent(model_path)
    # for i in range(16):
    #     explain_latent(model_path, dim=i, num_samples=10)