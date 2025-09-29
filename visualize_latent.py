from datetime import datetime
import random

import seaborn as sns
import numpy as np
import shap
import torch
from torch.utils.data import DataLoader
import matplotlib.pylab as plt
import matplotlib as mpl
from scipy.stats import spearmanr
from matplotlib.colors import TwoSlopeNorm

# we have more than 20 sensors! need to expand color cycle
colors = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

from model import AutoEncoder, Encoder
from dataset import AggregateIVDatasetForAutoEncoder, AggregateLatentDataset
from utils import Sensor, DATABASE_DIR, load_model_from_pth


def get_full_dataset():
    if not hasattr(get_full_dataset, "_dataset"):
        print("Initializing dataset...")
        get_full_dataset._dataset = AggregateIVDatasetForAutoEncoder(DATABASE_DIR, mode="full")
    return get_full_dataset._dataset


def get_compact_dataset():
    if not hasattr(get_compact_dataset, "_dataset"):
        print("Initializing dataset...")
        get_compact_dataset._dataset = AggregateIVDatasetForAutoEncoder(DATABASE_DIR, mode="compact")
    return get_compact_dataset._dataset


def plot_latent(model_path: str):
    """
    Generate plots of latent space annotated by environmental variables.
    
    Parameters
    ----------
    model_path : str
        The relative path to interested autoencoder model.
    
    Notes
    -----
    The latent is a high dimensional space. You can choose which 2 dimensions 
    to visualize by changing dim_1 and dim_2 in the body. If you added more 
    environmental vars to consider, update the code accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    dataset = get_full_dataset()

    model = load_model_from_pth(model_path, "Encoder", dataset.max_seq_len, device)
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
    for temp, date, iv_seq, humi, ramp_type, dura, seq_len, bd_v, i_at_100v, slope, offset, sensor_num, sensor_name in dataset:
        iv_seq = iv_seq.unsqueeze(0).to(device)
        latent = model(iv_seq[:,[1],:])
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
    fig_idx = 0
    for param, value_lst in zip(params, value_lsts):
        plt.figure(figsize=(12, 10))
        if param == "Ramp Type":
            ramp_map = {-1: "Down", float('inf'): "NA", 1: "Up"}
            for ramp_type in np.unique(value_lst):
                mask = (value_lst == ramp_type)
                plt.scatter(all_latent[mask][:, dim_1], all_latent[mask][:, dim_2], label=ramp_map[ramp_type])
            plt.legend()

        elif param == "Date":
            # convert ordinal to date string
            labels = [datetime.fromordinal(int(dataset.z_score_to_date_ordinal(d))).strftime("%Y-%m-%d") for d in
                      value_lst]
            uniq_dates = sorted(set(labels))
            # re-index dates
            label_to_int = {l: i for i, l in enumerate(uniq_dates)}
            numeric_vals = [label_to_int[l] for l in labels]
            sc = plt.scatter(all_latent[:, dim_1], all_latent[:, dim_2], c=numeric_vals, cmap="viridis")
            # set colorbar
            cbar = plt.colorbar(sc, ticks=range(len(uniq_dates)))
            cbar.ax.set_yticklabels(uniq_dates)
            cbar.set_label(param)

        elif param == "Sensor Number":
            for sensor_num in np.unique(value_lst):
                mask = (value_lst == sensor_num)
                plt.scatter(all_latent[mask][:, dim_1], all_latent[mask][:, dim_2],
                            label=dataset.sensor_number_to_name[sensor_num])
            plt.legend()

        else:  # all other continuous types
            sc = plt.scatter(all_latent[:, dim_1], all_latent[:, dim_2], c=value_lst, cmap="rainbow")
            plt.colorbar(sc, label=param)

        plt.title(f"Latent Space Dim {dim_2} vs {dim_1}")
        plt.xlabel(f"Latent Dim {dim_1}")
        plt.ylabel(f"Latent Dim {dim_2}")
        plt.tight_layout()
        plt.savefig(model_path.replace(".pth", f"dim{dim_2}_{dim_1}_{fig_idx}.png"))
        # plt.show()
        plt.close()
        fig_idx += 1


def explain_latent_on_data(model_path: str, dim: int = None, num_samples: int = None):
    """
    Run SHAP analysis on latents generated by autoencoder. Plot IV curves 
    colored with SHAP values.
    
    Parameters
    ----------
    model_path : str
        The relative path to autoencoder model.
    dim : int, optional
        Which dim of the latent space to examine. If not specified, plots all dims.
    num_samples : int, optional
        Number of IV curves to plot.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    dataset = get_compact_dataset()

    model = load_model_from_pth(model_path, "Encoder", dataset.max_seq_len, device)
    model.eval()

    if dim is not None:
        assert 0 <= dim < model.latent_dim
        dims = [dim]
    else:
        dims = [i for i in range(model.latent_dim)]

    if num_samples is None:
        num_samples = len(dataset) - 300

    print("Computing SHAP on background..")
    # select a set of background examples to take an expectation over
    background_idx = set(random.sample(range(len(dataset)), k=len(dataset) - num_samples))
    background = [dataset[i][0] for i in background_idx]  # take i curve only
    background = torch.stack(background, dim=0).float().to(device)
    e = shap.GradientExplainer(model, background)

    print("Computing SHAP on inputs to explain..")
    input_to_explain = [dataset[i] for i in range(len(dataset)) if i not in background_idx]
    iv_curves_to_explain = [out[0] for out in input_to_explain]
    seq_lens = [out[1] for out in input_to_explain]
    iv_curves_to_explain = torch.stack(iv_curves_to_explain, dim=0).float().to(device)

    # explain output of the encoder
    shap_values = e.shap_values(iv_curves_to_explain)  # shape (num_samples, 1, max_seq_len, latent_dim)

    volt_grid = np.arange(0, 400, 1)
    norm = TwoSlopeNorm(vmin=-np.abs(shap_values).max() / 7, vcenter=0, vmax=np.abs(shap_values).max() / 7)
    for dim in dims:
        plt.figure(figsize=(14, 10))
        for i in range(num_samples):
            plt.scatter(volt_grid[:seq_lens[i]], iv_curves_to_explain[i, 1, :seq_lens[i]].cpu().detach().squeeze(),
                        c=shap_values[i, 0, :seq_lens[i], dim].squeeze(), cmap=plt.get_cmap("RdBu").reversed(),
                        norm=norm)
        plt.ylim(bottom=-20)
        plt.title(f"SHAP Values on Dim {dim} on Training Data")
        plt.xlabel("Reverse Bias Voltage (V)")
        plt.ylabel("log(Pad Current (A))")
        plt.colorbar(label=f"SHAP Values on Dim {dim}")
        plt.tight_layout()
        plt.savefig(model_path.replace(".pth", f"dim{dim}.png"))
        # plt.show()
        plt.close()


def explain_latent_corr(model_path: str):
    """
    Compute Spearman Correlation Coefficient for every pair of environmental 
    variables and latent dims. Generate a heatmap. Values closer to 1 mean 
    stronger monotonic relation, and closer to 0 means no obvious monotonic 
    relation. The coefficients are not absolute, and hence should be compared across 
    different pairs.
    
    Parameters
    ----------
    model_path : str
        The relative path to autoencoder model.
    """
    dataset = AggregateLatentDataset(DATABASE_DIR, model_path)

    selected_metrics = [0, 1, 3, 4, 5, 8]
    X = [torch.tensor([dataset[i][j] for j in selected_metrics]) for i in range(len(dataset))]  # metrics
    Y = [dataset[i][-1].squeeze() for i in range(len(dataset))]  # latents
    X = torch.stack(X, dim=0)
    Y = torch.stack(Y, dim=0)

    corr_matrix = np.zeros((6, Y.shape[1]))
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            corr_matrix[i, j], _ = spearmanr(X[:, i], Y[:, j])

    plt.figure(figsize=(16, 6))
    input_names = ["Temperature", "Date", "Humidity", "Ramp Type", "Duration", "Sensor Number"]
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    sns.heatmap(corr_matrix, annot=True, cmap=plt.get_cmap("RdBu").reversed(),
                yticklabels=[input_names[i] for i in range(corr_matrix.shape[0])],
                xticklabels=[f"Dim {j}" for j in range(corr_matrix.shape[1])],
                fmt=".2f", norm=norm)
    plt.xlabel("Latent Dimensions")
    plt.ylabel("Environmental Conditions")
    plt.title("Spearman Correlation Coefficients")
    plt.tight_layout()
    # plt.show()
    plt.savefig(model_path.replace(".pth", f"corr.png"))


if __name__ == '__main__':
    model_path = "autoencoder_model/ivcvscans-2025-08-12-03-24-52/e125_l15.918.pth"
    # plot_latent(model_path)
    explain_latent_corr(model_path)
    # explain_latent_on_data(model_path)
    #
    # model_path = "autoencoder_model/ivcvscans-2025-08-12-21-21-55/e112_l21.576.pth"  # without attention
    # plot_latent(model_path)
    # explain_latent_corr(model_path)
    # explain_latent_on_data(model_path)
    #
    # model_path = "autoencoder_model/ivcvscans-2025-08-13-01-24-08/e102_l6.585.pth"  # with attention
    # plot_latent(model_path)
    # explain_latent_corr(model_path)
    # explain_latent_on_data(model_path)
