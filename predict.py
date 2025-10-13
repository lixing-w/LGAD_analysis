import random
import os

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from torch.utils.data import DataLoader
import matplotlib.pylab as plt
import numpy as np

from tqdm import tqdm
from datetime import datetime

from model import AutoEncoder, Decoder, EnvToLatent
from dataset import AggregateIVDatasetForAutoEncoder
from utils import Sensor, DATABASE_DIR, disable_top_and_right_bounds, load_model_from_pth
from analyze import analyze_sensor_iv

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

sigmoid = nn.Sigmoid()
dataset = AggregateIVDatasetForAutoEncoder(DATABASE_DIR, data_config_path="./ml_data_config.txt")


def predict_curve(autoencoder_model: str, mlp_path: str, temp=None, humi=None, ramp_type=None, date=None, duration=None,
                  sensor_no=None, plot=True, is_conditional=False):
    """
    Parameters
    ----------
    autoencoder_model : str
        Path to the autoencoder model.
    mlp_path : str
        Path to the MLP that predicts latent vector from environmental variables.
    temp : float, optional
    humi : float, optional
    ramp_type : int {-1, 0, 1}, optional
    date : int, optional
        Datetime ordinal.
    duration : float, optional
    sensor_no : int, optional
    plot : bool, optional
        Whether show the predicted IV curve. Defaults to True.
    is_conditional : bool, optional
        Whether the MLP model is conditional. Defaults to False.
    """
    # initialize model
    mlp_model = load_model_from_pth(mlp_path, "EnvToLatent", None, device=device)
    mlp_model.eval()

    if is_conditional:
        decoder = load_model_from_pth(autoencoder_model, "ConditionalDecoder", 316, device=device)
    else:
        decoder = load_model_from_pth(autoencoder_model, "Decoder", 316, device=device)
    decoder.eval()

    # normalize date according to database 
    if date is not None:
        normalized_date = (date - dataset.date_mean) / dataset.date_std

    if temp is None: temp = 0
    if humi is None: humi = 5
    if ramp_type is None: ramp_type = 0
    if date is None: normalized_date = 0
    if duration is None: duration = 0
    if sensor_no is None: sensor_no = 0

    # metrics = torch.tensor([temp, normalized_date, humi, ramp_type, duration, sensor_no]).float().to(device)
    metrics = torch.tensor([temp, humi, ramp_type, sensor_no, dataset.sensor_number_to_thickness.get(sensor_no, 0.0), dataset.sensor_number_to_type.get(sensor_no, 0)]).float().to(device)
    # metrics = torch.tensor([temp, humi, ramp_type, sensor_no, 50, dataset.sensor_number_to_type.get(sensor_no, 0)]).float().to(device)
    p_latent = mlp_model(metrics)
    p_latent = p_latent.unsqueeze(0)
    if is_conditional:
        p_curve = decoder(p_latent, metrics.unsqueeze(0))
    else:
        p_curve = decoder(p_latent)

    mask = sigmoid(p_curve[:, 1, :])
    mask = mask.squeeze().cpu().detach().numpy()
    mask[:30] = 1
    valid_seq_len = np.where(mask < 0.1)[0][0]
    p_curve = p_curve[:, 0, :valid_seq_len]
    p_curve = p_curve.squeeze().cpu().detach().numpy()
    volt_grid = torch.arange(0, valid_seq_len, 1).detach().numpy()
    denoised = savgol_filter(median_filter(p_curve, 3), 7, 1)
    if plot:
        # plot the predicted IV
        plt.figure(figsize=(10, 8))
        plt.plot(volt_grid, denoised, color="black", label="Denoised")
        plt.plot(volt_grid, p_curve, color="pink", label="Predicted")
        plt.xlabel("Reverse-bias Voltage (V)")
        plt.ylabel(f"log(Pad Current (A))")
        plt.title(rf"IV Prediction")
        disable_top_and_right_bounds(plt)
        plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    return volt_grid, p_curve, denoised


def fake_sensor(autoencoder_model: str, mlp_path: str, is_conditional: bool=False, analysis_var='temp'):
    # make fake IV scans based on given models
    path = "./fake_sensors"
    if not os.path.exists(path):
        os.mkdir(path)
    start_time = datetime.now()
    sensor_name = f"{autoencoder_model.split(os.sep)[-1].removesuffix('.pth')}-{mlp_path.split(os.sep)[-1].removesuffix('.pth')}-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}"

    if not os.path.exists(os.path.join(path, sensor_name)):
        os.mkdir(os.path.join(path, sensor_name))

    params_list = []  # add env vars for prediction!
    sensor_no = 9
    if analysis_var == 'temp':
        for temp in range(-120, 180, 10):
            humi = 10 # fix humidity for now
            params_list.append([temp, humi, 0, dataset.date_to_z_score(datetime(2024, 12, 6)), 0, sensor_no])
    elif analysis_var == 'humi':
        for humi in range(0, 40, 2):
            temp = 20  # fix temp for now
            params_list.append([temp, humi, 0, dataset.date_to_z_score(datetime(2024, 12, 6)), 0, sensor_no])
    else:
        raise ValueError("analysis_var must be 'temp' or 'humi'")

    scan_no = 0
    for temp, humi, ramp_type, normalized_date, duration, sensor_no in params_list:
        file_lines = [f"Temperature: {temp} C\n",
                      f"Humidity: {humi} %\n",
                      f"Date: 1970-01-01 00:00:00.000000\n",
                      f"voltage,pad,gr,totalCurrent\n"]
        volt_grid, p_curve, denoised = predict_curve(autoencoder_model, mlp_path, temp=temp, humi=humi, ramp_type=ramp_type, sensor_no=sensor_no, plot=False, is_conditional=is_conditional)
        for v, i in zip(volt_grid, denoised):
            if v > 25:  # ignores below 25V
                file_lines.append(f"{v},{10 ** i},{0},{10 ** i}\n")

        with open(os.path.join(path, sensor_name, f"fake-{scan_no}.txt"), mode='w') as f:
            f.writelines(file_lines)

        scan_no += 1

    print(f"Generated {scan_no} IV scans at {path}")
    return path, sensor_name


def analyze_fake(path, sensor_name, analysis_var='temp'):
    # needs to temporarily modify DATABASE_DIR to "./fake_sensors"
    sensor = Sensor(sensor_name, database=path)
    analyze_sensor_iv(sensor, var=analysis_var)


if __name__ == "__main__":
    # autoencoder_model = "autoencoder_model/ivcvscans-2025-08-12-03-24-52/e125_l15.918.pth"
    # mlp_model = "env_to_latent_model/ivcvscans-2025-08-12-14-10-45/e136_l0.177.pth"
    # mlp_model = "env_to_latent_model/ivcvscans-2025-10-13-08-52-22-e2e/e82_l1.162.pth"
    # mlp_model = "env_to_latent_model/ivcvscans-2025-10-13-09-09-32-e2e/e81_l2.403.pth"
    
    # autoencoder_model = "conditional_autoencoder_model/ivcvscans-2025-10-13-09-31-32/e105_l0.300.pth"
    # mlp_model = "env_to_latent_model/ivcvscans-2025-10-13-09-59-07-e2e/e63_l4.837.pth"
    
    # autoencoder_model = "conditional_autoencoder_model/ivcvscans-2025-10-13-10-37-16/e93_l13.541.pth"
    # mlp_model = "env_to_latent_model/ivcvscans-2025-10-13-11-05-23-e2e/e81_l2.623.pth"
    
    # autoencoder_model = "conditional_autoencoder_model/ivcvscans-2025-10-13-11-59-29/e108_l0.332.pth"
    # mlp_model = "env_to_latent_model/ivcvscans-2025-10-13-12-08-01-e2e/e60_l3.557.pth"
    
    autoencoder_model = "conditional_autoencoder_model/ivcvscans-2025-10-13-14-25-08/e97_l0.269.pth"
    mlp_model = "env_to_latent_model/ivcvscans-2025-10-13-14-38-28-e2e/e86_l3.081.pth"
    
    analysis_var = 'temp'  # 'temp' or 'humi'
    path, sensor_name = fake_sensor(autoencoder_model, mlp_model, is_conditional=True, analysis_var=analysis_var)
    analyze_fake(path, sensor_name, analysis_var=analysis_var)
