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
dataset = AggregateIVDatasetForAutoEncoder(DATABASE_DIR)


def predict_curve(autoencoder_model: str, mlp_path: str, temp=None, humi=None, ramp_type=None, date=None, duration=None,
                  sensor_no=None, plot=True):
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
    """
    # initialize model
    mlp_model = load_model_from_pth(mlp_path, "EnvToLatent", None, device=device)
    mlp_model.eval()

    decoder = load_model_from_pth(autoencoder_model, "Decoder", 400, device=device)
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

    metrics = torch.tensor([temp, normalized_date, humi, ramp_type, duration, sensor_no]).float().to(device)
    p_latent = mlp_model(metrics)
    p_latent = p_latent.unsqueeze(0)
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


def fake_sensor(autoencoder_model: str, mlp_path: str):
    # make fake IV scans based on given models
    path = "./fake_sensors"
    if not os.path.exists(path):
        os.mkdir(path)
    start_time = datetime.now()
    sensor_name = f"{autoencoder_model.split(os.sep)[-1].removesuffix('.pth')}-{mlp_path.split(os.sep)[-1].removesuffix('.pth')}-{start_time.strftime('%Y-%m-%d-%H-%M-%S')}"

    if not os.path.exists(os.path.join(path, sensor_name)):
        os.mkdir(os.path.join(path, sensor_name))

    params_list = []  # add env vars for prediction!
    # for temp in range(-40, 130, 10):
    #     humi = 10 # fix humidity for now
    #     params_list.append([temp, humi, 0, dataset.date_to_z_score(datetime(2024, 12, 6)), 0, 9])

    for humi in range(0, 40, 2):
        temp = 20  # fix temp for now
        params_list.append([temp, humi, 0, dataset.date_to_z_score(datetime(2024, 12, 6)), 0, 9])

    scan_no = 0
    for temp, humi, ramp_type, normalized_date, duration, sensor_no in params_list:
        file_lines = [f"Temperature: {temp} C\n",
                      f"Humidity: {humi} %\n",
                      f"Date: 1970-01-01 00:00:00.000000\n",
                      f"voltage,pad,gr,totalCurrent\n"]
        volt_grid, p_curve, denoised = predict_curve(autoencoder_model, mlp_path, temp, humi, sensor_no, plot=False)
        for v, i in zip(volt_grid, denoised):
            file_lines.append(f"{v},{10 ** i},{0},{10 ** i}\n")

        with open(os.path.join(path, sensor_name, f"fake-{scan_no}.txt"), mode='w') as f:
            f.writelines(file_lines)

        scan_no += 1

    print(f"Generated {scan_no} IV scans at {path}")
    return path, sensor_name


def analyze_fake(path, sensor_name):
    # needs to temporarily modify DATABASE_DIR to "./fake_sensors"
    sensor = Sensor(sensor_name, database=path)
    analyze_sensor_iv(sensor, var='humi')


if __name__ == "__main__":
    # autoencoder_model = "autoencoder_model/ivcvscans-2025-08-12-21-21-55/e112_l21.576.pth"
    # mlp_model = "env_to_latent_model/ivcvscans-2025-08-12-21-42-46/e136_l0.16.pth"
    autoencoder_model = "autoencoder_model/ivcvscans-2025-08-13-15-56-15/e102_l2.580.pth"
    mlp_model = "env_to_latent_model/ivcvscans-2025-08-13-16-12-29/e92_l0.117.pth"
    # autoencoder_model = "autoencoder_model/ivcvscans-2025-08-12-03-24-52/e125_l15.918.pth"
    # mlp_model = "env_to_latent_model/ivcvscans-2025-08-12-14-10-45/e136_l0.177.pth"
    # autoencoder_model = 'autoencoder_model/ivcvscans-2025-08-11-16-44-32/e97_l13.130.pth'
    # mlp_model = 'env_to_latent_model/ivcvscans-2025-08-11-17-13-08/e199_l0.177.pth'

    path, sensor_name = fake_sensor(autoencoder_model, mlp_model)
    analyze_fake(path, sensor_name)
