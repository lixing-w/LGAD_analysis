from preprocess import interpolate_nonan
import os
from datetime import datetime

import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

from model import Encoder
from utils import (
    Sensor, list_sensors, load_data_config, load_sensor_config, parse_file_iv, load_model_from_pth
)
from analyze import find_breakdown


class IVDataset(Dataset):

    def __init__(self, DATABASE_DIR, sensor_name):
        self.sensor = Sensor(sensor_name)
        self.sensor_path = self.sensor.sensor_dir
        load_sensor_config(DATABASE_DIR, [self.sensor])  # remember to set dep_v for this sensor!
        load_data_config(DATABASE_DIR, [self.sensor])  # set up analysis preference

        # list all IV scans
        self.all_iv_scans = []
        for scan_folder in self.sensor.data_dirs:
            for scan in os.listdir(scan_folder):
                if not (not scan.startswith(".") and (scan.endswith(".txt") or scan.endswith(".iv"))):
                    continue  # ignore files other than iv scans
                full_scan_path = os.path.join(scan_folder, scan)
                if not self.sensor.is_ignored(full_scan_path):
                    self.all_iv_scans.append(full_scan_path)

    def __len__(self):
        return len(self.all_iv_scans)

    def __getitem__(self, index):
        full_scan_path = self.all_iv_scans[index]
        temperature, date, data, humi, ramp_type = parse_file_iv(full_scan_path)
        xs = data["voltage"]
        ys = data["pad"]

        if np.median(xs) < 0:
            xs = -xs
        if np.median(ys) < 0:
            ys = -ys
        with np.errstate(divide='ignore', invalid='ignore'):  # suppress invalid value and division by zero err
            ys_log10 = np.log10(ys)  # we will remove nan and inf after

        xs, ys_log10 = interpolate_nonan(xs, ys_log10, 1)
        assert len(xs) != 0, full_scan_path

        dep_v = self.sensor.depletion_v
        if dep_v is None:
            dep_v = 25  # if not set, defaults to 25V
        set_params = self.sensor.query_conf(full_scan_path)
        if set_params is not None:  # configuration overrides
            if "DEP" in set_params:
                old_dep_v = dep_v
                dep_v = set_params["DEP"]

            if "RT" in set_params:
                ramp_type = set_params["RT"]

        try:
            first_idx_after_dep_v = np.where(xs > dep_v)[0][0]
        except:  # cannot find voltage after dep_v, either dep_v too high or voltage range too small
            print(
                f"Warning: Scan at {full_scan_path} voltage range too small. "
                f"Should be ignored or try a different depletion voltage.")
            plt.figure(figsize=(10, 6))
            # a. plot the scan itself
            plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", marker='o', markersize=3)
            plt.xlabel("Reverse-bias Voltage (V)")
            plt.ylabel(f"log(Pad Current (A))")
            plt.title(
                rf"IV Scan at {temperature}$^\circ$C: Unable to Find Breakdown V; Range too Small ({self.sensor.name} {date.strftime('%b %d, %Y')})")
            plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
            plt.tight_layout()
            plt.show()
            plt.close()
            return None

        xs = xs[first_idx_after_dep_v:]
        ys_log10 = ys_log10[first_idx_after_dep_v:]

        temperature_seq = np.ones_like(xs)
        humi_seq = np.ones_like(xs)
        humi_seq *= humi
        temperature_seq *= temperature
        v_temp_humi_seq = np.stack([xs, temperature_seq], axis=-1)
        v_temp_humi_seq = torch.from_numpy(v_temp_humi_seq).float()

        iv_seq = np.stack([xs, ys_log10], axis=0)
        iv_seq = torch.from_numpy(iv_seq).float()

        return v_temp_humi_seq, iv_seq


class AggregateIVDatasetForAutoEncoder(Dataset):
    """ A database consisting of all IV scans from DATABASE_DIR """

    # contains ALL IV scans from DATABASE_DIR, regardless what sensor they belong to
    def __init__(self, database_dir: str, mode: str = "compact"):
        """
        Initializes a database consisting of all IV scans from DATABASE_DIR.
        
        Parameters
        ----------
        database_dir : str 
            Relative path to your database.
        mode : str, optional, {"compact", "full"}
            If set to "compact", indexing into the database would return 
            iv_seq and seq_len. If set to "full", it would return 
            all other meta-information in addition to iv_seq and seq_len.
            Defaults to "compact"
        """
        super().__init__()
        assert mode in {"compact", "full"}
        self.mode = mode
        self.sensors = list_sensors()
        load_sensor_config(database_dir, self.sensors)
        load_data_config(database_dir, self.sensors)
        # every sensor is associated with a unique number for identification
        self.sensor_name_to_number = dict()
        self.sensor_number_to_name = dict()
        for i, sensor in enumerate(self.sensors):
            self.sensor_name_to_number[sensor.name] = i
            self.sensor_number_to_name[i] = sensor.name
        print(f"Found {len(self.sensors)} sensors in database.")
        print(self.sensor_number_to_name)
        # list all IV scans
        self.all_iv_scans = []
        self.all_iv_scans_from_sensor = []
        ignored_scan_count = 0
        for sensor in self.sensors:
            for scan_folder in sensor.data_dirs:
                for scan in os.listdir(scan_folder):
                    if not (not scan.startswith(".") and (scan.endswith(".txt") or scan.endswith(".iv"))):
                        continue  # ignore files other than iv scans
                    full_scan_path = os.path.join(scan_folder, scan)
                    if not sensor.is_ignored(full_scan_path):
                        self.all_iv_scans.append(full_scan_path)
                        self.all_iv_scans_from_sensor.append(self.sensor_name_to_number[sensor.name])
                    else:
                        ignored_scan_count += 1
        print(
            f"Found {len(self.all_iv_scans) + ignored_scan_count} IV scans, (ignoring {ignored_scan_count}, using {len(self.all_iv_scans)}).")
        # now, calculate longest sequence length for padding and cache data
        self.max_seq_len = 0
        self.data = []
        all_date_ordinal = []
        for full_scan_path in self.all_iv_scans:
            temperature, date, data, humi, ramp_type, duration = parse_file_iv(full_scan_path)

            # we normalize date by calculating z-score
            all_date_ordinal.append(date.toordinal())

            xs = data["voltage"]
            ys = data["pad"]
            # remove nan
            if np.median(xs) < 0:
                xs = -xs
            if np.median(ys) < 0:
                ys = -ys
            with np.errstate(divide='ignore', invalid='ignore'):  # suppress invalid value and division by zero err
                ys_log10 = np.log10(ys)  # we will remove nan and inf after
            xs, ys_log10 = interpolate_nonan(xs, ys_log10, 1)

            nan_mask = np.isnan(ys_log10) | np.isinf(ys_log10)
            xs = xs[~nan_mask]
            ys_log10 = ys_log10[~nan_mask]
            assert not np.any(nan_mask), f"{np.sum(nan_mask)}, {nan_mask.shape}"

            # estimate the breakdown pt 
            if mode == "full":
                lines, std = find_breakdown(xs, ys_log10, start_idx=30, bd_thresh=0.5, plot=False)
                slope, offset, bd_v = lines[0]
                c_at_100v = slope * 100 + offset
            else:
                slope, offset, bd_v = None, None, None
                c_at_100v = None

            # pytorch expects (B, Channel, Seq_len)
            data = np.stack([xs, ys_log10], axis=0)
            seq_len = data.shape[1]
            self.max_seq_len = max(self.max_seq_len, seq_len)
            self.data.append([temperature, date, data, humi, ramp_type, duration, seq_len,
                              bd_v, c_at_100v, slope, offset])

        # calculate z-scores for date
        all_date_ordinal = np.array(all_date_ordinal)
        self.date_mean = np.mean(all_date_ordinal)
        self.date_std = np.std(all_date_ordinal)
        normalized_date_ordinal = (all_date_ordinal - self.date_mean) / self.date_std

        # perform padding for all data seq 
        for i in range(len(self.data)):
            self.data[i][1] = normalized_date_ordinal[i]  # replace w/ z-score
            seq_len = self.data[i][6]
            self.data[i][2] = np.pad(self.data[i][2], [(0, 0), (0, self.max_seq_len - seq_len)], constant_values=0)
            assert self.data[i][2].shape[1] == self.max_seq_len

        print(f"IV sequence padded to max sequence length {self.max_seq_len}.")

    def z_score_to_date_ordinal(self, z):
        return z * self.date_std + self.date_mean

    def date_to_z_score(self, dt: datetime):
        return (dt.toordinal() - self.date_mean) / self.date_std

    def __len__(self):
        return len(self.all_iv_scans)

    def __getitem__(self, index):
        temperature, date, data, humi, ramp_type, duration, seq_len, bd_v, i_at_100v, slope, offset = self.data[index]
        iv_seq = torch.from_numpy(data).float()
        if self.mode == "compact":
            return iv_seq, seq_len
        elif self.mode == "full":
            sensor_number = self.all_iv_scans_from_sensor[index]
            sensor_name = self.sensor_number_to_name[sensor_number]
            if humi is None: humi = float('inf')
            if duration is None: duration = float('inf')
            if ramp_type == 0: ramp_type = float('inf')
            if temperature is None: temperature = float('inf')
            return temperature, date, iv_seq, humi, ramp_type, duration, seq_len, bd_v, i_at_100v, slope, offset, sensor_number, sensor_name


class AggregateLatentDataset(AggregateIVDatasetForAutoEncoder):
    """ A database consisting of all IV scans from DATABASE_DIR """

    # contains ALL IV scans from DATABASE_DIR, regardless what sensor they belong to
    def __init__(self, database_dir: str, model_path: str):
        """
        Initializes a database consisting of all IV scans from DATABASE_DIR.
        
        Parameters
        ----------
        database_dir : str 
            The relative path to your database.
        model_path : str
            The relative path to the autoencoder that generates latents.
        """
        super().__init__(database_dir, mode="full")

        # initialize autoencoder 
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        print(f"Using device: {device}")
        model = load_model_from_pth(model_path, "Encoder", self.max_seq_len, device=device)
        model.eval()

        for d in self.data:  # compute latent, then append to d
            iv_seq = torch.from_numpy(d[2]).float().to(device).unsqueeze(0)
            i_seq = iv_seq[:, [1], :]  # add batch dim
            latent = model(i_seq) # CHANGE THIS FOR DIFFERENT MODELS! Some operates on i_seq, some on iv_seq
            d.append(latent.cpu().detach())

    def z_score_to_date_ordinal(self, z):
        return z * self.date_std + self.date_mean

    def __len__(self):
        return len(self.all_iv_scans)

    def __getitem__(self, index):
        temperature, date, data, humi, ramp_type, duration, seq_len, bd_v, i_at_100v, slope, offset, latent = self.data[index]
        iv_seq = torch.from_numpy(data).float()
        sensor_number = self.all_iv_scans_from_sensor[index]
        sensor_name = self.sensor_number_to_name[sensor_number]
        if humi is None: humi = 0
        if duration is None: duration = 0
        if ramp_type == 0: ramp_type = 0
        if temperature is None: temperature = 25
        return temperature, date, iv_seq, humi, ramp_type, duration, seq_len, bd_v, sensor_number, sensor_name, latent
