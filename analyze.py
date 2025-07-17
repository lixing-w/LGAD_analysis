import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import random
from datetime import datetime
from Sensor import Sensor 
from utils import *

import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=plt.cm.tab20.colors)
# use 20 color cycle instead of the default 10 cycle

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="LGADs Analysis System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--iv',
        action='store_true',
        help='''Whether to analyze IV scans.''')
    parser.add_argument(
        '--cv',
        action='store_true',
        help='''Whether to analyze CV scans.''')
    parser.add_argument(
        '--file', '-f',
        nargs='+',
        default=None,
        help='''Analyze one or more specific scan files.'''
    )
    parser.add_argument(
        '--overwrite', 
        action='store_true',
        default=False,
        help='''For new analysis results, whether modify sensor_config.txt or overwrite old.'''
    )
    parser.add_argument(
        '--curr_type', 
        choices=['pad', 'gr', 'total'],
        default='pad',
        help='''Specify a current type for IV analysis. Defaults to 'pad'.'''
    )
    parser.add_argument(
        '--sensor', '--sensors', '-s',
        nargs='+',
        default=None,
        help='''Specify one or more sensors to analyze. If not given, analyze all sensors.'''
    )
    parser.add_argument(
        '--clear_plots', '--clear',
        action='store_true',
        help='''Whether to remove all plots (before analysis).'''
    )

    return parser.parse_args()

def clear_plots():
    """
    Removes all plots in DATABASE_DIR
    """
    for root, dirs, files in os.walk(DATABASE_DIR):
        for file in files:
            if file.lower().endswith('.png'):
                path = os.path.join(root, file)
                os.remove(path)
                print(f"Removed: {path}")

def load_data_config(path: str, sensors: list[Sensor]):
    """
    Loads scans that we ignore.
    
    Each line in the data_config.txt should be:
    - an expression starting with command: 'DR' (drop record) 'SPEC SET' (specify then set properties)
    - each expression should have the following format:
      [command] N:[Name] D:[Date]|[Date~Date]|[Date~]|[~Date] T:[Temp]|[Temp~Temp]|[Temp~]|[~Temp] F:[dir] R:[regex] RT:[1|-1|0]
    - all ranges are inclusive
    - a scan is ignored if it satisfies EVERY condition of SOME expression
    
    Examples:
    DR N:DC_W3058 T:-60 D:26/10/2023
    => drop scans of sensor DC_W3058, equal to -60C, and on date 26/10/2023
    DR N:DC_W3058 R:RoomTemp
    => drop scans of sensor DC_W3058 whose file name matches (at least partially) regex "RoomTemp"
    DR N:AC_W3096 T:20~40 RT:1
    => drop scans of sensor AC_W3096, whose temperature between 20C and 40C (inclusive), and ramp type is ramping up
    
    SPEC N:DC_W3058 D:30/10/2023 T:-20 RT:-1 SET DEP:70
    => specify scan of sensor DC_W3058 on date 30/10/2023, temperature being -20C, ramping down
    => and set the depletion voltage as 70V just for analyzing this scan
    """
    
    name_to_sensor = dict()
    for sensor in sensors:
        name_to_sensor[sensor.name] = sensor
        
    if not os.path.exists(os.path.join(path, "data_config.txt")):
        with open(os.path.join(path, "data_config.txt"), "w") as f:
            return 
        
    with open(os.path.join(path, "data_config.txt"), "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            line = line.removesuffix("\n")
            # ignore comments
            comment_idx = line.find('#') 
            if comment_idx != -1:
                line = line[:comment_idx].strip()
            # ignore empty lines
            if len(line) <= 1:
                continue

            # paring commands
            def parse_spec(info):
                # parses commands that specify scan(s)
                params = dict()
                if "D" in info: # date_range
                    if "~" not in info["D"]:
                        params['date_range']=datetime.strptime(info["D"], "%d/%m/%Y")
                    elif info["D"].endswith("~"):
                        params['date_range']=(datetime.strptime(info["D"][:-1], "%d/%m/%Y"), None)
                    elif info["D"].startswith("~"):
                        params['date_range']=(None, datetime.strptime(info["D"][1:], "%d/%m/%Y"))
                    else:
                        r = info["D"].split("~")
                        params['date_range']=(datetime.strptime(r[0], "%d/%m/%Y"), datetime.strptime(r[1], "%d/%m/%Y"))
                if "F" in info: # file
                    params['file'] = info["F"]
                if "T" in info: # temp_range
                    if "~" not in info["T"]:
                        params['temp_range']=float(info["T"])
                    elif info["T"].endswith("~"):
                        params['temp_range']=(float(info["T"][:-1]), None)
                    elif info["T"].startswith("~"):
                        params['temp_range']=(None, float(info["T"][1:]))
                    else:
                        r = info["T"].split("~")
                        params['temp_range']=(float(r[0]), float(r[1]))
                if "R" in info: # regex
                    params['regex']=info["R"]
                if "RT" in info: # ramp_type
                    params['ramp_type']=int(info['RT'])
                
                return params 
            
            def parse_set(info):
                # parses commands that specify properties of a scan
                set_params = dict()
                if "DEP" in info:
                    set_params['DEP'] = float(info['DEP'])
                if "RT" in info:
                    set_params['RT'] = int(info['RT'])
                return set_params
            
            if line.startswith("DR"):
                toks = line.removeprefix("DR ").split(' ')
                # ignore multiple spaces
                toks = [tok for tok in toks if len(tok)]
                file_info = dict()
                for tok in toks:
                    pair = tok.split(":")
                    file_info[pair[0]] = pair[1]
                assert "N" in file_info
                sensor = name_to_sensor[file_info["N"]]
                params = parse_spec(file_info)
                sensor.add_ignore(**params)
                
            elif line.startswith("SPEC"):
                toks = line.removeprefix("SPEC ").split(' ')
                # ignore multiple spaces
                toks = [tok for tok in toks if len(tok)]
                file_info = dict()
                set_info = dict()
                
                parsing_dict = file_info
                for tok in toks:
                    if tok == "SET":
                        parsing_dict = set_info # now starts parsing properties
                        continue
                    pair = tok.split(":")
                    parsing_dict[pair[0]] = pair[1]
                    
                assert "N" in file_info
                sensor = name_to_sensor[file_info["N"]]
                file_params = parse_spec(file_info)
                set_params = parse_set(set_info)
                params = file_params 
                params['set_params'] = set_params
                sensor.add_scan_conf(**params)
                
            else:
                raise ValueError(f"Invalid data_config command at line {i}: ...{line}...")
    
    print(f"Loaded file configurations")
    
    return 
            
def write_sensor_config(path: str, sensors: list[Sensor]):
    """
    Writes out sensor information to path/sensor_config.txt.
    """
    
    with open(os.path.join(path, "sensor_config.txt"), "w") as f:
        f.write("# This is config file for sensors.\n")
        f.write("# N:name T:type D:avg_dep_v C:c_after_depletion,freq,temp|... R:bd_thresh L:slope,offset,avg_sigma B:temp,avg_volt,avg_humi|... \n")
        for sensor in tqdm(sensors, desc="Writing sensor config"):
            f.write(f"N:{sensor.name:<20} T:{"None" if sensor.type is None else sensor.type:<4} ")

            f.write(f"D:{"None" if sensor.depletion_v is None else f'{sensor.depletion_v:.3f}'} ")
            if sensor.cv_scan_data is None or len(sensor.cv_scan_data) < 1:
                f.write(f"C:None")
            else:
                f.write(f"C:")
                for c_after_dep, freq, temp in sensor.cv_scan_data:
                    f.write(f"{c_after_dep:.3g},{freq:.3g},{temp:.1f}|")
            f.write(" ")

            f.write(f"R:{'None' if sensor.bd_thresh is None else f'{sensor.bd_thresh:.2f}'} ")

            if sensor.iv_scan_line is None or len(sensor.iv_scan_line) < 1:
                f.write(f"L:None ")
            else:
                f.write("L:")
                slope, offset, avg_sigma = sensor.iv_scan_line # unpack
                f.write(f"{slope:.3f},{offset:.3f},{avg_sigma:.3f}")
            f.write(" ")
            
            if sensor.iv_scan_data is None or len(sensor.iv_scan_data) < 1:
                f.write(f"B:None ")
            else:
                f.write("B:")
                for temp, volt, humi in sensor.iv_scan_data:
                    f.write(f"{temp:.1f},{volt:.3f},{humi:.3f}|")
            f.write(" \n")
                
def load_sensor_config(path: str, sensors: list[Sensor], load_iv=True, load_cv=True):
    """
    Set up sensor using path/sensor_config.txt.
    """
    name_to_sensor = dict()
    for sensor in sensors:
        name_to_sensor[sensor.name] = sensor
    
    config_path = os.path.join(path, "sensor_config.txt")
    if not os.path.exists(config_path): # if config file DNE, create
        with open(config_path, "w") as f:
            return 
        
    with open(config_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading sensor config"):
            # ignore comments
            comment_idx = line.find('#') 
            if comment_idx != -1:
                line = line[:comment_idx].strip()
            # ignore empty lines
            if len(line) <= 1:
                continue
            # # N:name T:type D:avg_dep_v C:c_after_depletion,freq,temp|... R:bd_thresh L:slope,offset,avg_sigma B:temp,avg_volt,avg_humi|...
            line = line.removesuffix('\n')
            line = line.split(' ')
            info = dict()
            for tok in line: 
                if tok == '': # happens with there're multiple spaces. ignore
                    continue 
                pair = tok.split(':')
                info[pair[0]] = pair[1]
            
            name, type, dep_v, cv_scan_data, bd_thresh, iv_scan_data, iv_scan_line = info["N"], info["T"], info["D"], info["C"], info["R"], info["B"], info["L"]
            sensor = name_to_sensor[name]
            sensor.name = name # name is assumed to be not None
            sensor.type = type if type != "None" else None
            sensor.bd_thresh = float(bd_thresh) if bd_thresh != "None" else None
            sensor.depletion_v = float(dep_v) if dep_v != "None" else None 
            if iv_scan_data != "None" and load_iv:
                bd_list = iv_scan_data.split('|')
                bd_list = bd_list[:-1] # remove the trailing empty list since the str ends with '|' 
                for i in range(len(bd_list)):
                    bd_list[i] = [float(tok) for tok in bd_list[i].split(',') if tok != '']
                sensor.iv_scan_data = bd_list
            if cv_scan_data != "None" and load_cv:
                cv_list = cv_scan_data.split('|')
                cv_list = cv_list[:-1] # remove the trailing empty list since the str ends with '|'
                for i in range(len(cv_list)):
                    cv_list[i] = [float(tok) for tok in cv_list[i].split(',') if tok != '']
                sensor.cv_scan_data = cv_list
            if iv_scan_line != "None" and load_iv:
                scan_line_list = [float(tok) for tok in iv_scan_line.split(',') if tok != '']
                sensor.iv_scan_line = scan_line_list
            
def list_sensors():
    """
    Initializes and returns a list of Sensor objects
    """
    path = DATABASE_DIR
    folders = [p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]
    sensors = [Sensor(folder) for folder in folders]
    sensors = sorted(sensors, key=lambda s:s.name)
    return sensors 

def get_min_uncertainty(sequence: np.ndarray, range: int=None):
    """
    Returns sequence common difference / sqrt(range). If range is not 
    given, defaults to length of the sequence
    """
    
    diff = determine_spacing(sequence)
    if range is None:
        range = len(sequence)
    return diff / np.sqrt(range)
    
def disable_top_and_right_bounds(plt):
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return 
    
def ransac(xs: np.ndarray, ys: np.ndarray, thresh: float=0.05, iter: int=2000):
    """
    RANSAC that fits a line for given x and y data
    
    Inputs:
    xs - x data 
    ys - y data 
    thresh - max y-distance to be considered as an inlier
    iter - number of RANSAC iterations
    
    Output:
    lines - a list of lists [slope, offset, inlier count, mse]
    
    Note:
    The output is not sorted.
    """
    assert len(xs) == len(ys)
    lines = [] # each entry is a list [slope, offset, inlier count, mse, predicted_ys]
    seen = set() # some dot pairs may have been selected. dedup
    index_list = range(len(xs))
    for _ in range(iter):
        index_pair = random.sample(index_list, 2)
        if (index_pair[0], index_pair[1]) in seen:
            continue
        else:
            seen.add((index_pair[0], index_pair[1]))
        slope = (ys[index_pair[1]] - ys[index_pair[0]]) / (xs[index_pair[1]] - xs[index_pair[0]])
        offset = ys[index_pair[0]] - slope * xs[index_pair[0]]
        
        pred_ys = xs * slope + offset # the y's for fitted line
        diff = np.abs(pred_ys - ys)
        is_close_map = diff < thresh 
        inlier_count = np.sum(is_close_map)
        
        # some data might be nan due to np.log10, so use np.nanmean to find mse
        mse = np.nanmean(np.square(diff))
        
        lines.append([slope, offset, inlier_count, mse])
    
    return lines

def find_breakdown(xs: np.ndarray, ys: np.ndarray, start_idx: int, path: str, bd_thresh: float, plot: bool=True):
    """
    Given a single IV scan, finds the breakdown voltage by fitting linear lines.
    
    Inputs:
    xs - voltage data
    ys - current data 
    start_idx - data before this index is ignored
    path - the path to this particular scan
    bd_thresh - threshold for determining breakdown
    plot - whether generate and save breakdown distribution plots
    
    Output:
    a dictionary containing parameters of the fitted lines, breakdown voltage, 
    uncertainty of breakdown voltage, and index of breakdown voltage.
    """
    use_ransac = True
    
    # truncate data 
    valid_xs = xs[start_idx:]
    valid_ys = ys[start_idx:]
    assert len(valid_xs) == len(valid_ys)
    
    if use_ransac:
        # repeat 2000 times, each time randomly select 2 dots from range, 
        # fit a line, and determine the number of inliers, given a threshold
        # (a different one from bd_thresh)
        # lines with more inliers are more likely to be true trend, thus we can 
        # find a distribution of breakdown voltage
        
        fit_range = len(valid_xs) // 2
        
        lines = ransac(valid_xs[:fit_range], valid_ys[:fit_range], 0.05, 2000)
        
        for line in lines:
            # append bd_voltage prediction to each sublist!
            slope, offset = line[0], line[1]
            inlier_count, mse = line[2], line[3]
            pred_ys = slope * valid_xs + offset
            
            # find the breakdown voltage with linear interpolation
            last_id_in_thresh = -1 # some lines are so ridiculous that we cant find last_id_in_thresh
            for i in range(len(valid_ys)):
                if abs(valid_ys[i] - pred_ys[i]) < bd_thresh:
                    last_id_in_thresh = i

            if last_id_in_thresh + 1 == len(valid_ys):
                # last data point! just use it as is
                bd_voltage = valid_xs[last_id_in_thresh]
            else: 
                # find the next data point, fit a line, solve for intersection
                slope_inter = (valid_ys[last_id_in_thresh+1] - valid_ys[last_id_in_thresh]) / (valid_xs[last_id_in_thresh+1] - valid_xs[last_id_in_thresh])
                offset_inter = valid_ys[last_id_in_thresh] - slope_inter * valid_xs[last_id_in_thresh]
                bd_voltage = (offset_inter - (offset + bd_thresh)) / (slope - slope_inter)
            
            line.append(bd_voltage)
        
        # now each sublist has structure [slope, offset, inlier_count, mse, bd_voltage]
        lines = np.array(lines)
        # sanity check, breakdown voltage must > 0
        mask = lines[:,4] > 0
        # sanity check #2, the fitted line must have slope > 0
        mask &= lines[:,0] > 0
        lines = lines[mask]
        if len(lines) == 0:
            # There's something wrong with the data!
            # We won't fit, just return!
            return None, -1
        
        # sort according to bd_voltage
        lines = lines[np.argsort(lines[:,4])]
        
        # cutoff extreme values
        cum_weights = np.cumsum(lines[:,2])
        total_weight = cum_weights[-1]
        
        trim_ratio = 0.10
        lower = trim_ratio * total_weight
        upper = (1 - trim_ratio) * total_weight

        mask = (cum_weights >= lower) & (cum_weights <= upper)

        # group line inliers and outliers
        filtered_lines = lines[mask]
        lines = lines[~mask]
        
        mean = np.average(filtered_lines[:,4], weights=filtered_lines[:,2])
        variance = np.average((filtered_lines[:,4] - mean)**2, weights=filtered_lines[:,2])
        # calculate uncertainty for bd_voltage
        std = np.sqrt(variance)
        std = max(std, get_min_uncertainty(valid_xs, fit_range))

        # sort against inlier counts so the line with most inliers is at index 0
        # then sort against mse (if inlier counts are same, use one with smaller mse)
        filtered_lines = filtered_lines[np.lexsort((filtered_lines[:, 3], -filtered_lines[:, 2]))]
        
        # separate into bins
        bins = np.histogram_bin_edges(lines[:, 4], bins=160)
        bin_idx = np.digitize(lines[:,4], bins) - 1
        bin_idx2 = np.digitize(filtered_lines[:,4], bins) - 1
        # calculate max inlier count by each bin
        bin_mic = [lines[:,2][bin_idx == i].max() if np.any(bin_idx == i) else np.nan for i in range(len(bins)-1)]
        bin_mic2 = [filtered_lines[:,2][bin_idx2 == i].max() if np.any(bin_idx2 == i) else np.nan for i in range(len(bins)-1)]
        # calculate frequency by each bin
        bin_tic = [(lines[:,2][bin_idx == i] >= 1).sum() if np.any(bin_idx == i) else np.nan for i in range(len(bins)-1)]
        bin_tic2 = [(filtered_lines[:,2][bin_idx2 == i] >= 1).sum() if np.any(bin_idx2 == i) else np.nan for i in range(len(bins)-1)]

        # calculate min mse by each bin # some data could be nan due to np.log10, so use nan-ignoring np.nanmin()
        bin_mrmse = np.sqrt([np.nanmin(lines[:,3][bin_idx == i]) if np.any(bin_idx == i) else np.nan for i in range(len(bins)-1)])
        bin_mrmse2 = np.sqrt([np.nanmin(filtered_lines[:,3][bin_idx2 == i]) if np.any(bin_idx2 == i) else np.nan for i in range(len(bins) - 1)])
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        if plot: # plot breakdown distribution (frequency) and rmse
            plt.figure(figsize=(8,6))
            ax1 = plt.gca()
            ax1.bar(bin_centers, bin_tic, width=np.diff(bins), color='black', alpha=0.3, label=f'Outliers (Top/Bottom {int(trim_ratio*100)}%)')
            ax1.bar(bin_centers, bin_tic2, width=np.diff(bins), color='purple', alpha=0.7)
            ax1.set_ylabel('Frequency')
            ax1.set_xlabel('Breakdown Voltage ($V$)')
            
            ax2 = ax1.twinx()
            ax2.plot(bin_centers, bin_mrmse, label='Outlier RMSE', color='black', alpha=0.4)
            ax2.plot(bin_centers, bin_mrmse2, label='RMSE', color='purple', alpha=1)
            ax2.set_ylabel('RMSE', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            ax1.axvline(filtered_lines[0][4], color="black", ls="--", label=f"Max Weight Min RMSE Choice: {filtered_lines[0][4]:.2f} $V$")
            
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
            plt.title('Breakdown Voltage Distribution (Frequency Per Bin)')
            plt.tight_layout()
            plt.savefig(f"{path.removesuffix(".txt").removesuffix(".iv")}_bdv_freq.png")
            plt.close()
        if plot: # plot breakdown distribution (max weight) and rmse
            plt.figure(figsize=(8,6))
            ax1 = plt.gca()
            ax1.bar(bin_centers, bin_mic, width=np.diff(bins), color='black', alpha=0.3, label=f'Outliers (Top/Bottom {int(trim_ratio*100)}%)')
            ax1.bar(bin_centers, bin_mic2, width=np.diff(bins), color='purple', alpha=0.7)
            ax1.set_ylabel('Max Weights')
            ax1.set_xlabel('Breakdown Voltage ($V$)')
            
            ax2 = ax1.twinx()
            ax2.plot(bin_centers, bin_mrmse, label='Outlier RMSE', color='black', alpha=0.4)
            ax2.plot(bin_centers, bin_mrmse2, label='RMSE', color='purple', alpha=1)
            ax2.set_ylabel('RMSE', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            ax1.axvline(filtered_lines[0][4], color="black", ls="--", label=f"Max Weight Min RMSE Choice: {filtered_lines[0][4]:.2f} $V$")
            
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
            plt.title('Breakdown Voltage Distribution (Max Weight Per Bin)')
            plt.tight_layout()
            plt.savefig(f"{path.removesuffix(".txt").removesuffix(".iv")}_bdv_max.png")
            plt.close()
        
        lines = filtered_lines
        lines = np.delete(lines, [2, 3], axis=1) 
        # now each line has structure [slope, offset, bd_voltage]
        
    else: # if not use ransac
        # repeat 100 times to calculate uncertainty
        rand_len = 100
        lines = [] # each entry is a list [slope, offset, bd_voltage]
        fit_range = len(valid_xs) // 2
        
        # calculate linear fit for full range
        fixed_popt, _, _, _ = linear_fit(valid_xs[:fit_range], valid_ys[:fit_range], p0=[1, -100])
        slope, offset = fixed_popt[0], fixed_popt[1]
        pred_ys = linear(valid_xs, slope, offset)

        # find the breakdown voltage with linear interpolation
        for i in range(len(valid_ys)):
            if abs(valid_ys[i] - pred_ys[i]) < bd_thresh:
                last_id_in_thresh = i

        if last_id_in_thresh + 1 == len(valid_ys):
            # last data point! just use it as is
            bd_voltage = valid_xs[last_id_in_thresh]
        else: 
            # find the next data point, fit a line, solve for intersection
            slope_inter = (valid_ys[last_id_in_thresh+1] - valid_ys[last_id_in_thresh]) / (valid_xs[last_id_in_thresh+1] - valid_xs[last_id_in_thresh])
            offset_inter = valid_ys[last_id_in_thresh] - slope_inter * valid_xs[last_id_in_thresh]
            bd_voltage = (offset_inter - (offset + bd_thresh)) / (slope - slope_inter)
        
        lines.append([fixed_popt[0], fixed_popt[1], bd_voltage])
        
        # repeat rand_len times to use width for uncertainty calculation
        select_from = range(fit_range)
        for rand_i in range(1, rand_len):
            indices = random.sample(select_from, k=fit_range//2) # sample half the data
            fixed_popt, _, _, _ = linear_fit(valid_xs[indices], valid_ys[indices], p0=[1, -100])
            slope, offset = fixed_popt[0], fixed_popt[1]
            pred_ys = linear(valid_xs, slope, offset)

            # find the breakdown voltage with linear interpolation
            for i in range(len(valid_ys)):
                if abs(valid_ys[i] - pred_ys[i]) < bd_thresh:
                    last_id_in_thresh = i

            if last_id_in_thresh + 1 == len(valid_ys):
                # last data point! just use it as is
                bd_voltage = valid_xs[last_id_in_thresh]
            else: 
                # find the next data point, fit a line, solve for intersection
                slope_inter = (valid_ys[last_id_in_thresh+1] - valid_ys[last_id_in_thresh]) / (valid_xs[last_id_in_thresh+1] - valid_xs[last_id_in_thresh])
                offset_inter = valid_ys[last_id_in_thresh] - slope_inter * valid_xs[last_id_in_thresh]
                bd_voltage = (offset_inter - (offset + bd_thresh)) / (slope - slope_inter)
                
            lines.append([fixed_popt[0], fixed_popt[1], bd_voltage])
        
        lines = np.array(lines)
        # calculate uncertainty for bd_voltage
        std = np.std(lines[:,2])
        std = max(std, get_min_uncertainty(valid_xs, fit_range))
            
        if plot: # plot breakdown distribution
            plt.figure()
            plt.hist(lines[:,2], histtype='step', bins=60)
            plt.ylabel('Frequency')
            plt.xlabel('Breakdown Voltage')
            plt.title('Breakdown Voltage Histogram (Linear Fit)')
            plt.tight_layout()
            plt.savefig(f"{path}_bdv.png")
            plt.close()

    return lines, std

def analyze_sensor_iv(sensor: Sensor, curr_type: str='pad', plot=True):
    # 1. sets up related constants for the sensor:
    # a. use the specified threshold for breakdown fits
    bd_thresh = sensor.bd_thresh 
    if bd_thresh is None:
        bd_thresh = 0.5 # defaults to 0.5
    # b. use specified depletion voltage
    dep_v = sensor.depletion_v
    if dep_v is None:
        dep_v = 25 # if not set, defaults to 25V
        
    all_data_in_dirs = [] # used for 3). a list [[[temp, date, data, humidity, ramp_type, bd_v, std],...]...]
    
    def get_curr_type_label():
        if curr_type == 'pad': return "Pad"
        elif curr_type == 'gr': return "Guard Ring"
        elif curr_type == 'total': return "Total Current"
        else: raise ValueError(f"Invalid Current Type: {curr_type}. Must be one of 'pad', 'gr', or 'total'.")
        
    # 2. loops thru all dirs containing scans
    for dir in tqdm(sensor.data_dirs, desc=f"Analyzing IV profiles for sensor {sensor.name:<20}"): 
        # Note: plt only has one buffer that's cleared whenever 
        # plt.figure() is called! 

        all_data_in_dir = [] # used for b). a list [[temp, date, data, humidity, ramp_type, bd_v, std],...], 
        # dir is relative path to a folder containing scans
        # a. loops thru all individual scans within the dir
        tot_scan_count = 0
        ignored_scan_count = 0
        
        for scan_path in os.listdir(dir):
            if scan_path.startswith("."): continue # ignore hidden files 
            if not (scan_path.endswith(".iv") or scan_path.endswith(".txt")): continue # ignore non .txt non .iv 
            tot_scan_count += 1

            set_params = sensor.query_conf(os.path.join(dir, scan_path))
            
            temperature, date, data, humidity, ramp_type = parse_file_iv(os.path.join(dir, scan_path))
            
            if set_params is not None: # configuration overrides
                if "DEP" in set_params:
                    old_dep_v = dep_v 
                    dep_v = set_params["DEP"]
                
                if "RT" in set_params:
                    ramp_type = set_params["RT"]
            
            # some voltage data is negative. normalize to absolute value 
            if np.median(data["voltage"]) < 0:
                data["voltage"] = -data["voltage"]
            xs = data["voltage"]
            if np.median(data[curr_type]) < 0:
                data[curr_type] = -data[curr_type]
            ys = data[curr_type]
            ys_log10 = np.log10(ys)
            # find the first index after dep_v 
            no_available_voltage = False
            try:
                first_idx_after_dep_v = np.where(xs > dep_v)[0][0]
            except: # cannot find voltage after dep_v, either dep_v too high or voltage range too small 
                save_dir = f"{os.path.join(dir, scan_path.removesuffix(".txt").removesuffix(".iv"))}_breakdown_warn.png"
                print(f"Warning: Scan at {os.path.join(dir, scan_path)} voltage range too small. Should be ignored or try a different depletion voltage. IV scan plot generated at {save_dir}.")
                plt.figure(figsize=(10, 6))
                # a. plot the scan itself
                plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature), marker='o', markersize=3)
                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel(f"log(Pad Current (A))")
                plt.title(rf"IV Scan at {temperature}$^\circ$C: Unable to Find Breakdown V; Range too Small ({sensor.name} {date.strftime("%b %d, %Y")})")
                disable_top_and_right_bounds(plt)
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(save_dir)
                plt.close()
                continue 

            if sensor.is_ignored(os.path.join(dir, scan_path)): # if the iv scan is ignored
                ignored_scan_count += 1
                # data ignored, just plot the individual scan, then continue
                save_dir = f"{os.path.join(dir, scan_path.removesuffix(".txt").removesuffix(".iv"))}_breakdown_ignored.png"
                print(f"Ignoring {os.path.join(dir, scan_path)}. IV scan plot generated at {save_dir}.")
                plt.figure(figsize=(10, 6))

                plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature), marker='o', markersize=3)
                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel(f"log(Pad Current (A))")
                plt.title(rf"IV Scan at {temperature}$^\circ$C: Ignored ({sensor.name} {date.strftime("%b %d, %Y")})")
                disable_top_and_right_bounds(plt)
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(save_dir)
                plt.close()
                continue
            
            lines, std = find_breakdown(xs, ys_log10, start_idx=first_idx_after_dep_v, path=os.path.join(dir,scan_path), bd_thresh=bd_thresh, plot=True)
            # lines is [[slope, offset, bd_voltage], ...]
            # sorted by inlier_count (decreasing), then by RMSE (increasing)
            
            # if line is None, something's wrong! We just plot the scan itself, and warn the user 
            if lines is None:
                save_dir = f"{os.path.join(dir, scan_path.removesuffix(".txt").removesuffix(".iv"))}_breakdown_warn.png"
                print(f"Warning: Scan at {os.path.join(dir, scan_path)} deprecated. Should be ignored or try a different current type. IV scan plot generated at {save_dir}.")
                plt.figure(figsize=(10, 6))
                # a. plot the scan itself
                plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature), marker='o', markersize=3)
                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel(f"log(Pad Current (A))")
                plt.title(rf"IV Scan at {temperature}$^\circ$C: Unable to Find Breakdown V ({sensor.name} {date.strftime("%b %d, %Y")})")
                disable_top_and_right_bounds(plt)
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(save_dir)
                plt.close()
                continue 

            primary_line = lines[0]
            
            if plot: # plots individual iv scan with fitted lines
                plt.figure(figsize=(10, 6))
                # a. plot the scan itself
                plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature), marker='o', markersize=3)
                # b. plot 50 auxiliary lines
                for i in range(1, min(len(lines)-1, 51)): 
                    pred_y = lines[i][0] * xs[first_idx_after_dep_v:] + lines[i][1]
                    plt.plot(xs[first_idx_after_dep_v:], pred_y, color='grey', alpha=0.1, linestyle='--', label='Linear Fits' if i == 1 else None)
                # c. plot auxiliary points
                for i in range(1, min(len(lines)-1, 51)): 
                    plt.axvline(lines[i][2], color='grey', alpha=0.1, ls='-', label='Breakdown Points' if i == 1 else None)
                
                # d. plot main fitted line and thresholding line, and main breakdown point
                pred_y = primary_line[0] * xs[first_idx_after_dep_v:] + primary_line[1]
                plt.plot(xs[first_idx_after_dep_v:], pred_y + bd_thresh, color='brown', linestyle='--', label='Primary Threshold')
                plt.plot(xs[first_idx_after_dep_v:], pred_y, color='black', linestyle='--', label='Primary Linear Fit')
                plt.axvline(primary_line[2], color='black', ls='-', label=f"Primary Breakdown Point")
                
                
                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel("log(Pad Current (A))")
                plt.title(rf"IV Scan at {temperature}$^\circ$C: Breakdown at {primary_line[2]:.2f} +/- {std:.2f} V ({sensor.name} {date.strftime("%b %d, %Y")})")
                disable_top_and_right_bounds(plt)
                plt.legend()
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(f"{os.path.join(dir, scan_path.removesuffix(".txt").removesuffix(".iv"))}_breakdown.png")
                plt.close()
            all_data_in_dir.append([temperature, date, data, humidity, ramp_type, primary_line[2], std])

            # roll-back the overrides 
            if set_params is not None:
                if "DEP" in set_params:
                    dep_v = old_dep_v
                    
        if ignored_scan_count == tot_scan_count: # error check
            print(f"Warning: {sensor.name} all scans in {dir} are ignored.")
            continue 
         
        all_data_in_dirs.append(all_data_in_dir)
        
        # b. for each dir containing multiple scans, plot just all ramp up scans together, then all ramp down scans together
        if plot: 
            # partition the data
            all_data_in_dir_ramp_up = [d for d in all_data_in_dir if d[4] == 1]
            all_data_in_dir_ramp_down = [d for d in all_data_in_dir if d[4] == -1]
            all_data_in_dir_not_given = [d for d in all_data_in_dir if d[4] == 0]
            # sort by temperature (decreasing)
            all_data_in_dir_ramp_up = sorted(all_data_in_dir_ramp_up, key=lambda d: -d[0])
            all_data_in_dir_ramp_down = sorted(all_data_in_dir_ramp_down, key=lambda d: -d[0])
            all_data_in_dir_not_given = sorted(all_data_in_dir_not_given, key=lambda d: -d[0])
            
            curr_type_str = get_curr_type_label()

            if len(all_data_in_dir_ramp_up) > 1:
                plt.figure(figsize=(10, 6))
                seen_label = set()
                for temperature, date, data, humidity, ramp_type, bd_v, std in all_data_in_dir_ramp_up:
                    label = rf"{temperature:.1f}$^\circ$C"
                    if label in seen_label:
                        plt.plot(data["voltage"], data[curr_type], marker='o', markersize=3, color=temperature_to_color(temperature))
                    else:
                        seen_label.add(label)
                        plt.plot(data["voltage"], data[curr_type], marker='o', markersize=3, label=label, color=temperature_to_color(temperature))
                plt.xlabel("Voltage (V)")
                plt.ylabel("Current (A)")
                plt.yscale('log')
                plt.legend()
                plt.title(f"IV Scan of {sensor.name} on {date.strftime("%b %d, %Y")} ({curr_type_str}, Ramp Up)")
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(dir, f"{sensor.name}_rampup_{curr_type}_all.png"))
                plt.close()
                
            if len(all_data_in_dir_ramp_down) > 1:
                plt.figure(figsize=(10, 6))
                seen_label = set()
                for temperature, date, data, humidity, ramp_type, bd_v, std in all_data_in_dir_ramp_down:
                    label = rf"{temperature:.1f}$^\circ$C"
                    if label in seen_label:
                        plt.plot(data["voltage"], data[curr_type], marker='o', markersize=3, color=temperature_to_color(temperature))
                    else:
                        seen_label.add(label)
                        plt.plot(data["voltage"], data[curr_type], marker='o', markersize=3, label=label, color=temperature_to_color(temperature))
                plt.xlabel("Voltage (V)")
                plt.ylabel("Current (A)")
                plt.yscale('log')
                plt.legend()
                plt.title(f"IV Scan of {sensor.name} on {date.strftime("%b %d, %Y")} ({curr_type_str}, Ramp Down)")
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(dir, f"{sensor.name}_rampdown_{curr_type}_all.png"))
                plt.close()
                
            if len(all_data_in_dir_not_given) > 1:
                plt.figure(figsize=(10, 6))
                seen_label = set()
                for temperature, date, data, humidity, ramp_type, bd_v, std in all_data_in_dir_not_given:
                    label = rf"{temperature:.1f}$^\circ$C"
                    if label in seen_label:
                        plt.plot(data["voltage"], data[curr_type], marker='o', markersize=3, color=temperature_to_color(temperature))
                    else:
                        seen_label.add(label)
                        plt.plot(data["voltage"], data[curr_type], marker='o', markersize=3, label=label, color=temperature_to_color(temperature))
                plt.xlabel("Voltage (V)")
                plt.ylabel("Current (A)")
                plt.yscale('log')
                plt.legend()
                plt.title(f"IV Scan of {sensor.name} on {date.strftime("%b %d, %Y")} ({curr_type_str})")
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(dir, f"{sensor.name}_{curr_type}_all.png"))
                plt.close()

    # 3. if the sensor folder contains multiple scans, consolidate then plot 
    # 1) all bdv vs temperature 2) avg bdv vs temperature w/ uncertainty and fitted line
    # 3) bdv at different temperature across time (scan number)
    if plot: # 1) plot all bdv vs temperature
        plt.figure(figsize=(10, 6))
        for all_data_in_dir in all_data_in_dirs:
            # partition the data 
            ramp_up = [d for d in all_data_in_dir if d[4] == 1]
            ramp_down = [d for d in all_data_in_dir if d[4] == -1]
            not_given = [d for d in all_data_in_dir if d[4] == 0]
            # sort the data by temperature 
            ramp_up = sorted(ramp_up, key=lambda d:d[0])
            ramp_down = sorted(ramp_down, key=lambda d:d[0])
            not_given = sorted(not_given, key=lambda d:d[0])
            
            if len(ramp_up) > 0:
                plt.plot([d[0] for d in ramp_up], [d[5] for d in ramp_up], marker='o', markersize=3, label=f"{ramp_up[0][1].date()} (Ramp Up)") 
                # [0] -> temp, [5] -> bd_v, [1] -> date
                # all data in all_data_in_dir should have same date; date is at index 1, i.e. d[1]
            if len(ramp_down) > 0:
                plt.plot([d[0] for d in ramp_down], [d[5] for d in ramp_down], marker='o', markersize=3, label=f"{ramp_down[0][1].date()} (Ramp Down)") 
            if len(not_given) > 0:
                plt.plot([d[0] for d in not_given], [d[5] for d in not_given], marker='o', markersize=3, label=f"{not_given[0][1].date()}") 
        
        plt.xlabel("Temperature ($C$)")
        plt.ylabel("Breakdown Voltage ($V$)")
        plt.title(f"{sensor.name} Breakdown Voltage vs. Temperature")
        disable_top_and_right_bounds(plt)
        plt.legend()
        plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(sensor.sensor_dir, f"{sensor.name}_all.png"))
        plt.close()
            
    # 2) preparing plot avg bdv vs temperature w/ uncertainty and fitted line 
    temp_to_bdv_std_dict = dict() # a dict that maps: temp -> [[bdv, std, humi], ...]
    all_data_in_dirs_flat = [data_lst for sublist in all_data_in_dirs for data_lst in sublist] # flatten
    for data_lst in all_data_in_dirs_flat:
        temp, bdv, std = data_lst[0], data_lst[5], data_lst[6]
        humi = data_lst[3]
        if temp == float('inf'): # some .iv scans did not record temp and were 'inf'
            continue
        if temp not in temp_to_bdv_std_dict:
            temp_to_bdv_std_dict[temp] = list()
        temp_to_bdv_std_dict[temp].append([bdv, std, humi])
    
    if len(temp_to_bdv_std_dict) == 0:
        print(f"Warning: No available temperature and breakdown voltage data for plotting. Either all data are ignored, or all temperatures are inf (not given).")
        return 
    
    # 2) loops thru all temps, compute weighted average for each temp 
    temp_mean_sigma = list() # a list [[temp, weighted_mean, weighted_sigma, weighted_humi], ...]
    for temp in temp_to_bdv_std_dict:
        all_std = np.array([d[1] for d in temp_to_bdv_std_dict[temp]])
        all_bdv = np.array([d[0] for d in temp_to_bdv_std_dict[temp]])
        all_humi = np.array([d[2] for d in temp_to_bdv_std_dict[temp]])
        weighted_sigma = np.sqrt(1 / np.sum(1 / np.square(all_std)))
        weighted_mean = np.sum(all_bdv / np.square(all_std)) * np.square(weighted_sigma)
        weighted_humi = np.sum(all_humi / np.square(all_std)) * np.square(weighted_sigma) # use same weights as weighted_mean
        temp_mean_sigma.append([temp, weighted_mean, weighted_sigma, weighted_humi])
    
    # write info to sensor
    temp_mean_sigma = np.array(temp_mean_sigma)
    temp_mean_sigma = temp_mean_sigma[np.argsort(temp_mean_sigma[:, 0])]
    sensor.iv_scan_data = np.delete(temp_mean_sigma, 2, axis=1) # remove weighted_sigma
    
    temp_mean_sigma = np.array(temp_mean_sigma)
    avg_uncertainty = np.mean(temp_mean_sigma[:,2])
    
    slope_err = None
    # 2) now plot, x is temp, y is weighted_mean, error_bar is weighted_sigma, then fit a line thru
    if len(temp_mean_sigma) >= 2: # plot bdv vs temperature (average trend)
        popt, perr, _, r2 = linear_fit(temp_mean_sigma[:,0], temp_mean_sigma[:,1], [1,150], sigmas=temp_mean_sigma[:,2])
        slope_err = perr[0]
        
        # write fitted line to sensor 
        sensor.iv_scan_line = [popt[0], popt[1], np.mean(temp_mean_sigma[:, 2])] # [slope, offset, avg_sigma]
        if plot:
            plt.figure(figsize=(10, 6))
            plt.errorbar(temp_mean_sigma[:,0], temp_mean_sigma[:,1], yerr=temp_mean_sigma[:,2], fmt='o', capsize=5, label=f'{sensor.name} data')
            plt.plot(temp_mean_sigma[:,0], linear(temp_mean_sigma[:,0], popt[0], popt[1]), label=f'r2 value: {r2:0.2f} Best Fit: y = {popt[0]:.2f}x + { popt[1]:.2f}', linestyle='--')
            disable_top_and_right_bounds(plt)
            plt.xlabel("Temperature (C)")
            plt.ylabel("Breakdown Voltage (V)")
            plt.title(f"{sensor.name} Breakdown Voltage vs. Temperature")
            disable_top_and_right_bounds(plt)
            plt.legend()
            plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(sensor.sensor_dir, f"{sensor.name}_avg.png"))
            plt.close()
    else: # some only has one scan, plot no trend
        print(f"Warning: Insufficient scans to plot BDV vs. Temp Trend ({sensor.name}). Could not fit a line.")
        # write fitted line to sensor 
        sensor.iv_scan_line = None
        if plot:
            plt.figure(figsize=(10, 6))
            plt.errorbar(temp_mean_sigma[:,0], temp_mean_sigma[:,1], yerr=temp_mean_sigma[:,2], fmt='o', capsize=5, label=f'{sensor.name} data')
            disable_top_and_right_bounds(plt)
            plt.xlabel("Temperature (C)")
            plt.ylabel("Breakdown Voltage (V)")
            plt.title(f"{sensor.name} Breakdown Voltage vs. Temperature")
            disable_top_and_right_bounds(plt)
            plt.legend()
            plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(sensor.sensor_dir, f"{sensor.name}_avg.png"))
            plt.close()
    
    if plot: # 3) bdv at different temperature across time (scan number)
        plt.figure(figsize=(10, 6))
        for temp in sorted(temp_to_bdv_std_dict.keys(), key=lambda temp:-temp):
            scan_idx = np.arange(len(temp_to_bdv_std_dict[temp]))
            # if sensor.name == 'DC_W3058':
            #     # ignore missing data
            #     if temp == -20: scan_idx = np.array([0,1,2,3,4,6,7,8,9,10,11,12,13,14,15])
            #     elif temp == -40: scan_idx = np.array([0,2,4,8,9,10,11,12,13,14,15])
            #     elif temp == -60: scan_idx = np.array([0,8,9,10,11,12,13,14,15])
            #     elif temp == 80: scan_idx = np.array([0,1,2,3,5,6,7,8,9,10,11,12,13,14,15])
            #     elif temp == 40: scan_idx = np.array([0,1,2,3,4,5,6,7,9,10,11,12,13,14,15])
            plt.plot(scan_idx, [d[0] for d in temp_to_bdv_std_dict[temp]], marker='o', color=temperature_to_color(temp), label=rf"{temp}$^\circ$C")
        plt.xlabel("Scan Number")
        plt.ylabel("Breakdown Voltage (V)")
        plt.title(f"{sensor.name} Breakdown Voltage vs. Temperature over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(sensor.sensor_dir, f"{sensor.name}_bdv_trend.png"))
        plt.close()

    # return slope_err, avg_sigma

def plot_humidity_scans(data_dir, bd_thresh):
    curr_type = 'pad'
    sensor='AC'
    plt.figure(figsize=(10, 6))
    humidities = []
    filenames = []
    for filename in os.listdir(data_dir):
        if filename.endswith("p__21.txt") and filename.startswith("rh_"):
            filenames.append(filename)
            humidities.append(float(filename.split('_')[1][:-1]))
    leakage_140 = []
    for i in np.argsort(humidities):
        _, _, data = parse_file(os.path.join(data_dir, filenames[i]))
        neg_idx = data[curr_type] < 0
        voltages = abs(data['voltage'][neg_idx])
        log_curr = np.log10(-1*data[curr_type][neg_idx])
        plt.scatter(voltages, log_curr, color=humidity_to_color(humidities[i]), label=str(humidities[i])+' % rh', s=20)
        leakage_140.append(log_curr[np.argmin(abs(voltages-140))])
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("log( pad current (A) )")
    plt.title(sensor+"-LGAD IV Scan as Function of Relative Humidity at 21 C")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("data/humidity_"+sensor+".png")
    plt.close()
    # plot leakage current at 140V (approximate operating voltage)
    plt.figure(figsize=(10, 6))
    plt.plot(np.sort(humidities), leakage_140, marker='o')
    plt.xlabel("Relative Humidity (%)")
    plt.ylabel("log( pad current (A) ) @ 140V")
    plt.title(sensor+"-LGAD Leakage Current at 140V as Function of Relative Humidity at 21 C")
    plt.tight_layout()
    plt.savefig("data/humidity_leakage_"+sensor+".png")
    plt.close()

def find_threshold(sensor: Sensor, max_bd_thresh=1, min_temp=0):
    """
    Finds the bd_thresh value that minimizes slope uncertainty for a given sensor,
    the bd_thresh is used for IV scan analysis.
    """
    min_slope_err = float('inf')
    min_threshold = float('inf')
    for thresh in np.linspace(max_bd_thresh, 0, 10, endpoint=False):
        slope_err, avg_err = analyze_sensor_iv(sensor, min_temp=min_temp, bd_thresh=thresh, plot=False)
        if slope_err < min_slope_err:
            min_slope_err = slope_err
            min_threshold = thresh
    print(f"Optimal bd_threshold for {sensor.name} is: {min_threshold:.4f}")
    print(f"Achieving slope uncertainty of: {min_slope_err:.4f}")
    return min_threshold

def analyze_sensor_cv(sensor: Sensor, plot: bool=True):
    """
    Given a specific Sensor, ignores its iv scans, 
    analyzes and plots all its cv scans.
    
    Inputs:
    sensor - a Sensor object
    plot - whether generates and saves plots
    
    Returns:
    """
    for dir in tqdm(sensor.data_dirs, desc=f"Analyzing CV profiles for sensor {sensor.name:<20}"):
        cv_paths = [os.path.join(dir, p) for p in os.listdir(dir) if p.endswith(".cv")]
        
        def find_high_derivative(c_invsq_diff_data):
                """
                Finds possibly where the linear relation starts
                """
                return np.argmax(c_invsq_diff_data)

        total_scan_count = 0
        ignored_scan_count = 0
        for path in cv_paths:
            temperature, date, data, frequency = parse_file_cv(path)

            v_data = data['voltage']
            # if v are in negative direction, switch to positive direction 
            if np.mean(v_data) < 0:
                v_data = -1 * v_data
            v_diff_data = np.diff(v_data)
            c_data = data['capacitance']
            c_invsq_data = 1 / np.square(data['capacitance'])
            c_invsq_diff_data = np.diff(c_invsq_data)
            
            if sensor.is_ignored(os.path.join(path)):
                ignored_scan_count += 1
                # data ignored, just plot the individual scan, then continue
                save_dir = path.replace(".cv", "_cv_ignored.png")
                print(f"Ignoring {os.path.join(dir, path)}. CV scan plot generated at {save_dir}.")
                plt.figure(figsize=(10, 6))

                plt.plot(v_data, c_data, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature), marker='o', markersize=3)
                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel(f"log(Pad Current (A))")
                plt.title(rf"IV Scan at {temperature}$^\circ$C: Ignored ({sensor.name} {date.strftime("%b %d, %Y")})")
                disable_top_and_right_bounds(plt)
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(save_dir)
                plt.close()
                continue
            
            
            def fit_depletion():
                # take last few data points, calculate mean, 
                # then threshold at 90% mean
                all_dep_v = []
                for i in range(len(c_invsq_data) // 2):
                    plateau_v = np.mean(c_invsq_data[-i:])
                    cutoff_v = 0.9 * plateau_v
                    mask = c_invsq_data > cutoff_v
                    first_above_id = np.where(mask)[0][0]
                    last_below_id = first_above_id - 1 
                    
                    # linear interpolation
                    # assuming the last_below_id is not the last data point
                    slope_inter = (c_invsq_data[first_above_id] - c_invsq_data[last_below_id]) / (v_data[first_above_id] - v_data[last_below_id])
                    offset_inter = c_invsq_data[last_below_id] - slope_inter * v_data[last_below_id]
                    dep_v = (cutoff_v - offset_inter) / slope_inter
                    all_dep_v.append(dep_v)
                    
                # discard top and bottom 10% extremes
                all_dep_v = sorted(all_dep_v)
                n = len(all_dep_v)
                all_dep_v = all_dep_v[int(n*0.1):n-int(n*0.1)]
                
                all_dep_v = np.array(all_dep_v)
                dep_v = np.median(all_dep_v)
                std = np.std(all_dep_v)
                std = max(std, get_min_uncertainty(v_data, len(c_invsq_data) // 2))
                
                if plot: # plot depletion distribution
                    plt.figure(figsize=(8,6))
                    plt.hist(all_dep_v, bins=60, color='purple')
                    plt.axvline(dep_v, color="black", ls="--", label=f"Median: {dep_v:.2f} $V$")
                    plt.legend()
                    plt.ylabel('Frequency')
                    plt.xlabel('Depletion Voltage ($V$)')
                    plt.title(f'Depletion Voltage Distribution')
                    plt.tight_layout()
                    plt.savefig(path.replace(".cv", "_dep_distribution.png"))
                    plt.close()
                
                return dep_v, std, all_dep_v
            
            dep_v, std, all_dep_v= fit_depletion()

            def find_depleted_c():
                # computes the mean capacitance after dep_v
                start_idx = np.where(v_data >= dep_v)[0][0]
                mean_c = np.mean(c_data[start_idx:])
                return mean_c 
            
            # writes info to sensor
            sensor.depletion_v = dep_v
            sensor.cv_scan_data.append([find_depleted_c(), frequency, temperature])
            
            def plot_median_and_distribution_v_lines():
                d_label = "Depletion Distribution"
                for v in all_dep_v:
                    plt.axvline(v, color='black', alpha=0.15, linewidth=0.3, label=d_label)
                    d_label = None
                plt.axvline(dep_v, color='black', label=rf'Median at {dep_v:.2f} $\pm$ {std:.2f} $V$')
                return 
            
            if plot: # C vs V
                plt.figure(figsize=(10, 6))
                plt.plot(v_data, c_data, label=rf"{temperature}$^\circ$C {frequency}Hz", color=temperature_to_color(temperature), marker='o', markersize=2)
                
                disable_top_and_right_bounds(plt)
                plot_median_and_distribution_v_lines()
                
                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel("Capacitance (F)")
                plt.title(rf"Capacitance vs. Voltage at {temperature}$^\circ$C {frequency}Hz ({sensor.name})")
                plt.legend()
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(path.replace(".cv", "_cv.png"))
                plt.close()
            if plot: # C^-2 vs V
                plt.figure(figsize=(10, 6))
                plt.plot(v_data, c_invsq_data, label=rf"{temperature}$^\circ$C {frequency}Hz", color=temperature_to_color(temperature), marker='o', markersize=2)
                
                disable_top_and_right_bounds(plt)
                plot_median_and_distribution_v_lines()
                
                plt.xlabel("Reverse-bias Voltage ($V$)")
                plt.ylabel("Inverse-squared Capacitance ($F^{-2}$)")
                plt.title(rf"Inverse-squared Capacitance vs. Voltage at {temperature}$^\circ$C {frequency}Hz ({sensor.name})")
                plt.legend()
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(path.replace(".cv", "_c-2v.png"))
                plt.close()
            if plot: # Derivative of C^-2 w.r.t V, d(C^-2)/dV
                plt.figure(figsize=(10, 6))
                plt.plot(v_data[:-1], c_invsq_diff_data / v_diff_data, label=rf"{temperature}$^\circ$C {frequency}Hz", color=temperature_to_color(temperature), marker='o', markersize=2)
                
                disable_top_and_right_bounds(plt)
                plot_median_and_distribution_v_lines()
                
                plt.xlabel("Reverse-bias Voltage ($V$)")
                plt.ylabel("$d C^{-2}/dV$ ($F^{-2}/V$)")
                plt.title(rf"$d C^{-2}/dV$ vs. $V$ at {temperature}$^\circ$C {frequency}Hz ({sensor.name})")
                plt.legend()
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(path.replace(".cv", "_c-2v_derivative.png"))
                plt.close()
                
    
def main():
    if ARGS.clear_plots:
        clear_plots()
    
    # loads all sensors and configs. 
    # Must load all sensors at once to properly write out to config at the end
    sensors = list_sensors()
    load_sensor_config(DATABASE_DIR, sensors, load_iv=(not ARGS.iv or (ARGS.iv and not ARGS.overwrite)), load_cv=(not ARGS.cv or (ARGS.cv and not ARGS.overwrite)))
    load_data_config(DATABASE_DIR, sensors)
    
    if ARGS.sensor is None:
        interested_sensor_names = set([sensor.name for sensor in sensors])
    else:
        # check if all sensor names are valid 
        all_names = set([sensor.name for sensor in sensors])
        interested_sensor_names = set(ARGS.sensor)
        if not interested_sensor_names.issubset(all_names):
            raise ValueError("Invalid sensor name(s).")
    
    # thresholds = {"AC_W3096": 0.3, "DC_W3058": 0.4, "DC_W3045": 0.4,
    #               "BNL_LGAD_513": 0.6, 
    #               "BNL_LGAD_W3076_9_13": 0.6,
    #               "BNL_LGAD_W3076_12_13": 0.5}
    
    # plot_humidity_scans("data/AC_W3096/Dec102024", thresholds["AC_W3096"])
    
    if ARGS.cv:
        for sensor in sensors:
            if sensor.name in interested_sensor_names:
                analyze_sensor_cv(sensor, plot=True)
                
    if ARGS.iv:
        for sensor in sensors:
            if sensor.name in interested_sensor_names:
                analyze_sensor_iv(sensor, ARGS.curr_type, plot=True)
    
    write_sensor_config(DATABASE_DIR, sensors)
    
    return 0

if __name__ == "__main__":
    ARGS = parse_args()
    main()