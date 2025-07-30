import os
import argparse
import math
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=plt.cm.tab20.colors)
# use 20 color cycle instead of the default 10 cycle

from utils import (
    Sensor, DATABASE_DIR, list_sensors, parse_file_cv, parse_file_cv_basic_info, 
    parse_file_iv_basic_info, parse_file_iv, temperature_to_color, 
    load_data_config, load_sensor_config, determine_spacing, linear, 
    linear_fit, disable_top_and_right_bounds, write_sensor_config, clear_plots, 
    calculate_weighted_mean
)

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
        '--file', '--files', '-f',
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
        
def get_min_uncertainty(sequence: np.ndarray, range: int=None):
    """
    Returns the minimum uncertainty of any estimation from a sequence of 
    data due to resolution limit.
    
    Parameters
    ----------
    sequence : 1D-array
        The sequence of data being analyzed (and thus has a minimum uncertainty).
        Assumed to be (roughly) an Arithmetic Sequence.
    range : int, optional
        The number of data points used or involved with the analysis. If range 
        is not given, defaults to length of the sequence
    
    Returns
    -------
    sequence common difference / sqrt(range)
    """
    
    diff = determine_spacing(sequence)
    if range is None:
        range = len(sequence)
    return diff / np.sqrt(range)
    
def ransac(xs: np.ndarray, ys: np.ndarray, thresh: float=0.5, iter: int=200):
    """
    Use a modified RANSAC to fit a line for given x and y data.
    In each iteration, randomly select 33% data pts, fit a line,
    and determine the number of inliers. Lines with more inliers are more 
    likely to be the true trend. Ignores data that are NaNs or Infs in the y 
    coordinate.
    
    Parameters
    ----------
    xs : 1D-array
        x data
    ys : 1D-array
        y data
    thresh : float
        Max y-distance to be considered as an inlier. Defaults to 0.5
    iter : int, optional 
        Number of RANSAC iterations. Defaults to 200.
    
    Returns
    -------
    lines : List[List[float]]
        A list of lists [[slope, offset, inlier count, mse], ...], each 
        sublist represents a trend line proposed by RANSAC
    
    Notes
    -----
    The output is not sorted in anyway.
    """
    assert xs.shape == ys.shape
    
    nan_mask = np.isnan(ys) | np.isinf(ys)
    xs = xs[~nan_mask]
    ys = ys[~nan_mask]
    assert len(xs) >= 2
    
    lines = [] # each entry is a list [slope, offset, inlier count, mse, predicted_ys]
    seen = set() # some dot pairs may have been selected. dedup
    index_list = range(len(xs))
    for _ in range(iter):
        indices = random.sample(index_list, max(2, len(xs)//3))
        popt, _, _, _ = linear_fit(xs[indices], ys[indices])
        slope, offset = popt
        
        # indices = random.sample(index_list, 2)
        # if (indices[0], indices[1]) in seen:
        #     continue
        # else:
        #     seen.add((indices[0], indices[1]))
        # slope = (ys[indices[1]] - ys[indices[0]]) / (xs[indices[1]] - xs[indices[0]])
        # offset = ys[indices[0]] - slope * xs[indices[0]]
        
        pred_ys = xs * slope + offset # the y's for fitted line
        diff = np.abs(pred_ys - ys)
        is_close_map = diff < thresh 
        inlier_count = np.sum(is_close_map)
        
        # some data might be nan due to np.log10, so use np.nanmean to find mse
        mse = np.nanmean(np.square(diff))
        
        lines.append([slope, offset, inlier_count, mse])
    
    return lines

def find_breakdown(xs: np.ndarray, ys: np.ndarray, start_idx: int, path: str, bd_thresh: float, dp_thresh: float, plot: bool=True, relative_dp_thresh: bool=False):
    """
    Given a single IV scan, finds the breakdown and depletion voltage by fitting 
    linear lines. Ignores data that are NaNs or Infs in ys.
    

    Parameters
    ----------
    xs : 1D-array
        Voltage data.
    ys : 1D-array
        Current data.
    start_idx : int 
        Index before which data will be ignored when fitting the line.
    path : str 
        The path to this particular scan.
    bd_thresh : float 
        Absolute value of threshold for determining breakdown. (+)
    dp_thresh : float 
        Absolute value of threshold for determining depletion. (+/-)
    plot : bool, optional 
        Whether to generate and save breakdown distribution plots. Defaults to True.
    relative_dp_thresh : bool, optional
        Whether to automatically determine dp_thresh based on how noisy the data is. Defaults to False.
    
    Returns
    -------
    lines : List[List[float]]
        A list of lines in the form [[slope, offset, bd_voltage], ...], each 
        sublist represents a trend line and its corresponding predicted 
        breakdown voltage.
    std : float 
        Uncertainty of estimation.
    """
    # truncate data 
    valid_xs = xs[start_idx:]
    valid_ys = ys[start_idx:]
    
    nan_mask = np.isnan(ys) | np.isinf(ys)
    xs = xs[~nan_mask]
    ys = ys[~nan_mask]
    nan_mask = np.isnan(valid_ys) | np.isinf(valid_ys)
    valid_xs = valid_xs[~nan_mask]
    valid_ys = valid_ys[~nan_mask]
    assert valid_xs.shape == valid_ys.shape
    
    fit_range = len(valid_xs) // 2
    raw_lines = ransac(valid_xs[:fit_range], valid_ys[:fit_range], 0.15, 200)
    lines = []
    for line in raw_lines:
        slope, offset = line[0], line[1]
        pred_ys = slope * valid_xs + offset
        
        # find the breakdown voltage with linear interpolation
        last_id_in_thresh = -1 # some lines are so ridiculous that we cant find last_id_in_thresh
        for i in range(len(valid_ys)):
            if abs(valid_ys[i] - pred_ys[i]) < bd_thresh:
                last_id_in_thresh = i
        if last_id_in_thresh == -1:
            continue # no pts is inside bd_thresh
        if last_id_in_thresh + 1 == len(valid_ys):
            # last data point! just use it as is
            bd_voltage = valid_xs[last_id_in_thresh]
        else: # find the next data point, fit a line, solve for intersection
            slope_inter = (valid_ys[last_id_in_thresh+1] - valid_ys[last_id_in_thresh]) / (valid_xs[last_id_in_thresh+1] - valid_xs[last_id_in_thresh])
            offset_inter = valid_ys[last_id_in_thresh] - slope_inter * valid_xs[last_id_in_thresh]
            bd_voltage = (offset_inter - (offset + bd_thresh)) / (slope - slope_inter)
        
        # calculate rmse of the line in the fitting range 
        if relative_dp_thresh:
            rmse_in_fit_range = np.sqrt(np.mean(np.square(slope * valid_xs[:fit_range] + offset - valid_ys[:fit_range])))
            dp_thresh = 5*rmse_in_fit_range # 5 sigma
        # find the depletion voltage with linear interpolation
        pred_ys = slope * xs + offset # we use full range of xs 
        first_id_in = -1

        # Going from right to left
        # if for 3 consecutive pts there are less than 2 inliers,
        # the end idx of the first such 3 pts is dep_v
        # Linear time algorithm using prefix sum
        n_inlier_upto_me = np.zeros((len(ys)//2,), dtype=int)
        n_inlier_upto_me[0] = int(pred_ys[0] - dp_thresh <= ys[0] <= pred_ys[0] + dp_thresh)
        for i in range(1, len(n_inlier_upto_me)):
            n_inlier_upto_me[i] = n_inlier_upto_me[i-1] + int(pred_ys[i] - dp_thresh <= ys[i] <= pred_ys[i] + dp_thresh)
        for i in range(len(n_inlier_upto_me)-1, 4, -1):
            if n_inlier_upto_me[i] - n_inlier_upto_me[i-4] <= 2:
                first_id_in = i
                break
        # print(first_id_in, n_inlier_upto_me, dp_thresh)
        
        # # Going from right to left,
        # # the first pt at which there no 2 consecutive inliers 
        # # is dep_v
        # for i in range(len(ys) // 2, 1, -1):
        #     if abs(pred_ys[i] - ys[i]) > dp_thresh:
        #         if abs(pred_ys[i-1] - ys[i-1]) > dp_thresh:
        #             first_id_in = i+1
        #             break
        if first_id_in == -1:
            print(dp_thresh, 11)
            continue # no satisfying pt
        if first_id_in + 1 == len(xs):
            # last data point! just use it as is 
            dp_voltage = xs[first_id_in]
        else:
            slope_inter = (ys[first_id_in+1] - ys[first_id_in]) / (xs[first_id_in+1] - xs[first_id_in])
            offset_inter = ys[first_id_in] - slope_inter * xs[first_id_in]
            if ys[first_id_in-1] > ys[first_id_in]:
                # use + dp_thresh
                dp_voltage = (offset_inter - (offset + dp_thresh)) / (slope - slope_inter)
            else: # use - dp_thresh
                dp_voltage = (offset_inter - (offset - dp_thresh)) / (slope - slope_inter)
            dp_voltage = xs[first_id_in]
        line.append(bd_voltage)
        line.append(dp_voltage)
        lines.append(line) # add the line from raw_lines to lines

    if len(lines) == 0:
        # There's something wrong with the data!
        # We can't fit anything! 
        return None, -1, -1
    # now each sublist has structure [slope, offset, inlier_count, mse, bd_voltage, dp_voltage]
    lines = np.array(lines)
    # sanity check, breakdown voltage must > 0
    mask = lines[:,4] > 0
    # sanity check #2, the fitted line must have slope > 0
    mask &= lines[:,0] > 0
    # sanity check #3, depletion voltage must > 0
    mask &= lines[:,5] > 0
    # sanity check #4, depletion voltage must < breakdown voltage 
    mask &= lines[:,5] < lines[:,4]
    lines = lines[mask]
    if len(lines) == 0:
        # There's something wrong with the data!
        # We can't fit anything! 
        return None, -1, -1
    
    # sort according to bd_voltage
    lines = lines[np.argsort(lines[:,4])]
    # cutoff extreme values for bd_voltage
    cum_weights = np.cumsum(lines[:,2])
    total_weight = cum_weights[-1]
    trim_ratio = 0.10
    lower = trim_ratio * total_weight
    upper = (1 - trim_ratio) * total_weight
    mask = (cum_weights >= lower) & (cum_weights <= upper)
    # sort according to dp_voltage
    lines = lines[np.argsort(lines[:,5])]
    # cutoff extreme values for dp_voltage 
    cum_weights = np.cumsum(lines[:,2])
    mask &= (cum_weights >= lower) & (cum_weights <= upper)
    
    # group line inliers and outliers
    filtered_lines = lines[mask]
    lines = lines[~mask]
    assert len(filtered_lines) != 0
    # calculate uncertainty for bd_voltage and dp_voltage
    def find_frequency_weighted_uncertainty(filtered_lines, col):
        # col is 4 for breakdown, 5 for depletion
        try: # the uncertainty is weighted by frequency in each bin, bin width = 1V
            frequencies = np.zeros_like(filtered_lines[:,col])
            min_bin_left_edge = math.floor(np.min(filtered_lines[:,col]))
            max_bin_left_edge = math.floor(np.max(filtered_lines[:,col]))
            for volt in range(min_bin_left_edge, max_bin_left_edge+1):
                lines_in_bin_mask = (filtered_lines[:,col] >= volt) & (filtered_lines[:,col] < volt+1)
                count = np.sum(lines_in_bin_mask)
                frequencies[lines_in_bin_mask] = count
                
            mean = np.average(filtered_lines[:,[4,5]], axis=0, weights=frequencies)
            variance = np.average((filtered_lines[:,[4,5]] - mean)**2, axis=0, weights=frequencies)
            std = max(np.sqrt(variance[0]), get_min_uncertainty(valid_xs, fit_range))
            return std
        except:
            # weights sum to zero, can't be normalized
            return None 
    
    bd_std = find_frequency_weighted_uncertainty(filtered_lines, 4)
    dp_std = find_frequency_weighted_uncertainty(filtered_lines, 5)
    
    # sort against inlier counts so the line with most inliers is at index 0
    # then sort against mse (if inlier counts are same, use one with smaller mse)
    filtered_lines = filtered_lines[np.lexsort((filtered_lines[:, 3], -filtered_lines[:, 2]))]

    # plot breakdown distribution, frequency and max weight
    def plot_entry(col, str, std):
        # col is 4 for breakdown, 5 for depletion
        # str is "Breakdown" or "Depletion"
        # std is the frequency weighted uncertainty to be included on graph
        save_str = "bdv" if str == "Breakdown" else "dpv"
        # separate into bins
        bins = np.histogram_bin_edges(lines[:, col], bins=160)
        bin_idx = np.digitize(lines[:,col], bins) - 1
        bin_idx2 = np.digitize(filtered_lines[:,col], bins) - 1
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
        
        def plot_rmse(ax1):
            ax2 = ax1.twinx()
            ax2.plot(bin_centers, bin_mrmse, label='Outlier RMSE', color='black', alpha=0.4)
            ax2.plot(bin_centers, bin_mrmse2, label='RMSE', color='purple', alpha=1)
            ax2.set_ylabel('RMSE', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            return ax2
        def setup_legend(ax1, ax2):
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
            
        if plot: # plot distribution (frequency) and rmse
            plt.figure(figsize=(8,6))
            ax1 = plt.gca()
            ax1.bar(bin_centers, bin_tic, width=np.diff(bins), color='black', alpha=0.3, label=f'Outliers')
            ax1.bar(bin_centers, bin_tic2, width=np.diff(bins), color='purple', alpha=0.7)
            ax1.set_ylabel('Frequency')
            ax1.set_xlabel(f'{str} Voltage ($V$)')
            
            ax2 = plot_rmse(ax1)
            ax1.axvline(filtered_lines[0][col], color="black", ls="--", label=f"Max Weight Min RMSE Choice: {filtered_lines[0][col]:.2f} $V$")
            
            setup_legend(ax1, ax2)
            plt.title(f'{str} Voltage Frequency Distribution (Best: {filtered_lines[0][col]:.2f} +/- {std:.2f} V)')
            plt.tight_layout()
            plt.savefig(f"{path.removesuffix('.txt').removesuffix('.iv')}_{save_str}_freq.png")
            plt.close()
        if plot: # plot distribution (max weight) and rmse
            plt.figure(figsize=(8,6))
            ax1 = plt.gca()
            ax1.bar(bin_centers, bin_mic, width=np.diff(bins), color='black', alpha=0.3, label=f'Outliers')
            ax1.bar(bin_centers, bin_mic2, width=np.diff(bins), color='purple', alpha=0.7)
            ax1.set_ylabel('Max Weights')
            ax1.set_xlabel(f'{str} Voltage ($V$)')
            
            ax2 = plot_rmse(ax1)
            
            ax1.axvline(filtered_lines[0][col], color="black", ls="--", label=f"Max Weight Min RMSE Choice: {filtered_lines[0][col]:.2f} $V$")
                
            setup_legend(ax1, ax2)
            plt.title(f'{str} Voltage Max Weight Distribution (Best: {filtered_lines[0][col]:.2f} +/- {std:.2f} V)')
            plt.tight_layout()
            plt.savefig(f"{path.removesuffix('.txt').removesuffix('.iv')}_{save_str}_max.png")
            plt.close()
        
    plot_entry(4, "Breakdown", bd_std)
    plot_entry(5, "Depletion", dp_std)
    
    lines = filtered_lines
    lines = np.delete(lines, [2, 3], axis=1) 
    # now each line has structure [slope, offset, bd_voltage, dp_voltage]

    return lines, bd_std, dp_std
    
def analyze_sensor_iv(sensor: Sensor, curr_type: str='pad', plot=True):
    """
    Analyze all IV scans of a specific sensor. 
    
    Parameters
    ----------
    sensor : Sensor 
        The interestedt sensor.
    curr_type : str {'pad', 'gr', 'total'}
        Specify the current type for analysis. Some scan may not support all the 
        types.
    plot : bool 
        Whether generate and save the plots. If set to True, a plot of the IV 
        scans with analysis, plots of distributions of estimated parameters, 
        plots of IV scan ramps in each subdirectory, 
        a plot of breakdown voltage vs temperature, 
        a plot of average breakdown voltage vs temperature, 
        and a plot of breakdown voltage vs scan number 
        will be generated, either alongside the scans or in the sensor's 
        primary directory.

        If the something went wrong during analysis, a 
        warning plot is generated regardless what value plot is.
    """
    # 1. sets up related constants for the sensor:
    # a. use the specified threshold for breakdown fits
    bd_thresh = sensor.bd_thresh 
    if bd_thresh is None:
        bd_thresh = 0.5 # defaults to 0.5
    # b. data before dep_v ignored when fitting (only when fitting)
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
            
            temperature, date, data, humidity, ramp_type, duration = parse_file_iv(os.path.join(dir, scan_path))
            
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
            with np.errstate(divide='ignore', invalid='ignore'): # suppress invalid value and division by zero err
                ys_log10 = np.log10(ys) # we will remove nan and inf after
            # find the first index after dep_v 
            try:
                first_idx_after_dep_v = np.where(xs > dep_v)[0][0]
            except: # cannot find voltage after dep_v, either dep_v too high or voltage range too small 
                save_dir = f"{os.path.join(dir, scan_path.removesuffix('.txt').removesuffix('.iv'))}_ivscan_warn.png"
                print(f"Warning: Scan at {os.path.join(dir, scan_path)} voltage range too small. Should be ignored or try a different depletion voltage. IV scan plot generated at {save_dir}.")
                plt.figure(figsize=(10, 6))
                # a. plot the scan itself
                plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature), marker='o', markersize=3)
                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel(f"log(Pad Current (A))")
                plt.title(rf"IV Scan at {temperature}$^\circ$C: Unable to Find Breakdown V; Range too Small ({sensor.name} {date.strftime('%b %d, %Y')})")
                disable_top_and_right_bounds(plt)
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(save_dir)
                plt.close()
                continue 

            if sensor.is_ignored(os.path.join(dir, scan_path)) and plot: # if the iv scan is ignored
                ignored_scan_count += 1
                # data ignored, just plot the individual scan, then continue
                save_dir = f"{os.path.join(dir, scan_path.removesuffix('.txt').removesuffix('.iv'))}_ivscan_ignored.png"
                print(f"Ignoring {os.path.join(dir, scan_path)}. IV scan plot generated at {save_dir}.")
                plt.figure(figsize=(10, 6))

                plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature), marker='o', markersize=3)
                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel(f"log(Pad Current (A))")
                plt.title(rf"IV Scan at {temperature}$^\circ$C: Ignored ({sensor.name} {date.strftime('%b %d, %Y')})")
                disable_top_and_right_bounds(plt)
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(save_dir)
                plt.close()
                continue

            lines, bd_std, dp_std = find_breakdown(xs, ys_log10, start_idx=first_idx_after_dep_v, path=os.path.join(dir,scan_path), bd_thresh=bd_thresh, dp_thresh=0.1, plot=plot, relative_dp_thresh=True)
            # lines is [[slope, offset, bd_voltage, dp_voltage], ...]
            # sorted by inlier_count (decreasing), then by RMSE (increasing)
            
            # if line is None, something's wrong! We just plot the scan itself, and warn the user 
            if lines is None:
                save_dir = f"{os.path.join(dir, scan_path.removesuffix('.txt').removesuffix('.iv'))}_ivscan_warn.png"
                print(f"Warning: Scan at {os.path.join(dir, scan_path)} deprecated. Should be ignored or try a different current type. IV scan plot generated at {save_dir}.")
                plt.figure(figsize=(10, 6))
                # a. plot the scan itself
                plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature), marker='o', markersize=3)
                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel(f"log(Pad Current (A))")
                plt.title(rf"IV Scan at {temperature}$^\circ$C: Unable to Find Breakdown V ({sensor.name} {date.strftime('%b %d, %Y')})")
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
                    pred_y = lines[i][0] * xs + lines[i][1]
                    plt.plot(xs, pred_y, color='grey', alpha=0.1, linestyle='--', label='Linear Fits' if i == 1 else None)
                # c. plot auxiliary points
                for i in range(1, min(len(lines)-1, 51)): 
                    plt.axvline(lines[i][2], color='grey', alpha=0.1, ls='-', label='Breakdown Points' if i == 1 else None)
                    plt.axvline(lines[i][3], color='grey', alpha=0.1, ls='-', label='Depletion Points' if i == 1 else None)
                
                # d. plot main fitted line and thresholding line, and main breakdown point
                pred_y = primary_line[0] * xs + primary_line[1]
                plt.plot(xs, pred_y + bd_thresh, color='brown', linestyle='--', label='Primary Breakdown Threshold')
                plt.plot(xs, pred_y, color='black', linestyle='--', label='Primary Linear Fit')
                plt.axvline(primary_line[2], color='black', ls='-', label=f"Primary Breakdown Point")
                plt.axvline(primary_line[3], color='black', ls='-', label=f"Primary Depletion Point")
                
                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel("log(Pad Current (A))")
                plt.title(rf"IV Scan at {temperature}$^\circ$C: Breakdown {primary_line[2]:.2f} +/- {bd_std:.2f} V, Depletion {primary_line[3]:.2f} +/- {dp_std:.2f} V ({sensor.name} {date.strftime('%b %d, %Y')})")
                disable_top_and_right_bounds(plt)
                plt.legend()
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(f"{os.path.join(dir, scan_path.removesuffix('.txt').removesuffix('.iv'))}_ivscan.png")
                plt.close()
            all_data_in_dir.append([temperature, date, data, humidity, ramp_type, primary_line[2], bd_std])

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
            all_ramp_types_here = set(d[4] for d in all_data_in_dir)
            partition = []
            for ramp_type in all_ramp_types_here:
                partition.append([d for d in all_data_in_dir if d[4] == ramp_type])
            # sort by temperature (decreasing)
            for i in range(len(partition)):
                partition[i] = sorted(partition[i], key=lambda d: -d[0])
            
            curr_type_str = get_curr_type_label()
            ramp_title_str = dict({0:"", 1:", Ramp Up", -1:", Ramp Down"})
            ramp_save_str = dict({0:"", 1:"rampup", -1:"rampdown"})
            
            for all_data_in_dir_ramp_type in partition:
                if len(all_data_in_dir_ramp_type) > 1:
                    plt.figure(figsize=(10, 6))
                    seen_label = set()
                    for temperature, date, data, humidity, ramp_type, bd_v, std in all_data_in_dir_ramp_type:
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
                    plt.title(f"IV Scan of {sensor.name} on {date.strftime('%b %d, %Y')} ({curr_type_str}{ramp_title_str[ramp_type]})")
                    plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(dir, f"{sensor.name}_{ramp_save_str[ramp_type]}_{curr_type}_all.png"))
                    plt.close()

    # 3. if the sensor folder contains multiple scans, consolidate then plot 
    # 1) all bdv vs temperature 2) avg bdv vs temperature w/ uncertainty and fitted line
    # 3) bdv at different temperature across time (scan number)
    if plot: # 1) plot all bdv vs temperature
        plt.figure(figsize=(10, 6))
        for all_data_in_dir in all_data_in_dirs:
            # partition the data 
            all_ramp_types_here = set(d[4] for d in all_data_in_dir)
            partition = []
            for ramp_type in all_ramp_types_here:
                partition.append([d for d in all_data_in_dir if d[4] == ramp_type])
            # sort the data by temperature (increasing)
            for i in range(len(partition)):
                partition[i] = sorted(partition[i], key=lambda d: d[0])
            ramp_title_str = dict({0:"", 1:" (Ramp Up)", -1:" (Ramp Down)"})
            for ramp in partition:
                if len(ramp) > 0:
                    plt.plot([d[0] for d in ramp], [d[5] for d in ramp], marker='o', markersize=3, label=f"{ramp[0][1].date()}{ramp_title_str[ramp[0][4]]}") 
                    # [0] -> temp, [5] -> bd_v, [1] -> date
                    # all data in the same all_data_in_dir should have same date; date is at index 1, i.e. d[1]
            
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
        
        weighted_mean, weighted_sigma = calculate_weighted_mean(all_bdv, all_std)
        weighted_humi, _ = calculate_weighted_mean(all_humi, all_std)  # use same weights as weighted_mean

        temp_mean_sigma.append([temp, weighted_mean, weighted_sigma, weighted_humi])
    
    # write info to sensor
    temp_mean_sigma = np.array(temp_mean_sigma)
    temp_mean_sigma = temp_mean_sigma[np.argsort(temp_mean_sigma[:, 0])]
    sensor.iv_scan_data = np.delete(temp_mean_sigma, 2, axis=1) # remove weighted_sigma
    
    temp_mean_sigma = np.array(temp_mean_sigma)
    avg_uncertainty = np.mean(temp_mean_sigma[:,2])
    
    slope_err = None
    # 2) now plot, x is temp, y is weighted_mean, error_bar is weighted_sigma, then fit a line thru
    if plot:
        plt.figure(figsize=(10, 6))
        plt.errorbar(temp_mean_sigma[:,0], temp_mean_sigma[:,1], yerr=temp_mean_sigma[:,2], fmt='o', capsize=5, label=f'{sensor.name} data')
        if len(temp_mean_sigma) >= 2: # plot bdv vs temperature (average trend)
            popt, perr, _, r2 = linear_fit(temp_mean_sigma[:,0], temp_mean_sigma[:,1], [1,150], sigmas=temp_mean_sigma[:,2])
            slope_err = perr[0]
            # write fitted line to sensor 
            sensor.iv_scan_line = [popt[0], popt[1], np.mean(temp_mean_sigma[:, 2])] # [slope, offset, avg_sigma]
            plt.plot(temp_mean_sigma[:,0], linear(temp_mean_sigma[:,0], popt[0], popt[1]), label=f'r2 value: {r2:0.2f} Best Fit: y = {popt[0]:.2f}x + { popt[1]:.2f}', linestyle='--')
        else: # some only has one scan, cannot fit a line
            print(f"Warning: Insufficient scans to plot BDV vs. Temp Trend ({sensor.name}). Could not fit a line.")
            sensor.iv_scan_line = None
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
            plt.plot(scan_idx, [d[0] for d in temp_to_bdv_std_dict[temp]], marker='o', color=temperature_to_color(temp), label=rf"{temp}$^\circ$C")
        plt.xlabel("Scan Number")
        plt.title(f"{sensor.name} Breakdown Voltage vs. Temperature over Time")
        plt.ylabel("Breakdown Voltage (V)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(sensor.sensor_dir, f"{sensor.name}_bdv_trend.png"))
        plt.close()

    # return slope_err, avg_sigma

def analyze_file_iv(sensor: Sensor, path: str, curr_type: str='pad', plot=True):
    """
    Analyze a specific IV scan file. 
    
    Parameters
    ----------
    sensor : Sensor 
        The sensor associated with this scan.
    path : str
        The relative path to this scan.
    curr_type : str {'pad', 'gr', 'total'}
        Specify the current type for analysis. Some scan may not support all the 
        types.
    plot : bool 
        Whether generate and save the plots. If set to True, a plot of the IV 
        scan with analysis is generated to where the same directory the 
        scan is located. If the something went wrong during analysis, a 
        warning plot is generated regardless what value plot is.
    """
    
    # 1. sets up related constants for the sensor:
    # a. use the specified threshold for breakdown fits
    bd_thresh = sensor.bd_thresh 
    if bd_thresh is None:
        bd_thresh = 0.5 # defaults to 0.5
    # b. data before dep_v ignored when fitting (only when fitting)
    dep_v = sensor.depletion_v
    if dep_v is None:
        dep_v = 25 # if not set, defaults to 25V
        
    def get_curr_type_label():
        if curr_type == 'pad': return "Pad"
        elif curr_type == 'gr': return "Guard Ring"
        elif curr_type == 'total': return "Total Current"
        else: raise ValueError(f"Invalid Current Type: {curr_type}. Must be one of 'pad', 'gr', or 'total'.")

    if path.split(os.sep)[-1].startswith("."): return # ignore hidden files 
    if not (path.endswith(".iv") or path.endswith(".txt")): return # ignore non .txt non .iv 

    set_params = sensor.query_conf(path)
    
    temperature, date, data, humidity, ramp_type, duration = parse_file_iv(path)
    
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
    with np.errstate(divide='ignore', invalid='ignore'): # suppress invalid value and division by zero err
        ys_log10 = np.log10(ys) # we will remove nan and inf after
    # find the first index after dep_v 
    no_available_voltage = False
    try:
        first_idx_after_dep_v = np.where(xs > dep_v)[0][0]
    except: # cannot find voltage after dep_v, either dep_v too high or voltage range too small 
        save_dir = f"{path.removesuffix('.txt').removesuffix('.iv')}_ivscan_warn.png"
        print(f"Warning: Scan at {path} voltage range too small. Should be ignored or try a different depletion voltage. IV scan plot generated at {save_dir}.")
        plt.figure(figsize=(10, 6))
        # a. plot the scan itself
        plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature), marker='o', markersize=3)
        plt.xlabel("Reverse-bias Voltage (V)")
        plt.ylabel(f"log(Pad Current (A))")
        plt.title(rf"IV Scan at {temperature}$^\circ$C: Unable to Find Breakdown V; Range too Small ({sensor.name} {date.strftime('%b %d, %Y')})")
        disable_top_and_right_bounds(plt)
        plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_dir)
        plt.close()
        return 

    if sensor.is_ignored(path) and plot: # if the iv scan is ignored
        ignored_scan_count += 1
        # data ignored, just plot the individual scan, then continue
        save_dir = f"{path.removesuffix('.txt').removesuffix('.iv')}_ivscan_ignored.png"
        print(f"Ignoring {path}. IV scan plot generated at {save_dir}.")
        plt.figure(figsize=(10, 6))
        plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature), marker='o', markersize=3)
        plt.xlabel("Reverse-bias Voltage (V)")
        plt.ylabel(f"log(Pad Current (A))")
        plt.title(rf"IV Scan at {temperature}$^\circ$C: Ignored ({sensor.name} {date.strftime('%b %d, %Y')})")
        disable_top_and_right_bounds(plt)
        plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_dir)
        plt.close()
        return

    lines, bd_std, dp_std = find_breakdown(xs, ys_log10, start_idx=first_idx_after_dep_v, path=path, bd_thresh=bd_thresh, dp_thresh=0.1, plot=True, relative_dp_thresh=True)
    # lines is [[slope, offset, bd_voltage, dp_voltage], ...]
    # sorted by inlier_count (decreasing), then by RMSE (increasing)
    # if line is None, something's wrong! We just plot the scan itself, and warn the user 
    if lines is None:
        save_dir = f"{path.removesuffix('.txt').removesuffix('.iv')}_ivscan_warn.png"
        print(f"Warning: Scan at {path} deprecated. Should be ignored or try a different current type. IV scan plot generated at {save_dir}.")
        plt.figure(figsize=(10, 6))
        # a. plot the scan itself
        plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature), marker='o', markersize=3)
        plt.xlabel("Reverse-bias Voltage (V)")
        plt.ylabel(f"log(Pad Current (A))")
        plt.title(rf"IV Scan at {temperature}$^\circ$C: Unable to Find Breakdown V ({sensor.name} {date.strftime('%b %d, %Y')})")
        disable_top_and_right_bounds(plt)
        plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_dir)
        plt.close()
        return 

    primary_line = lines[0]
    
    if plot: # plots individual iv scan with fitted lines
        plt.figure(figsize=(10, 6))
        # a. plot the scan itself
        plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature), marker='o', markersize=3)
        # b. plot 50 auxiliary lines
        for i in range(1, min(len(lines)-1, 51)): 
            pred_y = lines[i][0] * xs + lines[i][1]
            plt.plot(xs, pred_y, color='grey', alpha=0.1, linestyle='--', label='Linear Fits' if i == 1 else None)
        # c. plot auxiliary points
        for i in range(1, min(len(lines)-1, 51)): 
            plt.axvline(lines[i][2], color='grey', alpha=0.1, ls='-', label='Breakdown Points' if i == 1 else None)
            plt.axvline(lines[i][3], color='grey', alpha=0.1, ls='-', label='Depletion Points' if i == 1 else None)
        
        # d. plot main fitted line and thresholding line, and main breakdown point
        pred_y = primary_line[0] * xs + primary_line[1]
        plt.plot(xs, pred_y + bd_thresh, color='brown', linestyle='--', label='Primary Breakdown Threshold')
        plt.plot(xs, pred_y, color='black', linestyle='--', label='Primary Linear Fit')
        plt.axvline(primary_line[2], color='black', ls='-', label=f"Primary Breakdown Point")
        plt.axvline(primary_line[3], color='black', ls='-', label=f"Primary Depletion Point")
        
        plt.xlabel("Reverse-bias Voltage (V)")
        plt.ylabel("log(Pad Current (A))")
        plt.title(rf"IV Scan at {temperature}$^\circ$C: Breakdown {primary_line[2]:.2f} +/- {bd_std:.2f} V, Depletion {primary_line[3]:.2f} +/- {dp_std:.2f} V ({sensor.name} {date.strftime('%b %d, %Y')})")
        disable_top_and_right_bounds(plt)
        plt.legend()
        plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{path.removesuffix('.txt').removesuffix('.iv')}_ivscan.png")
        plt.close()

    # roll-back the overrides 
    if set_params is not None:
        if "DEP" in set_params:
            dep_v = old_dep_v

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
    
    Parameters
    ----------
    sensor : Sensor
        a Sensor object
    plot : bool
        Whether generates and saves plots
    """
    for dir in tqdm(sensor.data_dirs, desc=f"Analyzing CV profiles for sensor {sensor.name:<20}"):
        cv_paths = [os.path.join(dir, p) for p in os.listdir(dir) if p.endswith(".cv")]
        
        total_scan_count = 0
        ignored_scan_count = 0
        
        all_cv_info = []
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
            
            set_params = sensor.query_conf(path)
            
            if set_params is not None: # configuration overrides
                if "DEP" in set_params: # minimum volt to consider
                    if np.any(v_data >= set_params["DEP"]):
                        min_idx_after_dep = np.where(v_data >= set_params["DEP"])[0][0]
                        v_data = v_data[min_idx_after_dep:]
                        v_diff_data = v_diff_data[min_idx_after_dep:]
                        c_data = c_data[min_idx_after_dep:]
                        c_invsq_data = c_invsq_data[min_idx_after_dep:]
                        c_invsq_diff_data = c_invsq_diff_data[min_idx_after_dep:]
                    else:
                        # DEP is too high! ignore
                        print(f"Warning: DEP too high for CV scan at {path}. Ignoring config. You should edit data_config.")
                        
                if "MAX" in set_params: # max volt to consider 
                    if np.any(v_data <= set_params["MAX"]):
                        max_idx_before_max = np.where(v_data <= set_params["MAX"])[0][-1]
                        idx_to_cut = len(v_data) - max_idx_before_max - 1
                        v_data = v_data[:-idx_to_cut]
                        v_diff_data = v_diff_data[:-idx_to_cut]
                        c_data = c_data[:-idx_to_cut]
                        c_invsq_data = c_invsq_data[:-idx_to_cut]
                        c_invsq_diff_data = c_invsq_diff_data[:-idx_to_cut]
                    else: 
                        # MAX is too low! ignore 
                        print(f"Warning: MAX too low for CV scan at {path}. Ignoring config. You should edit data_config.")
                
                if "RT" in set_params:
                    # RT is irrelevant to CV analysis tho, but we keep it here for readability
                    ramp_type = set_params["RT"]
                    
            if sensor.is_ignored(os.path.join(path)):
                ignored_scan_count += 1
                # data ignored, just plot the individual scan, then continue
                save_dir = path.replace(".cv", "_cv_ignored.png")
                print(f"Ignoring {os.path.join(dir, path)}. CV scan plot generated at {save_dir}.")
                plt.figure(figsize=(10, 6))

                plt.plot(v_data, c_data, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature), marker='o', markersize=3)
                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel(f"log(Pad Current (A))")
                plt.title(rf"IV Scan at {temperature}$^\circ$C: Ignored ({sensor.name} {date.strftime('%b %d, %Y')})")
                disable_top_and_right_bounds(plt)
                plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                plt.tight_layout()
                plt.savefig(save_dir)
                plt.close()
                continue
            
            total_scan_count += 1
            
            def fit_depletion():
                # take last few data points, calculate mean, 
                # then threshold at 90% mean
                all_dep_v = []
                for i in range(2, len(c_data) // 2):
                    plateau_v = np.mean(c_data[-i:])
                    smallest_plateau_v = np.min(c_data[-i:])
                    cutoff_v = plateau_v + 50*np.abs(plateau_v - smallest_plateau_v)
                    mask = c_data < cutoff_v
                    if not np.any(mask):
                        # cutoff_v is too low, skip 
                        continue
                    first_below_id = np.where(mask)[0][0]
                    last_above_id = first_below_id - 1 
                    
                    # linear interpolation
                    # assuming the last_below_id is not the last data point
                    slope_inter = (c_data[first_below_id] - c_data[last_above_id]) / (v_data[first_below_id] - v_data[last_above_id])
                    offset_inter = c_data[last_above_id] - slope_inter * v_data[last_above_id]
                    dep_v = (cutoff_v - offset_inter) / slope_inter
                    # sanity check
                    if dep_v < 0 or dep_v > np.max(v_data):
                        continue 
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
                mask = v_data >= dep_v
                if not np.any(mask):
                    # something's wrong! dep_v is too high
                    return None
                start_idx = np.where(v_data >= dep_v)[0][0]
                mean_c = np.mean(c_data[start_idx:])
                return mean_c 
            
            # writes info to the overall list 
            all_cv_info.append([dep_v, std, ])
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
             
        # now, calculate a weighted sum for dep_v, and write to sensor
        if total_scan_count != 0:
            all_cv_info = np.array(all_cv_info)
            weighted_mean, weighted_sigma = calculate_weighted_mean(all_cv_info[:,0], all_cv_info[:,1])   
            sensor.depletion_v = weighted_mean
        
def main():
    # thresholds = {"AC_W3096": 0.3, "DC_W3058": 0.4, "DC_W3045": 0.4,
    #               "BNL_LGAD_513": 0.6, 
    #               "BNL_LGAD_W3076_9_13": 0.6,
    #               "BNL_LGAD_W3076_12_13": 0.5}
    
    # plot_humidity_scans("data/AC_W3096/Dec102024", thresholds["AC_W3096"])
    
    if ARGS.clear_plots:
        clear_plots()
    
    # loads all sensors and configs. 
    # Must load ALL sensors at once to properly write out to config at the end
    sensors = list_sensors()
    name_to_sensors = dict([(sensor.name, sensor) for sensor in sensors])
    load_sensor_config(DATABASE_DIR, sensors, load_iv=(not ARGS.iv or (ARGS.iv and not ARGS.overwrite)), load_cv=(not ARGS.cv or (ARGS.cv and not ARGS.overwrite)))
    load_data_config(DATABASE_DIR, sensors)
    
    if ARGS.file is None: # analyze sensors directly
        if ARGS.sensor is None:
            interested_sensor_names = set([sensor.name for sensor in sensors])
        else:
            # check if all sensor names are valid 
            all_names = set([sensor.name for sensor in sensors])
            interested_sensor_names = set(ARGS.sensor)
            if not interested_sensor_names.issubset(all_names):
                raise ValueError("Invalid sensor name(s).")
        
        if ARGS.cv:
            for sensor in sensors:
                if sensor.name in interested_sensor_names:
                    analyze_sensor_cv(sensor, plot=True)
                    
        if ARGS.iv:
            for sensor in sensors:
                if sensor.name in interested_sensor_names:
                    analyze_sensor_iv(sensor, curr_type=ARGS.curr_type, plot=True)
        
        write_sensor_config(DATABASE_DIR, sensors)
    else:
        print("Warning: In file mode, no information will be written to sensor_config.")
        for file_path in tqdm(ARGS.file, desc="Analyzing specified scans"):
            # find which sensor this scan belongs to
            if not file_path.startswith(DATABASE_DIR):
                raise ValueError(f"Path must start with database directory {DATABASE_DIR}, given {file_path}")
            elif not os.path.exists(file_path):
                raise ValueError(f"Path does not exist, given {file_path}")
            sensor_name = file_path.removeprefix(DATABASE_DIR).split('/')[1]
            sensor = name_to_sensors[sensor_name]
            if file_path.endswith(".txt") or file_path.endswith(".iv"):
                analyze_file_iv(sensor, file_path, curr_type=ARGS.curr_type, plot=True)
            elif file_path.endswith(".cv"):
                analyze_file_cv(sensor, file_path, plot=True)    
    
    return 0

if __name__ == "__main__":
    ARGS = parse_args()
    main()