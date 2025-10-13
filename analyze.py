import argparse
import math
import os
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=plt.cm.tab20.colors)
# use 20 color cycle instead of the default 10 cycle

from utils import (
    Sensor, DATABASE_DIR, list_sensors, parse_file_cv, parse_file_iv, temperature_to_color,
    load_data_config, load_sensor_config, determine_spacing, linear,
    linear_fit, disable_top_and_right_bounds, write_sensor_config, clear_plots,
    calculate_weighted_mean, dog_1d, humidity_to_color
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
        '--var',
        choices=['temp', 'humi'],
        default='temp',
        help='''Interested independent variable to analyze in IV scan.'''
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


def get_min_uncertainty(sequence: np.ndarray, ndata: int = None):
    """
    Returns the minimum uncertainty of any estimation from a sequence of 
    data due to resolution limit.
    
    Parameters
    ----------
    sequence : 1D-array
        The sequence of data being analyzed (and thus has a minimum uncertainty).
        Assumed to be (roughly) an Arithmetic Sequence.
    ndata : int, optional
        The number of data points used or involved with the analysis. If range 
        is not given, defaults to length of the sequence
    
    Returns
    -------
    sequence common difference / sqrt(range)
    """

    diff = determine_spacing(sequence)
    if ndata is None:
        ndata = len(sequence)
    return diff / np.sqrt(ndata)


def remove_ys_nan(xs: np.ndarray, ys: np.ndarray):
    """
    Removes data points with NaN values in ys and returns clean arrays.
    """
    nan_mask = np.isnan(ys) | np.isinf(ys)
    xs = xs[~nan_mask]
    ys = ys[~nan_mask]
    return xs, ys


def ransac(xs: np.ndarray, ys: np.ndarray, thresh: float = 0.5, niter: int = 300):
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
    niter : int, optional
        Number of RANSAC iterations. Defaults to 300.
    
    Returns
    -------
    lines : List[List[float]]
        A list of lists [[slope, offset, inlier count, mse], ...], each 
        sublist represents a trend line proposed by RANSAC
    
    Notes
    -----
    The output is not sorted in any way.
    """
    assert xs.shape == ys.shape

    xs, ys = remove_ys_nan(xs, ys)
    assert len(xs) >= 2

    lines = []  # each entry is a list [slope, offset, inlier count, mse, predicted_ys]
    seen = set()  # some dot pairs may have been selected. dedup
    index_list = range(len(xs))
    for _ in range(niter):
        indices = random.sample(index_list, max(2, len(xs) // 3))
        popt, _, _, _ = linear_fit(xs[indices], ys[indices])
        slope, offset = popt

        pred_ys = xs * slope + offset  # the y's for fitted line
        diff = np.abs(pred_ys - ys)
        is_close_map = diff < thresh
        inlier_count = np.sum(is_close_map)

        # some data might be nan due to np.log10, so use np.nanmean to find mse
        mse = np.nanmean(np.square(diff))

        lines.append([slope, offset, inlier_count, mse])

    return lines


def find_breakdown(xs: np.ndarray, ys: np.ndarray, start_idx: int, bd_thresh: float, plot: bool = True,
                   path: str = None):
    """
    Given a single IV scan, finds the breakdown voltage by fitting linear lines.
    Ignores data that are NaNs or Infs in ys.
    

    Parameters
    ----------
    xs : 1D-array
        Voltage data.
    ys : 1D-array
        Log10 of current data.
    start_idx : int 
        Index before which data will be ignored when fitting the line.
    bd_thresh : float 
        Absolute value of threshold for determining breakdown. (+)
    plot : bool, optional 
        Whether to generate and save breakdown distribution plots. Defaults to True.
    path : str, optional
        The path to this particular scan. Only required when plot is set to True. Defaults to None.
    
    Returns
    -------
    lines : List[List[float]]
        A list of lines in the form [[slope, offset, bd_voltage], ...], each 
        sublist represents a trend line and its corresponding predicted 
        breakdown voltage. Ranked according to confidence. Better lines at 
        the top.
    std : float 
        Uncertainty of estimation.
    """
    # truncate data 
    valid_xs = xs[start_idx:]
    valid_ys = ys[start_idx:]

    xs, ys = remove_ys_nan(xs, ys)
    valid_xs, valid_ys = remove_ys_nan(valid_xs, valid_ys)
    assert valid_xs.shape == valid_ys.shape

    fit_range = len(valid_xs) // 2
    raw_lines = ransac(valid_xs[:fit_range], valid_ys[:fit_range], 0.15, 200)
    lines = []
    for line in raw_lines:
        slope, offset = line[0], line[1]
        pred_ys = slope * valid_xs + offset

        # find the breakdown voltage with linear interpolation
        last_id_in_thresh = -1  # some lines are so ridiculous that we cant find last_id_in_thresh
        for i in range(len(valid_ys)):
            if abs(valid_ys[i] - pred_ys[i]) < bd_thresh:
                last_id_in_thresh = i
        if last_id_in_thresh == -1:
            continue  # no pts is inside bd_thresh
        if last_id_in_thresh + 1 == len(valid_ys):
            # last data point! just use it as is
            bd_voltage = valid_xs[last_id_in_thresh]
        else:  # find the next data point, fit a line, solve for intersection
            slope_inter = (valid_ys[last_id_in_thresh + 1] - valid_ys[last_id_in_thresh]) / (
                    valid_xs[last_id_in_thresh + 1] - valid_xs[last_id_in_thresh])
            offset_inter = valid_ys[last_id_in_thresh] - slope_inter * valid_xs[last_id_in_thresh]
            bd_voltage = (offset_inter - (offset + bd_thresh)) / (slope - slope_inter)

        line.append(bd_voltage)
        lines.append(line)  # add the line from raw_lines to lines

    if len(lines) == 0:
        # There's something wrong with the data!
        # We can't fit anything! 
        return [], -1
    # now each sublist has structure [slope, offset, inlier_count, mse, bd_voltage]
    lines = np.array(lines)
    # sanity check, breakdown voltage must > 0
    mask = lines[:, 4] > 0
    # sanity check #2, the fitted line must have slope > 0
    mask &= lines[:, 0] > 0
    lines = lines[mask]
    if len(lines) == 0:
        # There's something wrong with the data!
        # We can't fit anything! 
        return [], -1

    # sort according to bd_voltage
    lines = lines[np.argsort(lines[:, 4])]
    # cutoff extreme values for bd_voltage
    cum_weights = np.cumsum(lines[:, 2])
    total_weight = cum_weights[-1]
    trim_ratio = 0.10
    lower = trim_ratio * total_weight
    upper = (1 - trim_ratio) * total_weight
    mask = (cum_weights >= lower) & (cum_weights <= upper)

    if np.sum(mask) <= 0:
        mask = np.ones_like(lines).astype(bool)  # just keep all then
    # group line inliers and outliers
    filtered_lines = lines[mask]
    lines = lines[~mask]
    assert len(filtered_lines) != 0

    # calculate uncertainty for bd_voltage and dp_voltage
    def find_frequency_weighted_uncertainty(filtered_lines, col):
        # col is 4 for breakdown, 5 for depletion
        try:  # the uncertainty is weighted by frequency in each bin, bin width = 1V
            frequencies = np.zeros_like(filtered_lines[:, col])
            min_bin_left_edge = math.floor(np.min(filtered_lines[:, col]))
            max_bin_left_edge = math.floor(np.max(filtered_lines[:, col]))
            for volt in range(min_bin_left_edge, max_bin_left_edge + 1):
                lines_in_bin_mask = (filtered_lines[:, col] >= volt) & (filtered_lines[:, col] < volt + 1)
                count = np.sum(lines_in_bin_mask)
                frequencies[lines_in_bin_mask] = count
            mean = np.average(filtered_lines[:, 4], weights=frequencies)
            variance = np.average((filtered_lines[:, 4] - mean) ** 2, weights=frequencies)
            std = max(np.sqrt(variance), get_min_uncertainty(valid_xs, fit_range))
            return std
        except:
            # weights sum to zero, can't be normalized
            return None

    bd_std = find_frequency_weighted_uncertainty(filtered_lines, 4)
    if bd_std is None:  # something's wrong
        return [], -1
    # sort against inlier counts so the line with most inliers is at index 0
    # then sort against mse (if inlier counts are same, use one with smaller mse)
    filtered_lines = filtered_lines[np.lexsort((filtered_lines[:, 3], -filtered_lines[:, 2]))]

    # plot breakdown distribution, frequency and max weight
    def plot_entry(col, str, std):
        # col is 4 for breakdown
        # str is "Breakdown"
        # std is the frequency weighted uncertainty to be included on graph
        save_str = "bdv"
        # separate into bins
        bins = np.histogram_bin_edges(lines[:, col], bins=120)
        bin_idx = np.digitize(lines[:, col], bins) - 1
        bin_idx2 = np.digitize(filtered_lines[:, col], bins) - 1
        # calculate max inlier count by each bin
        bin_mic = [lines[:, 2][bin_idx == i].max() if np.any(bin_idx == i) else np.nan for i in range(len(bins) - 1)]
        bin_mic2 = [filtered_lines[:, 2][bin_idx2 == i].max() if np.any(bin_idx2 == i) else np.nan for i in
                    range(len(bins) - 1)]
        # calculate frequency by each bin
        bin_tic = [(lines[:, 2][bin_idx == i] >= 1).sum() if np.any(bin_idx == i) else np.nan for i in
                   range(len(bins) - 1)]
        bin_tic2 = [(filtered_lines[:, 2][bin_idx2 == i] >= 1).sum() if np.any(bin_idx2 == i) else np.nan for i in
                    range(len(bins) - 1)]
        # calculate min mse by each bin # some data could be nan due to np.log10, so use nan-ignoring np.nanmin()
        bin_mrmse = np.sqrt(
            [np.nanmin(lines[:, 3][bin_idx == i]) if np.any(bin_idx == i) else np.nan for i in range(len(bins) - 1)])
        bin_mrmse2 = np.sqrt(
            [np.nanmin(filtered_lines[:, 3][bin_idx2 == i]) if np.any(bin_idx2 == i) else np.nan for i in
             range(len(bins) - 1)])

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

        if plot:  # plot distribution (frequency) and rmse
            plt.figure(figsize=(8, 6))
            ax1 = plt.gca()
            ax1.bar(bin_centers, bin_tic, width=np.diff(bins), color='black', alpha=0.3, label=f'Outliers')
            ax1.bar(bin_centers, bin_tic2, width=np.diff(bins), color='purple', alpha=0.7)
            ax1.set_ylabel('Frequency')
            ax1.set_xlabel(f'{str} Voltage (V)')

            ax2 = plot_rmse(ax1)
            ax1.axvline(filtered_lines[0][col], color="black", ls="--",
                        label=f"Max Weight Min RMSE Choice: {filtered_lines[0][col]:.2f} $V$")

            setup_legend(ax1, ax2)
            plt.title(f'{str} Voltage Frequency Distribution (Best: {filtered_lines[0][col]:.2f} +/- {std:.2f} V)')
            plt.tight_layout()
            plt.savefig(f"{path.removesuffix('.txt').removesuffix('.iv')}_{save_str}_freq.png")
            plt.close()
        if plot:  # plot distribution (max weight) and rmse
            plt.figure(figsize=(8, 6))
            ax1 = plt.gca()
            ax1.bar(bin_centers, bin_mic, width=np.diff(bins), color='black', alpha=0.3, label=f'Outliers')
            ax1.bar(bin_centers, bin_mic2, width=np.diff(bins), color='purple', alpha=0.7)
            ax1.set_ylabel('Max Weights')
            ax1.set_xlabel(f'{str} Voltage (V)')

            ax2 = plot_rmse(ax1)

            ax1.axvline(filtered_lines[0][col], color="black", ls="--",
                        label=f"Max Weight Min RMSE Choice: {filtered_lines[0][col]:.2f} $V$")

            setup_legend(ax1, ax2)
            plt.title(f'{str} Voltage Max Weight Distribution (Best: {filtered_lines[0][col]:.2f} +/- {std:.2f} V)')
            plt.tight_layout()
            plt.savefig(f"{path.removesuffix('.txt').removesuffix('.iv')}_{save_str}_max.png")
            plt.close()

    plot_entry(4, "Breakdown", bd_std)

    lines = filtered_lines
    lines = np.delete(lines, [2, 3], axis=1)
    # now each line has structure [slope, offset, bd_voltage]

    return lines, bd_std


def find_depletion(xs: np.ndarray, ys: np.ndarray, path: str, line: list[float] = None, start_idx: int = None,
                   plot: bool = True):
    """
    Given a single IV scan, finds the depletion voltage by convolving with DoG 
    and applying Non-maximum suppresion. Ignores data that are NaNs or Infs in ys.
    

    Parameters
    ----------
    xs : 1D-array
        Voltage data.
    ys : 1D-array
        Log10 of current data.
    path : str 
        The path to this particular scan.
    line: list[float], optional
        A line returned by find_breakdown().
    start_idx: int, optional
        The start_idx for which line fitting begins.
    plot : bool, optional 
        Whether to generate and save breakdown distribution plots. Defaults to True.
        
    Returns
    -------
    primary_dep_v : float
        The primary (most confident) depletion point.
    deps : List[float]
        A list of candidate depletion points
    std : float 
        Uncertainty of estimation.
    """
    # truncate data 

    xs, ys = remove_ys_nan(xs, ys)
    assert xs.shape == ys.shape

    ys = median_filter(ys, size=3)

    # find the depletion voltage with that linear line
    slope, offset, bd_v = line
    pred_ys = slope * xs + offset  # we use full range of xs
    first_id_in = -1
    fit_range = (len(xs) - start_idx) // 2
    rmse_in_fit_range = np.sqrt(np.mean(np.square(slope * xs[start_idx:fit_range] + offset - ys[start_idx:fit_range])))
    dp_thresh = 3 * rmse_in_fit_range  # 3 sigma
    # Going from right to left
    # if for 3 consecutive pts there are less than 2 inliers,
    # the end idx of the first such 3 pts is dep_v
    # Linear time algorithm using prefix sum
    n_inlier_upto_me = np.zeros((len(ys) // 2,), dtype=int)
    n_inlier_upto_me[0] = int(pred_ys[0] - dp_thresh <= ys[0] <= pred_ys[0] + dp_thresh)
    for i in range(1, len(n_inlier_upto_me)):
        n_inlier_upto_me[i] = n_inlier_upto_me[i - 1] + int(pred_ys[i] - dp_thresh <= ys[i] <= pred_ys[i] + dp_thresh)
    for i in range(len(n_inlier_upto_me) - 1, 4, -1):
        if n_inlier_upto_me[i] - n_inlier_upto_me[i - 4] <= 2:
            first_id_in = i
            break

    first_diff = np.diff(ys, 1)
    # remove data pts before 10V
    min_v = 10
    min_v_idx = np.where(xs > min_v)[0][0]
    first_diff[:min_v_idx] = 0
    first_diff = np.pad(first_diff, (1, 0), mode="edge")
    dog_arr = np.convolve(first_diff, dog_1d(3), mode="same")  # conv with dog
    # focus on region near first_id_in 
    dog_arr[:first_id_in - 3] = 0
    dog_arr[first_id_in + 3:] = 0
    dog_arr = np.abs(dog_arr)  # at depletion, slope can rise or drop!
    # nms_arr = nms_1d(dog_arr) # non-max suppression
    nms_arr = dog_arr
    nms_arr[int(len(nms_arr) * 0.3):] = 0  # truncate tail

    primary_dep_idx = np.argmax(nms_arr)  # np.where(nms_arr > 0)[0][-1]
    primary_dep_v = xs[primary_dep_idx]
    std = max(determine_spacing(xs) / np.sqrt(12), np.std(nms_arr))
    dep_vs = xs[nms_arr > 0]
    dep_weights = nms_arr[nms_arr > 0]
    args = np.argsort(-dep_weights)
    dep_vs = list(dep_vs[args])

    if plot:
        save_str = "dpv"
        plt.figure(figsize=(8, 6))
        # plt.bar(dep_vs, dep_weights, color='purple', alpha=0.7)
        plt.bar(xs, nms_arr, color='purple', alpha=0.7)
        plt.ylabel('Max Weights')
        plt.xlabel(f'Voltage (V)')
        plt.title(f'Detected Transient (Best: {primary_dep_v:.2f} +/- {std:.2f} V)')
        plt.tight_layout()
        plt.savefig(f"{path.removesuffix('.txt').removesuffix('.iv')}_{save_str}_max.png")
        plt.close()

    return primary_dep_v, dep_vs, std


def warn_plot_iv(xs, ys_log10, temperature, date, sensor, msg, save_dir):
    plt.figure(figsize=(10, 6))
    # plot the scan itself
    plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature),
             marker='o', markersize=3)
    plt.xlabel("Reverse-bias Voltage (V)")
    plt.ylabel(f"log(Pad Current (A))")
    plt.title(rf"IV Scan at {temperature}$^\circ$C: {msg} ({sensor.name} {date.strftime('%b %d, %Y')})")
    disable_top_and_right_bounds(plt)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close()


def analyze_sensor_iv(sensor: Sensor, curr_type: str = 'pad', var: str = 'temp', plot: bool = True,
                      file_mode: bool = False, path: str = None):
    """
    Analyze all IV scans of a specific sensor. 
    
    Parameters
    ----------
    sensor : Sensor
        The interested sensor.
    curr_type : str {'pad', 'gr', 'total'}, optional
        Specify the current type for analysis. Some scan may not support all the 
        types. Defaults to 'pad'.
    var : str {'temp', 'humi'}, optional
        Specify which environmental variable to analyze. Defaults to 'temp'.
    plot : bool, optional
        Whether generate and save the plots. If set to True, a plot of the IV 
        scans with analysis, plots of distributions of estimated parameters, 
        plots of IV scan ramps in each subdirectory, 
        a plot of breakdown voltage vs temperature, 
        a plot of average breakdown voltage vs temperature, 
        and a plot of breakdown voltage vs scan number 
        will be generated, either alongside the scans or in the sensor's 
        primary directory. Defaults to True.

        If the something went wrong during analysis, a 
        warning plot is generated regardless what value plot is.
    file_mode : bool, optional
        Whether operate in file mode. Defaults to False.
    path : str, optional
        The path to specified IV scan. Used with file_mode=True.
    """
    assert var == "temp" or var == "humi"
    if file_mode: assert path is not None, "path is required in file_mode"
    # 1. sets up related constants for the sensor:
    # a. use the specified threshold for breakdown fits
    bd_thresh = sensor.bd_thresh
    if bd_thresh is None:
        bd_thresh = 0.5  # defaults to 0.5
    # b. data before dep_v ignored when fitting (only when fitting)
    dep_v = sensor.depletion_v
    if dep_v is None:
        dep_v = 25  # if not set, defaults to 25V

    all_data_in_dirs = []  # used for 3). a list [[[temp, date, data, humidity, ramp_type, bd_v, std],...]...]

    def get_curr_type_label():
        if curr_type == 'pad':
            return "Pad"
        elif curr_type == 'gr':
            return "Guard Ring"
        elif curr_type == 'total':
            return "Total Current"
        else:
            raise ValueError(f"Invalid Current Type: {curr_type}. Must be one of 'pad', 'gr', or 'total'.")

    # 2. loops thru all dirs containing scans
    for ndir, dir in enumerate(tqdm(sensor.data_dirs, desc=f"Analyzing IV profiles for sensor {sensor.name:<20}")):
        # Note: plt only has one buffer that's cleared whenever 
        # plt.figure() is called! 
        if file_mode and ndir:
            break
        all_data_in_dir = []  # used for b). a list [[temp, date, data, humidity, ramp_type, bd_v, std],...],
        # dir is relative path to a folder containing scans
        # a. loops thru all individual scans within the dir
        tot_scan_count = 0
        ignored_scan_count = 0

        for nscan, scan_path in enumerate(os.listdir(dir)):
            if file_mode and nscan:
                break
            if file_mode:
                scan_path = path  # analyze the specified file
            if scan_path.startswith("."):
                continue  # ignore hidden files
            if not (scan_path.endswith(".iv") or scan_path.endswith(".txt")):
                continue  # ignore non .txt non .iv
            tot_scan_count += 1

            set_params = sensor.query_conf(os.path.join(dir, scan_path))

            temperature, date, data, humidity, ramp_type, duration = parse_file_iv(os.path.join(dir, scan_path))
            if temperature is None:  # not all scans have complete data!
                temperature = float('nan')
            if humidity is None:
                humidity = float('nan')
            if duration is None:
                duration = float('nan')
            if set_params is not None:  # configuration overrides
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
            with np.errstate(divide='ignore', invalid='ignore'):  # suppress invalid value and division by zero err
                ys_log10 = np.log10(ys)  # we will remove nan and inf after
            # find the first index after dep_v 
            try:
                first_idx_after_dep_v = np.where(xs > dep_v)[0][0]
            except:  # cannot find voltage after dep_v, either dep_v too high or voltage range too small
                save_dir = f"{os.path.join(dir, scan_path.removesuffix('.txt').removesuffix('.iv'))}_ivscan_warn.png"
                warn_plot_iv(xs, ys_log10, temperature, date, sensor,
                             "Unable to Find Breakdown or Depletion; Range too Small", save_dir)
                print(
                    f"Warning: Scan at {os.path.join(dir, scan_path)} unable to analyze. "
                    f"Should be ignored or try with different config. IV scan plot generated at {save_dir}.")
                continue

            if sensor.is_ignored(os.path.join(dir, scan_path)) and plot:  # if the iv scan is ignored
                ignored_scan_count += 1
                # data ignored, just plot the individual scan, then continue
                save_dir = f"{os.path.join(dir, scan_path.removesuffix('.txt').removesuffix('.iv'))}_ivscan_ignored.png"
                print(f"Ignoring {os.path.join(dir, scan_path)}. IV scan plot generated at {save_dir}.")
                warn_plot_iv(xs, ys_log10, temperature, date, sensor, "Ignored", save_dir)
                continue

            bd_lines, bd_std = find_breakdown(xs, ys_log10, start_idx=first_idx_after_dep_v,
                                              path=os.path.join(dir, scan_path), bd_thresh=bd_thresh, plot=plot)
            # lines is [[slope, offset, bd_voltage], ...]
            # sorted by inlier_count (decreasing), then by RMSE (increasing)
            primary_v, dp_vs, dp_std = find_depletion(xs, ys_log10, os.path.join(dir, scan_path),
                                                      bd_lines[0] if len(bd_lines) else None, first_idx_after_dep_v,
                                                      plot=plot)

            # if both are empty, something's wrong! we just plot the scan and warn the user
            if len(bd_lines) == 0 and len(dp_vs) == 0:
                save_dir = f"{os.path.join(dir, scan_path.removesuffix('.txt').removesuffix('.iv'))}_ivscan_warn.png"
                print(
                    f"Warning: Scan at {os.path.join(dir, scan_path)} cannot estimate depletion and breakdown. "
                    f"IV scan plot generated at {save_dir}.")
                warn_plot_iv(xs, ys_log10, temperature, date, sensor, "Unable to Find Depletion and Breakdown",
                             save_dir)
                continue

            # plot the scan itself with estimated bd and dp
            plt.figure(figsize=(10, 6))
            plt.xlabel("Reverse-bias Voltage (V)")
            plt.ylabel("log(Pad Current (A))")
            plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
            disable_top_and_right_bounds(plt)
            # a plot the scan itself
            plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature),
                     marker='o', markersize=3)

            if len(bd_lines) != 0:
                primary_line = bd_lines[0]
                if plot:
                    # b. plot 50 auxiliary lines
                    for i in range(1, min(len(bd_lines) - 1, 51)):
                        pred_y = bd_lines[i][0] * xs + bd_lines[i][1]
                        plt.plot(xs[first_idx_after_dep_v:], pred_y[first_idx_after_dep_v:], color='grey', alpha=0.1,
                                 linestyle='--', label='Linear Fits' if i == 1 else None)
                    # c. plot auxiliary points
                    for i in range(1, min(len(bd_lines) - 1, 51)):
                        plt.axvline(bd_lines[i][2], color='grey', alpha=0.1, ls='-',
                                    label='Breakdown Points' if i == 1 else None)
                    # d. plot main fitted line and thresholding line, and main breakdown point
                    pred_y = primary_line[0] * xs + primary_line[1]
                    plt.plot(xs[first_idx_after_dep_v:], (pred_y + bd_thresh)[first_idx_after_dep_v:], color='brown',
                             linestyle='--', label='Primary Breakdown Threshold')
                    plt.plot(xs[first_idx_after_dep_v:], pred_y[first_idx_after_dep_v:], color='black', linestyle='--',
                             label='Primary Linear Fit')
                    plt.axvline(primary_line[2], color='black', ls='-', label=f"Primary Breakdown Point")
            if len(dp_vs) != 0:
                if plot:
                    # e. plot the main depletion point
                    plt.axvline(primary_v, color='purple', ls='-', label=f"Primary Depletion Point")
                    # f. plot auxiliary points
                    # for i in range(1, min(len(dp_vs)-1, 51)):
                    #     plt.axvline(dp_vs[i], color='purple', ls='-', alpha=0.1, label='Depletion Points' if i == 1 else None)
            if len(bd_lines) != 0 and len(dp_vs) != 0:
                plt.title(
                    rf"IV Scan at {temperature}$^\circ$C: Breakdown {primary_line[2]:.2f} +/- {bd_std:.2f} V, "
                    rf"Depletion {primary_v:.2f} +/- {dp_std:.2f} V ({sensor.name} {date.strftime('%b %d, %Y')})")
            elif len(bd_lines) != 0 and len(dp_vs) == 0:
                plt.title(
                    rf"IV Scan at {temperature}$^\circ$C: Breakdown {primary_line[2]:.2f} +/- {bd_std:.2f} V, "
                    rf"Unable to Find Depletion ({sensor.name} {date.strftime('%b %d, %Y')})")
            elif len(bd_lines) == 0 and len(dp_vs) != 0:
                plt.title(
                    rf"IV Scan at {temperature}$^\circ$C: Unable to Find Breakdown, "
                    rf"Depletion {primary_v:.2f} +/- {dp_std:.2f} V ({sensor.name} {date.strftime('%b %d, %Y')})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{os.path.join(dir, scan_path.removesuffix('.txt').removesuffix('.iv'))}_ivscan.png")
            plt.close()

            if len(bd_lines) != 0 and len(dp_vs) != 0:
                all_data_in_dir.append(
                    [temperature, date, data, humidity, ramp_type, primary_line[2], bd_std, primary_v, dp_std])
            elif len(bd_lines) != 0 and len(dp_vs) == 0:
                print(
                    "Warning: no satisfying depletion estimation, "
                    "primary depletion point and dp_std set to float('nan')")
                all_data_in_dir.append(
                    [temperature, date, data, humidity, ramp_type, primary_line[2], bd_std, float('nan'), float('nan')])
            elif len(bd_lines) == 0 and len(dp_vs) != 0:
                print(
                    "Warning: no satisfying breakdown estimation, "
                    "primary breakdown point and bd_std set to float('nan')")
                all_data_in_dir.append(
                    [temperature, date, data, humidity, ramp_type, float('nan'), float('nan'), primary_v, dp_std])

            # roll-back the overrides 
            if set_params is not None:
                if "DEP" in set_params:
                    dep_v = old_dep_v

        if file_mode:
            # single file analysis finished here
            return

        if ignored_scan_count == tot_scan_count:  # error check
            print(f"Warning: {sensor.name} all scans in {dir} are ignored.")
            continue

        all_data_in_dirs.append(all_data_in_dir)

        # b. for each dir containing multiple scans, plot just all ramp up scans together,
        # then all ramp down scans together
        if plot:
            # partition the data
            all_ramp_types_here = set(d[4] for d in all_data_in_dir)
            partition = []
            for ramp_type in all_ramp_types_here:
                partition.append([d for d in all_data_in_dir if d[4] == ramp_type])
            # sort by temperature (decreasing), then humidity (increasing)
            for i in range(len(partition)):
                partition[i] = sorted(partition[i], key=lambda d: d[3])
                partition[i] = sorted(partition[i], key=lambda d: -d[0])

            curr_type_str = get_curr_type_label()
            ramp_title_str = dict({0: "", 1: ", Ramp Up", -1: ", Ramp Down"})
            ramp_save_str = dict({0: "", 1: "rampup", -1: "rampdown"})

            for all_data_in_dir_ramp_type in partition:
                if len(all_data_in_dir_ramp_type) > 1:
                    plt.figure(figsize=(10, 6))  # temp
                    seen_label = set()
                    for temperature, date, data, humidity, ramp_type, bd_v, bd_std, dp_v, dp_std in all_data_in_dir_ramp_type:
                        label = rf"{temperature:.1f}$^\circ$C"
                        if label in seen_label:
                            plt.plot(data["voltage"], data[curr_type], marker='o', markersize=3,
                                     color=temperature_to_color(temperature))
                        else:
                            seen_label.add(label)
                            plt.plot(data["voltage"], data[curr_type], marker='o', markersize=3, label=label,
                                     color=temperature_to_color(temperature))
                    plt.xlabel("Voltage (V)")
                    plt.ylabel("Current (A)")
                    plt.yscale('log')
                    plt.legend()
                    disable_top_and_right_bounds(plt)
                    plt.title(
                        f"IV Scan of {sensor.name} on {date.strftime('%b %d, %Y')} ({curr_type_str}{ramp_title_str[ramp_type]})")
                    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
                    plt.tight_layout()
                    plt.savefig(os.path.join(dir, f"{sensor.name}_{ramp_save_str[ramp_type]}_{curr_type}_all.png"))
                    plt.close()

                    plt.figure(figsize=(10, 6))  # humi
                    seen_label = set()
                    for temperature, date, data, humidity, ramp_type, bd_v, bd_std, dp_v, dp_std in all_data_in_dir_ramp_type:
                        label = rf"{humidity:.1f} %"
                        if label in seen_label:
                            plt.plot(data["voltage"], data[curr_type], marker='o', markersize=3,
                                     color=humidity_to_color(humidity))
                        else:
                            seen_label.add(label)
                            plt.plot(data["voltage"], data[curr_type], marker='o', markersize=3, label=label,
                                     color=humidity_to_color(humidity))
                    plt.xlabel("Voltage (V)")
                    plt.ylabel("Current (A)")
                    plt.yscale('log')
                    plt.legend()
                    disable_top_and_right_bounds(plt)
                    plt.title(
                        f"IV Scan of {sensor.name} on {date.strftime('%b %d, %Y')} ({curr_type_str}{ramp_title_str[ramp_type]})")
                    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
                    plt.tight_layout()
                    plt.savefig(os.path.join(dir, f"{sensor.name}_{ramp_save_str[ramp_type]}_{curr_type}_all2.png"))
                    plt.close()

    if var == "temp":
        var_idx = 0
    elif var == "humi":
        var_idx = 3
    else:
        raise ValueError(f"Unknown variable, given {var}")

    def autoset_xlabel(plt):
        if var == "temp":
            plt.xlabel("Temperature (C)")
        elif var == "humi":
            plt.xlabel("Humidity (%)")

    def get_var_name():
        if var == "temp":
            return "Temperature"
        elif var == "humi":
            return "Humidity"

    # 3. if the sensor folder contains multiple scans, consolidate then plot
    # 1) all bdv vs interested var 2) avg bdv vs interested var w/ uncertainty and fitted line
    # 3) bdv at different vals of interested var across time (scan number)
    if plot:  # 1) plot all bdv vs interested var
        plt.figure(figsize=(10, 6))
        for all_data_in_dir in all_data_in_dirs:
            # partition the data 
            all_ramp_types_here = set(d[4] for d in all_data_in_dir)
            partition = []
            for ramp_type in all_ramp_types_here:
                partition.append([d for d in all_data_in_dir if d[4] == ramp_type])
            # sort the data by interested var (increasing)
            for i in range(len(partition)):
                partition[i] = sorted(partition[i], key=lambda d: d[var_idx])
            ramp_title_str = dict({0: "", 1: " (Ramp Up)", -1: " (Ramp Down)"})
            for ramp in partition:
                if len(ramp) > 0:
                    plt.plot([d[var_idx] for d in ramp], [d[5] for d in ramp], marker='o', markersize=3,
                             label=f"{ramp[0][1].date()}{ramp_title_str[ramp[0][4]]}")
                    # [0] -> temp, [5] -> bd_v, [1] -> date
                    # all data in the same all_data_in_dir should have same date; date is at index 1, i.e. d[1]

        autoset_xlabel(plt)
        plt.ylabel("Breakdown Voltage (V)")
        plt.title(f"{sensor.name} Breakdown Voltage vs. {get_var_name()}")
        disable_top_and_right_bounds(plt)
        plt.legend()
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(sensor.sensor_dir, f"{sensor.name}_all.png"))
        plt.close()

    # 2) preparing plot avg bdv vs interested var w/ uncertainty and fitted line
    var_to_bdv_std_dict = dict()  # a dict that maps: var_val -> [[bdv, std], ...]
    all_data_in_dirs_flat = [data_lst for sublist in all_data_in_dirs for data_lst in sublist]  # flatten
    for data_lst in all_data_in_dirs_flat:
        var_val, bdv, std = data_lst[var_idx], data_lst[5], data_lst[6]
        if var_val == float('inf'):  # some .iv scans did not record temp and were 'inf'
            continue
        var_to_bdv_std_dict.setdefault(var_val, list()).append([bdv, std])

    if len(var_to_bdv_std_dict) == 0:
        print(
            "Warning: No available temperature and breakdown voltage data for plotting. "
            "Either all data are ignored, or all temperatures are inf (not given).")
        return

    # 2) loops thru all temps, compute weighted average for each var_val
    var_mean_sigma = list()  # a list [[var_val, weighted_mean, weighted_sigma], ...]
    for var_val in var_to_bdv_std_dict:
        all_std, all_bdv = [], []
        for d in var_to_bdv_std_dict[var_val]:
            if np.isnan(d[0]) or np.isnan(d[1]):
                continue  # invalid sample
            all_std.append(d[1])
            all_bdv.append(d[0])
        if len(all_std) == 0:  # note: the 3 list should have same len
            continue
        all_std = np.array(all_std)
        all_bdv = np.array(all_bdv)
        weighted_mean, weighted_sigma = calculate_weighted_mean(all_bdv, all_std)

        var_mean_sigma.append([var_val, weighted_mean, weighted_sigma])

    # write info to sensor
    var_mean_sigma = np.array(var_mean_sigma)
    var_mean_sigma = var_mean_sigma[np.argsort(var_mean_sigma[:, 0])]
    sensor.iv_scan_data = np.delete(var_mean_sigma, 2, axis=1)  # remove weighted_sigma

    var_mean_sigma = np.array(var_mean_sigma)
    avg_uncertainty = np.mean(var_mean_sigma[:, 2])

    slope_err = None
    # 2) now plot, x is temp, y is weighted_mean, error_bar is weighted_sigma, then fit a line thru
    if plot:
        plt.figure(figsize=(10, 6))
        plt.errorbar(var_mean_sigma[:, 0], var_mean_sigma[:, 1], yerr=var_mean_sigma[:, 2], fmt='o', capsize=5,
                     label=f'{sensor.name} data')
        if len(var_mean_sigma) >= 2:  # plot bdv vs var_val (average trend)
            assert var_mean_sigma[:, 1].shape == var_mean_sigma[:, 2].shape
            assert not np.any(np.isnan(var_mean_sigma[:, 2]) | np.isinf(var_mean_sigma[:, 2]))
            popt, perr, _, r2 = linear_fit(var_mean_sigma[:, 0], var_mean_sigma[:, 1], [1, 150],
                                           sigmas=var_mean_sigma[:, 2])
            slope_err = perr[0]
            # write fitted line to sensor 
            sensor.iv_scan_line = [popt[0], popt[1], np.mean(var_mean_sigma[:, 2])]  # [slope, offset, avg_sigma]
            plt.plot(var_mean_sigma[:, 0], linear(var_mean_sigma[:, 0], popt[0], popt[1]),
                     label=f'r2 value: {r2:0.2f} Best Fit: y = {popt[0]:.2f}x + {popt[1]:.2f}', linestyle='--')
        else:  # some only has one scan, cannot fit a line
            print(f"Warning: Insufficient scans to plot BDV vs. Temp Trend ({sensor.name}). Could not fit a line.")
            sensor.iv_scan_line = None
        autoset_xlabel(plt)
        plt.ylabel("Breakdown Voltage (V)")
        plt.title(f"{sensor.name} Breakdown Voltage vs. {get_var_name()}")
        disable_top_and_right_bounds(plt)
        plt.legend()
        plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(sensor.sensor_dir, f"{sensor.name}_avg.png"))
        plt.close()

    if plot:  # 3) bdv at different temperature across time (scan number)
        plt.figure(figsize=(10, 6))
        for var_val in sorted(var_to_bdv_std_dict.keys(), key=lambda temp: -temp):
            scan_idx = np.arange(len(var_to_bdv_std_dict[var_val]))
            plt.plot(scan_idx, [d[0] for d in var_to_bdv_std_dict[var_val]], marker='o',
                     color=temperature_to_color(var_val), label=rf"{var_val}$^\circ$C")
        plt.xlabel("Scan Number")
        plt.title(f"{sensor.name} Breakdown Voltage vs. {get_var_name()} over Time")
        plt.ylabel("Breakdown Voltage (V)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(sensor.sensor_dir, f"{sensor.name}_bdv_trend.png"))
        plt.close()

    # return slope_err, avg_sigma


def plot_humidity_scans(data_dir, bd_thresh):
    curr_type = 'pad'
    sensor = 'AC'
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
        log_curr = np.log10(-1 * data[curr_type][neg_idx])
        plt.scatter(voltages, log_curr, color=humidity_to_color(humidities[i]), label=str(humidities[i]) + ' % rh',
                    s=20)
        leakage_140.append(log_curr[np.argmin(abs(voltages - 140))])
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("log( pad current (A) )")
    plt.title(sensor + "-LGAD IV Scan as Function of Relative Humidity at 21 C")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("data/humidity_" + sensor + ".png")
    plt.close()
    # plot leakage current at 140V (approximate operating voltage)
    plt.figure(figsize=(10, 6))
    plt.plot(np.sort(humidities), leakage_140, marker='o')
    plt.xlabel("Relative Humidity (%)")
    plt.ylabel("log( pad current (A) ) @ 140V")
    plt.title(sensor + "-LGAD Leakage Current at 140V as Function of Relative Humidity at 21 C")
    plt.tight_layout()
    plt.savefig("data/humidity_leakage_" + sensor + ".png")
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


def analyze_sensor_cv(sensor: Sensor, plot: bool = True):
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

            if set_params is not None:  # configuration overrides
                if "DEP" in set_params:  # minimum volt to consider
                    if np.any(v_data >= set_params["DEP"]):
                        min_idx_after_dep = np.where(v_data >= set_params["DEP"])[0][0]
                        v_data = v_data[min_idx_after_dep:]
                        v_diff_data = v_diff_data[min_idx_after_dep:]
                        c_data = c_data[min_idx_after_dep:]
                        c_invsq_data = c_invsq_data[min_idx_after_dep:]
                        c_invsq_diff_data = c_invsq_diff_data[min_idx_after_dep:]
                    else:
                        # DEP is too high! ignore
                        print(
                            f"Warning: DEP too high for CV scan at {path}. Ignoring config. You should edit data_config.")

                if "MAX" in set_params:  # max volt to consider
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
                        print(
                            f"Warning: MAX too low for CV scan at {path}. Ignoring config. You should edit data_config.")

                if "RT" in set_params:
                    # RT is irrelevant to CV analysis tho, but we keep it here for readability
                    ramp_type = set_params["RT"]

            if sensor.is_ignored(os.path.join(path)):
                ignored_scan_count += 1
                # data ignored, just plot the individual scan, then continue
                save_dir = path.replace(".cv", "_cv_ignored.png")
                print(f"Ignoring {os.path.join(dir, path)}. CV scan plot generated at {save_dir}.")
                plt.figure(figsize=(10, 6))

                plt.plot(v_data, c_data, label=rf"Scan at {temperature}$^\circ$C",
                         color=temperature_to_color(temperature), marker='o', markersize=3)
                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel(f"log(Pad Current (A))")
                plt.title(rf"IV Scan at {temperature}$^\circ$C: Ignored ({sensor.name} {date.strftime('%b %d, %Y')})")
                disable_top_and_right_bounds(plt)
                plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
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
                    cutoff_v = plateau_v + 50 * np.abs(plateau_v - smallest_plateau_v)
                    mask = c_data < cutoff_v
                    if not np.any(mask):
                        # cutoff_v is too low, skip 
                        continue
                    first_below_id = np.where(mask)[0][0]
                    last_above_id = first_below_id - 1

                    # linear interpolation
                    # assuming the last_below_id is not the last data point
                    slope_inter = (c_data[first_below_id] - c_data[last_above_id]) / (
                            v_data[first_below_id] - v_data[last_above_id])
                    offset_inter = c_data[last_above_id] - slope_inter * v_data[last_above_id]
                    dep_v = (cutoff_v - offset_inter) / slope_inter
                    # sanity check
                    if dep_v < 0 or dep_v > np.max(v_data):
                        continue
                    all_dep_v.append(dep_v)

                # discard top and bottom 10% extremes
                all_dep_v = sorted(all_dep_v)
                n = len(all_dep_v)
                all_dep_v = all_dep_v[int(n * 0.1):n - int(n * 0.1)]

                all_dep_v = np.array(all_dep_v)
                dep_v = np.median(all_dep_v)
                std = np.std(all_dep_v)
                std = max(std, get_min_uncertainty(v_data, len(c_invsq_data) // 2))

                if plot:  # plot depletion distribution
                    plt.figure(figsize=(8, 6))
                    plt.hist(all_dep_v, bins=60, color='purple')
                    plt.axvline(dep_v, color="black", ls="--", label=f"Median: {dep_v:.2f} $V$")
                    plt.legend()
                    plt.ylabel('Frequency')
                    plt.xlabel('Depletion Voltage (V)')
                    plt.title(f'Depletion Voltage Distribution')
                    plt.tight_layout()
                    plt.savefig(path.replace(".cv", "_dep_distribution.png"))
                    plt.close()

                return dep_v, std, all_dep_v

            dep_v, std, all_dep_v = fit_depletion()

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

            if plot:  # C vs V
                plt.figure(figsize=(10, 6))
                plt.plot(v_data, c_data, label=rf"{temperature}$^\circ$C {frequency}Hz",
                         color=temperature_to_color(temperature), marker='o', markersize=2)

                disable_top_and_right_bounds(plt)
                plot_median_and_distribution_v_lines()

                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel("Capacitance (F)")
                plt.title(rf"Capacitance vs. Voltage at {temperature}$^\circ$C {frequency}Hz ({sensor.name})")
                plt.legend()
                plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
                plt.tight_layout()
                plt.savefig(path.replace(".cv", "_cv.png"))
                plt.close()
            if plot:  # C^-2 vs V
                plt.figure(figsize=(10, 6))
                plt.plot(v_data, c_invsq_data, label=rf"{temperature}$^\circ$C {frequency}Hz",
                         color=temperature_to_color(temperature), marker='o', markersize=2)

                disable_top_and_right_bounds(plt)
                plot_median_and_distribution_v_lines()

                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel("Inverse-squared Capacitance ($F^{-2}$)")
                plt.title(
                    rf"Inverse-squared Capacitance vs. Voltage at {temperature}$^\circ$C {frequency}Hz ({sensor.name})")
                plt.legend()
                plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
                plt.tight_layout()
                plt.savefig(path.replace(".cv", "_c-2v.png"))
                plt.close()
            if plot:  # Derivative of C^-2 w.r.t V, d(C^-2)/dV
                plt.figure(figsize=(10, 6))
                plt.plot(v_data[:-1], c_invsq_diff_data / v_diff_data, label=rf"{temperature}$^\circ$C {frequency}Hz",
                         color=temperature_to_color(temperature), marker='o', markersize=2)

                disable_top_and_right_bounds(plt)
                plot_median_and_distribution_v_lines()

                plt.xlabel("Reverse-bias Voltage (V)")
                plt.ylabel("$d C^{-2}/dV$ ($F^{-2}/V$)")
                plt.title(rf"$d C^{-2}/dV$ vs. $V$ at {temperature}$^\circ$C {frequency}Hz ({sensor.name})")
                plt.legend()
                plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
                plt.tight_layout()
                plt.savefig(path.replace(".cv", "_c-2v_derivative.png"))
                plt.close()

        # now, calculate a weighted sum for dep_v, and write to sensor
        if total_scan_count != 0:
            all_cv_info = np.array(all_cv_info)
            weighted_mean, weighted_sigma = calculate_weighted_mean(all_cv_info[:, 0], all_cv_info[:, 1])
            sensor.depletion_v = weighted_mean


def main(ARGS):
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
    load_sensor_config(DATABASE_DIR, sensors, load_iv=(not ARGS.iv or (ARGS.iv and not ARGS.overwrite)),
                       load_cv=(not ARGS.cv or (ARGS.cv and not ARGS.overwrite)))
    load_data_config(DATABASE_DIR, sensors)

    if ARGS.file is None:  # analyze sensors directly
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
                    analyze_sensor_iv(sensor, curr_type=ARGS.curr_type, var=ARGS.var, plot=True)

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
                analyze_sensor_iv(sensor, curr_type=ARGS.curr_type, var=ARGS.var, plot=True, file_mode=True, path=file_path)
            elif file_path.endswith(".cv"):
                analyze_file_cv(sensor, file_path, plot=True)

    return 0


if __name__ == "__main__":
    ARGS = parse_args()
    main()
    f()
