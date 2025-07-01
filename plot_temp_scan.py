import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
from datetime import datetime
import shutil
import random

class Sensor:
    """
    Defines a Sensor object
    """
    def __init__(self, type: str, name: str):
        """
        type - either "AC" or "DC"
        name - a string, e.g. "AC_W3096"
        """
        self.type = type
        self.name = name
        
        data_dir = [os.path.join(f"data/{name}", scan) for scan in os.listdir(f"data/{name}") if not scan.startswith(".")]
        self.data_dir = sorted(data_dir, key=lambda s: self.parse_file_name_order(s))
        
        self.depletion_v = None
    
    def parse_file_name_order(self, file_name: str):
        try:
            return datetime.strptime(file_name.split('/')[-1], '%b%d%Y') # use date
        except: 
            return file_name.removesuffix(".iv").split('_')[-1] # use no.
            
    def __str__(self):
        return self.name

def is_close(a: float, b: float, tol: float):
    # Compares two floats to see if their difference is less than tol
    return abs(abs(a) - b) < tol
   
def temperature_to_color(temperature):
    # Map temperature to a rainbow color (purple to red)
    min_temp, max_temp = -60, 120
    norm_temp = (temperature - min_temp) / (max_temp - min_temp)
    return plt.cm.rainbow(norm_temp)

def humidity_to_color(rh):
    min_rh, max_rh = 0, 36
    norm_rh = (rh - min_rh) / (max_rh - min_rh)
    return plt.cm.rainbow(norm_rh)
     
def parse_file(filepath: str):
    if filepath.endswith(".txt"):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # Extract metadata
        if 'DC' in filepath: 
            temperature = float(lines[0].split(': ')[1].split(' ')[0])
        else: 
            temperature = float(lines[0].split(':')[1].strip().replace(" C", ""))
        date = lines[2].split(':', 1)[1].strip().split(' ')[0]
        # Extract table data
        data = np.genfromtxt(lines[4:], delimiter=',', names=['voltage','pad','gr','totalCurrent'])
    
    elif filepath.endswith(".iv"):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        table_begin_idx = 0
        # extract metadata 
        for i in range(len(lines)):
            if ":temperature[C]" in lines[i]:
                temperature = float(lines[i+1])
            elif ":start" in lines[i]:
                date = lines[i+1].split(' ')[0]
            elif "- compliance [A or V]:" in lines[i]:
                compliance = float(lines[i].split(" ")[-1])
            elif "BEGIN" in lines[i]:
                table_begin_idx = i+1
                break
        # extract table data 

        data = np.genfromtxt(lines[table_begin_idx:-1], delimiter=None, names=['voltage', 'totalCurrent', 'pad'])
        
        for i in range(data.shape[0]):
            if is_close(data[i]['totalCurrent'], compliance, 1E-7):
                data = data[:i]
                break
            
    else:
        raise ValueError(f"File must be .txt or .iv, given {filepath}")
    
    return temperature, date, data

def linear(x, m, b):
    return m*x + b

def linear_fit(x_data, y_data, p0, sigmas=None):
    popt, pconv = curve_fit(linear, xdata = x_data , ydata = y_data, p0=p0, sigma=sigmas)
    perr= np.sqrt(np.diag(pconv))
    residual_squared = np.sum(np.square(y_data - linear(x_data, *popt)))
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (residual_squared / ss_tot)
    return popt, perr, np.sqrt(residual_squared), r_squared

def fit_breakdown(xs, ys, start_idx, dist_file=None, bd_thresh=0.5, plot=True):
    """
    Given a single IV scan, finds the breakdown voltage by fitting linear lines.
    
    Inputs:
    xs - voltage data
    ys - current data 
    start_idx - data before this index is ignored
    dist_file - where to save the plots
    bd_thresh - threshold for determining breakdown
    plot - whether generates and saves plots
    
    Output:
    a dictionary containing parameters of the fitted lines, breakdown voltage, 
    uncertainty of breakdown voltage, and index of breakdown voltage.
    """
    use_ransac = True
    
    # set up data structures
    results = {'random': {'color': 'black'}}
    
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
        
        lines = [] # each entry is a list [slope, offset, inlier count, bd_voltage]
        seen = set() # some dot pairs may have been selected. dedup
        index_list = range(fit_range)
        ransac_thresh = 0.05
        for _ in range(2000):
            index_pair = random.sample(index_list, 2)
            if (index_pair[0], index_pair[1]) in seen:
                continue
            else:
                seen.add((index_pair[0], index_pair[1]))
            slope = (valid_ys[index_pair[1]] - valid_ys[index_pair[0]]) / (valid_xs[index_pair[1]] - valid_xs[index_pair[0]])
            offset = valid_ys[index_pair[0]] - slope * valid_xs[index_pair[0]]
            
            pred_ys = valid_xs * slope + offset # the y's for fitted line
            is_close_map = np.abs(pred_ys - valid_ys) < ransac_thresh 
            inlier_count = np.sum(is_close_map)

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
                
            if bd_voltage < 0: # just a sanity check
                continue
            lines.append([slope, offset, inlier_count, bd_voltage])
        
        lines = np.array(lines)
        # sort according to bd_voltage
        lines = lines[np.argsort(lines[:,3])]
        # calculate uncertainty for bd_voltage
        # cutoff extreme values
        cum_weights = np.cumsum(lines[:,2])
        total_weight = cum_weights[-1]
        
        trim_ratio = 0.10
        lower = trim_ratio * total_weight
        upper = (1 - trim_ratio) * total_weight

        mask = (cum_weights >= lower) & (cum_weights <= upper)

        filtered_lines = lines[mask]
        
        mean = np.average(filtered_lines[:,3], weights=filtered_lines[:,2])
        variance = np.average((filtered_lines[:,3] - mean)**2, weights=filtered_lines[:,2])
        std = np.sqrt(variance)
        std = std + 1.01

        if plot:
            # plot breakdown distribution
            plt.figure()
            # plt.hist(vals, histtype='step', bins=80, weights=weights, density=True)
            plt.scatter(lines[~mask][:,3], lines[~mask][:,2], color='purple', label=f'Outliers (Top/Bottom {int(trim_ratio*100)}%)')
            plt.scatter(filtered_lines[:,3], filtered_lines[:,2])
            plt.legend()
            plt.ylabel('Weights')
            plt.xlabel('Breakdown Voltage')
            plt.title('Breakdown Voltage Scatter (RANSAC-Based)')
            plt.tight_layout()
            plt.savefig(dist_file)
            plt.close()
        
        # need this so the line with most inliers is at index 0
        lines = filtered_lines
        lines = lines[np.argsort(-lines[:,2])]
        
        lines = np.delete(lines, 2, axis=1) # remove inlier_count for consistent data structure
        
    else:
        # repeat 100 times to calculate uncertainty
        rand_len = 100
        lines = [] # each entry is a list [slope, offset, bd_voltage]
        fit_range = len(valid_xs) // 2
        
        # calculate linear fit for full range
        fixed_popt, _, _, _ = linear_fit(valid_xs[:fit_range], valid_ys[:fit_range], p0=[1, -100])
        liny_vals = linear(valid_xs, *fixed_popt)
        
        # calculate breakdown voltage
        for i, liny in enumerate(liny_vals):
            if abs(valid_ys[i] - liny) < bd_thresh:
                bd_voltage = valid_xs[i]
        
        lines.append([fixed_popt[0], fixed_popt[1], bd_voltage])
        
        # repeat rand_len times to use width for uncertainty calculation
        select_from = range(fit_range)
        for rand_i in range(1, rand_len):
            indices = random.sample(select_from, k=fit_range//2) # sample half the data
            fixed_popt, _, _, _ = linear_fit(valid_xs[indices], valid_ys[indices], p0=[1, -100])
            liny_vals = linear(valid_xs, *fixed_popt)

            for i, liny in enumerate(liny_vals):
                if abs(valid_ys[i] - liny) < bd_thresh:
                    bd_voltage = valid_xs[i]
                
            lines.append([fixed_popt[0], fixed_popt[1], bd_voltage])
        
        lines = np.array(lines)
        # calculate uncertainty for bd_voltage
        std = np.std(lines[:,2]) + 0.34
        
        if plot:
            # plot breakdown distribution
            plt.figure()
            plt.hist(lines[:,2], histtype='step', bins=20)
            plt.ylabel('Frequency')
            plt.xlabel('Breakdown Voltage')
            plt.title('Breakdown Voltage Distribution From Random Ranges')
            plt.tight_layout()
            plt.savefig(dist_file)
            plt.close()
        
    # saving breakdown and plotting information
    results['random']['x'] = valid_xs
    results['random']['y'] = linear(results['random']['x'], lines[0][0], lines[0][1])
    results['random']['bd'] = lines[0][2] # breakdown voltage
    results['random']['bderr'] = std # breakdown uncertainty
    results['random']['bdy'] = lines[0][0] * lines[0][2] + lines[0][1] # y-coordinate of breakdown
        
    # keep plotting information for 10 of the random fits (100 would make plot look bad)
    for rand_i in range(min(10, lines.shape[0])):
        r_name = 'random'+str(rand_i)
        results[r_name] = dict()
        results[r_name]['x'] = results['random']['x']
        results[r_name]['y'] = linear(results['random']['x'], lines[rand_i][0], lines[rand_i][1])
        results[r_name]['bd'] = lines[rand_i][2]
        results[r_name]['bderr'] = None
        results[r_name]['bdy'] = lines[rand_i][0] * lines[rand_i][2] + lines[rand_i][1]
    return results

def plot_scans(data_dir: str, curr_type: str, min_temp: float=0, bd_thresh: float=0.5, plot=True):
    """
    Given a specific folder containing scans, plots breakdown distribution 
    and linear breakdown fits for each scan.
    
    Inputs:
    data_dir - directory to the folder
    curr_type - current type, either "pad", "totalCurrent", or "gr"
    min_temp - only affects AC_W3096, minimum temperature for analysis
    bd_thresh - threshold for determining breakdown voltage
    plot - whether generates and saves plots
    
    Returns:
    temp_dict - a dict that maps ramp direction to a list of temperatures
    bdv_dict - a dict that maps ramp direction to a list of breakdown values
    bdverr_dict - a dict that maps ramp direction to a list of breakdown uncertainties 
    
    Note that these lists are "aligned". Data at the same index represents 
    coordinates of the same data point. 
    """
    file_groups = {"ru_": [], "rd_": []}
    for filename in os.listdir(data_dir):
        if not (filename.endswith(".txt") or filename.endswith(".iv")):
            continue 
        
        if "test_" in filename or "RoomTemp" in filename:
            continue # ignore test measurements
        elif "RampUp" in filename or "ru_" in filename:
            file_groups['ru_'].append(filename)
        elif "RampDown" in filename or "rd_" in filename:
            file_groups['rd_'].append(filename)
        else: 
            # either DC_W3045/Sep212023 or DC_W3058 or BNL_LGAD_XXXX
            if "W3045" in data_dir and "IV" in filename:
                file_groups['ru_'].append(filename)
            elif 'DC_W3058' in data_dir and int(filename.split("_")[0]) > 0:
                key = "ru_" if int(filename.split("_")[0]) < 11 else "rd_"
                file_groups[key].append(filename)
            elif "BNL_LGAD_" in data_dir:
                
                file_groups['rd_'].append(filename)
                
            
    # Loop through file groups and plot
    temp_dict = dict()
    bdv_dict = dict()
    bdverr_dict = dict()

    for ramp_type, files in file_groups.items():
        if files == []: continue
        
        plots = []  # To store plot data for sorting
        for filename in sorted(files):
            filepath = os.path.join(data_dir, filename)
            temp, date, data = parse_file(filepath)
            if temp < min_temp and 'AC_W3096' in data_dir: 
                continue # bad data for cold temps on this sensor
            
            if curr_type not in data.dtype.names:
                raise ValueError(f"curr_type {curr_type} not supported on scan {date} {temp}")
            
            color = temperature_to_color(temp)
            # Plot pad current and store data for later sorting
            neg_idx = data[curr_type] < 0
            
            if 'DC' in data_dir and curr_type != 'totalCurrent': 
                plots.append((abs(data['voltage'][~neg_idx]), data[curr_type][~neg_idx], temp, color))
            else: 
                plots.append((abs(data['voltage'][neg_idx]), -1*data[curr_type][neg_idx], temp, color))
        
        # Sort by temperature (numerical order)
        plots.sort(key=lambda x: float(x[2]))  # Sorting by temperature
        if plot:
            # Plot each sorted line
            plt.figure(figsize=(10, 6))
            for volt, curr, temp, color in plots:
                plt.plot(volt, curr, label=str(temp)+" C", color=color, marker='o', markersize=3)
            # Plot configuration
            plt.xlabel("Voltage (V)")
            plt.ylabel(curr_type+" Current (A)")
            plt.yscale('log')
            if curr_type == 'pad':
                title_curr_type = "Pad"
            elif curr_type == 'gr':
                title_curr_type = 'Guard Ring'
            elif curr_type == 'totalCurrent':
                title_curr_type = "Total Current"
            plt.title(f"{date} - {title_curr_type} Current Temperature Scan ({'Ramp Up' if ramp_type == 'ru_' else 'Ramp Down'})")
            plt.legend(title="Temperature", loc='best')
            plt.grid(True)
            plt.tight_layout()
            # Save the plot
            plot_filename = data_dir+"/"+f"{ramp_type.strip('_')}_plot"+curr_type+".png"
            plt.savefig(plot_filename)
            plt.close()
        if curr_type != 'pad': continue
        breakdown = dict()

        if 'W3045' in data_dir: depletion=25
        elif 'BNL_LGAD_W3076_12_13' in data_dir: depletion=10
        elif 'BNL_LGAD_W3076_9_13' in data_dir: depletion=25
        elif 'BNL_LGAD_513' in data_dir: depletion=10
        elif 'AC_W3096' in data_dir: depletion=25
        else: depletion=36
        for volt, curr, temp, color in plots:
            if temp < min_temp and 'AC_W3096' in data_dir: continue
            elif temp <= -20 and 'W3058' in data_dir and ramp_type == 'rd_' and date == '27/10/2023': continue
            elif temp == -60 and 'W3058' in data_dir and date == '27/10/2023': continue
            elif temp == -40 and 'W3058' in data_dir and ramp_type == 'rd_' and date == '26/10/2023': continue
            elif temp == -60 and 'W3058' in data_dir and date == '26/10/2023': continue
            elif temp < -20 and 'W3058' in data_dir and ramp_type == 'rd_' and date == '25/10/2023': continue
            elif temp < -20 and 'W3058' in data_dir and date == '30/10/2023': continue
            log_curr = np.log10(curr)
            # identify start point for baseline regression
            if temp == -20 and 'W3058' in data_dir and ramp_type == 'rd_' and date == '27/10/2023': temp_depletion = 66
            elif temp == -20 and 'W3058' in data_dir and ramp_type == 'rd_' and date == '30/10/2023': temp_depletion = 70
            else: temp_depletion = depletion
            for v_idx, voltage in enumerate(volt):
                if voltage >= temp_depletion:
                    base_start_idx = v_idx
                    break
            # print('performing fit for', date, ramp_type, temp)
            plot_filename = data_dir+"/"+f"{ramp_type.strip('_')}_"+str(temp)+"_breakdown.png"
            dist_file = data_dir+"/"+f"{ramp_type.strip('_')}_"+str(temp)+"_bd_dist.png"
            # fit the breakdown
            results = fit_breakdown(volt, log_curr, base_start_idx, dist_file=dist_file, bd_thresh=bd_thresh, plot=plot)
            breakdown[temp] = dict()
            for key, result in results.items():
                if key == 'random':
                    # store breakdown voltage and uncertainty for further analysis
                    breakdown[temp][key] = result['bd']
                    breakdown[temp][key+'err'] = result['bderr']
            # plot
            if plot:
                plt.figure(figsize=(10, 6))
                plt.plot(volt[1:], log_curr[1:], label=f"{temp} C", color=color, marker='o', markersize=3)
                plotted_random = False
                for key, result in results.items():
                    if key == 'random':
                        # plot actual breakdown fit
                        plt.plot(result['x'], result['y'], color=result['color'], linestyle='--', label='linear fit')
                        plt.plot(result['x'], result['y']+bd_thresh, color='brown', linestyle='-.', label='threshold')
                        plt.scatter([result['bd']], [[result['bdy']]], color=result['color'], marker='*')
                    else: # plot some of the random samples to visualize
                        if plotted_random: # plot without label
                            plt.plot(result['x'], result['y'], color='purple', linestyle=':')
                        else: # plot with label once
                            plt.plot(result['x'], result['y'], color='purple', linestyle=':', label='random samples')
                            plotted_random = True
                        plt.scatter([result['bd']], [[result['bdy']]], color='purple', marker='*')

                plt.xlabel("Voltage (V)")
                plt.ylabel("log(Pad Current (A))")
                plt.title(f"IV Scan with breakdown for {temp} degrees: {breakdown[temp]['random']:.1f} +/- {breakdown[temp]['randomerr']:.1f} V")
                valid_ylims = log_curr[1:][np.isfinite(log_curr[1:])]
                plt.ylim(np.min(valid_ylims)-.5, np.max(valid_ylims)+.5)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(plot_filename)
                plt.close()
        # get aggregate data for plotting
        temperatures, breakdown_vs, breakdown_errs = [], [], []
        for temp, bd_voltage in breakdown.items():
            temperatures.append(temp)
            breakdown_vs.append(bd_voltage['random'])
            breakdown_errs.append(bd_voltage['randomerr'])
        temperatures = np.array(temperatures)
        breakdown_vs = np.array(breakdown_vs)

        popt, perr, _, r2 = linear_fit(temperatures, breakdown_vs, [1,150], sigmas=breakdown_errs)
        #plt.scatter(temperatures, breakdown_vs, label=key, color="purple")
        if plot:
            plt.figure(figsize=(10, 6))
            plt.errorbar(temperatures, breakdown_vs, yerr=breakdown_errs, fmt='o', capsize=5, label=key, color="black")
            plt.plot(temperatures, linear(temperatures, popt[0], popt[1]), label=f'r2 value: {r2:0.2f} Best Fit: y = {popt[0]:.2f}x + { popt[1]:.2f}', color='purple', linestyle='--')
            plt.xlabel("Temperature (C)")
            plt.ylabel("Breakdown Voltage (V)")
            plt.title("Breakdown Voltage as a Function of Temperature")
            plt.legend()
            plt.tight_layout()
            plt.savefig(data_dir+"/"+f"{ramp_type.strip('_')}_fit_breakdown.png")
            plt.close()
        # return fits so that they can be plotted together
        temp_dict[ramp_type] = temperatures
        bdv_dict[ramp_type] = breakdown_vs
        bdverr_dict[ramp_type] = breakdown_errs
    return temp_dict, bdv_dict, bdverr_dict

def plot_sensor(sensor: Sensor, curr_type: str='pad', bd_thresh: float=0.5, min_temp=0, plot=True):
    # plot all scans together for a given sensor
    means = dict()
    sigmas = dict()
    # retrieve and plot the raw data
    ramp_types, ramp_temps, ramp_bdvs, ramp_bdsigs = [], [], [], []
    for dir in tqdm(sensor.data_dir): 
        # Note: plt only has one buffer that's cleared whenever 
        # plt.figure() is called! 
        temps, bdv, bdverr = plot_scans(dir, curr_type=curr_type, bd_thresh=bd_thresh, min_temp=min_temp, plot=plot)
        for ramp_type, ramp_temp in temps.items():
            ramp_temps.append(ramp_temp)
            ramp_types.append(ramp_type)
            ramp_bdvs.append(bdv[ramp_type])
            ramp_bdsigs.append(bdverr[ramp_type])
            # add each individual measurement to the dictionaries
            for j, temp in enumerate(ramp_temp):
                means.setdefault(temp, []).append(bdv[ramp_type][j]) # setdefault: if DNE, create
                sigmas.setdefault(temp, []).append(bdverr[ramp_type][j])

    dirs = [dir for dir in sensor.data_dir for _ in range(2)]
    # add scan to plot
    if plot:
        plt.figure(figsize=(10, 6))
        for dir, ramp_type, ramp_temp, ramp_bdv in zip(dirs, ramp_types, ramp_temps, ramp_bdvs):
            plt.plot(ramp_temp, ramp_bdv, marker='o', markersize=3, label=f"{dir.split("/")[-1]} {"(Ramp Up)" if ramp_type == 'ru_' else "(Ramp Down)"}")
                
        plt.xlabel("Temperature (C)")
        plt.ylabel("Breakdown Voltage (V)")
        plt.title(f"{sensor.name} Breakdown Voltage as a Function of Temperature")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"data/all_breakdown_fits_{sensor.name}.png")
        plt.close()
    
        # plot breakdown voltages over time (as a function of scan number)
        plt.figure(figsize=(10, 6))
        for temp, bdvs in means.items():
            scan_idx = np.arange(len(bdvs))
            if sensor.name == 'DC_W3058':
                # ignore missing data
                if temp == -20: scan_idx = np.array([0,1,2,3,4,6,7,8,9,10,11,12,13,14,15])
                elif temp == -40: scan_idx = np.array([0,2,4,8,9,10,11,12,13,14,15])
                elif temp == -60: scan_idx = np.array([0,8,9,10,11,12,13,14,15])
                elif temp == 80: scan_idx = np.array([0,1,2,3,5,6,7,8,9,10,11,12,13,14,15])
                elif temp == 40: scan_idx = np.array([0,1,2,3,4,5,6,7,9,10,11,12,13,14,15])
            plt.plot(scan_idx, bdvs, marker='o', color=temperature_to_color(temp), label=f"{temp} C")
        plt.xlabel("Scan Number")
        plt.ylabel("Breakdown Voltage (V)")
        plt.title(f"{sensor.name} Breakdown Voltage by Temperature over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"data/breakdown_over_time_{sensor.name}.png")
        plt.close()
    
    # plot the overall fit
    # calculate weighted mean and its uncertainty across measurements
    avg_sigma = 0
    num_sigmas = 0
    fit_temps, fit_bds, fit_sigs = [], [], []
    for temp, sigma in sigmas.items():
        sigma = np.array(sigma)
        fit_temps.append(temp)
        fit_sigs.append(np.sqrt(1/np.sum(1/(sigma**2))))
        fit_bds.append(np.sum(np.array(means[temp])/(sigma**2))*(fit_sigs[-1]**2))
        print(fit_bds[-1], fit_sigs[-1])
        avg_sigma += fit_sigs[-1]
        num_sigmas += 1
    if num_sigmas > 0: avg_sigma /= num_sigmas
    print(f"{sensor.name} AVERAGE UNCERTAINTY", avg_sigma)
    # calculate and plot overall fit
    fit_temps = np.array(fit_temps)
    fit_bds = np.array(fit_bds)
    fit_sigs = np.array(fit_sigs)
    popt, perr, _, r2 = linear_fit(fit_temps, fit_bds, [1,150], sigmas=fit_sigs)
    slope_err = perr[0]
    print(f"{sensor.name} SLOPE UNCERTAINTY", slope_err)
    if plot:
        plt.figure(figsize=(10, 6))
        plt.errorbar(fit_temps, fit_bds, yerr=fit_sigs, fmt='o', capsize=5, label=f'{sensor.name} data')
        plt.plot(fit_temps, linear(fit_temps, popt[0], popt[1]), label=f'r2 value: {r2:0.2f} Best Fit: y = {popt[0]:.2f}x + { popt[1]:.2f}', linestyle='--')
        plt.xlabel("Temperature (C)")
        plt.ylabel("Breakdown Voltage (V)")
        plt.title(f"{sensor.name} Breakdown Voltage as a Function of Temperature")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"data/averaged_fit_{sensor.name}.png")
        plt.close()

    return slope_err, avg_sigma

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
    min_slope_err = float('inf')
    min_threshold = float('inf')
    for thresh in np.linspace(max_bd_thresh, 0, 10, endpoint=False):
        slope_err, avg_err = plot_sensor(sensor, min_temp=min_temp, bd_thresh=thresh, plot=False)
        if slope_err < min_slope_err:
            min_slope_err = slope_err
            min_threshold = thresh
    print(f"Optimal bd_threshold for {sensor.name} is: {min_threshold:.4f}")
    print(f"Achieving slope uncertainty of: {min_slope_err:.4f}")
    return min_threshold

def main():
    sensors = [
        Sensor("AC", "AC_W3096"), Sensor("DC", "DC_W3045"), 
        Sensor("DC", "DC_W3058"), Sensor("DC", "BNL_LGAD_513"),
        Sensor("DC", "BNL_LGAD_W3076_9_13"), 
        Sensor("DC", "BNL_LGAD_W3076_12_13"),
    ]
    # sensors = [Sensor("DC", "BNL_LGAD_W3076_12_13")]
    # test_data_dirs = ["AC_W3096/Dec112024", "DC_W3058/Nov012023", "DC_W3045/Sep252023"]
    # find_threshold(Sensor("DC", "BNL_LGAD_W3076_12_13"))
    
    thresholds = {"AC_W3096": 0.5, "DC_W3058": 0.4, "DC_W3045": 0.4,
                  "BNL_LGAD_513": 0.6, 
                  "BNL_LGAD_W3076_9_13": 0.6,
                  "BNL_LGAD_W3076_12_13": 0.5}
    
    for sensor in sensors:
        plot_sensor(sensor, 'pad', bd_thresh=thresholds.get(sensor.name), min_temp=0, plot=True)
    plot_humidity_scans("data/AC_W3096/Dec102024", thresholds["AC_W3096"])

    return 0

if __name__ == "__main__":
    main()
    pass