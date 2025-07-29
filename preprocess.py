from utils import * 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import math #, pywt
from scipy.signal import butter, filtfilt

"""
def butter_highpass_filter(data, cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

from scipy.interpolate import UnivariateSpline

def spline_denoise(x, y, s=1):
    spline = UnivariateSpline(x, y, s=s)
    return spline(x)

# path = "data/ivcvscans/AC_W3096/Dec52024/rd_-40.txt"
# path = "data/ivcvscans/AC_W3096/Dec52024/rd_60.txt"
# path = "data/ivcvscans/AC_W3096/Dec52024/ru_20.txt"
path = "data/ivcvscans/AC_W3096/Dec52024/rd_-20.txt"
temperature, date, data, humi, ramp_type = parse_file_iv(path)

curr_type = "pad"

xs = data["voltage"]
ys = data[curr_type]
ys_log10 = np.log10(ys)

# some voltage data is negative. normalize to absolute value 
if np.median(data["voltage"]) < 0:
    data["voltage"] = -data["voltage"]
xs = data["voltage"]
if np.median(data[curr_type]) < 0:
    data[curr_type] = -data[curr_type]
ys = data[curr_type]
ys_log10 = np.log10(ys)

nan_mask = ~np.isnan(ys_log10)
xs = xs[nan_mask]
ys_log10 = ys_log10[nan_mask]
    
def sg_denoise(xs: np.ndarray, ys: np.ndarray, window_size: int, order):
    # output uniformly w/ interval 0.5V
    assert window_size & 1, "Window size must be odd"
    assert len(xs) == len(ys)
    # for each voltage point:
    # slide window, fit a polynomial, then evaluate at this voltage point
    # to get denoised current value
    interval = 1
    half_window = window_size // 2 
    
    max_volt = np.max(xs)
    nan_mask = ~np.isnan(ys)
    xs = xs[nan_mask]
    ys = ys[nan_mask]
    
    volt_grid = np.arange(0, max_volt, interval)
    denoised_cur = np.zeros_like(volt_grid)
    
    cur_volt = 0
    min_volt_above_idx = 0 # use min above not max below, because most scans start at 0+epsilon volts
    for i in range(len(volt_grid)):
        while cur_volt > xs[min_volt_above_idx]:
            min_volt_above_idx += 1
        
        range_left = max(0, min_volt_above_idx - half_window)
        range_right = min(len(xs), min_volt_above_idx + half_window +1)
        
        v_win = xs[range_left:range_right]
        c_win = ys[range_left:range_right]
        
        params = np.polyfit(v_win, c_win, order)
        cur_curr = np.polyval(params, cur_volt)
        denoised_cur[i] = cur_curr
        
        cur_volt += interval

    return volt_grid, denoised_cur

def wavelet_denoise(xs: np.ndarray, ys:np.ndarray):
    # pywt.wavedec only accepts 1D array.
    # linear interpolate ys first.
    assert len(xs) == len(ys)
    assert len(xs) >= 2, "Cannot interpolate"
    
    interval = 0.5
    
    def interpolate(xs, ys, interval):
        # linear interpolation.
        # if encounter np.nan data pt, use the closest non-nan pt, 
        # equivalently, remove all nan data pts before interpolation
        
        nan_mask = ~np.isnan(ys)
        xs = xs[nan_mask]
        ys = ys[nan_mask]
        max_volt = np.max(xs)
        volt_grid = np.arange(0, max_volt, interval)
        interp_cur = np.zeros_like(volt_grid)
        
        cur_volt = 0
        min_volt_above_idx = 0 # use min above not max below, because most scans start at 0+epsilon volts
        for i in range(len(volt_grid)):
            while cur_volt > xs[min_volt_above_idx]:
                min_volt_above_idx += 1
            
            if min_volt_above_idx == 0:
                # use pts at idx 0 and 1
                slope_inter = (ys[1]-ys[0]) / (xs[1]-xs[0])
                offset_inter = ys[0] - slope_inter * xs[0]
                interp_cur[i] = slope_inter * cur_volt + offset_inter 
            else:
                max_volt_below_idx = min_volt_above_idx - 1
                # fit a line. 
                slope_inter = (ys[max_volt_below_idx] - ys[min_volt_above_idx]) / (xs[max_volt_below_idx] - xs[min_volt_above_idx])
                offset_inter = ys[max_volt_below_idx] - slope_inter * xs[max_volt_below_idx]
                interp_cur[i] = slope_inter * cur_volt + offset_inter 
            
            cur_volt += interval

        return volt_grid, interp_cur
    
    # return interpolate(xs, ys, interval)
    volt_grid, interp_cur = interpolate(xs, ys, interval)
    # now interp_cur stores interpolated current 
    
    wavelet = 'coif5'
    params = pywt.wavedec(interp_cur, wavelet=wavelet, level=5)
    thresholded = [params[0]] + [pywt.threshold(c, value=1, substitute=1, mode='less') for c in params[1:]]
    denoised_cur = pywt.waverec(thresholded, wavelet=wavelet)
    
    denoised_cur = denoised_cur[:len(volt_grid)]
    return volt_grid, denoised_cur

def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))

def curve(x, a, b, c, d, e, f):
    return (a*x + b) * (1 - sigmoid(x, c, d)) + (e*x + f) * sigmoid(x, c, d)

def parametrize(xs: np.ndarray, ys:np.ndarray):
    start = 12
    p0 = [0.005, 2, 2, 150, 5, 1]
    bounds = (
        [0, 0, 0.1, 100, 5, -np.inf],
        [0.01, 2000, 20, 300, np.inf, np.inf]
    )
    nan_mask = ~np.isnan(ys)
    xs = xs[nan_mask]
    ys = ys[nan_mask]
    popt, _ = curve_fit(curve, xdata=xs[start:], ydata=ys[start:], bounds=bounds, p0=p0)
    return popt 

def get_volt_grid(xs: np.ndarray, interval):
    max_volt = np.max(xs)
    volt_grid = np.arange(0, max_volt, interval)
    return volt_grid 

plt.figure(figsize=(10, 6))
# plot the scan itself
plt.plot(xs, ys_log10, label=rf"Scan at {temperature}$^\circ$C", color=temperature_to_color(temperature), marker='o', markersize=3)

# plot the sg-denoised curve 
order = 1
window_size = 3
volt_grid, denoised_cur = sg_denoise(xs, ys_log10, window_size, order)
# plt.plot(volt_grid, denoised_cur, label=rf"SG Denoised Scan (Window size {window_size}, Order {order})", markersize=3, linestyle='--', color='blue')

# plot wavelet denoise
volt_grid, denoised_cur = wavelet_denoise(xs, ys_log10)
# plt.plot(volt_grid, denoised_cur, label=rf"Wavelet Denoised Scan", markersize=3, linestyle='--', color='black')

# plot paramterization
params = parametrize(xs, ys_log10)
print(params)
volt_grid = get_volt_grid(xs, 2)
# volt_grid = np.linspace(0, 300, 300)
# plt.plot(volt_grid, curve(volt_grid, *params), label=rf"Parametrized Curve", linestyle='--', marker='o', markersize=3, color='black')

plt.plot(xs, spline_denoise(xs, ys_log10, 1))

plt.legend()
plt.xlabel("Reverse-bias Voltage (V)")
plt.ylabel(f"log(Pad Current (A))")
# plt.title(rf"IV Scan at {temperature}$^\circ$C: Unable to Find Breakdown V ({sensor.name} {date.strftime("%b %d, %Y")})")
disable_top_and_right_bounds(plt)
plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
plt.tight_layout()
# plt.savefig(save_dir)
plt.show()
plt.close()"""

def interpolate_nonan(xs: np.ndarray, ys: np.ndarray, interval: float):
    """
    Interpolates a curve. Ignores NaNs and Infs in y values.
    
    Parameters
    ----------
    xs : 1D-array
        x data
    ys : 1D-array
        y data
    interval : float
        Common difference of the new x sequence.
        
    Returns
    -------
    volt_grid : 1D-array
        An Arithmetic Sequence with common difference interval.
    interp_cur : 1D-array 
        Linearly interpolated y values at new x values.
    """
    nan_mask = np.isnan(ys) | np.isinf(ys)
    xs = xs[~nan_mask]
    ys = ys[~nan_mask]
    max_volt = np.max(xs)
    volt_grid = np.arange(0, max_volt, interval)
    interp_cur = np.zeros_like(volt_grid)
    
    cur_volt = 0
    min_volt_above_idx = 0 # use min above not max below, because most scans start at 0+epsilon volts
    for i in range(len(volt_grid)):
        while cur_volt > xs[min_volt_above_idx]:
            min_volt_above_idx += 1
        
        if min_volt_above_idx == 0:
            # use pts at idx 0 and 1
            slope_inter = (ys[1]-ys[0]) / (xs[1]-xs[0])
            offset_inter = ys[0] - slope_inter * xs[0]
            interp_cur[i] = slope_inter * cur_volt + offset_inter 
        else:
            max_volt_below_idx = min_volt_above_idx - 1
            # fit a line. 
            slope_inter = (ys[max_volt_below_idx] - ys[min_volt_above_idx]) / (xs[max_volt_below_idx] - xs[min_volt_above_idx])
            offset_inter = ys[max_volt_below_idx] - slope_inter * xs[max_volt_below_idx]
            interp_cur[i] = slope_inter * cur_volt + offset_inter 
        
        cur_volt += interval

    return volt_grid, interp_cur