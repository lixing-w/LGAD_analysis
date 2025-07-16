import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime
"""
A code lib for miscellaneous constants and functions:

DATABASE_DIR - directory for data base

is_close - Checks if two floats' abs value are close 
temperature_to_color - Maps temperature to a color
humidity_to_color - Maps humidity to a color 
parse_file - Parses .txt, .iv, or .cv files
linear - A linear function
linear_fit - Fits data with a linear line
determine_spacing - Tells the common interval of data in Arithmetic Progression
"""

# DATABASE_DIR = "./data/ivcvscans"
# DATABASE_DIR = "./data/IVCV_UNIGE_STRANGE_FEATURES/W3045"
# DATABASE_DIR = "./data/IVCV_UNIGE_STRANGE_FEATURES/Batch_3076_destructiontests"
# DATABASE_DIR = "./data/IVCV_UNIGE_STRANGE_FEATURES/ClimateTests_3045C"
DATABASE_DIR = "./data/IVCV_UNIGE_STRANGE_FEATURES/destruction_deionization"

def is_close(a: float, b: float, tol: float):
    # Checks if two floats' abs value are close 
    return abs(abs(a) - abs(b)) < tol

def temperature_to_color(temperature: float):
    # Map temperature to a rainbow color (purple to red)
    min_temp, max_temp = -60, 120
    norm_temp = (temperature - min_temp) / (max_temp - min_temp)
    return plt.cm.rainbow(norm_temp)

def humidity_to_color(rh: float):
    min_rh, max_rh = 0, 36
    norm_rh = (rh - min_rh) / (max_rh - min_rh)
    return plt.cm.rainbow(norm_rh)

def parse_file_iv(filepath: str):
    """
    Retrieves temperature, date, data, humidity, and ramp_type from a IV scan
    
    Returns:
    temperature - the temperature of measurement
    date - date of the measurement
    data - a numpy array of data
    humidity - the humidity
    ramp_type - either 1 (ramp_up), -1 (ramp_down), or 0 (not given)
    """
    ramp_type = 0 
    
    if filepath.endswith(".txt"):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # Extract metadata
        temperature = float(lines[0].split(' ')[1])
        humidity = float(lines[1].split(' ')[1])
        date_str = lines[2].split(' ')[1]
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except:
            date = datetime.strptime(date_str, '%d/%m/%Y')
        # Extract table data
        data = np.genfromtxt(lines[4:], delimiter=',', names=['voltage','pad','gr','total'])

        if "RampUp" in filepath: ramp_type = 1
        elif "RampDown" in filepath: ramp_type = -1 
        elif "ru" in filepath: ramp_type = 1 
        elif "rd" in filepath: ramp_type = -1
        elif "Up" in lines[0]: ramp_type = 1 
        elif "Down" in lines[0]: ramp_type = -1
        
    elif filepath.endswith(".iv"):
        # ramp_type is usually not given for any .iv files
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        table_begin_idx = 0
        humidity = float('nan') # humidities are not included in any .iv 
        # extract metadata 
        for i in range(len(lines)):
            if ":temperature[C]" in lines[i]:
                temperature = float(lines[i+1])
            elif ":start" in lines[i]:
                date_str = lines[i+1].split(' ')[0]
                date = datetime.strptime(date_str, "%d/%m/%Y")
            elif "- compliance [A or V]:" in lines[i]:
                compliance = float(lines[i].split(" ")[-1])
            elif "BEGIN" in lines[i]:
                table_begin_idx = i+1
                break
            
        # extract table data 
        data = np.genfromtxt(lines[table_begin_idx:-1], delimiter=None, names=['voltage', 'total', 'pad'])
        
        for i in range(data.shape[0]):
            if is_close(data[i]['total'], compliance, 1E-8):
                data = data[:i]
                break
    
    else:
        raise ValueError(f"File must be .txt or .iv, given {filepath}")
    
    return temperature, date, data, humidity, ramp_type

def parse_file_iv_basic_info(filepath: str):
    """
    Retrives temperature, date, and humidity from a IV scan
    """
    ramp_type = 0
    if filepath.endswith(".txt"):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # Extract metadata
        temperature = float(lines[0].split(' ')[1])
        humidity = float(lines[1].split(' ')[1])
        date_str = lines[2].split(' ')[1]
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except:
            date = datetime.strptime(date_str, '%d/%m/%Y')
        if "RampUp" in filepath: ramp_type = 1
        elif "RampDown" in filepath: ramp_type = -1 
        elif "ru" in filepath: ramp_type = 1 
        elif "rd" in filepath: ramp_type = -1
        elif "Up" in lines[0]: ramp_type = 1 
        elif "Down" in lines[0]: ramp_type = -1
        
    elif filepath.endswith(".iv"):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        humidity = float('nan') # humidities are not included in any .iv 
        # extract metadata 
        for i in range(len(lines)):
            if ":temperature[C]" in lines[i]:
                temperature = float(lines[i+1])
            elif ":start" in lines[i]:
                date_str = lines[i+1].split(' ')[0]
                date = datetime.strptime(date_str, "%d/%m/%Y")
            elif "BEGIN" in lines[i]:
                break

    else:
        raise ValueError(f"File must be .txt or .iv, given {filepath}")
    
    return temperature, date, humidity, ramp_type

def parse_file_cv_basic_info(filepath: str):
    if filepath.endswith(".cv"):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        humidity = float('nan') # humidities are not included in any .iv 
        # extract metadata 
        for i in range(len(lines)):
            if ":temperature[C]" in lines[i]:
                temperature = float(lines[i+1])
            elif ":start" in lines[i]:
                date_str = lines[i+1].split(' ')[0]
                date = datetime.strptime(date_str, "%d/%m/%Y")
            elif "- compliance [A or V]:" in lines[i]:
                compliance = float(lines[i].split(" ")[-1])
            elif "- Frequency:" in lines[i]:
                frequency = float(lines[i].split(' ')[-2])
            elif "BEGIN" in lines[i]:
                break
        # extract table data 
        
    else:
        raise ValueError(f"File must be .cv, given {filepath}")
    
    return temperature, date, frequency

def parse_file_cv(filepath: str):
    if filepath.endswith(".cv"):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        table_begin_idx = 0
        humidity = float('nan') # humidities are not included in any .iv 
        # extract metadata 
        for i in range(len(lines)):
            if ":temperature[C]" in lines[i]:
                temperature = float(lines[i+1])
            elif ":start" in lines[i]:
                date_str = lines[i+1].split(' ')[0]
                date = datetime.strptime(date_str, "%d/%m/%Y")
            elif "- compliance [A or V]:" in lines[i]:
                compliance = float(lines[i].split(" ")[-1])
            elif "- Frequency:" in lines[i]:
                frequency = float(lines[i].split(' ')[-2])
            elif "BEGIN" in lines[i]:
                table_begin_idx = i+1
                break
        # extract table data 

        data = np.genfromtxt(lines[table_begin_idx:-1], delimiter=None, names=['voltage', 'capacitance', 'conductivity', 'bias', 'current_power_supply'])
        
        for i in range(data.shape[0]):
            if is_close(data[i]['current_power_supply'], compliance, 1E-8):
                data = data[:i]
                break
        
    else:
        raise ValueError(f"File must be .cv, given {filepath}")
    
    return temperature, date, data, frequency

def linear(x, m, b):
    return m*x + b

def linear_fit(x_data: np.ndarray, y_data: np.ndarray, p0=None, sigmas=None):
    popt, pconv = curve_fit(linear, xdata = x_data , ydata = y_data, p0=p0, sigma=sigmas)
    perr= np.sqrt(np.diag(pconv))
    residual_squared = np.sum(np.square(y_data - linear(x_data, *popt)))
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (residual_squared / ss_tot)
    return popt, perr, np.sqrt(residual_squared), r_squared

def determine_spacing(xs: np.ndarray):
    diff = np.diff(xs)
    space = np.abs(diff[:10].mean())
    return round(space)