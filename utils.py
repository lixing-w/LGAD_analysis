import os
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime
from tqdm import tqdm

"""
A code lib for miscellaneous constants and functions:

DATABASE_DIR - directory for data base

is_close - Checks if two floats' abs values are close 
temperature_to_color - Maps temperature to a color
humidity_to_color - Maps humidity to a color 
parse_file_iv - Parses .txt, or .iv files, returns basic info and data
parse_file_iv_basic_info - Parses .txt, or .iv files, returns basic info
parse_file_cv - Parses .cv files, returns basic info and data 
parse_file_cv_basic_info - Parses .cv files, returns basic info
linear - A linear function
linear_fit - Fits data with a linear line
determine_spacing - Tells the avg interval of data
"""

# Specify database directory. Note: must not end in '/' or '\'
DATABASE_DIR = "./data/ivcvscans"
# DATABASE_DIR = "./data/IVCV_UNIGE_STRANGE_FEATURES/W3045"
# DATABASE_DIR = "./data/IVCV_UNIGE_STRANGE_FEATURES/Batch_3076_destructiontests"
# DATABASE_DIR = "./data/IVCV_UNIGE_STRANGE_FEATURES/ClimateTests_3045C"
# DATABASE_DIR = "./data/IVCV_UNIGE_STRANGE_FEATURES/destruction_deionization"

print(f"Using Database at {DATABASE_DIR}")

class Sensor:
    """
    Defines a Sensor object.
    
    Fields
    ------
    self.name : str
        A unique identifier
    self.type : str 
        Either AC or DC
    self.depletion_v : float
    self.sensor_dir : str 
        Directory to sensor's entire data
    self.data_dirs : List[str] 
        A list of directories, each contain just scans
    """
    def __init__(self, name: str):
        """
        name - a string, e.g. "AC_W3096"
        """
        self.name = name
        self.type = None # either AC or DC. optional
        self.depletion_v = None # should be a float
        self.iv_scan_data = list() # a list [[temp, avg_volt, avg_humi], ...]
        self.iv_scan_line = list() # a list [slope, offset, avg_sigma]
        self.bd_thresh = None
        self.cv_scan_data = list() # a list [[c_after_dep, freq, temp], ...]
        self.ignore_cond = list() # a list [{date_range:.., temp_range:.., file:..., regex:..}]
        self.scan_conf = list() # a list [{date_range:.., temp_range:.., file:..., regex:.., set_params:..}]
        
        # some sensor folders contain secondary directories, each contains 
        # measurements from a specific date.
        # other sensor folders just contain 1 or 2 .iv scans, and/or .cv scans
        # directly. in this case, self.data_dir is just the sensor dir.
        
        # directories are relative to project root
        self.sensor_dir = os.path.join(DATABASE_DIR, self.name)
        def has_subdirectories(folder_path):
            for item in os.listdir(folder_path):
                if os.path.isdir(os.path.join(folder_path, item)):
                    return True
            return False
        
        if has_subdirectories(self.sensor_dir):
            self.data_dirs = sorted(
                [os.path.join(self.sensor_dir, scan) for scan in os.listdir(self.sensor_dir) if not scan.startswith(".") and not scan.endswith(".png")],
                key=lambda s: self.sort_file_name_order(s)
                )
        else:
            self.data_dirs = [self.sensor_dir]
    
    def sort_file_name_order(self, file_name: str):
        try:
            # if sensor dir contains secondary dirs, they are formatted 
            # as date of measurements
            return datetime.strptime(file_name.split('/')[-1], '%b%d%Y') 
        except: 
            raise ValueError(f"Unsupported folder format: {file_name}")

    def add_cond(self, conf_lst, date_range=None, temp_range=None, file=None, regex=None, ramp_type=None, set_params=None):
        """
        Adds a condition to a conf_lst. 
        
        Inputs:
        date_range - either a datetime, or a tuple of datetime to indicate range (inclusive). If a single datetime, then 
                     scans need to be on that DAY
        temp_range - either a float, a int, or a tuple of floats or ints to indicate range (inclusive). If a single float, then
                     scans need to have exactly that number
        file - the relative path to a particular scan
        regex - a single regex to match scan file name
        ramp_type - either -1 (ramp down) or 1 (ramp up) or 0 (not given)
        set_params - a dict() specifying additional params to use (used in SPEC SET)
        """
        new_cond = dict()
        if date_range is not None:
            if isinstance(date_range, tuple):
                new_cond['date_range'] = date_range
            elif isinstance(date_range, datetime):
                new_cond['date_range'] = (date_range,)
            else:
                raise ValueError("date_range must be a datetime, or a tuple of two datetime or None")
        
        if temp_range is not None:
            if isinstance(temp_range, tuple):
                new_cond['temp_range'] = temp_range
            elif isinstance(temp_range, float) or isinstance(temp_range, int):
                new_cond['temp_range'] = (temp_range,)
            else:
                raise ValueError("temp_range must be a float, int, or a tuple of two floats or ints or None")

        if file is not None:
            new_cond['file'] = file
        
        if regex is not None:
            new_cond['regex'] = regex 
        
        if ramp_type is not None:
            new_cond['ramp_type'] = ramp_type
        
        if set_params is not None:
            new_cond['set_params'] = set_params
        
        if len(new_cond):
            conf_lst.append(new_cond)
            
        return 
        
    def is_specified_by(self, conf_lst, dir):
        """
        Given the relative path to a particular scan, returns the cond if 
        this scan is specified in the conf_lst, None otherwise
        """
        
        if dir.endswith(".iv") or dir.endswith(".txt"):
            temperature, date, humidity, rt, duration = parse_file_iv_basic_info(dir)
            frequency = None
        elif dir.endswith(".cv"):
            temperature, date, frequency = parse_file_cv_basic_info(dir)
            humidity = None 
            rt = None 
        else:
            raise ValueError(f"Unknown file format, given {dir}")
        
        for cond in conf_lst:
            # cond is a dict: 'date_range'->tuple, 'temp_range'->tuple, 'regex'->regex 
            
            if 'date_range' in cond: # we only compare by DAY!
                date_range = cond['date_range']
                if len(date_range) == 1 and date.date() != date_range[0].date(): 
                    continue # file not specified by this cond
                if len(date_range) == 2:
                    if date_range[0] is None and date_range[1] is not None and date.date() > date_range[1].date():
                        continue 
                    if date_range[0] is not None and date.date() < date_range[0].date() and date_range[1] is None:
                        continue
                    if (date_range[0] is not None and date_range[1] is not None and (date.date() > date_range[1].date() or date.date() < date_range[0].date())):
                        continue # file not specified by this cond

            if 'temp_range' in cond:
                temp_range = cond['temp_range']
                if len(temp_range) == 1 and temperature != temp_range[0]:
                    continue 
                if len(temp_range) == 2:
                    if temp_range[0] is None and temp_range[1] is not None and temperature > temp_range[1]:
                        continue 
                    if temp_range[0] is not None and temperature < temp_range[0] and temp_range[1] is None:
                        continue
                    if (temp_range[0] is not None and temp_range[1] is not None and (temperature > temp_range[1] or temperature < temp_range[0])):
                        continue 
            
            if 'file' in cond:
                if cond['file'] != dir:
                    continue 
                
            if 'regex' in cond:
                regex = cond['regex']
                file_name = dir.split(os.sep)[-1]
                if not re.search(regex, file_name):
                    continue # did not match
            
            if 'ramp_type' in cond:
                ramp_type = cond['ramp_type']
                if ramp_type != rt:
                    continue # did not match
                
            return cond 
        
        return None
           
    def add_ignore(self, date_range=None, temp_range=None, file=None, regex=None, ramp_type=None):
        """
        Adds a condition specifying scan(s) to ignore.
        """
        self.add_cond(self.ignore_cond, date_range=date_range, temp_range=temp_range, file=file, regex=regex, ramp_type=ramp_type)
        
    def is_ignored(self, dir):
        """
        Given the relative path to a particular scan, returns True if 
        this scan is specified in the self.ignore_cond
        """
        return (self.is_specified_by(self.ignore_cond, dir) is not None)
    
    def add_scan_conf(self, date_range=None, temp_range=None, file=None, regex=None, ramp_type=None, set_params=None):
        """
        Adds a condition specifying scan(s) for further configuration.
        """
        self.add_cond(self.scan_conf, date_range=date_range, temp_range=temp_range, file=file, regex=regex, ramp_type=ramp_type, set_params=set_params)

    def query_conf(self, dir):
        """
        Given the relative path to a particular scan, returns related 
        configuration specified by SPEC..SET.., or None if there's no 
        further configuration.
        """
        cond = self.is_specified_by(self.scan_conf, dir)
        if cond is None:
            return None 
        return cond['set_params']
    
def list_sensors():
    """
    Initializes and returns a list of Sensor objects in DATABASE_DIR
    
    Returns
    -------
    sensors : List[Sensor]
        A list of sensor objects.
    """
    path = DATABASE_DIR
    folders = [p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]
    folders = sorted(folders)
    sensors = [Sensor(folder) for folder in folders]
    sensors = sorted(sensors, key=lambda s:s.name)
    return sensors 

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

def disable_top_and_right_bounds(plt):
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return 
    
def load_data_config(DATABASE_DIR: str, sensors: list[Sensor]):
    """
    Loads scans that we ignore.
    
    Each line in the data_config.txt should be:
    - an expression starting with command: 'DR' (drop record) 'SPEC SET' (specify then set properties)
    - each expression should have the following format:
      [command] N:[Name] D:[Date]|[Date~Date]|[Date~]|[~Date] T:[Temp]|[Temp~Temp]|[Temp~]|[~Temp] F:[dir] R:[regex] RT:[1|-1|0]
    - all ranges are inclusive
    - a scan is ignored if it satisfies EVERY condition of SOME expression
    
    Parameters
    ----------
    DATABASE_DIR : str 
        The directory to database.
    sensors : list[Sensor]
        A list of sensors to which the config will load. This list is usually 
        the one returned by list_sensors().
        
    Example Config Commands
    -----------------------
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
        
    if not os.path.exists(os.path.join(DATABASE_DIR, "data_config.txt")):
        with open(os.path.join(DATABASE_DIR, "data_config.txt"), "w") as f:
            return 
        
    with open(os.path.join(DATABASE_DIR, "data_config.txt"), "r") as f:
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
                if "MAX" in info:
                    set_params['MAX'] = float(info['MAX'])
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
                if file_info["N"] not in name_to_sensor:
                    # not interested, skip
                    continue
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
                if file_info["N"] not in name_to_sensor:
                    # not interested, skip
                    continue
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
            if name not in name_to_sensor:
                # this sensor is not interested, skip
                continue 
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

def calculate_weighted_mean(seq: np.ndarray, sigmas: np.ndarray):
    """
    Computes the weighted mean and weighted sigma from a given sequence of 
    data and their corresponding sigmas.
    
    Parameters
    ----------
    seq : 1D-array
        A sequence of data.
    sigmas : 1D-array 
        A sequence of sigmas.
    
    Returns
    -------
    weighted_mean : float
    weighted_sigma : float
    """
    assert seq.shape == sigmas.shape
    weighted_sigma = np.sqrt(1 / np.sum(1 / np.square(sigmas)))
    weighted_mean = np.sum(seq / np.square(sigmas)) * np.square(weighted_sigma)
    return weighted_mean, weighted_sigma 

def is_close(a: float, b: float, tol: float, cmp_abs=True):
    # Checks if two floats' abs value are close 
    if cmp_abs:
        return abs(abs(a) - abs(b)) <= tol
    else:
        return abs(a - b) <= tol

def temperature_to_color(temperature: float):
    # Map temperature to a rainbow color (purple to red)
    min_temp, max_temp = -60, 120
    norm_temp = (temperature - min_temp) / (max_temp - min_temp)
    return plt.cm.rainbow(norm_temp)

def humidity_to_color(rh: float):
    min_rh, max_rh = 0, 36
    norm_rh = (rh - min_rh) / (max_rh - min_rh)
    return plt.cm.rainbow(norm_rh)

def parse_file_iv(filepath: str, returns_data: bool=True):
    """
    Retrieves temperature, date, data, humidity, and ramp_type from a IV scan
    
    Parameters
    ----------
    filepath : str 
        The path to the scan.
    returns_data : bool
        Whether parse and return the voltage and current data. Defaults to True.
        
    Returns
    -------
    temperature : float
        Temperature of the measurement, or None if not provided.
    date : datetime 
        Time at which the measurement started.
    data : 2D-array
        A numpy array of data.
    humidity : float 
        Humidity of the measurement, or None if not provided.
    ramp_type : int 
        Either 1 (ramp_up), -1 (ramp_down), or 0 (not given).
    duration : float 
        Number of seconds the measurement lasted, or None if not given.
    """
    ramp_type = 0 
    
    if filepath.endswith(".txt"):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # Extract metadata
        if 'Inf' in lines[0].split(' ')[1]:
            temperature = None
        else:
            temperature = float(lines[0].split(' ')[1])
        humidity = float(lines[1].split(' ')[1])
        date_str = lines[2].split(' ',maxsplit=1)[1].strip()
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
        except:
            date = datetime.strptime(date_str, '%d/%m/%Y %I:%M %p')
        # Extract table data
        if returns_data:
            data = np.genfromtxt(lines[4:], delimiter=',', names=['voltage','pad','gr','total'])
        else:
            data = None
        if "RampUp" in filepath: ramp_type = 1
        elif "RampDown" in filepath: ramp_type = -1 
        elif "ru" in filepath: ramp_type = 1 
        elif "rd" in filepath: ramp_type = -1
        elif "Up" in lines[0]: ramp_type = 1 
        elif "Down" in lines[0]: ramp_type = -1

        duration = None 
        
    elif filepath.endswith(".iv"):
        # ramp_type is usually not given for any .iv files
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        table_begin_idx = 0
        humidity = None # humidities are not included in .iv 
        # extract metadata 
        for i in range(len(lines)):
            if ":temperature[C]" in lines[i]:
                if 'Inf' in lines[i+1]:
                    temperature = None 
                else:
                    temperature = float(lines[i+1])
            elif ":elapsed" in lines[i]:
                duration = float(lines[i+1])
            elif ":start" in lines[i]:
                date_str = lines[i+1].strip()
                date = datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")
            elif "- compliance [A or V]:" in lines[i]:
                compliance = float(lines[i].split(" ")[-1])
            elif "BEGIN" in lines[i]:
                table_begin_idx = i+1
                break
            
        # extract table data 
        if returns_data:
            data = np.genfromtxt(lines[table_begin_idx:-1], delimiter=None, names=['voltage', 'total', 'pad'])
            for i in range(data.shape[0]):
                if is_close(data[i]['total'], compliance, 1E-8):
                    data = data[:i]
                    break
        else:
            data = None 
    
    else:
        raise ValueError(f"File must be .txt or .iv, given {filepath}")
    assert temperature != float('inf')
    return temperature, date, data, humidity, ramp_type, duration

def parse_file_iv_basic_info(filepath: str):
    """
    Retrives temperature, date, and humidity from a IV scan
    """
    temperature, date, _, humidity, ramp_type, duration = parse_file_iv(filepath, False)
    return temperature, date, humidity, ramp_type, duration

def parse_file_cv_basic_info(filepath: str):
    
    temperature, date, _, frequency = parse_file_cv(filepath, False)
    return temperature, date, frequency

def parse_file_cv(filepath: str, returns_data: bool=True):
    if filepath.endswith(".cv"):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        table_begin_idx = 0
        humidity = None # humidities are not included in any .iv 
        # extract metadata 
        for i in range(len(lines)):
            if ":temperature[C]" in lines[i]:
                if 'Inf' in lines[i+1]:
                    temperature = None 
                else:
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
        if returns_data:
            data = np.genfromtxt(lines[table_begin_idx:-1], delimiter=None, names=['voltage', 'capacitance', 'conductivity', 'bias', 'current_power_supply'])
            for i in range(data.shape[0]):
                if is_close(data[i]['current_power_supply'], compliance, 1E-8):
                    data = data[:i]
                    break    
        else:
            data = None
            
        
    else:
        raise ValueError(f"File must be .cv, given {filepath}")
    
    return temperature, date, data, frequency

def linear(x, m, b):
    return m*x + b

def linear_fit(x_data: np.ndarray, y_data: np.ndarray, p0=None, sigmas=None):
    """
    Fits a linear line through given x and y data, ignoring NaNs in y data.
    
    Parameters
    ----------
    x_data : 1D-array 
    y_data : 1D-array
    p0 : array_like
        Initial guess of parameters to use.
    sigmas : 1D-array
        Uncertainties of data.
    
    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared residuals of f(xdata, *popt) - ydata is minimized.
    perr : array
        One standard deviation errors of parameters in popt
    residual : float 
        Average residual of the fit
    r_sqaured : float
    """
    nan_mask = np.isnan(y_data)
    x_data = x_data[~nan_mask]
    y_data = y_data[~nan_mask]
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