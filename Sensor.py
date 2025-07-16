from datetime import datetime
import re 
import os

from utils import *

class Sensor:
    """
    Defines a Sensor object.
    
    Fields:
    self.name - A unique identifier
    self.type - Either AC or DC
    self.depletion_v 
    self.sensor_dir - Directory to sensor's entire data
    self.data_dirs - A list of directories, each contain just scans
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
            temperature, date, humidity, rt = parse_file_iv_basic_info(dir)
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