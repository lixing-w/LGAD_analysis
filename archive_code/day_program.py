import serial
import time
import matplotlib.pyplot as plt
import datetime as dt
import os
import winsound
import numpy as np
from scipy.odr import Model, RealData, ODR
import shutil
from serial.tools import list_ports

port = list(list_ports.comports())
for p in port:
    print(p.device)

location1 = ''#input('Keithley 1 (Used as Voltage Source) path: ')
if location1 == '':
    print("Enter Path for Keithley 1")
    location1 = "COM7"

location2 = ''#input('Keithley 2 (Used as Picometer) path: ')
if location2 == '':
    print("Enter Path for Keithley 2")
    location2 = "COM6"

class Keithley1():
    def __init__(self,location=location1,baudrate=9600,outtime=1):
        try:
            self.device=serial.Serial(location, baudrate, timeout=outtime)
        except serial.SerialException:
            print("can't open the keithley, is it connected on "+location+" ?")
    def __del__(self):
        try:
            self.device.close()
        except AttributeError:
            pass

    def GetType(self):
        query="*IDN?\n"

        self.device.write(query.encode())
        idn=self.device.readline()
        print(idn)
        return idn


    def TurnOn(self):
        query=":OUTP ON\n"
        try:
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device turn on failed")

    def TurnOff(self):
        query=":OUTP OFF\n"
        try:
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device turn off failed")

    def SetCurrentCompliance(self,compliance):
        try:
            query=":SENS:CURR:PROT "+str(compliance)+"\n"
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device setting compliance current to "+str(compliance)+" failed")

    def SetVoltageCompliance(self,compliance):
        try:
            query=":SENS:VOLT:PROT "+str(compliance)+"\n"
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device setting compliance voltage to "+str(compliance)+" failed")

    def ReadData(self):
        try:
            query=":READ?\n"
            self.device.write(query.encode())
            data=self.device.readline()
            # print(data)
            return data
        except serial.SerialTimeoutException:
            print("Device Read failed")
            return None
    def FetchData(self):
        try:
            query=":FETCh?\n"
            self.device.write(query.encode())
            data=self.device.readline()
            # print(data)
            return data
        except serial.SerialTimeoutException:
            print("Device Fetch failed")
            return None
    def SetVoltageRange(self,volt):
        try:
            query=":SOUR:VOLT:RANG "+ str(volt)+"\n"
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device voltage range failed")

    def SetCurrentRange(self,current):
        try:
            query=":SOUR:CURR:RANG "+ str(current)+"\n"
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device current range failed")

    def SetVoltageLevel(self,volt):
        try:
            query=":SOUR:VOLT:LEV "+str(volt)+"\n"
            self.device.write(query.encode())
        except serial.SerialTimeoutException:
            print("Device voltage level failed")


    def SetCurrentLevel(self,current):
        try:
            query=":SOUR:CURR:LEV "+str(current)+"\n"
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device current level failed")

    def GetCurrent(self):
        try:
            query=":MEAS:CURR?\n"
            self.device.write(query.encode())
            
            results=self.device.readline()
            current=results.decode().split(",")[1]
            return float(current)
        except serial.SerialTimeoutException:
            print("Device get current failed")

    def GetVoltage(self):
        try:
            query=":MEAS:VOLT?\n"
            self.device.write(query.encode())
            results=self.device.readline()
            print('Getting Voltage', results)
            voltage=results.decode().split(",")[0]
            print(results)
            return float(voltage)
        except serial.SerialTimeoutException:
            print("Device get voltage failed")
            return ""
    def GetResistance(self):
        try:
            query=":MEAS:RES?\n"
            self.device.write(query.encode())
            return self.device.readline()
        except serial.SerialTimeoutException:
            print("Device get resistance failed")
            return ""
  
    def ToRear(self):
        query=":ROUT:TERM REAR\n"
        self.device.write(query.encode())

    def ToFront(self):
        query=":ROUT:TERM FRON\n"
        self.device.write(query.encode())

class Keithley2():
    def __init__(self,location=location2,baudrate=9600,outtime=1):
        try:
            self.device=serial.Serial(location, baudrate, timeout=outtime)
        except serial.SerialException:
            print("can't open the keithley, is it connected on "+location+" ?")
    def __del__(self):
        try:
            self.device.close()
        except AttributeError:
            pass

    def GetType(self):
        query="*IDN?\n"

        self.device.write(query.encode())
        idn=self.device.readline()
        print(idn)
        return idn
    
    def Restore(self):
        query="*RST\n"

        self.device.write(query.encode())
        # idn=self.device.readline()
        # print(idn)
        return


    def TurnOn1(self):
        query=":OUTP1 ON\n"
        try:
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device turn on failed")

    def TurnOff1(self):
        query=":OUTP1 OFF\n"
        try:
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device turn off failed")

    def TurnOn2(self):
        query=":OUTP2 ON\n"
        try:
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device turn on failed")

    def TurnOff2(self):
        query=":OUTP2 OFF\n"
        try:
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device turn off failed")

    def SetCurrentCompliance(self,compliance):
        try:
            query=":SENS:CURR:PROT "+str(compliance)+"\n"
            self.device.write(query.encode())
            query=":SENS2:CURR:PROT "+str(compliance)+"\n"
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device setting compliance current to "+str(compliance)+" failed")

    def SetVoltageCompliance(self,compliance):
        try:
            query=":SENS:VOLT:PROT "+str(compliance)+"\n"
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device setting compliance voltage to "+str(compliance)+" failed")

    def ReadData(self):
        try:
            query=":READ?\n"
            self.device.write(query.encode())
            results=self.device.readline()

            # data=data.decode()
            # print("Printing result", results)
            data=results.decode().split(",")[0]
            # print(data)
            return data
        except serial.SerialTimeoutException:
            print("Device Read failed")
            return None
    def FetchData(self):
        try:
            query=":FETCh?\n"
            self.device.write(query.encode())
            data=self.device.readline()
            data=data.decode()
            # results.decode().split(",")[1]
            # print(data)
            return data
        except serial.SerialTimeoutException:
            print("Device Fetch failed")
            return None
    def SetVoltageRange(self,volt):
        try:
            query=":SOUR:VOLT:RANG "+ str(volt)+"\n"
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device voltage range failed")

    def SetCurrentRange1(self,current):
        try:
            query=":SENS1:CURR:RANG "+ str(current)+"\n"
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device current range failed")

    def SetCurrentRange2(self,current):
        try:
            query=":SENS2:CURR:RANG "+ str(current)+"\n"
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device current range failed")

    def SetVoltageLevel(self,volt):
        try:
            query=":SOUR:VOLT:LEV "+str(volt)+"\n"
            self.device.write(query.encode())
        except serial.SerialTimeoutException:
            print("Device voltage level failed")


    def SetCurrentLevel(self,current):
        try:
            query=":SOUR:CURR:LEV "+str(current)+"\n"
            self.device.write(query.encode())
            print(self.device.readline())
        except serial.SerialTimeoutException:
            print("Device current level failed")

    def GetCurrent(self):
        try:
            query=":MEAS?\n"
            self.device.write(query.encode())
            
            results=self.device.readline()
            current=results.decode().split(",")[0]
            return float(current)
        except serial.SerialTimeoutException:
            print("Device get current failed")


    def GetVoltage(self):
        try:
            query=":MEAS:VOLT?\n"
            self.device.write(query.encode())
            results=self.device.readline()
            print('Getting Voltage', results)
            voltage=results.decode().split(",")[0]
            print(results)
            return float(voltage)
        except serial.SerialTimeoutException:
            print("Device get voltage failed")
            return ""
    def GetResistance(self):
        try:
            query=":MEAS:RES?\n"
            self.device.write(query.encode())
            return self.device.readline()
        except serial.SerialTimeoutException:
            print("Device get resistance failed")
            return ""
  
    def ToRear(self):
        query=":ROUT:TERM REAR\n"
        self.device.write(query.encode())

    def ToFront(self):
        query=":ROUT:TERM FRON\n"
        self.device.write(query.encode())

    def selectChannel(self, channel):
        query=":FORM:ELEM CURR"+str(channel)+"\n"
        self.device.write(query.encode())

    def setGround(self): # Set both channels to independent operation mode globally 
        #query = ":SYST:CHAN:CPL 0\n" # Set channel coupling to independent 
        query = 'SOUR:GCON ON\n'
        self.device.write(query.encode())
        query = 'SOUR2:GCON ON\n'
        self.device.write(query.encode())
        print("Channels set to ground connect")
        
class kController():

    def __init__(self, data_dir, depletionV):
        if depletionV == '': depletionV = 30
        self.step_time = 0
        self.kth=Keithley1()
        self.kthCurrent=Keithley2()
        print(self.kth.GetType())
        print(self.kthCurrent.GetType())
        self.kthCurrent.Restore()
        self.kth.TurnOn()
        self.kthCurrent.TurnOn1()
        self.kthCurrent.TurnOn2()
        self.kthCurrent.setGround()
        self.kth.SetVoltageRange(200)
        print("[info] Done Setting Voltage Range")
        self.temp = 21
        self.name = 'test'
        self.data_dir = data_dir
        dir_path = '../data/'+self.data_dir+'/'
        if not os.path.exists(dir_path): os.makedirs(dir_path)
        self.depletionV = float(depletionV)
        self.temp_range = {'ru': range(-60,140,20), 'rd': range(120,-80,-20)}
        self.max_voltages = [130, 140, 150, 160, 170, 180, 190, 200, 210, 220]
        self.complianceCurrs = [5e-6, 5e-6, 5e-6, 1e-5, 1e-5, 1e-5, 5e-5, 1e-4, 1e-3, 1e-3]
        self.picoComplianceCurrs = [5e-6, 5e-6, 5e-6, 1e-5, 1e-5, 1e-5, 5e-5, 1e-4, 1e-3, 1e-3]
        # input the times for each step of the day program here (the total time including the )
        self.ambient_times = {'ru': [2460, 960, 960, 960, 960, 960, 1020, 1020, 1020, 1020],
                              'rd': [7560, 1260, 960, 960, 960, 960, 1020, 1320, 1320, 1620]}
        
        self.end_time = 1200
        # these are the times in seconds to add to each step of the day program
        self.iv_times = [310, 320, 330, 340, 350, 360, 370, 380, 390, 400]

    def set_temp_index(self):
        self.temp_idx = max(0, min(9, (self.temp + 60) // 20))
        self.kthCurrent.SetCurrentCompliance(self.picoComplianceCurrs[self.temp_idx])
        self.kth.SetCurrentCompliance(self.complianceCurrs[self.temp_idx])
        print("Compliance currents set!")

    def increase_max_voltages(self):
        for i in range(self.temp_idx, len(self.max_voltages)):
            self.max_voltages[i] += 4

    def add_vpoint(self, f, curr_volt):
        self.kth.SetVoltageLevel(-curr_volt)
        v=self.kth.GetVoltage()
        time.sleep(0.2)
        totalCurrent=self.kth.GetCurrent()
        self.kthCurrent.selectChannel(1)
        padCurr = self.kthCurrent.ReadData().strip('\r')
        self.kthCurrent.selectChannel(2)
        grCurr = self.kthCurrent.ReadData().strip('\r')
        f.write(str(v)+","+str(padCurr)+","+str(grCurr)+","+str(totalCurrent)+"\n")
        if abs(totalCurrent) > self.complianceCurrs[self.temp_idx]*0.97:
            return False
        elif abs(float(padCurr)) > self.picoComplianceCurrs[self.temp_idx]*0.97:
            return False
        elif abs(float(grCurr)) > self.picoComplianceCurrs[self.temp_idx]*0.97:
            return False
        else:
            return True

    def run_scan(self):
        time_init = time.time()
        # set up file
        self.set_temp_index()
        print("Max voltage: ", str(self.max_voltages[self.temp_idx]))
        f = open(self.file_path, "w")
        date_and_time = dt.datetime.now()
        f.write('Temperature: '+str(self.temp)+' C\n')
        f.write('Humidity: '+self.humidity+' %\n')
        f.write('Date: '+str(date_and_time)+'\n')
        f.write("voltage,pad,gr,totalCurrent\n")
        # ramp down to 0 voltage
        curr_volt = abs(int(self.kth.GetVoltage()))
        while curr_volt > 5:
            curr_volt -= 5
            self.kth.SetVoltageLevel(-curr_volt)
            time.sleep(0.2)
        curr_volt = 0
        self.kth.SetVoltageLevel(-curr_volt)
        # ramp up til we meet compliance current
        compliant = self.add_vpoint(f, curr_volt)
        step = 2
        while compliant and (curr_volt < self.depletionV):
            curr_volt += step
            compliant = self.add_vpoint(f, curr_volt)
        step = 4
        while compliant and (curr_volt < self.max_voltages[self.temp_idx]-50):
            curr_volt += step
            compliant = self.add_vpoint(f, curr_volt)
        step = 2
        while compliant and (curr_volt < self.max_voltages[self.temp_idx]):
            curr_volt += step
            compliant = self.add_vpoint(f, curr_volt)
        # save data or rerun if breakdown never reached
        final_voltage = curr_volt
        f.close()
        while curr_volt > 55:
            curr_volt -= 5
            self.kth.SetVoltageLevel(-curr_volt)
            time.sleep(0.2)
        self.kth.SetVoltageLevel(-curr_volt)
        if compliant:
            print("Current never reached compliance!! Manual adjustment needed!")
            #os.remove(self.file_path)
            #self.increase_max_voltages()
            #self.run_scan()
        else:
            print("Current has reached compliance!")
            print("Maximum voltage reached: "+str(int(final_voltage)))
            print("Scan took "+str(time.time()-time_init)+" seconds")
            winsound.Beep(500, 1000)

    def turn_off(self, end_program):
        curr_volt = abs(int(self.kth.GetVoltage()))
        if end_program == "weekend":
            while curr_volt > 5:
                curr_volt -= 5
                self.kth.SetVoltageLevel(-curr_volt)
                time.sleep(0.2)
            self.kth.SetVoltageLevel(0)
            self.kth.TurnOff()
        else:
            while curr_volt < 150:
                curr_volt += 5
                self.kth.SetVoltageLevel(-curr_volt)
                time.sleep(0.2)
            self.kth.SetVoltageLevel(-150)
        self.kthCurrent.TurnOff1()
        self.kthCurrent.TurnOff2()


    def test_IV(self):
        print("Ready for test scan at "+str(self.temp)+"C!")
        self.file_path = "../data/"+self.data_dir+"/"+self.name+"_"+str(int(self.temp))+".txt"
        if os.path.exists(self.file_path):
            print("Just kidding, the file already exists! Skipping measurement!")
        else: 
            self.humidity = input('Relative humidity[%]: ')
            self.run_scan()
        print("Done testing! Time for a short break!")

    def ramp(self, name):
        self.name = name
        if name == 'ru': self.start_time = time.time()
        for i, temp in enumerate(self.temp_range[name]):
            self.temp = temp
            self.step_time += self.ambient_times[name][i]
            print("Ready for next measurement at "+str(self.temp)+"C! Waiting for thermalization...")
            time.sleep(self.start_time + self.step_time - time.time() - self.iv_times[i] - 5) #5s buffer to be safe
            print("Beginning measurements!")
            self.file_path = "../data/"+self.data_dir+"/"+self.name+"_"+str(int(self.temp))+".txt"
            if os.path.exists(self.file_path):
                print("Just kidding, the file already exists! Skipping measurement!")
            else: 
                self.humidity = 0 # THIS IS WHERE THE HUMIDITY NEEDS TO GO
                # alternatively if the humidity can be retrieved at every time interval and given in the file as a function of time
                #input('Relative humidity[%]: ')
                self.run_scan()
        if name == 'ru': print("Done ramping up! Time for a break!")
        else: print("Done ramping down! Almost done for the day!")


# plotting functions
def parse_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    temperature = float(lines[0].split(':')[1].strip().replace(" C", ""))
    date = lines[2].split(':', 1)[1].strip()
    date = date.split(' ')[0]
    header = lines[3].strip().split(',')
    data = np.genfromtxt(lines[4:], delimiter=',', names=header)
    return temperature, date, data
# Map temperature to a rainbow color (purple to red)
def temperature_to_color(temperature):
    min_temp, max_temp = -60, 100
    norm_temp = (temperature - min_temp) / (max_temp - min_temp)
    return plt.cm.rainbow(norm_temp)
# Process files in the directory
def plot_scans(data_dir):
    file_groups = {"ru_": [], "rd_": []}
    for filename in os.listdir("../data/"+data_dir):
        if filename.startswith("ru_") or filename.startswith("rd_"):
            key = "ru_" if filename.startswith("ru_") else "rd_"
            file_groups[key].append(os.path.join("../data/"+data_dir, filename))
    # Loop through file groups and plot
    for ramp_type, files in file_groups.items():
        plt.figure(figsize=(10, 6))
        plots = []  # To store plot data for sorting
        labels = []  # To store labels for sorting
        for filepath in sorted(files):
            temp, date, data = parse_file(filepath)
            color = temperature_to_color(temp)
            label = f"{temp} C"
            # Plot pad current and store data for later sorting
            plots.append((abs(data['voltage']), abs(data['pad']), label, color))
        # Sort by temperature (numerical order)
        plots.sort(key=lambda x: float(x[2].split()[0]))  # Sorting by temperature in label    
        # Plot each sorted line
        for volt, pad, label, color in plots:
            plt.plot(volt, pad, label=label, color=color)
        # Plot configuration
        plt.xlabel("Voltage (V)")
        plt.ylabel("Pad Current (A)")
        plt.yscale('log')
        plt.title(f"{date} - {'Ramp Up' if ramp_type == 'ru_' else 'Ramp Down'}")
        # Create legend (now sorted by temperature)
        plt.legend(title="Temperature", loc='best')
        plt.grid(True)
        plt.tight_layout()
        # Save the plot
        plot_filename = "../data/"+data_dir+"/"+f"{ramp_type.strip('_')}_plot.png"

        plt.savefig(plot_filename)
        plt.clf()
        print(f"Saved plot: {plot_filename}")

def linear(b, x):
    return b[0]*x + b[1]

def analyze(data_dirs, temps, base_i, base_j, bd_i, bd_j):
    # collect all data
    data = dict()
    for temp in temps:
        data[temp] = []
    for data_dir in data_dirs:
        for temp in temps:
            if os.path.exists("../data/"+data_dir+"/rd_"+str(temp)+".txt"):
                _, _, file_data = parse_file("../data/"+data_dir+"/rd_"+str(temp)+".txt")
                data[temp].append(file_data)
            if os.path.exists("../data/"+data_dir+"/ru_"+str(temp)+".txt"):
                _, _, file_data = parse_file("../data/"+data_dir+"/ru_"+str(temp)+".txt")
                data[temp].append(file_data)
    # average data for each temperature at each data point
    means = dict()
    stds = dict()
    breakdown = dict()
    sigma_breakdown = dict()
    min_std = {'voltage': 1, 'pad': 1e-12, 'gr': 1e-12, 'totalCurrent': 1e-12}
    model = Model(linear)
    for temp_i, temp in enumerate(temps):
        # initialize vector to keep track of index in each file
        file_v_idx = [1 for k in range(len(data[temp]))]
        vpoint = 1
        means[temp] = {'voltage': [], 'pad': [], 'gr': [], 'totalCurrent': []}
        stds[temp] = {'voltage': [], 'pad': [], 'gr': [], 'totalCurrent': []}
        max_voltage = max([-1*file_data['voltage'][-1] for file_data in data[temp]])
        # loop over voltage points in groups of 2
        while vpoint < max_voltage:
            data_point = {'voltage': [], 'pad': [], 'gr': [], 'totalCurrent': []}
            for i_file, file_data in enumerate(data[temp]):
                # add data to voltage point if voltage is less than 2V more than vpoint
                if len(file_data['voltage']) > file_v_idx[i_file]:
                    if file_data['voltage'][file_v_idx[i_file]] + vpoint > -2:
                        for key in means[temp].keys():
                            data_point[key].append(file_data[key][file_v_idx[i_file]])
                        file_v_idx[i_file] += 1
            if len(data_point['voltage']) > 0:
                for key in means[temp].keys():
                    means[temp][key].append(np.mean(data_point[key]))
                    stds[temp][key].append(max([np.std(data_point[key]), min_std[key]]))
            vpoint += 2
        for key in means[temp].keys():
            means[temp][key] = np.array(means[temp][key])*-1
            stds[temp][key] = np.array(stds[temp][key])
        # identify the baseline
        base_x = means[temp]['voltage'][base_i[temp_i]:base_j[temp_i]]
        base_xerr = stds[temp]['voltage'][base_i[temp_i]:base_j[temp_i]]
        base_y = np.log10(means[temp]['pad'][base_i[temp_i]:base_j[temp_i]])
        base_yerr = stds[temp]['pad'][base_i[temp_i]:base_j[temp_i]] / (base_y * np.log(10))
        rdata = RealData(base_x, base_y, sx=base_xerr, sy=base_yerr)
        odr = ODR(rdata, model, beta0=[0.01, -8])
        output = odr.run()
        base_m, base_b = output.beta
        base_merr, base_berr = np.sqrt(np.diag(output.cov_beta))
        # do the same for the breakdown line (3rd and 2nd to last points)
        bd_x = means[temp]['voltage'][bd_i[temp_i]:bd_j[temp_i]]
        bd_xerr = stds[temp]['voltage'][bd_i[temp_i]:bd_j[temp_i]]
        bd_y = np.log10(means[temp]['pad'][bd_i[temp_i]:bd_j[temp_i]])
        bd_yerr = stds[temp]['pad'][bd_i[temp_i]:bd_j[temp_i]] / (bd_y * np.log(10))
        rdata = RealData(bd_x, bd_y, sx=bd_xerr, sy=bd_yerr)
        odr = ODR(rdata, model, beta0=[1, -100])
        output = odr.run()
        bd_m, bd_b = output.beta
        bd_merr, bd_berr = np.sqrt(np.diag(output.cov_beta))
        # calculate breakdown voltage and uncertainty
        breakdown[temp] = (bd_b - base_b) / (base_m - bd_m)
        inv_mdiff = 1 / (base_m - bd_m)
        sigma_breakdown[temp] = np.sqrt(
            (breakdown[temp]*inv_mdiff * base_merr)**2 +
            (breakdown[temp]*inv_mdiff * bd_merr)**2 +
            (inv_mdiff * base_berr)**2 +
            (inv_mdiff * bd_berr)**2
        )
        # plot individual breakdown plots
        plt.figure(figsize=(10, 6))
        full_x = means[temp]['voltage']
        full_y = np.log10(means[temp]['pad'])
        full_xerr = stds[temp]['voltage']
        full_yerr = stds[temp]['pad'] / (full_y * np.log(10))
        plt.errorbar(full_x, full_y, xerr=full_xerr, yerr=abs(full_yerr), fmt='-o', markersize=2, label='Average', color='blue')
        plt.plot(means[temp]['voltage'], linear([base_m, base_b], means[temp]['voltage']), label='baseline', color='red')
        example_xs = np.array([breakdown[temp], means[temp]['voltage'][-1]])
        plt.plot(example_xs, linear([bd_m, bd_b], example_xs), label='breakdown', color='black')
        plt.xlabel("Voltage (V)")
        plt.ylabel("log(Pad Current (A))")
        #plt.ylim(-13,-3)
        #plt.xlim(0,250)
        plt.title("IV Scan with breakdown for "+str(temp)+" degrees: "+str(int(breakdown[temp]))+" V")
        # Create legend (now sorted by temperature)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # Save the plot
        plot_filename = "../data/plots/breakdown_"+str(temp)+".png"
        plt.savefig(plot_filename)
        plt.clf()
    # plot overall averages
    for curr_type in ["pad", "gr", "totalCurrent"]:
        plt.figure(figsize=(10, 6))
        for temp in temps:
            plt.plot(means[temp]['voltage'], means[temp][curr_type], label=str(temp)+" C", color=temperature_to_color(temp), markersize=2)
        plt.xlabel("Voltage (V)")
        plt.ylabel(curr_type+" Current (A)")
        plt.yscale('log')
        plt.ylim(1e-13,1e-3)
        plt.title("IV Scan Averaged across all days "+curr_type)
        # Create legend (now sorted by temperature)
        plt.legend(title="Temperature", loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("../data/plots/average_"+curr_type+".png")
        plt.clf()
    # make and plot breakdown fit
    temp_err = np.ones_like(breakdown.keys())/2
    temperatures = np.array(temps)
    breakdown_vs = np.array([breakdown[temp] for temp in temps])
    sig_breakdown_vs = np.array([sigma_breakdown[temp] for temp in temps])
    rdata = RealData(temperatures, breakdown_vs, sx=temp_err, sy=sig_breakdown_vs)
    odr = ODR(rdata, model, beta0=[1, 150])
    output = odr.run()
    breakdown_m, breakdown_b = output.beta
    plt.figure(figsize=(10, 6))
    plt.errorbar(temperatures, breakdown_vs, xerr=temp_err, yerr=sig_breakdown_vs, fmt='o', capsize=5, label='Observed', color='blue')
    plt.plot(temps, linear([breakdown_m, breakdown_b], temperatures), label=f'Best Fit: y = {breakdown_m:.2f}x + {breakdown_b:.2f}', color='red')
    plt.xlabel("Temperature (C)")
    plt.ylabel("Breakdown Voltage (V)")
    plt.title("Breakdown Voltage as a Function of Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../data/plots/breakdown_fit.png")
    plt.clf()

# get user inputs
winsound.Beep(1000, 1000)
data_dir = input("Where to store data (Usually today's date): ")
assert(data_dir != '')
depletionV = input('Approximate depletion voltage (Default = 30): ')
end_program = input('Which program will be running after measurements? \'night\' (default) or \'weekend?\'')
run_analysis = input('Do you want to analyze the data to calculate breakdown voltages? If yes (y) please verify the options available at the bottom of the code: ')
if end_program != 'weekend': end_program = 'night'
# initialize scanner
scanner = kController(data_dir, depletionV)
# run IV scans
scanner.test_IV()
input("Test complete! Start the ambient chamber day program and immediately press enter if everything looks good: ")
scanner.ramp('ru')
scanner.ramp('rd')
# all done
scanner.turn_off(end_program)
print("All done with the measurements!")

# run analysis

plot_scans(data_dir)
if run_analysis == 'yes' or run_analysis == 'y':
    data_dirs = ["Dec112024", "Dec122024", "Dec92024", "Dec62024", "Dec52024", "Dec42024", "Nov272024"]
    temps = [-40, -20, 0, 20,  40,  60,  80,  100,  120]
    base_i = [-5, 30, 10, 10,  10,  12,  12,  11,   12]
    base_j = [-2, -10, -20, -24, -25, -24, -36, -24,  -37]
    bd_i = [-3, -3, -3, -3,  -3,  -4,  -5,  -5,   -5]
    bd_j = [-1, -1, -1, None,  -1,  None,  -1,  -1,   -1]
    analyze(data_dirs, temps, base_i, base_j, bd_i, bd_j)

print("Done plotting! Zipping data...")
shutil.make_archive("../data/"+data_dir, 'zip', "../data/"+data_dir)
shutil.make_archive("../data/plots", 'zip', "../data/plots")


print("All done! Set the temperature to 21C and voltage to -150V, then you can go home!")