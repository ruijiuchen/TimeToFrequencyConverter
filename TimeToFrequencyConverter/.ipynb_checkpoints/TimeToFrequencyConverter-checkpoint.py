import sys
import time
import numpy as np
import random
from ROOT import TCanvas, gROOT, TGraph, TApplication,TH1F, gPad, gStyle, TFile
from prettytable import PrettyTable
import argparse

class TimeToFrequencyConverter:
    def __init__(self):
        self.simulated_data_list=[]
        self.sampling_amplitudes = []
        self.frequencies = []
        self.fft_result = []
        self.canvas = None  # Initialize canvas attribute
        #self.canvas.Divide(1,2)
        self.g_sampling_amplitude_time= TGraph()
        self.h_fft_result_frequencies = TH1F()
        
    def load_sampling_amplitudes(self):
        try:
            self.sampling_amplitudes = np.load("sampling_amplitudes.npy")
            return True
        except FileNotFoundError:
            print("### Error: File 'sampling_amplitudes.npy' not found.")
            return False
        except Exception as e:
            print(f"### Error loading 'sampling_amplitudes.npy': {e}")
            return False
    
    def load_frequencies(self):
        try:
            self.frequencies = np.load("frequencies.npy")
            return True
        except FileNotFoundError:
            print("### Error: File 'frequencies.npy' not found.")
            return False
        except Exception as e:
            print(f"### Error loading 'frequencies.npy': {e}")
            return False
        
    def load_fft_result(self):
        try:
            self.fft_result = np.load("fft_result.npy")
            return True
        except FileNotFoundError:
            print("### Error: File 'fft_result.npy' not found.")
            return False
        except Exception as e:
            print(f"### Error loading 'fft_result.npy': {e}")
            return False
        
    def ReadTimeData_root(self,SimulatedDataFile,TOF):
        """
        Read time data from a file and convert it to a list of values.
    
        Parameters:
        - SimulatedDataFile (str): The name of the file containing the simulated data.
        - TOF : TOF=0: single TOF; 1:DTOF0, 2:DTOF1
        Returns:
        - simulated_data_list: A list of simulation values.
        """
        print("SimulatedDataFile:", SimulatedDataFile, " TOF:",TOF)
        
        file = TFile(SimulatedDataFile)
        #file.ls()
        t = file.Get("t")
        #t.Print()
        point_number=TOF
        NEntries = t.GetEntries()
        print("NEntries:",NEntries, " point_number:", point_number)
        SimulatedData = []
        for i in range(0, NEntries):
            t.GetEntry(i)
            if t.Point==point_number:
                if t.Turn < 10:
                    print(" Turn= ",t.Turn, " point ",t.Point," time = ","%20.6f"%t.time," x = ", "%10.6f"%t.BE_X," y = ", "%10.6f"%t.BE_Y)
                self.simulated_data_list.insert(i,t.time/1e12)
        if len(self.simulated_data_list) == 0:
            return False
        return True

    def ReadTimeData(self,SimulatedDataFile):
        """
        Read time data from a file and convert it to a list of values.
    
        Parameters:
        - SimulatedDataFile (str): The name of the file containing the simulated data.
        Returns:
        - simulated_data_list: A list of simulation values.
        """
       # Read time data
        try:
            with open(SimulatedDataFile, "r") as file:
                lines = file.readlines()
        except FileNotFoundError:
            print(f"### Error: File '{SimulatedDataFile}' not found.")
            return False
        except Exception as e:
            print(f"### Error opening file '{SimulatedDataFile}': {e}")
            return False
        #for line in lines:
        for line_num, line in enumerate(lines):
             try:
                 # Convert each line of data to a list
                 data = eval(line.strip())
                 # Get time data (assuming time is in the third column of each line)
                 simulated_data = float(data[2])/1e12 # convert to second.
                 # Append simulated_data to the list
                 self.simulated_data_list.append(simulated_data)
             except (ValueError, IndexError, TypeError) as e:
                 print(f"### Error processing line: {line.strip()}. Details: {e}")
                 return False
        if len(simulated_data) == 0:
            return False
        return True
    
    def generate_sampling_amplitudes(self, sample_rate=20e6, signal_width = 100e-12, noise_level = 0):
        """
        Generate sampling signal based on simulated data.
    
        Parameters:
        - sample_rate (float): Sampling rate in Hertz. Default is 20MHz.
        - signal_width: width of gaussion function
        - noise_level: noise level

        Returns:
        - sampling_amplitudes, a list of sampling amplitutde.
        """
        time_step = 1/ sample_rate
        sampling_time_max = self.simulated_data_list[-1]
        sampling_points = int(sampling_time_max/time_step)
        print("time_step:",time_step," sampling_points:",sampling_points)
        self.sampling_amplitudes = [0] * sampling_points
        
        for i in range(0, len(self.simulated_data_list)):
            center_time=self.simulated_data_list[i]
            # Assume each data point represents a Gaussian signal
            i_min = max(int((center_time - 5*signal_width)/time_step),0)
            i_max = min(int((center_time + 5*signal_width)/time_step),sampling_points-1)
            
            if (i<=10 or i%10000==0):
                print("sampling signal, %3.1f"%(i/len(self.simulated_data_list)*100) ,"[%] completed...")
                
            i = i+1
            for i in range (i_min, i_max):
                sampling_time = i * time_step
                sampling_amplitude = (1 / (np.sqrt(2 * np.pi) * signal_width)) * np.exp(-0.5 * ((sampling_time - center_time) / signal_width) ** 2)
                self.sampling_amplitudes[i] = self.sampling_amplitudes[i] + sampling_amplitude
        # add white noise to sampling signal
        for i in range (0, len(self.sampling_amplitudes)):
            random_number = random.random()*noise_level
            self.sampling_amplitudes[i] = self.sampling_amplitudes[i] + random_number
    
    def generate_sampling_amplitudes_sin(self,sample_rate):
        """
        Generate time inputs based on simulated data.
        
        Parameters:
        - sample_rate (float): Sampling rate in Hertz. Default is 20MHz.
        
        Returns:
        """
        time_step = 1/ sample_rate
        sampling_points = int(100/time_step)
        sampling_amplitudes = np.zeros(sampling_points)  # Initialize an array with zeros
        for i in range(0,sampling_points):
            sampling_time      = i *time_step
            sampling_amplitude = np.sin(2 * np.pi * 0.2 * sampling_time)  # Example sine function, adjust as needed
            sampling_amplitudes[i] = sampling_amplitude
        return sampling_amplitudes
    
    def convert_to_frequency(self, sample_rate):
        """
        Convert a list of time points to a list of frequencies using FFT.
    
        Parameters:
        - sample_rate (float): The rate at which the signal is sampled.
    
        Returns:
        - frequencies: A list of corresponding frequencies.
        - fft_result: noise power density
        """
        # Calculate the number of samples
        num_samples = len(self.sampling_amplitudes)
    
        # Perform FFT on the sampling_amplitudes
        self.fft_result = np.abs(np.fft.fft(self.sampling_amplitudes))
        
        # Calculate the frequencies corresponding to the FFT result
        self.frequencies = np.fft.fftfreq(num_samples, 1.0 / sample_rate)
        
    def plot_spectrum(self, sample_rate, plot_time_min, plot_time_max, plot_fre_min, plot_fre_max, plot_opt):
        """
        Plot the frequency spectrum.
    
        Parameters:
        - sample_rate (float): The sample rate.
        - plot_opt: plot option. plot_opt==2: plot up and down panel.
                                 plot_opt!=2: only plot down panel.
        - plot_time_min, plot_time_max, the time range of the plot
        - plot_fre_min, plot_fre_max, the frequency range of the plot
        """
        if self.canvas == None or not self.canvas.IsZombie():
            self.canvas = TCanvas("canvas","cavas",0,0,600,600)  # Initialize canvas attribute
            self.canvas.Divide(1,2)
        self.canvas.cd(1)
        gPad.SetTopMargin(0.07)
        gPad.SetBottomMargin(0.15)
        gPad.SetRightMargin(0.08)
        print("1. Ploting the sampling signal.")
        
        self.g_sampling_amplitude_time.SetName("g_sampling_amplitude_time")
        title = "sampling signal with sampling rate" + str(sample_rate)
        self.g_sampling_amplitude_time.SetTitle(title)
        self.g_sampling_amplitude_time.Set(0)  # Clear the data points
        if plot_opt == 2:
            time_step = 1 / sample_rate
            index_plot_time_min = int(plot_time_min/time_step)
            index_plot_time_max = min(int(plot_time_max/time_step), len(self.sampling_amplitudes))
            for i in range(index_plot_time_min, index_plot_time_max):
                sampling_time = i * time_step*1e9
                self.g_sampling_amplitude_time.SetPoint(i-index_plot_time_min, sampling_time, self.sampling_amplitudes[i])
        	    
            if self.g_sampling_amplitude_time.GetN()<1e6:
        	    self.g_sampling_amplitude_time.Draw("apl")
        	    self.g_sampling_amplitude_time.SetMarkerStyle(20)
        	    self.g_sampling_amplitude_time.SetMarkerColor(2)
        	    self.g_sampling_amplitude_time.SetMarkerSize(0.5)
            else:
        	    self.g_sampling_amplitude_time.Draw("al")
        	    
            self.g_sampling_amplitude_time.GetXaxis().SetTitle("Time [ns]")
            self.g_sampling_amplitude_time.GetXaxis().CenterTitle()
            self.g_sampling_amplitude_time.GetXaxis().SetTitleSize(0.06)
            self.g_sampling_amplitude_time.GetXaxis().SetLabelSize(0.06)
            
            self.g_sampling_amplitude_time.GetYaxis().SetTitle("Amplitude")
            self.g_sampling_amplitude_time.GetYaxis().CenterTitle()
            self.g_sampling_amplitude_time.GetYaxis().SetTitleSize(0.06)
            self.g_sampling_amplitude_time.GetYaxis().SetLabelSize(0.06)
            self.g_sampling_amplitude_time.GetYaxis().SetTitleOffset(0.8)
        	
        self.canvas.cd(2)
        gPad.SetTopMargin(0.05)
        gPad.SetBottomMargin(0.15)
        gPad.SetRightMargin(0.08)
        gPad.SetLogy(1)
        gStyle.SetOptTitle(0)
        gStyle.SetOptStat(0)
        
        print("2. Ploting the frequency spectrum")
        # Assuming frequencies is a NumPy array
        half_index = len(self.frequencies) // 2
        # Find the indices for plot_fre_min and plot_fre_max
        index_min = np.searchsorted(self.frequencies[:half_index], plot_fre_min)
        index_max = np.searchsorted(self.frequencies[:half_index], plot_fre_max)
        
        # Adjust indices for the full frequencies array
        index_max = min(index_max, half_index - 1)
    
        fre_N = index_max - index_min
        fre_min = self.frequencies[index_min]
        fre_max = self.frequencies[index_max]

        #if  self.h_fft_result_frequencies==None:
        #self.h_fft_result_frequencies = TH1F("h_fft_result_frequencies","h_fft_result_frequencies",int(fre_N),fre_min,fre_max)
        self.h_fft_result_frequencies.SetName("h_fft_result_frequencies")
        self.h_fft_result_frequencies.SetTitle("h_fft_result_frequencies")
        self.h_fft_result_frequencies.SetBins(int(fre_N),fre_min,fre_max)
        self.h_fft_result_frequencies.Reset()
        
        for i in range(index_min,index_max):
            x = self.frequencies[i]
            y = np.abs(self.fft_result[i])
            Nx = self.h_fft_result_frequencies.GetXaxis().FindBin(x)
            self.h_fft_result_frequencies.SetBinContent(Nx,y)
    
        self.h_fft_result_frequencies.Draw("hist")
        self.h_fft_result_frequencies.GetXaxis().SetRangeUser(0, sample_rate/2)
        self.h_fft_result_frequencies.GetXaxis().SetTitle("Frequency [Hz]")
        self.h_fft_result_frequencies.GetXaxis().CenterTitle()
        self.h_fft_result_frequencies.GetXaxis().SetTitleSize(0.06)
        self.h_fft_result_frequencies.GetXaxis().SetLabelSize(0.06)
        self.h_fft_result_frequencies.GetYaxis().SetTitle("Noise power density")
        self.h_fft_result_frequencies.GetYaxis().CenterTitle()
        self.h_fft_result_frequencies.GetYaxis().SetTitleSize(0.06)
        self.h_fft_result_frequencies.GetYaxis().SetLabelSize(0.06)
        self.h_fft_result_frequencies.GetYaxis().SetTitleOffset(0.8)
        self.canvas.Update()
        
    def run(self, sampling_method,sample_rate, signal_width,  noise_level, plot_opt, plot_time_min, plot_time_max, plot_fre_min, plot_fre_max, start_step, SimulatedDataFile, TOF):
        """
        Run the TimeToFrequencyConverter.

        Args:
        sampling_method (int): The method used for sampling.
        sample_rate (float): The sampling rate in Hertz.
        signal_width (float): The width of the signal in seconds.
        noise_level (float): The amplitude of noise level.
        plot_opt (int): The plotting option.
        plot_time_min (float): The minimum time for plotting.
        plot_time_max (float): The maximum time for plotting.
        plot_fre_min (float): The minimum frequency for plotting.
        plot_fre_max (float): The maximum frequency for plotting.
        start_step (int): The starting step for the conversion process.
        SimulatedDataFile (str): The file containing simulated data.

        Returns:
        the ROOT canvas, 
        sampling amplitude graph, 
        FFT result histogram, and a PrettyTable with runtimes.
    """
        print("\n################################################")
        print("### Welcome to use TimeToFrequencyConverter. ###")
        print("################################################\n")
        start_time = time.time()  # Record the start time    
        # Create a PrettyTable to display runtimes
        table = PrettyTable()
        table.field_names = ["Step", "Runtime (seconds)"]

        self.simulated_data_list=[]
        self.sampling_amplitudes = []
        self.frequencies = []
        self.fft_result = []

        if start_step == 1:
            print("-step 1. Read simulated data and creating sampling signal.")
            start_time1 = time.time()  # Record the end time
            self.ReadTimeData_root(SimulatedDataFile, TOF)
            if len(self.simulated_data_list) == 0:
                print("Error: simulated_data_list is emplty. Please check the --SimulatedDataFile and  --TOF options.")
                return False
            
            if sampling_method == 1:
                self.generate_sampling_amplitudes_sin(sample_rate)
            else:
                self.generate_sampling_amplitudes(sample_rate, signal_width, noise_level)
            end_time1 = time.time()  # Record the end time
            runtime1 = end_time1 - start_time1
            np.save("sampling_amplitudes.npy", self.sampling_amplitudes)
            print(f"Runtime: {runtime1:.4f} seconds")
            table.add_row(["1.Reading and Generating Data", f"{runtime1:.4f}"])
            
        if start_step >= 1 and start_step <= 2:
            print("-step 2. Do fft with the sampling data.")
            start_time2 = time.time()  # Record the end time
            # Load sampling_amplitudes from binary file
            if start_step == 2:
                self.load_sampling_amplitudes()
            self.convert_to_frequency(sample_rate)
            end_time2 = time.time()  # Record the end time
            runtime2 = end_time2 - start_time2
            np.save("frequencies.npy", self.frequencies)
            np.save("fft_result.npy", self.fft_result)
            print(f"Runtime: {runtime2:.4f} seconds")
            table.add_row(["2.FFT Calculation", f"{runtime2:.4f}"])
            
        if start_step >= 1 and start_step <= 3:
            print("-step 3. Plot the spectrum and save results.")
            start_time3 = time.time()  # Record the end time
            # Load sampling_amplitudes from binary file
            flag_data1 = True
            flag_data2 = True
            flag_data3 = True
            
            if start_step == 3:
                flag_data1 = self.load_sampling_amplitudes()
                flag_data2 = self.load_frequencies()
                flag_data3 = self.load_fft_result()
                
            if flag_data1==True and flag_data2==True and flag_data3==True:
                self.plot_spectrum(sample_rate,plot_time_min,plot_time_max,plot_fre_min,plot_fre_max, plot_opt)
                self.canvas.Print("plot_spectrum.png")
                fout = TFile("plot_spectrum.root","recreate")
                self.canvas.Write()
                self.g_sampling_amplitude_time.Write()
                self.h_fft_result_frequencies.Write()
                fout.Close()
            else:
                print("No data exist for plot. Please press Run bottom first!")
                return False
            
            end_time3 = time.time()  # Record the end time
            runtime3 = end_time3 - start_time3
            print(f"Runtime: {runtime3:.4f} seconds")
            table.add_row(["3.Plotting Spectrum", f"{runtime3:.4f}"])
                
        end_time = time.time()  # Record the end time
        total_runtime = end_time - start_time
        print(f"Total total_runtime: {total_runtime:.4f} seconds")
        table.add_row(["4.Total Runtime", f"{total_runtime:.4f}"])
        print(table)
        return True
