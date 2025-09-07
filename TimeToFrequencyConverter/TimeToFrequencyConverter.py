import sys
import time
import numpy as np
import random
from ROOT import TCanvas, gROOT, TGraph, TApplication, TH1F, gPad, gStyle, TFile
from prettytable import PrettyTable
import argparse
import os

class TimeToFrequencyConverter:
    def __init__(self):
        self.simulated_data_list = []  # Time values
        self.amplitudes = []          # Amplitude values from CSV
        self.sampling_amplitudes = []
        self.frequencies = []
        self.fft_result = []
        self.canvas = None
        self.g_sampling_amplitude_time = TGraph()
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

    def ReadTimeData_root(self, SimulatedDataFile, TOF,plot_time_min,plot_time_max):
        """
        Read time data from a ROOT file and convert it to a list of values.
        
        Parameters:
        - SimulatedDataFile (str): The name of the ROOT file containing the simulated data.
        - TOF (int): TOF=0: single TOF; 1:DTOF0, 2:DTOF1
        Returns:
        - bool: True if successful, False otherwise.
        """
        print("SimulatedDataFile:", SimulatedDataFile, " TOF:", TOF)
        file = TFile(SimulatedDataFile)
        t = file.Get("t")
        point_number = TOF
        NEntries = t.GetEntries()
        print("NEntries:", NEntries, " point_number:", point_number)
        self.simulated_data_list = []
        self.amplitudes = []  # Initialize amplitudes list for consistency
        for i in range(0, NEntries):
            t.GetEntry(i)
            if t.Point == point_number and t.time/ 1e12>plot_time_min and t.time/ 1e12<plot_time_max:
                if t.Turn < 10:
                    print(" Turn= ", t.Turn, " point ", t.Point, " time = ", "%20.6f" % t.time, " x = ", "%10.6f" % t.BE_X, " y = ", "%10.6f" % t.BE_Y)
                self.simulated_data_list.append(t.time / 1e12)
                self.amplitudes.append(1.0)  # Default amplitude for ROOT data
        file.Close()
        if len(self.simulated_data_list) == 0:
            print("### Error: No data found for the specified TOF.")
            return False
        return True

    def ReadTimeData_csv(self, SimulatedDataFile, plot_time_min, plot_time_max):
            """
            Read time and amplitude data from a CSV file, skipping the header row, and filter data within the specified time range.
           
            Parameters:
            - SimulatedDataFile (str): The name of the CSV file containing time and amplitude.
            - plot_time_min (float): Minimum time value (in seconds) for filtering data.
            - plot_time_max (float): Maximum time value (in seconds) for filtering data.
            Returns:
            - bool: True if successful, False otherwise.
            """
            try:
                with open(SimulatedDataFile, "r") as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"### Error: File '{SimulatedDataFile}' not found.")
                return False
            except Exception as e:
                print(f"### Error opening file '{SimulatedDataFile}': {e}")
                return False
            self.simulated_data_list = []
            self.amplitudes = []
            # Skip the first line (header)
            for line_num, line in enumerate(lines[1:], start=1): # Start from index 1 to skip header
                try:
                    # Assuming CSV format: time,amplitude
                    data = line.strip().split(',')
                    if len(data) != 2:
                        print(f"### Error: Line {line_num + 1} does not contain exactly 2 columns: {line.strip()}")
                        return False
                    time_value = float(data[0]) / 1e12 # Convert time to seconds
                    # Check if time_value is within the specified range
                    if plot_time_min <= time_value <= plot_time_max:
                        amplitude_value = float(data[1]) # Amplitude
                        self.simulated_data_list.append(time_value)
                        self.amplitudes.append(amplitude_value)
                except (ValueError, IndexError) as e:
                    print(f"### Error processing line {line_num + 1}: {line.strip()}. Details: {e}")
                    return False
            if len(self.simulated_data_list) == 0:
                print("### Error: No valid data found in the CSV file within the specified time range.")
                return False
            return True

    def generate_sampling_amplitudes(self, sample_rate=20e6, signal_width=100e-12, noise_level=0):
        """
        Generate sampling signal based on simulated data, using amplitudes from CSV.
        
        Parameters:
        - sample_rate (float): Sampling rate in Hertz. Default is 20MHz.
        - signal_width (float): Width of Gaussian function.
        - noise_level (float): Noise level.
        Returns:
        - None (updates self.sampling_amplitudes).
        """
        time_step = 1 / sample_rate
        sampling_time_max = self.simulated_data_list[-1]
        sampling_points = int(sampling_time_max / time_step)
        print("time_step:", time_step, " sampling_points:", sampling_points)
        self.sampling_amplitudes = [0] * sampling_points

        for i in range(0, len(self.simulated_data_list)):
            center_time = self.simulated_data_list[i]
            amplitude = self.amplitudes[i]  # Use amplitude from CSV or ROOT (default 1.0 for ROOT)
            i_min = max(int((center_time - 5 * signal_width) / time_step), 0)
            i_max = min(int((center_time + 5 * signal_width) / time_step), sampling_points - 1)

            if i <= 10 or i % 10000 == 0:
                print("sampling signal, %3.1f" % (i / len(self.simulated_data_list) * 100), "[%] completed...")

            for j in range(i_min, i_max):
                sampling_time = j * time_step
                #sampling_amplitude = amplitude * (1 / (np.sqrt(2 * np.pi) * signal_width)) * np.exp(-0.5 * ((sampling_time - center_time) / signal_width) ** 2)
                sampling_amplitude = amplitude * np.exp(-0.5 * ((sampling_time - center_time) / signal_width) ** 2)
                self.sampling_amplitudes[j] += sampling_amplitude

        for i in range(0, len(self.sampling_amplitudes)):
            random_number = random.random() * noise_level
            self.sampling_amplitudes[i] += random_number

    def generate_sampling_amplitudes_sin(self, sample_rate):
        """
        Generate sinusoidal sampling amplitudes scaled by CSV amplitudes.
        
        Parameters:
        - sample_rate (float): Sampling rate in Hertz.
        Returns:
        - numpy.ndarray: Array of sampling amplitudes.
        """
        time_step = 1 / sample_rate
        sampling_points = int(100 / time_step)
        self.sampling_amplitudes = np.zeros(sampling_points)
        for i in range(0, sampling_points):
            sampling_time = i * time_step
            # Find the closest time in simulated_data_list and use corresponding amplitude
            if len(self.simulated_data_list) > 0:
                closest_idx = min(range(len(self.simulated_data_list)), key=lambda j: abs(self.simulated_data_list[j] - sampling_time))
                amplitude = self.amplitudes[closest_idx]
            else:
                amplitude = 1.0  # Default amplitude if no data
            sampling_amplitude = amplitude * np.sin(2 * np.pi * 0.2 * sampling_time)
            self.sampling_amplitudes[i] = sampling_amplitude
        return self.sampling_amplitudes


    def generate_sampling_amplitudes_exp(self, sample_rate, tau=2000.0, noise_level=0):
        """
        Generate exponential build-up and decay sampling amplitudes based on simulated data.
      
        Parameters:
        - sample_rate (float): Sampling rate in Hertz.
        - tau (float): Time constant for exponential decay/build-up (in seconds).
        - noise_level (float): Noise level.
        Returns:
        - None (updates self.sampling_amplitudes).
        """
        print("Generate exponential build-up and decay sampling amplitudes based on simulated data.")
        time_step = 1 / sample_rate
        sampling_time_max = self.simulated_data_list[-1] if self.simulated_data_list else 10.0  # Default to 10s if no data
        sampling_points = int(sampling_time_max / time_step)
        self.sampling_amplitudes = np.zeros(sampling_points)
        
        total_signals = len(self.simulated_data_list)
        for signal_idx, center_time in enumerate(self.simulated_data_list):
            # Determine amplitude for this signal
            amplitude = self.amplitudes[signal_idx] if len(self.simulated_data_list) > 0 else 1.0
            
            # Define the range based on 5 tau width
            i_min = max(int((center_time - 5 * tau) / time_step), 0)
            i_max = min(int((center_time + 5 * tau) / time_step), sampling_points - 1)
            
            # Calculate peak time index for this signal
            peak_time_idx = int(center_time / time_step)
            
            # Generate signal and add to sampling_amplitudes
            for i in range(i_min, i_max + 1):
                sampling_time = i * time_step
                if i <= peak_time_idx:
                    # Build-up phase: V/V_0 = 1 - e^(-(t - t_start)/tau)
                    t_relative = sampling_time - (i_min * time_step)
                    sampling_amplitude = amplitude * (1 - np.exp(-t_relative / tau))
                else:
                    # Decay phase: V/V_0 = e^(-(t - t_peak)/tau)
                    t_relative = sampling_time - center_time
                    sampling_amplitude = amplitude * np.exp(-abs(t_relative) / tau)
                self.sampling_amplitudes[i] += sampling_amplitude
            
            # Update progress bar
            progress = (signal_idx + 1) / total_signals * 100
            bar_length = 50
            blocks = int(bar_length * signal_idx / total_signals)
            print(f"\rProgress: [{'#' * blocks}{' ' * (bar_length - blocks)}] {progress:.1f}%", end="")
        
        # Add noise after all signals are superimposed
        for i in range(len(self.sampling_amplitudes)):
            random_number = random.random() * noise_level
            self.sampling_amplitudes[i] += random_number
        
        print("\rProgress: [########################################] 100.0%")  # Complete the progress bar
    
    def convert_to_frequency(self, sample_rate):
        """
        Convert a list of time points to a list of frequencies using FFT.
        
        Parameters:
        - sample_rate (float): The rate at which the signal is sampled.
        Returns:
        - None (updates self.frequencies and self.fft_result).
        """
        num_samples = len(self.sampling_amplitudes)
        self.fft_result = np.abs(np.fft.fft(self.sampling_amplitudes)) ** 2
        self.frequencies = np.fft.fftfreq(num_samples, 1.0 / sample_rate)

    def plot_spectrum(self, sample_rate, plot_time_min, plot_time_max, plot_fre_min, plot_fre_max, plot_opt, output):
        """
        Plot the frequency spectrum.
        
        Parameters:
        - sample_rate (float): The sample rate.
        - plot_time_min, plot_time_max (float): The time range of the plot.
        - plot_fre_min, plot_fre_max (float): The frequency range of the plot.
        - plot_opt (int): Plot option (2 for time and frequency, else only frequency).
        """
        if self.canvas is None:
            self.canvas = TCanvas("canvas", "canvas", 0, 0, 600, 600)
            self.canvas.Divide(1, 2)
        
        self.canvas.cd(1)
        gPad.SetTopMargin(0.07)
        gPad.SetBottomMargin(0.15)
        gPad.SetRightMargin(0.08)
        print("1. Plotting the sampling signal.")
        print("plot_time_min = ",plot_time_min)
        print("plot_time_max = ",plot_time_max)
        self.g_sampling_amplitude_time.SetName("g_sampling_amplitude_time")
        title = f"sampling signal with sampling rate {sample_rate}"
        self.g_sampling_amplitude_time.SetTitle(title)
        self.g_sampling_amplitude_time.Set(0)
        if plot_opt == 2:
            time_step = 1 / sample_rate
            index_plot_time_min = int(plot_time_min / time_step)
            index_plot_time_max = min(int(plot_time_max / time_step), len(self.sampling_amplitudes))
            for i in range(index_plot_time_min, index_plot_time_max):
                sampling_time = i * time_step * 1e9
                self.g_sampling_amplitude_time.SetPoint(i - index_plot_time_min, sampling_time, self.sampling_amplitudes[i])
                
            if self.g_sampling_amplitude_time.GetN() < 1e6:
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

        print("2. Plotting the frequency spectrum")
        half_index = len(self.frequencies) // 2
        index_min = np.searchsorted(self.frequencies[:half_index], plot_fre_min)
        index_max = np.searchsorted(self.frequencies[:half_index], plot_fre_max)
        index_max = min(index_max, half_index - 1)

        fre_N = index_max - index_min
        fre_min = self.frequencies[index_min]
        fre_max = self.frequencies[index_max]
        self.h_fft_result_frequencies.SetName("h_fft_result_frequencies")
        self.h_fft_result_frequencies.SetTitle("h_fft_result_frequencies")
        self.h_fft_result_frequencies.SetBins(int(fre_N), fre_min, fre_max)
        self.h_fft_result_frequencies.Reset()

        for i in range(index_min, index_max):
            x = self.frequencies[i]
            y = np.abs(self.fft_result[i])
            Nx = self.h_fft_result_frequencies.GetXaxis().FindBin(x)
            self.h_fft_result_frequencies.SetBinContent(Nx, y)

        self.h_fft_result_frequencies.Draw("hist")
        self.h_fft_result_frequencies.GetXaxis().SetRangeUser(0, sample_rate / 2)
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

    def run(self, sampling_method, sample_rate, signal_width, noise_level, plot_opt, plot_time_min, plot_time_max, plot_fre_min, plot_fre_max, start_step, SimulatedDataFile, TOF, Output):
        """
        Run the TimeToFrequencyConverter.
        
        Args:
        - sampling_method (int): The method used for sampling (1 for sine, 2 for exp, else Gaussian).
        - sample_rate (float): The sampling rate in Hertz.
        - signal_width (float): The width of the signal in seconds.
        - noise_level (float): The amplitude of noise level.
        - plot_opt (int): The plotting option.
        - plot_time_min (float): The minimum time for plotting.
        - plot_time_max (float): The maximum time for plotting.
        - plot_fre_min (float): The minimum frequency for plotting.
        - plot_fre_max (float): The maximum frequency for plotting.
        - start_step (int): The starting step for the conversion process.
        - SimulatedDataFile (str): The file containing simulated data.
        - TOF (int): TOF value for ROOT files.
        - Output (str): Output file name.
        
        Returns:
        - bool: True if successful, False otherwise.
        """
        # Convert sampling_method to integer
        try:
            sampling_method = int(sampling_method)
        except (ValueError, TypeError):
            print("### Error: Invalid sampling_method value. Must be an integer.")
            return False
        print("\n################################################")
        print("### Welcome to use TimeToFrequencyConverter. ###")
        print("################################################\n")
        start_time = time.time()
        table = PrettyTable()
        table.field_names = ["Step", "Runtime (seconds)"]
        self.simulated_data_list = []
        self.amplitudes = []
        self.sampling_amplitudes = []
        self.frequencies = []
        self.fft_result = []

        if start_step == 1:
            print("-step 1. Read simulated data and creating sampling signal.")
            start_time1 = time.time()
            file_extension = os.path.splitext(SimulatedDataFile)[1].lower()
            if file_extension == '.root':
                success = self.ReadTimeData_root(SimulatedDataFile, TOF,plot_time_min,plot_time_max)
            elif file_extension == '.csv':
                success = self.ReadTimeData_csv(SimulatedDataFile,plot_time_min,plot_time_max)
            else:
                print(f"### Error: Unsupported file extension '{file_extension}'. Supported extensions are '.root' and '.csv'.")
                return False
            if not success or len(self.simulated_data_list) == 0:
                print("Error: simulated_data_list is empty. Please check the --SimulatedDataFile and --TOF (for ROOT) options.")
                return False
            print("chenrj sampling_method ", sampling_method, "type:", type(sampling_method))
            if sampling_method == 1:
                self.sampling_amplitudes = self.generate_sampling_amplitudes_sin(sample_rate)
            elif sampling_method == 2:
                print("chenrj sampling_method2 ", sampling_method, "type:", type(sampling_method))
                self.generate_sampling_amplitudes_exp(sample_rate, tau=signal_width, noise_level=noise_level)
            else:
                print("chenrj else branch, sampling_method:", sampling_method, "type:", type(sampling_method))
                self.generate_sampling_amplitudes(sample_rate, signal_width, noise_level)
        
            end_time1 = time.time()
            runtime1 = end_time1 - start_time1
            np.save("sampling_amplitudes.npy", self.sampling_amplitudes)
            print(f"Runtime: {runtime1:.4f} seconds")
            table.add_row(["1.Reading and Generating Data", f"{runtime1:.4f}"])

        if start_step >= 1 and start_step <= 2:
            print("-step 2. Do fft with the sampling data.")
            start_time2 = time.time()
            if start_step == 2:
                if not self.load_sampling_amplitudes():
                    return False
            self.convert_to_frequency(sample_rate)
            end_time2 = time.time()
            runtime2 = end_time2 - start_time2
            np.save("frequencies.npy", self.frequencies)
            np.save("fft_result.npy", self.fft_result)
            print(f"Runtime: {runtime2:.4f} seconds")
            table.add_row(["2.FFT Calculation", f"{runtime2:.4f}"])

        if start_step >= 1 and start_step <= 3:
            print("-step 3. Plot the spectrum and save results.")
            start_time3 = time.time()
            flag_data1 = flag_data2 = flag_data3 = True
            if start_step == 3:
                flag_data1 = self.load_sampling_amplitudes()
                flag_data2 = self.load_frequencies()
                flag_data3 = self.load_fft_result()

            if flag_data1 and flag_data2 and flag_data3:
                self.plot_spectrum(sample_rate, plot_time_min, plot_time_max, plot_fre_min, plot_fre_max, plot_opt, output = Output)
                fout = TFile(Output, "recreate")
                self.canvas.Write()
                self.g_sampling_amplitude_time.Write()
                self.h_fft_result_frequencies.Write()
                fout.Close()
                Output_png = Output.replace(".root", ".png")
                self.canvas.Print(Output_png)
            else:
                print("No data exist for plot. Please run earlier steps first!")
                return False

            end_time3 = time.time()
            runtime3 = end_time3 - start_time3
            print(f"Runtime: {runtime3:.4f} seconds")
            table.add_row(["3.Plotting Spectrum", f"{runtime3:.4f}"])

        end_time = time.time()
        total_runtime = end_time - start_time
        print(f"Total runtime: {total_runtime:.4f} seconds")
        table.add_row(["4.Total Runtime", f"{total_runtime:.4f}"])
        print(table)
        return True