import sys
import time
import numpy as np
import random
from ROOT import TCanvas, gROOT, TGraph, TApplication, TH1F, gPad, gStyle, TFile
from prettytable import PrettyTable
import argparse
import os
from numba import cuda
import math

from scipy import signal
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
            if t.Point == point_number:
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
                    #if plot_time_min <= time_value <= plot_time_max:
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



    def apply_bandpass_filter(self, signal_data, sample_rate, low_freq_hz=2.4e8, high_freq_hz=2.5e8, numtaps=101, window='hamming'):
        """
        应用 FIR 带通滤波器，仅允许指定频率范围的信号通过。
    
        参数:
        - signal_data (numpy.ndarray): 输入信号数据。
        - sample_rate (float): 采样率（赫兹）。
        - low_freq_hz (float): 通带下限频率（赫兹），默认 240 MHz。
        - high_freq_hz (float): 通带上限频率（赫兹），默认 250 MHz。
        - numtaps (int): 滤波器阶数（奇数），默认 101。
        - window (str): 窗函数类型，默认 'hamming'。
    
        返回:
        - numpy.ndarray: 滤波后的信号。
        """
        import numpy as np
        from scipy import signal
    
        # 确保参数是标量
        try:
            sample_rate = float(self.parse_scalar(sample_rate))
            low_freq_hz = float(self.parse_scalar(low_freq_hz))
            high_freq_hz = float(self.parse_scalar(high_freq_hz))
            
        except (ValueError, TypeError) as e:
            print(f"### Error: Invalid parameter type in apply_bandpass_filter - {e}")
            return signal_data
    
        # 检查奈奎斯特准则
        if sample_rate < 2 * high_freq_hz or (low_freq_hz==0 and high_freq_hz==0):
            print(f"警告：采样率 {sample_rate/1e6:.1f} MHz 低于 {2*high_freq_hz/1e6:.1f} MHz，无法有效滤波 {low_freq_hz/1e6:.1f}-{high_freq_hz/1e6:.1f} MHz 信号！")
            return signal_data
    
        print(f"应用 {low_freq_hz/1e6:.1f}-{high_freq_hz/1e6:.1f} MHz 带通滤波器...")
        nyquist_rate = sample_rate / 2
        low_freq = low_freq_hz / nyquist_rate  # 归一化频率
        high_freq = high_freq_hz / nyquist_rate  # 归一化频率
        if numtaps % 2 == 0:
            numtaps += 1  # 确保奇数阶
        taps = signal.firwin(numtaps, [low_freq, high_freq], pass_zero=False, window=window)
        filtered_signal = signal.filtfilt(taps, 1.0, signal_data)
        print("带通滤波完成！")
        return filtered_signal

    def generate_sampling_amplitudes_exp(self, sample_rate, tau=2000.0, noise_level=0):
        """
        生成基于模拟数据的指数型上升和衰减采样幅度，并应用 240-250 MHz 带通滤波器。
    
        参数:
        - sample_rate (float): 采样率（赫兹）。
        - tau (float): 指数衰减/上升的时间常数（秒）。
        - noise_level (float): 噪声水平。
        返回:
        - 无（更新 self.sampling_amplitudes）。
        """
        print("生成基于模拟数据的指数型上升和衰减采样幅度。")
        time_step = 1 / sample_rate
        sampling_time_max = self.simulated_data_list[-1] if self.simulated_data_list else 10.0
        sampling_points = int(sampling_time_max / time_step)
        self.sampling_amplitudes = np.zeros(sampling_points)
    
        # 将列表转换为数组以进行向量化操作
        center_times = np.array(self.simulated_data_list)
        amplitudes = np.array(self.amplitudes) if hasattr(self, 'amplitudes') and len(self.amplitudes) == len(self.simulated_data_list) else np.ones(len(self.simulated_data_list))
    
        # 创建所有采样点的时间数组
        time_array = np.arange(sampling_points) * time_step
    
        total_signals = len(center_times)
        for signal_idx, (center_time, amplitude) in enumerate(zip(center_times, amplitudes)):
            # 基于 5 * tau 定义范围
            i_min = max(int((center_time - 5 * tau) / time_step), 0)
            i_max = min(int((center_time + 5 * tau) / time_step), sampling_points - 1)
            if i_min >= i_max:
                continue  # 跳过无效范围
    
            # 提取该信号的时间片
            time_slice = time_array[i_min:i_max + 1]
            peak_time_idx = int(center_time / time_step)
    
            # 向量化幅度计算
            sampling_amplitude = np.zeros_like(time_slice)
            build_up_mask = time_slice <= center_time
            decay_mask = ~build_up_mask
    
            # 上升阶段：V/V_0 = 1 - e^(-(t - t_start)/tau)
            t_relative_build_up = time_slice[build_up_mask] - (i_min * time_step)
            sampling_amplitude[build_up_mask] = amplitude * (1 - np.exp(-t_relative_build_up / tau))
    
            # 衰减阶段：V/V_0 = e^(-(t - t_peak)/tau)
            t_relative_decay = time_slice[decay_mask] - center_time
            sampling_amplitude[decay_mask] = amplitude * np.exp(-np.abs(t_relative_decay) / tau)
    
            # 添加到 sampling_amplitudes
            self.sampling_amplitudes[i_min:i_max + 1] += sampling_amplitude
    
            # 减少进度条更新频率
            if (signal_idx + 1) % 100 == 0 or signal_idx == total_signals - 1:
                progress = (signal_idx + 1) / total_signals * 100
                bar_length = 50
                blocks = int(bar_length * signal_idx / total_signals)
                print(f"\r进度: [{'#' * blocks}{' ' * (bar_length - blocks)}] {progress:.1f}%", end="", flush=True)
    
        # 向量化添加噪声
        if noise_level > 0:
            self.sampling_amplitudes += np.random.uniform(0, noise_level, sampling_points)
        
        print("\r进度: [########################################] 100.0%")

    @staticmethod
    @cuda.jit
    def _generate_exp_kernel(simulated_data_list, amplitudes, sample_rate, tau, sampling_amplitudes):
        """
        CUDA kernel to compute exponential build-up and decay for each signal in parallel.
        """
        signal_idx = cuda.grid(1)  # Get the thread index
        if signal_idx < len(simulated_data_list):
            center_time = simulated_data_list[signal_idx]
            amplitude = amplitudes[signal_idx]
            time_step = 1.0 / sample_rate
            sampling_points = len(sampling_amplitudes)

            # Define the range based on 5 tau width
            i_min = max(int((center_time - 5 * tau) / time_step), 0)
            i_max = min(int((center_time + 5 * tau) / time_step), sampling_points - 1)
            peak_time_idx = int(center_time / time_step)

            for i in range(i_min, i_max + 1):
                sampling_time = i * time_step
                if i <= peak_time_idx:
                    # Build-up phase: V/V_0 = 1 - e^(-(t - t_start)/tau)
                    t_relative = sampling_time - (i_min * time_step)
                    sampling_amplitude = amplitude * (1 - math.exp(-t_relative / tau))
                else:
                    # Decay phase: V/V_0 = e^(-(t - t_peak)/tau)
                    t_relative = sampling_time - center_time
                    sampling_amplitude = amplitude * math.exp(-abs(t_relative) / tau)
                # Use atomic addition to avoid race conditions
                cuda.atomic.add(sampling_amplitudes, i, sampling_amplitude)

    def generate_sampling_amplitudes_exp_gpu(self, sample_rate, tau=2000.0, noise_level=0):
        """
        Generate exponential build-up and decay sampling amplitudes using CUDA.
        """
        print("Generate exponential build-up and decay sampling amplitudes based on simulated data using CUDA.")
        time_step = 1 / sample_rate
        sampling_time_max = self.simulated_data_list[-1] if self.simulated_data_list else 10.0
        sampling_points = int(sampling_time_max / time_step)
        self.sampling_amplitudes = np.zeros(sampling_points, dtype=np.float64)

        # Transfer data to GPU
        d_simulated_data_list = cuda.to_device(np.array(self.simulated_data_list, dtype=np.float64))
        d_amplitudes = cuda.to_device(np.array(self.amplitudes, dtype=np.float64))
        d_sampling_amplitudes = cuda.to_device(self.sampling_amplitudes)

        # Configure CUDA grid and block sizes
        threads_per_block = 256
        blocks_per_grid = math.ceil(len(self.simulated_data_list) / threads_per_block)

        # Launch the CUDA kernel
        TimeToFrequencyConverter._generate_exp_kernel[blocks_per_grid, threads_per_block](
            d_simulated_data_list, d_amplitudes, sample_rate, tau, d_sampling_amplitudes
        )

        # Copy results back to host
        self.sampling_amplitudes = d_sampling_amplitudes.copy_to_host()

        # Add noise on CPU
        for i in range(len(self.sampling_amplitudes)):
            random_number = random.random() * noise_level
            self.sampling_amplitudes[i] += random_number

        print("Progress: [########################################] 100.0%")


    def convert_to_frequency(self, sample_rate, frame_length_seconds, overlap=0.5):
        """
        Convert a list of time points to a 3D spectrogram using FFT with sliding windows,
        and compute the average power spectrum across all frames.
        
        Parameters:
        - sample_rate (float): The rate at which the signal is sampled (samples per second).
        - frame_length_seconds (float): Length of each FFT frame in seconds.
        - overlap (float): Fraction of frame overlap (default 0.5 for 50% overlap).
        
        Returns:
        - None (updates self.frequencies, self.spectrogram, and self.avg_spectrum).
        """
        # Convert frame length from seconds to samples
        frame_length = int(frame_length_seconds * sample_rate)
        if frame_length < 1:
            raise ValueError("Frame length in samples must be at least 1. Increase frame_length_seconds or sample_rate.")
    
        # Initialize parameters
        num_samples = len(self.sampling_amplitudes)
        hop_size = int(frame_length * (1 - overlap))
        if hop_size < 1:
            raise ValueError("Hop size must be at least 1. Decrease overlap or increase frame_length_seconds.")
    
        # Calculate number of frames
        num_frames = (num_samples - frame_length) // hop_size + 1
        if num_frames < 1:
            raise ValueError("Not enough samples for at least one frame. Increase signal length or decrease frame_length_seconds.")
    
        # Initialize spectrogram array (3D: frequency x time x power)
        self.spectrogram = np.zeros((frame_length // 2 + 1, num_frames))
        
        # Compute FFT for each frame
        for i in range(num_frames):
            start = i * hop_size
            end = start + frame_length
            if end > num_samples:
                break
            frame = self.sampling_amplitudes[start:end]
            fft_result = np.abs(np.fft.fft(frame))[:frame_length // 2 + 1] ** 2
            self.spectrogram[:, i] = fft_result
        
        # Compute frequency bins (same for all frames)
        self.frequencies = np.fft.fftfreq(frame_length, 1.0 / sample_rate)[:frame_length // 2 + 1]
        
        # Compute average spectrum across all frames (average along time axis)
        self.avg_spectrum = np.mean(self.spectrogram, axis=1)

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
            y = np.abs(self.avg_spectrum[i])
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

    def parse_scalar(self, value):
        """将输入解析为标量浮点数"""
        import numpy as np
        try:
            if isinstance(value, str):
                # 处理字符串输入，如 "1000000000.0" 或 "[1000000000.0]"
                value = value.strip('[]').replace(' ', '')
                return float(value)
            elif isinstance(value, (list, np.ndarray)):
                # 处理数组或列表，取第一个元素
                return float(value[0]) if len(value) > 0 else float(value)
            elif isinstance(value, np.floating):
                # 处理 numpy 浮点类型
                return float(value)
            return float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid input value: {value} - {e}")
    
    def run(self, sampling_method, sample_rate, signal_width, noise_level, low_freq_hz, high_freq_hz, plot_opt, plot_time_min, plot_time_max,frame_length_seconds, plot_fre_min, plot_fre_max, start_step, SimulatedDataFile, TOF, Output):
        """
        Run the TimeToFrequencyConverter.
        
        Args:
        - sampling_method (int): The method used for sampling (1 for sine, 2 for exp, else Gaussian).
        - sample_rate (float): The sampling rate in Hertz.
        - signal_width (float): The width of the signal in seconds.
        - noise_level (float): The amplitude of noise level.
        - low_freq_hz, high_freq_hz, low and high frequency of bandpass_filter
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
        # 参数类型转换和验证
        try:
            sampling_method = int(self.parse_scalar(sampling_method))
            sample_rate = self.parse_scalar(sample_rate)
            signal_width = self.parse_scalar(signal_width)
            noise_level = self.parse_scalar(noise_level)
            low_freq_hz = self.parse_scalar(low_freq_hz)
            high_freq_hz = self.parse_scalar(high_freq_hz)
            print(f"Parsed inputs: sample_rate={sample_rate} ({type(sample_rate)}), low_freq_hz={low_freq_hz} ({type(low_freq_hz)}), high_freq_hz={high_freq_hz} ({type(high_freq_hz)})")
        except (ValueError, TypeError) as e:
            print(f"### Error: Invalid input parameter type - {e}")
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
            if sampling_method == 1:
                self.sampling_amplitudes = self.generate_sampling_amplitudes_sin(sample_rate)
            elif sampling_method == 2:
                self.generate_sampling_amplitudes_exp(sample_rate, tau=signal_width, noise_level=noise_level)
            elif sampling_method == 3:
                self.generate_sampling_amplitudes_exp_gpu(sample_rate, tau=signal_width, noise_level=noise_level)
            else:
                self.generate_sampling_amplitudes(sample_rate, signal_width, noise_level)
            
            # 应用带通滤波器
            self.sampling_amplitudes = self.apply_bandpass_filter(self.sampling_amplitudes, sample_rate, low_freq_hz, high_freq_hz)
        
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
            
            self.convert_to_frequency(sample_rate,frame_length_seconds)
            
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
                if plot_opt == 2: 
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