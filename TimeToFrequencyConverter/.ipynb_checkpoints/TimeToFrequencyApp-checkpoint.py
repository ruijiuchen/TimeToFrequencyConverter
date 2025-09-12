import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import filedialog
import toml
from ROOT import TCanvas, gROOT, TGraph, TFile
import numpy as np
from . import TimeToFrequencyConverter as TFC

class TimeToFrequencyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Time to frequency converter GUI")
     
        # Configure ttk Style before creating widgets
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TButton", font=("Helvetica", 30))
        self.style.configure("TLabel", font=("Helvetica", 30))
        self.style.configure("TEntry", font=("Helvetica", 30))
        
        # Initialize variables
        self.sampling_method_var = tk.StringVar()
        self.start_step_var = tk.StringVar()
        self.sample_rate_var = tk.DoubleVar()
        self.signal_width_var = tk.DoubleVar()
        self.noise_level_var = tk.DoubleVar()
        self.low_freq_hz_var = tk.DoubleVar()
        self.high_freq_hz_var = tk.DoubleVar()
        self.plot_opt_var = tk.IntVar()
        self.plot_time_min_var = tk.DoubleVar()
        self.plot_time_max_var = tk.DoubleVar()
        self.frame_length_seconds_var = tk.DoubleVar()
        self.plot_fre_min_var = tk.DoubleVar()
        self.plot_fre_max_var = tk.DoubleVar()
        self.start_step_var = tk.IntVar()
        self.simulated_data_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.tof_var = tk.IntVar()
        self.enable_freq_filter_var = tk.BooleanVar()  # New variable for checkbox
        
        # Set default values
        self.sampling_method_var.set(0)
        self.start_step_var.set(0)
        self.sample_rate_var.set(0)
        self.signal_width_var.set(0)
        self.noise_level_var.set(0)
        self.low_freq_hz_var.set(0)
        self.high_freq_hz_var.set(0)
        self.plot_opt_var.set(0)
        self.plot_time_min_var.set(0)
        self.plot_time_max_var.set(0)
        self.frame_length_seconds_var.set(0)
        self.plot_fre_min_var.set(0)
        self.plot_fre_max_var.set(0)
        self.simulated_data_var.set("")
        self.tof_var.set(0)
        self.enable_freq_filter_var.set(False)  # Checkbox off by default
     
        self.create_widgets()
        self.converter = TFC.TimeToFrequencyConverter()
     
    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_propagate(False)
        main_frame.config(width=500, height=500)
        
        # Create a menu bar
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save", command=self.save_file)
        index_row = 0
     
        # Create input fields with ttk controls
        ttk.Label(main_frame, text="Start_step[1.raw data/2.FFT/3.Plot]").grid(row=index_row, column=0, sticky=tk.W)
        start_step_entry = ttk.Entry(main_frame, textvariable=self.start_step_var)
        start_step_entry.grid(row=index_row, column=1)
        index_row += 1
        
        ttk.Label(main_frame, text="Sampling Method[1.sin/2.exp/3.gaus]").grid(row=index_row, column=0, sticky=tk.W)
        sampling_method_entry = ttk.Entry(main_frame, textvariable=self.sampling_method_var)
        sampling_method_entry.grid(row=index_row, column=1)
        index_row += 1
        
        ttk.Label(main_frame, text="Sampling Rate [Hz]").grid(row=index_row, column=0, sticky=tk.W)
        sample_rate_entry = ttk.Entry(main_frame, textvariable=self.sample_rate_var)
        sample_rate_entry.grid(row=index_row, column=1)
        index_row += 1
        
        ttk.Label(main_frame, text="Signal Width [s]").grid(row=index_row, column=0, sticky=tk.W)
        signal_width_entry = ttk.Entry(main_frame, textvariable=self.signal_width_var)
        signal_width_entry.grid(row=index_row, column=1)
        index_row += 1
        
        ttk.Label(main_frame, text="noise level").grid(row=index_row, column=0, sticky=tk.W)
        noise_level_entry = ttk.Entry(main_frame, textvariable=self.noise_level_var)
        noise_level_entry.grid(row=index_row, column=1)
        index_row += 1
       
        # Checkbox to enable/disable frequency filter entries
        ttk.Checkbutton(
            main_frame, 
            text="Enable Frequency Filter",
            variable=self.enable_freq_filter_var,
            command=self.toggle_freq_entries
        ).grid(row=index_row, column=0, columnspan=2, sticky=tk.W)
        index_row += 1
       
        ttk.Label(main_frame, text="low_freq[filter,Hz]").grid(row=index_row, column=0, sticky=tk.W)
        self.low_freq_hz_entry = ttk.Entry(main_frame, textvariable=self.low_freq_hz_var, state='disabled')
        self.low_freq_hz_entry.grid(row=index_row, column=1)
        index_row += 1
       
        ttk.Label(main_frame, text="high_freq[filter,Hz]").grid(row=index_row, column=0, sticky=tk.W)
        self.high_freq_hz_entry = ttk.Entry(main_frame, textvariable=self.high_freq_hz_var, state='disabled')
        self.high_freq_hz_entry.grid(row=index_row, column=1)
        index_row += 1
        
        ttk.Label(main_frame, text="Simulation Data[.cvs/.root]").grid(row=index_row, column=0, sticky=tk.W)
        simulated_data_entry = ttk.Entry(main_frame, textvariable=self.simulated_data_var)
        simulated_data_entry.grid(row=index_row, column=1)
        index_row += 1
        
        ttk.Label(main_frame, text="TOF [point]").grid(row=index_row, column=0, sticky=tk.W)
        tof_entry = ttk.Entry(main_frame, textvariable=self.tof_var)
        tof_entry.grid(row=index_row, column=1)
        index_row += 1
        
        ttk.Label(main_frame, text="Output [***.root]").grid(row=index_row, column=0, sticky=tk.W)
        output_entry = ttk.Entry(main_frame, textvariable=self.output_var)
        output_entry.grid(row=index_row, column=1)
        index_row += 1
        
        run_button = ttk.Button(main_frame, text="Run", command=self.run_worker)
        run_button.grid(row=index_row, column=1)
        index_row += 1
        
        ttk.Label(main_frame, text="Plot Option[1/2/3]").grid(row=index_row, column=0, sticky=tk.W)
        plot_opt_entry = ttk.Entry(main_frame, textvariable=self.plot_opt_var)
        plot_opt_entry.grid(row=index_row, column=1)
        index_row += 1
        
        ttk.Label(main_frame, text="Plot Time Min [s]").grid(row=index_row, column=0, sticky=tk.W)
        plot_time_min_entry = ttk.Entry(main_frame, textvariable=self.plot_time_min_var)
        plot_time_min_entry.grid(row=index_row, column=1)
        index_row += 1
        
        ttk.Label(main_frame, text="Plot Time Max [s]").grid(row=index_row, column=0, sticky=tk.W)
        plot_time_max_entry = ttk.Entry(main_frame, textvariable=self.plot_time_max_var)
        plot_time_max_entry.grid(row=index_row, column=1)
        index_row += 1
        
        ttk.Label(main_frame, text="Frame length[s]").grid(row=index_row, column=0, sticky=tk.W)
        frame_length_seconds_entry = ttk.Entry(main_frame, textvariable=self.frame_length_seconds_var)
        frame_length_seconds_entry.grid(row=index_row, column=1)
        index_row += 1
        
        ttk.Label(main_frame, text="Plot Frequency Min [Hz]").grid(row=index_row, column=0, sticky=tk.W)
        plot_fre_min_entry = ttk.Entry(main_frame, textvariable=self.plot_fre_min_var)
        plot_fre_min_entry.grid(row=index_row, column=1)
        index_row += 1
        
        ttk.Label(main_frame, text="Plot Frequency Max [Hz]").grid(row=index_row, column=0, sticky=tk.W)
        plot_fre_max_entry = ttk.Entry(main_frame, textvariable=self.plot_fre_max_var)
        plot_fre_max_entry.grid(row=index_row, column=1)
        index_row += 1
        
        plot_button = ttk.Button(main_frame, text="Re-Plot", command=self.plot_worker)
        plot_button.grid(row=index_row, column=1)
    
    def toggle_freq_entries(self):
        """Enable or disable frequency filter entries based on checkbox state."""
        state = 'normal' if self.enable_freq_filter_var.get() else 'disabled'
        self.low_freq_hz_entry.configure(state=state)
        self.high_freq_hz_entry.configure(state=state)
        if not self.enable_freq_filter_var.get():
            self.low_freq_hz_var.set(0)
            self.high_freq_hz_var.set(0)
    
    def open_file(self, file_path=None):
        if file_path is None:
            file_path = filedialog.askopenfilename(title="Open Parameter File", filetypes=[("TOML files", "*.toml")])
        if file_path:
            try:
                with open(file_path, "r") as toml_file:
                    params = toml.load(toml_file)['parameters']
                    self.start_step_var.set(params.get("start_step", 2))
                    self.sampling_method_var.set(params.get("sampling_method", 2))
                    self.sample_rate_var.set(params.get("sample_rate", 6e6))
                    self.signal_width_var.set(params.get("signal_width", 1e-8))
                    self.noise_level_var.set(params.get("noise_level", 0))
                    self.low_freq_hz_var.set(params.get("low_freq_hz", 0))
                    self.high_freq_hz_var.set(params.get("high_freq_hz", 0))
                    self.plot_opt_var.set(params.get("plot_opt", 2))
                    self.plot_time_min_var.set(params.get("plot_time_min", 0))
                    self.plot_time_max_var.set(params.get("plot_time_max", 1300e-9))
                    self.frame_length_seconds_var.set(params.get("frame_length_seconds", 1300e-9))
                    self.plot_fre_min_var.set(params.get("plot_fre_min", 0))
                    self.plot_fre_max_var.set(params.get("plot_fre_max", 2e6))
                    self.start_step_var.set(params.get("start_step", 1))
                    self.simulated_data_var.set(params.get("simulated_data", "out.root"))
                    self.tof_var.set(params.get("TOF", 0))
                    self.output_var.set(params.get("output", "simulation_result.root"))
                    # Update checkbox state based on loaded values
                    self.enable_freq_filter_var.set(
                        params.get("low_freq_hz", 0) != 0 or params.get("high_freq_hz", 0) != 0
                    )
                    self.toggle_freq_entries()  # Update entry states
                    return True
            except Exception as e:
                print(f"Error while loading parameters: {e}")
                return False
         
    def save_file(self):
        file_path = filedialog.asksaveasfilename(title="Save Parameter File", defaultextension=".toml", filetypes=[("TOML files", "*.toml")])
        if file_path:
            try:
                parameters = {
                    'start_step': self.start_step_var.get(),
                    'sampling_method': self.sampling_method_var.get(),
                    'sample_rate': self.sample_rate_var.get(),
                    'signal_width': self.signal_width_var.get(),
                    'noise_level': self.noise_level_var.get(),
                    'low_freq_hz': self.low_freq_hz_var.get(),
                    'high_freq_hz': self.high_freq_hz_var.get(),
                    'plot_opt': self.plot_opt_var.get(),
                    'plot_time_min': self.plot_time_min_var.get(),
                    'plot_time_max': self.plot_time_max_var.get(),
                    'frame_length_seconds': self.frame_length_seconds_var.get(),
                    'plot_fre_min': self.plot_fre_min_var.get(),
                    'plot_fre_max': self.plot_fre_max_var.get(),
                    'simulated_data': self.simulated_data_var.get(),
                    'TOF': self.tof_var.get(),
                    'output': self.output_var.get()
                }
                with open(file_path, "w") as toml_file:
                    toml.dump({'parameters': parameters}, toml_file)
            except Exception as e:
                print(f"Error while saving parameters: {e}")
            finally:
                print("Saved!")
             
    def run_worker(self):
        start_step = 1
        self.converter.run(int(self.sampling_method_var.get()), self.sample_rate_var.get(), self.signal_width_var.get(),
                           self.noise_level_var.get(), self.low_freq_hz_var.get(), self.high_freq_hz_var.get(), 
                           self.plot_opt_var.get(), self.plot_time_min_var.get(), self.plot_time_max_var.get(), self.frame_length_seconds_var.get(), 
                           self.plot_fre_min_var.get(), self.plot_fre_max_var.get(), int(self.start_step_var.get()),
                           self.simulated_data_var.get(), self.tof_var.get(), self.output_var.get())
     
    def plot_worker(self):
        start_step = 3
        self.converter.run(int(self.sampling_method_var.get()), self.sample_rate_var.get(), self.signal_width_var.get(),
                           self.noise_level_var.get(), self.low_freq_hz_var.get(), self.high_freq_hz_var.get(), 
                           self.plot_opt_var.get(), self.plot_time_min_var.get(), self.plot_time_max_var.get(), self.frame_length_seconds_var.get(), 
                           self.plot_fre_min_var.get(), self.plot_fre_max_var.get(), start_step,
                           self.simulated_data_var.get(), self.tof_var.get(), self.output_var.get())
        print("updated!")