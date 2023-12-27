import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import filedialog
import toml  # Add this import
from ROOT import TCanvas, gROOT, TGraph, TFile
import numpy as np
from . import TimeToFrequencyConverter as TFC

class TimeToFrequencyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Time to frequency converter GUI")
        
        # Initialize variables
        self.sampling_method_var = tk.StringVar()
        self.sample_rate_var = tk.DoubleVar()
        self.signal_width_var = tk.DoubleVar()
        self.plot_opt_var = tk.IntVar()
        self.plot_time_min_var = tk.DoubleVar()
        self.plot_time_max_var = tk.DoubleVar()
        self.plot_fre_min_var = tk.DoubleVar()
        self.plot_fre_max_var = tk.DoubleVar()
        self.start_step_var = tk.IntVar()
        self.simulated_data_var = tk.StringVar()
        self.tof_var = tk.IntVar()
        
        # Set default values (you can adjust these as needed)
        self.sampling_method_var.set(0)
        self.sample_rate_var.set(0)
        self.signal_width_var.set(0)
        self.plot_opt_var.set(0)
        self.plot_time_min_var.set(0)
        self.plot_time_max_var.set(0)
        self.plot_fre_min_var.set(0)
        self.plot_fre_max_var.set(0)
        self.simulated_data_var.set("")
        self.tof_var.set(0)
        
        self.create_widgets()
        self.converter = TFC.TimeToFrequencyConverter()
        
    def create_widgets(self):
        # Create and configure the main frame
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        # Set the width and height of the main frame explicitly
        #main_frame.geometry("500x300")  # Adjust the width and height as needed
        # Set the size of the main frame
        main_frame.grid_propagate(False)  # Disable size propagation
        main_frame.config(width=250, height=230)  # Set the desired size

        # Create a menu bar
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        # Create a "File" menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        # Add "Open" option to the "File" menu
        file_menu.add_command(label="Open", command=self.open_file)
        
        # Add "Save" option to the "File" menu
        file_menu.add_command(label="Save", command=self.save_file)

        
        # Create input fields
        ttk.Label(main_frame, text="Sampling Method").grid(row=0, column=0, sticky=tk.W)
        sampling_method_entry = ttk.Entry(main_frame, textvariable=self.sampling_method_var)
        sampling_method_entry.grid(row=0, column=1)

        ttk.Label(main_frame, text="Sampling Rate [Hz]").grid(row=1, column=0, sticky=tk.W)
        sample_rate_entry = ttk.Entry(main_frame, textvariable=self.sample_rate_var)
        sample_rate_entry.grid(row=1, column=1)

        ttk.Label(main_frame, text="Signal Width").grid(row=2, column=0, sticky=tk.W)
        signal_width_entry = ttk.Entry(main_frame, textvariable=self.signal_width_var)
        signal_width_entry.grid(row=2, column=1)
        
        ttk.Label(main_frame, text="Simulation Data").grid(row=3, column=0, sticky=tk.W)
        simulated_data_entry = ttk.Entry(main_frame, textvariable=self.simulated_data_var)
        simulated_data_entry.grid(row=3, column=1)
        
        ttk.Label(main_frame, text="TOF[point]").grid(row=4, column=0, sticky=tk.W)
        tof_entry = ttk.Entry(main_frame, textvariable=self.tof_var)
        tof_entry.grid(row=4, column=1)
        
        # Create a button to run the code
        run_button = ttk.Button(main_frame, text="Run", command=self.run_worker)
        run_button.grid(row=5, column=1)
        
        ttk.Label(main_frame, text="Plot Option").grid(row=6, column=0, sticky=tk.W)
        plot_opt_entry = ttk.Entry(main_frame, textvariable=self.plot_opt_var)
        plot_opt_entry.grid(row=6, column=1)
        
        ttk.Label(main_frame, text="Plot Time Min").grid(row=7, column=0, sticky=tk.W)
        plot_time_min_entry = ttk.Entry(main_frame, textvariable=self.plot_time_min_var)
        plot_time_min_entry.grid(row=7, column=1)
        
        ttk.Label(main_frame, text="Plot Time Max").grid(row=8, column=0, sticky=tk.W)
        plot_time_max_entry = ttk.Entry(main_frame, textvariable=self.plot_time_max_var)
        plot_time_max_entry.grid(row=8, column=1)
        
        ttk.Label(main_frame, text="Plot Frequency Min").grid(row=9, column=0, sticky=tk.W)
        plot_fre_min_entry = ttk.Entry(main_frame, textvariable=self.plot_fre_min_var)
        plot_fre_min_entry.grid(row=9, column=1)
        
        ttk.Label(main_frame, text="Plot Frequency Max").grid(row=10, column=0, sticky=tk.W)
        plot_fre_max_entry = ttk.Entry(main_frame, textvariable=self.plot_fre_max_var)
        plot_fre_max_entry.grid(row=10, column=1)
        
        # Create a button to plot the result
        plot_button = ttk.Button(main_frame, text="Re-Plot", command=self.plot_worker)
        plot_button.grid(row=11, column=1)
        
    #def open_file(self):
    #    file_path = filedialog.askopenfilename(title="Open Parameter File", filetypes=[("Text files", "*.txt")])
        #if file_path:
            # Handle opening the file, e.g., read the content and update your application
    def open_file(self):
        file_path = filedialog.askopenfilename(title="Open Parameter File", filetypes=[("TOML files", "*.toml")])
        if file_path:
            try:
                with open(file_path, "r") as toml_file:
                    params = toml.load(toml_file)['parameters']

                    self.sampling_method_var.set(params.get("sampling_method", 2))
                    self.sample_rate_var.set(params.get("sample_rate", 6e6))
                    self.signal_width_var.set(params.get("signal_width", 1e-8))
                    self.plot_opt_var.set(params.get("plot_opt", 2))
                    self.plot_time_min_var.set(params.get("plot_time_min", 0))
                    self.plot_time_max_var.set(params.get("plot_time_max", 1300e-9))
                    self.plot_fre_min_var.set(params.get("plot_fre_min", 0))
                    self.plot_fre_max_var.set(params.get("plot_fre_max", 2e6))
                    self.start_step_var.set(params.get("start_step", 1))
                    self.simulated_data_var.set(params.get("simulated_data", "out.root"))
                    self.tof_var.set(params.get("TOF", 0))
                    
                    # Optionally, you can update your GUI elements with the loaded parameters
                    # For example, update the Entry widgets or labels accordingly
                    # entry_sampling_method.delete(0, tk.END)
                    # entry_sampling_method.insert(0, self.sampling_method_var.get())
                    # Repeat for other parameters
            
            except Exception as e:
                print(f"Error while loading parameters: {e}")

    def save_file(self):
        file_path = filedialog.asksaveasfilename(title="Save Parameter File", defaultextension=".toml", filetypes=[("TOML files", "*.toml")])
        if file_path:
            try:
                # Extract the parameter values from your variables
                parameters = {
                    'sampling_method': self.sampling_method_var.get(),
                    'sample_rate': self.sample_rate_var.get(),
                    'signal_width': self.signal_width_var.get(),
                    'plot_opt': self.plot_opt_var.get(),
                    'plot_time_min': self.plot_time_min_var.get(),
                    'plot_time_max': self.plot_time_max_var.get(),
                    'plot_fre_min': self.plot_fre_min_var.get(),
                    'plot_fre_max': self.plot_fre_max_var.get(),
                    'simulated_data': self.simulated_data_var.get(),
                    'TOF': self.tof_var.get()
                }
    
                # Save parameters to the TOML file
                with open(file_path, "w") as toml_file:
                    toml.dump({'parameters': parameters}, toml_file)
    
            except Exception as e:
                print(f"Error while saving parameters: {e}")
    
            
    def run_worker(self):
        # Update the ROOT plot
        start_step=1
        self.converter.run(self.sampling_method_var.get(), self.sample_rate_var.get(), self.signal_width_var.get(), self.plot_opt_var.get(), self.plot_time_min_var.get(), self.plot_time_max_var.get(), self.plot_fre_min_var.get(), self.plot_fre_max_var.get(), start_step, self.simulated_data_var.get(),self.tof_var.get())
        
    def plot_worker(self):
        start_step=3
        self.converter.run(self.sampling_method_var.get(), self.sample_rate_var.get(), self.signal_width_var.get(), self.plot_opt_var.get(), self.plot_time_min_var.get(), self.plot_time_max_var.get(), self.plot_fre_min_var.get(), self.plot_fre_max_var.get(), start_step, self.simulated_data_var.get(),self.tof_var.get())
        print("updated!")
