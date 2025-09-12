import sys
import time
import numpy as np
from ROOT import TCanvas, gROOT, TGraph, TApplication,TH1F, gPad, gStyle, TFile
from prettytable import PrettyTable
import argparse
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from . import TimeToFrequencyConverter as TFC
from . import TimeToFrequencyApp as TFApp

def load_arguments():
    parser = argparse.ArgumentParser(description='TimeToFrequencyConverter. No parameters given, run in GUI mode. e.g. if you type command: $TimeToFrequencyConverter. The program will run in GUI mode. If you type command: $TimeToFrequencyConverter --start_step 1 --sampling_method 2 --sample_rate 5e6 --signal_width 1e-8 --noise_level 1e6 --low_freq_hz 240e6 --high_freq_hz 250e6 --plot_opt 2 --plot_time_min 0 --plot_time_max 5e-3 --plot_fre_min 0 --plot_fre_max 2e6 --start_step 1 --SimulatedDataFile out.root --TOF 147, the program will run in command mode.')
    # Define command-line arguments
    parser.add_argument('--sampling_method', type=int, help='Sampling method (1=sine wave or 2=gaussian)')
    parser.add_argument('--sample_rate', type=float, help='Sampling rate in Hertz')
    parser.add_argument('--signal_width', type=float, help='Signal width')
    parser.add_argument('--noise_level', type=float, help='noise level')
    parser.add_argument('--low_freq_hz', type=float, help='low_freq_hz')
    parser.add_argument('--high_freq_hz', type=float, help='high_freq_hz')
    parser.add_argument('--plot_opt', type=int, help='plot option (0, 1, 2) = (no plot/plot only up pannel/ plot up and bottom pannels.)')
    parser.add_argument('--plot_time_min', type=float, help='Minimum time for plotting (s)')
    parser.add_argument('--plot_time_max', type=float, help='Maximum time for plotting (s)')
    parser.add_argument('--frame_length_seconds', type=float, help='time of one frame (s)')
    parser.add_argument('--plot_fre_min', type=float, help='Minimum frequency for plotting')
    parser.add_argument('--plot_fre_max', type=float, help='Maximum frequency for plotting')
    parser.add_argument('--start_step', type=int, help='start step (1, 2, or 3).')
    parser.add_argument('--SimulatedDataFile', type=str, help='Path to simulated data file')
    parser.add_argument('--TOF', type=int, help='TOF (147, 230, or 16).')
    parser.add_argument('--Output', type=str, help='Path to output data file (Spectrum.root)')
    args = parser.parse_args()
    return args

def run_command_line_mode(args):
    # Add code here for direct command line execution
    print("Running in command line mode")
    #args = load_arguments()
    print(args)
    converter = TFC.TimeToFrequencyConverter()
    converter.run(args.sampling_method, args.sample_rate, args.signal_width, args.noise_level,args.low_freq_hz,args.high_freq_hz, args.plot_opt, args.plot_time_min, args.plot_time_max,args.frame_length_seconds, args.plot_fre_min, args.plot_fre_max, args.start_step, args.SimulatedDataFile, args.TOF, args.Output)


def run_gui_mode():
    # Add code here to open the GUI
    print("Running in GUI mode")
    root = tk.Tk()
    app = TFApp.TimeToFrequencyApp(root)
    root.mainloop()

def main():
    # Number of command line arguments
    num_args = len(sys.argv) - 1  # Subtracting the script name
    args = load_arguments()
    
    if num_args == 0:
        run_gui_mode()
    else:
        # Process command line arguments
        # You can use argparse or manually handle sys.argv
        # Here, we just print the arguments for demonstration
        print("Command line arguments:", sys.argv[1:])
        run_command_line_mode(args)
    
if __name__ == "__main__":    
    main()
