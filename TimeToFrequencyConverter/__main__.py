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
    parser = argparse.ArgumentParser(description='TimeToFrequencyConverter')
    # Define command-line arguments
    parser.add_argument('--sampling_method', type=int, help='Sampling method (1=sine wave or 2=gaussian)')
    parser.add_argument('--sample_rate', type=float, help='Sampling rate in Hertz')
    parser.add_argument('--signal_width', type=float, help='Signal width')
    parser.add_argument('--plot_opt', type=int, help='plot option (0, 1, 2) = (no plot/plot only up pannel/ plot up and bottom pannels.)')
    parser.add_argument('--plot_time_min', type=float, help='Minimum time for plotting')
    parser.add_argument('--plot_time_max', type=float, help='Maximum time for plotting')
    parser.add_argument('--plot_fre_min', type=float, help='Minimum frequency for plotting')
    parser.add_argument('--plot_fre_max', type=float, help='Maximum frequency for plotting')
    parser.add_argument('--start_step', type=int, help='start step (1, 2, or 3).')
    parser.add_argument('--SimulatedDataFile', type=str, help='Path to simulated data file')
    args = parser.parse_args()
    return args

    
def main():
    #args = load_arguments()
    #converter = TFC.TimeToFrequencyConverter()
    #canvas,g_sampling_amplitude_time,h_fft_result_frequencies, table = converter.run(args.sampling_method, args.sample_rate, args.signal_width, args.plot_opt, args.plot_time_min, args.plot_time_max, args.plot_fre_min, args.plot_fre_max, args.start_step, args.SimulatedDataFile)
        
    root = tk.Tk()
    app = TFApp.TimeToFrequencyApp(root)
    root.mainloop()
    
if __name__ == "__main__":    
    main()
