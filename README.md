# TimeToFrequencyConverter
 
This repository contains a Time to Frequency Converter tool that allows users to perform various operations related to time and frequency conversion. Below are instructions on how to download, install, and execute the code.

## 1. Downloading the Code

To download the code, use the following command in your terminal:

    git clone https://github.com/ruijiuchen/TimeToFrequencyConverter.git

## 2. Installation

After downloading the code, navigate to the project directory and install the required dependencies using:

    pip install -r requirements.txt
    pip install ./

## 3. Executing the Code

Once installed, you can execute the Time to Frequency Converter by running the following command:

    TimeToFrequencyConverter

This will launch the tool and provide you with the necessary interfaces for performing time and frequency conversion operations. Make sure to follow any additional instructions provided in the tool's interface for specific use cases.

## Parameters

### 1. Sampling Method

- **Description:** The method used for sampling.
- **Command Line Argument:** `--sampling_method`
- **Type:** Integer
- **Values:** 1 (sine wave) or 2 (gaussian)

### 2. Sample Rate

- **Description:** The sampling rate in Hertz.
- **Command Line Argument:** `--sample_rate`
- **Type:** Float

### 3. Signal Width

- **Description:** The width of the signal in seconds.
- **Command Line Argument:** `--signal_width`
- **Type:** Float

### 4. Plot Option

- **Description:** The plotting option.
- **Command Line Argument:** `--plot_opt`
- **Type:** Integer
- **Values:** 0 (no plot), 1 (plot only up panel), 2 (plot up and bottom panels)

### 5. Plot Time Range

- **Description:** The time range for plotting.
- **Command Line Argument:** `--plot_time_min` and `--plot_time_max`
- **Type:** Float
- **Units:** Seconds

### 6. Plot Frequency Range

- **Description:** The frequency range for plotting.
- **Command Line Argument:** `--plot_fre_min` and `--plot_fre_max`
- **Type:** Float
- **Units:** Hertz

### 7. Start Step

- **Description:** The starting step for the conversion process.
- **Command Line Argument:** `--start_step`
- **Type:** Integer
- **Values:** 1, 2, or 3

### 8. Simulated Data File

- **Description:** The file containing simulated data.
- **Command Line Argument:** `--SimulatedDataFile`
- **Type:** String
- **Example:** Path to the simulated data file

Feel free to explore and contribute to the TimeToFrequencyConverter project!

For inquiries, please contact:

  Ruijiu Chen
  Email: r.chen@gsi.de
  Email: chenrj13@impcas.ac.cn