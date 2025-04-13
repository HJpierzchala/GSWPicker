## Version
**Release v1.0 – April 13, 2025**

---

## Description

**GSW Picker** is an open-source Python-based tool for analyzing high-rate (≥1 Hz) GNSS velocity data to extract S-wave arrival times and ground shaking parameters (amplitude and duration).  
Designed for large-scale, research-grade seismo-geodetic analysis, it uses asynchronous parallel processing (`concurrent.futures`) for fast and scalable performance.

### Key Features

- **Input format support**:
  - **VarioPy** (`.varout`) – Position difference time series
  - **PRIDE-PPP** (`kin_*`) – PPP position time series
  - **Custom `.oy` format** – For user-defined position difference series

- **Pre-analysis tools**:
  - Noise analysis (Power Spectral Density and autocorrelation)
  - Time/frequency domain inspection (spectrograms)

- **Output formats**:
  - **Numerical**: `.csv` files
  - **Visual**: Plots of station noise, S-wave arrivals, and shaking duration

GSW Picker is ideal for large GNSS datasets and enables efficient, scalable seismic analysis.

---

## Installation & Running

### Installation

First, clone the repository to your local machine:
git clone git@github.com:HJpierzchala/GSWPicker.git

You can install **GSW Picker** in one of two ways:

1. **Automatic Installation** (macOS and Linux)  
   Run the provided `install.sh` script to create a conda environment (`gsw_env`) and install all required dependencies from `requirements.txt`.

2. **Manual Installation**  
   Create a new conda environment with **Python 3.11**, then install the packages listed in `requirements.txt`.

    A detailed step-by-step setup guide will be provided by Hubert.

!!! warning
    Be sure to use **Python 3.11**, and match the exact package versions specified in `requirements.txt`.

---

### Running the Software

To launch the GUI:

1. Open your terminal (Anaconda PowerShell Prompt on Windows or Terminal on macOS/Linux).
2. Activate the conda environment:
    ```
    conda activate gsw_env
    ```
3. Navigate to the GSW Picker directory:
    ```
    cd ./GSWPicker/
    ```
4. Run the GUI:
    ```
    python GSWPicker_GUI.py
     ```
---

### Verifying Installation
To confirm that the installation was successful:

1. Launch the GUI.
2. Load the test dataset: `./GSWPicker/vel_data/test`
3. Use the configurations from file: `./GSWPicker/results/test/1D/logs/log_test.txt` or manually input the parameters shown in the figure below.
4. Process the data and compare your output file `<PROJECTID>_MAD_1D.csv` with the reference:  
   `./GSWPicker/results/test/1D/reports/TEST_NORCIA_MAD_1D.csv`

If the files match, your installation is successful!

![Test GUI](../imgs/test_gui.png)

---

### Important Parameters

When configuring processing settings in the GUI, the following parameters are key:

- **`MODE`** – Method for ground shaking extraction:
- `MAD`, `SLOPE`, or `W-TEST`
- **`Window size (sec)`** – Length of the residual smoother’s sliding window.  
Recommended: **5 seconds** for high-rate (1–20 Hz) GNSS velocity data.
- **`Shaking length (sec)`** – Minimum duration of a shaking event.  
Recommended: **3 seconds** for high-rate (1–20 Hz) GNSS velocity data.

These parameters significantly impact performance and detection sensitivity. For high-rate GNSS velocity datasets, the recommended values (5s window, 3s shaking length) offer a good starting point.
