## Version
**Release v1.0 – April 13, 2025**

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

