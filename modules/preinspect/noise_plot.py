import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from .pre_input import read_file
from ..prep import sampfreq, sigma_filter,kin2oyeah
from ..time import gps_to_utc, utc_to_gps
from scipy.signal import lombscargle
from ..config import CFG
import pandas as pd

def plot_gnssvel_periodogram(file_path, event_date, sta_name, output_path, eq_time=None, output_type='frequency', plot_scale='log', CFG=CFG):
    """
    Plots the Lomb-Scargle periodogram for GNSS velocity noise time series.

    This function computes and visualizes the Lomb-Scargle periodogram for GNSS velocity data, 
    helping to analyze the frequency content of velocity noise signals.

    Parameters
    ----------
    file_path : str
        Path to the GNSS velocity data file (CSV format or similar).
    event_date : datetime.datetime
        Event date in UTC (YYYY-MM-DD format) for GPS time conversion.
    sta_name : str
        Station name (used for labeling and file saving).
    output_path : str
        Directory path to save the generated plot.
    eq_time : datetime.datetime, optional
        Event time in GPS seconds of the week (SOW). If provided, noise before this time is filtered out.
        Default is None (no filtering).
    output_type : {'frequency', 'period'}, optional
        Determines whether the x-axis represents 'frequency' (Hz) or 'period' (seconds).
        Default is 'frequency'.
    plot_scale : {'log', 'normal'}, optional
        Specifies the scale of the plot. 'log' creates a log-log scale, 
        while 'normal' creates a linear scale. Default is 'log'.

    Returns
    -------
    None
        The function saves the periodogram plot to the specified output directory 
        but does not return any values.

    Notes
    -----
    - The function automatically detects the GNSS velocity sampling frequency.
    - If `eq_time` is provided, data before this timestamp is filtered out.
    - The power spectral density is computed using the Lomb-Scargle method.
    - If the sampling frequency is below 1 Hz, the function exits with an error message.
    - The function supports both logarithmic and linear plotting.
    
    Saves
    -----
    A PNG image of the Lomb-Scargle periodogram is saved in the `output_path` directory.

    """
    if file_path.endswith('oy'):
        df = pd.read_csv(file_path)
        sow, vel, sigma = (df['t_gsow'].to_numpy(), df[['e','n','u']].to_numpy(),
                           df[['std_e','std_n','std_u']].to_numpy())
        print(type(sow[0]))
        print(type(vel[0, 0]))
    elif file_path.endswith('varout'):
        sow, vel, sigma = read_file(file_path)
    elif file_path.endswith('kin') or 'kin_' in file_path:
        df = kin2oyeah(kin_path=file_path)
        sow = df['t_gsow'].to_numpy()
        print(type(sow[0]))
        vel = df[['e', 'n', 'u']].to_numpy()
        print(type(vel[0, 0]))
        sigma = df[['std_e', 'std_n', 'std_u']].to_numpy()
    gpsw, starting_gps_sow = utc_to_gps(event_date)

    # Convert event time if provided
    eq_time = utc_to_gps(eq_time) if eq_time is not None else None
    utc = gps_to_utc(gpsw, sow)

    mode_sampf_freq, sampf_freq = sampfreq(sow)
    vel *= sampf_freq  # Scale velocity data

    # Apply sigma filtering to the velocity data
    if eq_time:
        indices = sow < eq_time
        sow, vel = sow[indices], vel[indices]

    e_noise, _ = sigma_filter(vel[:, 0])
    n_noise, _ = sigma_filter(vel[:, 1])
    u_noise, _ = sigma_filter(vel[:, 2])

    # Noise level calculation (1-sigma)
    noise_levels = {
        'vEast': np.nanstd(e_noise),
        'vNorth': np.nanstd(n_noise),
        'vUp': np.nanstd(u_noise)
    }

    # Handle error for too low sampling frequency
    if sampf_freq < 1:
        print("Error: Sampling frequency must be at least 1 Hz.")
        return
    # print(f'{sta_name} SAMPLING FREQ: {sampf_freq}')
    # Define frequency range
    nyquist_frequency = sampf_freq / 2.0
    n_freq_points = len(e_noise)

    frequencies = np.logspace(-2, np.log10(nyquist_frequency), CFG['PREINSPECT']['PSD_NFFT_freq'] if sampf_freq > 1 else n_freq_points)
    angular_frequencies = 2 * np.pi * frequencies

    # Plot setup
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    components = ['vEast', 'vNorth', 'vUp']
    fnt_size = 16

    # Plot borders
    for a in ax:
        for spine in a.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

    # Plot for each component
    for i, (noise_data, component) in enumerate(zip([e_noise, n_noise, u_noise], components)):
        mask = ~np.isnan(noise_data)
        time_clean = sow[mask] - np.min(sow[mask])
        noise_clean = noise_data[mask]

        power = lombscargle(time_clean, noise_clean, angular_frequencies) / (sampf_freq * 2)

        x_values = 1 / frequencies if output_type == 'period' else frequencies
        x_label = 'Period (seconds)' if output_type == 'period' else 'Frequency (Hz)'

        # Plot with log or normal scale
        plot_func = ax[i].loglog if plot_scale == 'log' else ax[i].plot
        plot_func(x_values, power, color='black', alpha=0.7, lw=2)

        ax[i].set_title(f'{component}', fontsize=fnt_size + 2)
        #ax[i].set_ylabel(component, fontsize=fnt_size)
        ax[i].grid(True)
        ax[i].tick_params(axis='both', which='major', labelsize=fnt_size - 2)

    # Add labels and title
    fig.supylabel('Power [$(m/s)^2/Hz$]', fontsize=fnt_size)
    ax[2].set_xlabel(x_label, fontsize=fnt_size)
    plt.suptitle(f'Lomb-Scargle Periodogram of GNSS Velocity Noise ({sta_name}), ({utc[0].date()})',
                 fontsize=fnt_size + 2)
    plt.tight_layout()
    plt.savefig(f'{output_path}/{sta_name}_noise_prg_PRE.png')
    plt.close()


def plot_gnssvel_autocorrelation(file_path, event_date, sta_name, output_path, log_path, eq_time=None, sigma_factor=3, consecutive_count=10):
    """
    Performs autocorrelation analysis on GNSS velocity noise time series.

    This function calculates the autocorrelation function of GNSS velocity noise 
    and determines the decorrelation time for East, North, and Up velocity components.

    Parameters
    ----------
    file_path : str
        Path to the GNSS velocity data file (CSV or similar format).
    event_date : str
        Event date in UTC (YYYY-MM-DD format) for GPS time conversion.
    sta_name : str
        Station name (used for labeling plots and log files).
    output_path : str
        Directory path where the autocorrelation plot will be saved.
    log_path : str
        Directory path where the log file with analysis results will be saved.
    eq_time : float, optional
        Event time in GPS seconds of the week (SOW). If provided, 
        noise before this time is filtered out. Default is None (no filtering).
    sigma_factor : int, optional
        Factor used to determine the decorrelation threshold as a multiple of the 
        standard deviation of the autocorrelation function. Default is 3.
    consecutive_count : int, optional
        Minimum number of consecutive lags below the threshold required 
        to confirm significant decorrelation. Default is 10.

    Returns
    -------
    dict
        A dictionary containing:
        - 'decorrelation_time': Estimated decorrelation time in seconds for each velocity component.
        - 'index': Index of the decorrelation time lag.
        - 'threshold': Threshold value used for decorrelation detection.
        - 'autocorr': Autocorrelation function values.

    Notes
    -----
    - The function automatically detects the GNSS velocity sampling frequency.
    - If `eq_time` is provided, data before this timestamp is filtered out.
    - The decorrelation lag is identified using a threshold-based method.
    - The function supports saving the autocorrelation plot in PNG format.
    - A log file summarizing the results is saved in `log_path`.

    Saves
    -----
    - Autocorrelation plots in the `output_path` directory.
    - A log file containing noise levels and decorrelation times in the `log_path` directory.

    """
    # print('')
    # print('Autocorrelation analysis results - ' + sta_name)
    # print('----------------------------------------')
    # Load GNSS velocity data
    if file_path.endswith('oy'):
        df = pd.read_csv(file_path)
        sow, vel, sigma = (df['t_gsow'].to_numpy(), df[['e', 'n', 'u']].to_numpy(),
                           df[['std_e', 'std_n', 'std_u']].to_numpy())
    elif file_path.endswith('varout'):
        sow, vel, sigma = read_file(file_path)
    elif file_path.endswith('kin') or 'kin_' in file_path:
        df = kin2oyeah(kin_path=file_path)
        sow = df['t_gsow'].to_numpy()
        vel = df[['e', 'n', 'u']].to_numpy()
        sigma = df[['std_e', 'std_n', 'std_u']].to_numpy()
    gpsw, starting_gps_sow = utc_to_gps(event_date)
    if eq_time is not None:
        eq_time = utc_to_gps(eq_time)
    utc = gps_to_utc(gpsw, sow)

    # Get sampling frequency
    mode_sampf_freq, sampf_freq = sampfreq(sow)
    vel *= sampf_freq  # Scale velocity data

    # Extract GNSS velocity noise
    if eq_time is None:
        e_noise, _ = sigma_filter(arr=vel[:, 0])
        n_noise, _ = sigma_filter(arr=vel[:, 1])
        u_noise, _ = sigma_filter(arr=vel[:, 2])
    else:
        indices = np.where(sow < eq_time)
        sow = sow[indices] - np.min(sow[indices])
        e_noise, _ = sigma_filter(arr=vel[:, 0])
        n_noise, _ = sigma_filter(arr=vel[:, 1])
        u_noise, _ = sigma_filter(arr=vel[:, 2])

    # Noise level - 1sigma
    e_nlevel = np.nanstd(e_noise)
    n_nlevel = np.nanstd(n_noise)
    u_nlevel = np.nanstd(u_noise)
    sigma_e = f"{e_nlevel:.4f} m/s"
    sigma_n = f"{n_nlevel:.4f} m/s"
    sigma_u = f"{u_nlevel:.4f} m/s"
    # print('-Noise level-')
    # print(f'E sigma: {sigma_e}')
    # print(f'N sigma: {sigma_n}')
    # print(f'U sigma: {sigma_u}')
    # print('')

    # Function to compute autocorrelation
    def compute_autocorrelation(signal):
        valid_indices = ~np.isnan(signal)
        signal = signal[valid_indices]
        autocor_fnc = np.correlate(signal - np.mean(signal), signal - np.mean(signal), mode='full')
        autocor_fnc /= autocor_fnc.max()
        return autocor_fnc[len(autocor_fnc) // 2:]  # Keep positive lags only

    # Function to find decorrelation lag using positive envelope
    def find_decorr_lag(autocorr, sigma_factor, consecutive_count):
        positive_envelope = np.abs(autocorr[len(autocorr) // 2:])  # Positive lags
        mean_autocorr = np.nanmean(positive_envelope)
        std_autocorr = np.nanstd(positive_envelope)
        threshold = sigma_factor * std_autocorr

        for i in range(len(positive_envelope) - consecutive_count):
            if positive_envelope[i] > threshold and np.all(
                    positive_envelope[i + 1:i + 1 + consecutive_count] < threshold):
                return i, threshold
        return None, threshold

    # Compute autocorrelation and find decorrelation lag for each component
    components = {'vEast': e_noise, 'vNorth': n_noise, 'vUp': u_noise}
    results = {}

    # print('-Decorrelation lag-')
    crp_dict ={}
    for comp, noise_data in components.items():
        autocorr = compute_autocorrelation(noise_data)
        last_decorr_idx, threshold_value = find_decorr_lag(autocorr, sigma_factor, consecutive_count)

        cross_point_sec = last_decorr_idx * (1 / sampf_freq) if last_decorr_idx is not None else None
        results[comp] = {'decorrelation_time': cross_point_sec, 'index': last_decorr_idx, 'threshold': threshold_value,
                         'autocorr': autocorr}
        #
        # if cross_point_sec is not None:
        #     print(f"{comp} decorrelation lag: {round(cross_point_sec, 2)} sec")
        # else:
        #     print(f"No significant decorrelation point found for {comp}")
        crp_dict[comp] = round(cross_point_sec, 2)

    # Plotting
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fnt_size = 16
    # Increase the outline (border) thickness of all subplots
    for a in ax:
        for spine in a.spines.values():
            spine.set_edgecolor('black')  # Set color of the border to black
            spine.set_linewidth(2)  # Set thickness of the border

    for i, comp in enumerate(["vEast", "vNorth", "vUp"]):
        autocorr = results[comp]["autocorr"]
        time_lags = np.arange(0, len(autocorr)) * (1 / sampf_freq)
        last_idx = results[comp]["index"]
        threshold = results[comp]["threshold"]

        # Modify the legend to include the noise level (sigma)
        if comp == "vEast":
            ax[i].plot(time_lags[:len(autocorr)], autocorr, color='black', alpha=0.7, lw=2,
                       label=f'{comp} - $\sigma_e$={sigma_e}')
        elif comp == "vNorth":
            ax[i].plot(time_lags[:len(autocorr)], autocorr, color='black', alpha=0.7, lw=2,
                       label=f'{comp} - $\sigma_n$={sigma_n}')
        elif comp == "vUp":
            ax[i].plot(time_lags[:len(autocorr)], autocorr, color='black', alpha=0.7, lw=2,
                       label=f'{comp} - $\sigma_u$={sigma_u}')

        ax[i].axhline(y=threshold, color='black', linestyle='dashed', alpha=0.2)  # Threshold line
        if last_idx is not None:
            ax[i].axvline(x=last_idx * (1 / sampf_freq), color='black', linestyle='dashed', alpha=0.2)

        ax[i].set_title(f'{comp}', fontsize=fnt_size + 2)
        # ax[i].set_ylabel(comp, fontsize=fnt_size)
        ax[i].grid(True)
        ax[i].tick_params(axis='both', which='major', labelsize=fnt_size - 2)
        ax[i].legend(fontsize=fnt_size - 2)

    fig.supylabel(r'Autocorrelation $R(\tau)$', fontsize=fnt_size)
    ax[2].set_xlabel('Lag Time (s)', fontsize=fnt_size)
    plt.suptitle(f'GNSS Autocorrelation Analysis {sta_name} ({utc[0].date()})',
                 fontsize=fnt_size + 2)
    plt.tight_layout()
    plt.savefig(f'{output_path}/{sta_name}_noise_ac_PRE.png')
    plt.close()
    # print('----------------------------------------')
    # print('')
    t = datetime.now()
    log_filename = f"log_autocorr_{sta_name}.txt"
    full_log_path = os.path.join(log_path, log_filename)

    # Zapis wyników do pliku
    with open(full_log_path, "w") as log_file:
        log_file.write("Autocorrelation analysis results - " + sta_name + "\n")
        log_file.write("----------------------------------------\n")
        log_file.write("-Noise level-\n")
        log_file.write(f"E sigma: {sigma_e}\n")
        log_file.write(f"N sigma: {sigma_n}\n")
        log_file.write(f"U sigma: {sigma_u}\n")
        log_file.write(f"\n")
        log_file.write("-Decorrelation lag Tau-\n")



        if cross_point_sec is not None:
            for k,v in crp_dict.items():
                log_file.write(f"{k} decorrelation lag: {v} sec\n")
        else:
            log_file.write(f"No significant decorrelation point found for {comp}\n")



    return results
