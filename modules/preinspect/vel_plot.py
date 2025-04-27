import datetime
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from ..time import gps_to_utc, utc_to_gps
from ..prep import sigma_filter, sampfreq, kin2oyeah
from .pre_input import read_file
import os

def plot_gnssvel_data(file_path, event_date, sta_name, output_path,mode, time_format='gsow', eq_time=None, log_dir=None):
    """
    Plot GNSS velocity data from a .varout file.

    This function visualizes GNSS velocity components (East, North, Up) and provides 
    options to display time in either Seconds of Week (SOW) or UTC format. It also 
    filters noise levels and computes velocity magnitudes for different modes (1D, 2D, 3D).

    Parameters
    ----------
    file_path : str
        Path to the GNSS data file (.varout format).
    event_date : datetime.datetime
        UTC date of the event, used for converting to GPS time.
    sta_name : str
        Name of the GNSS station.
    output_path : str
        Path where the output plots will be saved.
    mode : {'1D', '2D', '3D'}
        Mode of velocity visualization:
        - '1D': Plots individual East, North, and Up components.
        - '2D': Plots horizontal velocity magnitude (sqrt(V_East² + V_North²)).
        - '3D': Plots full velocity magnitude (sqrt(V_East² + V_North² + V_Up²)).
    time_format : {'gsow', 'UTC'}, optional
        Time format for the x-axis. 'gsow' (default) uses Seconds of Week, 
        while 'UTC' displays human-readable timestamps.
    eq_time : datetime.datetime, optional
        Event time used to filter noise before the event and for relative O.T time format. Default is None.

    Returns
    -------
    None
        The function generates and saves a plot but does not return a value.

    Notes
    -----
    - Noise levels are computed as 1-sigma standard deviation.
    - If `eq_time` is provided, the time series is adjusted to start from the event.
    - The plot is saved in both PNG and SVG formats.
    """
    # Read data
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
    gpsw, starting_gps_sow = utc_to_gps([event_date])

    # Convert event time to GPS time if eq_time is provided
    if eq_time is not None:
        _, eq_time = utc_to_gps(eq_time)
        if time_format =='rel':
            sow_relative = sow - eq_time

    utc = gps_to_utc(gpsw, sow)
    mode_sampf_freq, sampf_freq = sampfreq(sow)
    # Zapis wyników do pliku
    log_filename = f"log_velocity_{sta_name}.txt"
    full_log_path = os.path.join(log_dir, log_filename)
    with open(full_log_path, "w") as log_file:
        log_file.write("Velocity time series analysis results - " + sta_name + "\n")
        log_file.write("----------------------------------------\n")
        log_file.write(f"-Sampling frequency: {sampf_freq} Hz-\n")
        log_file.write(f"\n")
    # Scale velocity data
    vel *= sampf_freq

    # Extract GNSS velocity noise based on event time (if eq_time is provided)
    if eq_time is None:
        e_noise, n_noise, u_noise = [sigma_filter(arr=vel[:, i])[0] for i in range(3)]
    else:
        mask = sow < eq_time  # tablica True/False długości sow.shape[0]

        e_noise, n_noise, u_noise = [
            sigma_filter(arr=vel[mask, i])[0]
            for i in range(3)
        ]
    # Compute noise levels (1 sigma)
    noise_levels = np.nanstd([e_noise, n_noise, u_noise], axis=1)
    sigma_labels = [f"{level:.4f} m/s" for level in noise_levels]

    if mode == '3D':
        # Poprawione obliczenie prędkości 2D: sqrt(v_east^2 + v_north^2)
        vel_3d = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2+ vel[:,2]**2)


        noise_level = np.sqrt(noise_levels[0] ** 2 + noise_levels[1] ** 2 + noise_levels[2]**2)

        sigma_label = f"{noise_level:.4f} m/s"
        fnt_size = 16
        max_abs_value = np.max(np.abs(vel_3d))
        # Create subplot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
        # Ustawienie właściwości ramek wykresu
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        # Ustalenie danych osi x w zależności od formatu czasu
        if time_format == 'utc':
            x_data = [pd.to_datetime(u) for u in utc]
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif time_format == 'gsow':
            x_data = sow
        elif time_format=='rel':
            x_data = sow_relative

        velocity_label = 'v 3D'
        ax.plot(x_data, vel_3d, color='black', alpha=0.7, lw=2, label=f'3D Velocity- $\sigma$={sigma_label}')
        ax.grid(True)
        offset = max_abs_value / 10
        ax.set_ylim(-max_abs_value - offset, max_abs_value + offset)
        ax.legend(fontsize=fnt_size - 2)
        ax.tick_params(axis='both', which='major', labelsize=fnt_size - 2)  # Zmniejszenie czcionki o 2 punkty
        ax.set_title(f'{velocity_label}', fontsize=fnt_size + 2)
        if time_format == 'utc':
            # 'GSOW [sec]' if time_format == 'gsow' else
            ax.set_xlabel('UTC Time', fontsize=fnt_size)
        elif time_format == 'rel':
            ax.set_xlabel('Relative O.T [sec]', fontsize=fnt_size)
        elif time_format == 'gsow':
            ax.set_xlabel('GSOW [sec]', fontsize=fnt_size)

        # Ustawienia tytułów i layoutu
        fig.supylabel(r'Velocity [m/s]', fontsize=fnt_size)
        plt.suptitle(f'Instantaneous Velocity Time Series, {sta_name}, ({utc[0].date()})', fontsize=fnt_size + 2)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{output_path}/{sta_name}_time_PRE.png')
        plt.savefig(f'{output_path}/{sta_name}_time_PRE.svg')
        plt.close()
    if mode == '2D':
        # Poprawione obliczenie prędkości 2D: sqrt(v_east^2 + v_north^2)
        vel_2d = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)
        noise_level = np.sqrt(noise_levels[0] ** 2 + noise_levels[1] ** 2)
        sigma_label = f"{noise_level:.4f} m/s"
        fnt_size = 16
        max_abs_value = np.max(np.abs(vel_2d))
        # Create subplot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
        # Ustawienie właściwości ramek wykresu
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

        # Ustalenie danych osi x w zależności od formatu czasu
        if time_format == 'UTC':
            x_data = [pd.to_datetime(u) for u in utc]
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif time_format == 'gsow':
            x_data = sow
        elif time_format =='rel':
            x_data=sow_relative
        velocity_label = 'v 2D'
        ax.plot(x_data, vel_2d, color='black', alpha=0.7, lw=2, label=f'2D Velocity- $\sigma$={sigma_label}')
        ax.grid(True)
        offset = max_abs_value / 10
        ax.set_ylim(-max_abs_value - offset, max_abs_value + offset)
        ax.legend(fontsize=fnt_size - 2)
        ax.tick_params(axis='both', which='major', labelsize=fnt_size - 2)  # Zmniejszenie czcionki o 2 punkty
        ax.set_title(f'{velocity_label}', fontsize=fnt_size + 2)
        if time_format =='utc':
            #'GSOW [sec]' if time_format == 'gsow' else
            ax.set_xlabel('UTC Time', fontsize=fnt_size)
        elif time_format =='rel':
            ax.set_xlabel('Relative O.T [sec]', fontsize=fnt_size)
        elif time_format =='gsow':
            ax.set_xlabel('GSOW [sec]', fontsize=fnt_size)
        # Ustawienia tytułów i layoutu
        fig.supylabel(r'Velocity [m/s]', fontsize=fnt_size)
        plt.suptitle(f'Instantaneous Velocity Time Series, {sta_name}, ({utc[0].date()})', fontsize=fnt_size + 2)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{output_path}/{sta_name}_time_PRE.png')
        plt.savefig(f'{output_path}/{sta_name}_time_PRE.svg')
        plt.close()


    if mode == '1D':


        # Plotting settings
        fnt_size = 16
        max_abs_value = np.max(np.abs(vel))
        # Create subplots
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Increase the outline (border) thickness of all subplots
        for a in ax:
            for spine in a.spines.values():
                spine.set_edgecolor('black')  # Set color of the border to black
                spine.set_linewidth(2)  # Set thickness of the border

        # Set x-data based on the time format
        if time_format == 'utc':
            x_data = [pd.to_datetime(u) for u in utc]
            ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif time_format == 'gsow':
            x_data = sow
        elif time_format == 'rel':
            x_data = sow_relative

        # Plot velocity components in a loop to avoid repetitive code
        velocity_labels = ['vEast', 'vNorth', 'vUp']
        for i, comp in enumerate(velocity_labels):
            ax[i].plot(x_data, vel[:, i], color='black', alpha=0.7, lw=2, label=f'{comp} - $\sigma$={sigma_labels[i]}')
            # ax[i].set_ylabel(comp, fontsize=fnt_size)
            ax[i].grid(True)
            offset = max_abs_value / 10
            ax[i].set_ylim(-max_abs_value - offset, max_abs_value + offset)
            ax[i].legend(fontsize=fnt_size - 2)
            ax[i].tick_params(axis='both', which='major', labelsize=fnt_size - 2)  # Reduce fontsize by 2 points
            ax[i].set_title(f'{comp}', fontsize=fnt_size + 2)

        if time_format =='utc':
            #'GSOW [sec]' if time_format == 'gsow' else
            ax[2].set_xlabel('UTC Time', fontsize=fnt_size)
        elif time_format =='rel':
            ax[2].set_xlabel('Relative O.T [sec]', fontsize=fnt_size)
        elif time_format =='gsow':
            ax[2].set_xlabel('GSOW [sec]', fontsize=fnt_size)
        # Title and layout
        fig.supylabel(r'Velocity [m/s]', fontsize=fnt_size)
        plt.suptitle(f'Instantaneous Velocity Time Series, {sta_name}, ({utc[0].date()})', fontsize=fnt_size + 2)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'{output_path}/{sta_name}_time_PRE.png')
        plt.close()