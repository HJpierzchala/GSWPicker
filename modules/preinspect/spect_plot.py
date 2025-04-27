from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from ..time import gps_to_utc, utc_to_gps
from .pre_input import read_file
from ..prep import sampfreq, kin2oyeah
from scipy.signal import spectrogram

def plot_gnssvel_spectrum(file_path, event_date, sta_name, output_path, mode, time_format='gsow', eq_time=None):
    """
    Generates and saves the spectrogram of GNSS velocity components (East, North, Up) 
    from a .varout file, with options to display 1D (individual components), 
    2D (horizontal velocity), or 3D (total velocity).

    Parameters
    ----------
    file_path : str
        Path to the GNSS data file (.varout format).
    event_date : datetime.datetime
        UTC date of the event, used for GPS time conversion.
    sta_name : str
        Name of the GNSS station.
    output_path : str
        Directory path where the spectrogram plots will be saved.
    mode : {'1D', '2D', '3D'}
        Specifies the velocity representation:
        - '1D': Spectrograms for individual velocity components (East, North, Up).
        - '2D': Spectrogram of the horizontal velocity (computed as sqrt(vEast² + vNorth²)).
        - '3D': Spectrogram of the total velocity (computed as sqrt(vEast² + vNorth² + vUp²)).
    time_format : {'SOW', 'UTC'}, optional
        Format for the x-axis time representation:
        - 'SOW' (default): Displays time in Seconds of Week.
        - 'UTC': Displays time in Coordinated Universal Time (UTC).

    eq_time: datetime.datetime
        Eqrthquake origin time, used for relative o.t time format

    Returns
    -------
    None
        The function generates and saves spectrogram plots in PNG and SVG formats. 
        No direct output is returned.
    """

    # Read the GNSS velocity data
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
    utc = gps_to_utc(gpsw, sow)

    mode_sampf_freq, sampf_freq = sampfreq(sow)
    vel *= sampf_freq  # Scale velocity data

    try:
        if sampf_freq < 1:
            raise ValueError("Sampling frequency must be >= 1 Hz to generate spectrogram.")

        # Set NFFT and noverlap based on sampling frequency
        NFFT, noverlap = (25, 12) if 1 <= sampf_freq < 10 else (256, 128)
        window = ('tukey', 0.5)

        fnt_size = 16
        if mode == '1D':
            fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

            # Enhance plot borders for individual component plots
            for a in ax:
                for spine in a.spines.values():
                    spine.set_edgecolor('black')  # Set border color to black
                    spine.set_linewidth(2)  # Set border thickness

            components = ['vEast', 'vNorth', 'vUp']

            # Loop over the velocity components to generate spectrograms
            for i, comp in enumerate(components):
                freqs, times, Sxx = spectrogram(vel[:, i], fs=sampf_freq, window=window,
                                                nperseg=NFFT, noverlap=noverlap, mode='psd', scaling='density')
                times_mapped = sow[0] + times  # Adjust time scale


                # Format time based on selected time_format
                if time_format == 'utc':
                    utc_mapped = gps_to_utc(gpsw, times_mapped)
                    time_data = [pd.to_datetime(u) for u in utc_mapped]
                    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                elif time_format=='gsow':
                    time_data = times_mapped
                elif time_format=='rel':
                    _,eq_time_sow = utc_to_gps(utc_times=[eq_time])
                    time_data = times_mapped-eq_time_sow

                # Plot the spectrogram for the current component
                pcm = ax[i].pcolormesh(time_data, freqs, 10 * np.log10(Sxx), cmap='jet')
                ax[i].set_ylabel('Frequency (Hz)', fontsize=fnt_size)
                ax[i].set_title(f'{comp}', fontsize=fnt_size + 2)
                ax[i].grid(True)

                # Colorbar and tick settings
                cbar = fig.colorbar(pcm, ax=ax[i])
                cbar.set_label('Power/Frequency (dB/Hz)', fontsize=fnt_size - 4)
                ax[i].tick_params(axis='both', which='major', labelsize=fnt_size - 2)

            # Set common x-axis label for individual component plots
            if time_format =='utc':
                ax[-1].set_xlabel('UTC Time', fontsize=fnt_size)
            elif time_format=='gsow':
                ax[-1].set_xlabel('GSOW [sec]', fontsize=fnt_size)
            elif time_format == 'rel':
                ax[-1].set_xlabel('Relative O.T [sec]', fontsize=fnt_size)
            # Title and layout for individual component plots
            plt.suptitle(f'GNSS Instantaneous Velocity Spectrogram, {sta_name}, ({utc[0].date()})',
                         fontsize=fnt_size + 2)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f'{output_path}/{sta_name}_freq_PRE.png')
            plt.savefig(f'{output_path}/{sta_name}_freq_PRE.svg')
            plt.close()

        if mode =='2D':
            # 2D: Horizontal (vEast and vNorth)
            vel_2d = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
            for spine in ax2.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)
            freqs, times, Sxx = spectrogram(vel_2d, fs=sampf_freq, window=window,
                                             nperseg=NFFT, noverlap=noverlap, mode='psd', scaling='density')
            times_mapped = sow[0] + times
            utc_mapped = gps_to_utc(gpsw, times_mapped)
            # Format time based on selected time_format
            if time_format == 'utc':
                utc_mapped = gps_to_utc(gpsw, times_mapped)
                time_data = [pd.to_datetime(u) for u in utc_mapped]
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            elif time_format == 'gsow':
                time_data = times_mapped
            elif time_format == 'rel':
                _, eq_time_sow = utc_to_gps(utc_times=[eq_time])
                time_data = times_mapped - eq_time_sow
            pcm = ax2.pcolormesh(time_data, freqs, 10 * np.log10(Sxx), cmap='jet')
            ax2.set_ylabel('Frequency (Hz)', fontsize=fnt_size)
            ax2.set_title('vHorizontal (2D)', fontsize=fnt_size + 2)
            ax2.grid(True)
            cbar = fig2.colorbar(pcm, ax=ax2)
            cbar.set_label('Power/Frequency (dB/Hz)', fontsize=fnt_size - 4)
            ax2.tick_params(axis='both', which='major', labelsize=fnt_size - 2)
            if time_format =='utc':
                ax2.set_xlabel('UTC Time', fontsize=fnt_size)
            elif time_format=='gsow':
                ax2.set_xlabel('GSOW [sec]', fontsize=fnt_size)
            elif time_format == 'rel':
                ax2.set_xlabel('Relative O.T [sec]', fontsize=fnt_size)
            plt.suptitle(f'GNSS Instantaneous Velocity Spectrogram (2D), {sta_name}, ({utc[0].date()})', fontsize=fnt_size + 2)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f'{output_path}/{sta_name}_freq_2D_PRE.png')
            plt.savefig(f'{output_path}/{sta_name}_freq_2D_PRE.svg')
            plt.close()
        if mode == '3D':
            # 3D: Total velocity (vEast, vNorth, vUp)
            vel_3d = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2 + vel[:, 2]**2)
            fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
            for spine in ax3.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)
            freqs, times, Sxx = spectrogram(vel_3d, fs=sampf_freq, window=window,
                                             nperseg=NFFT, noverlap=noverlap, mode='psd', scaling='density')
            times_mapped = sow[0] + times
            utc_mapped = gps_to_utc(gpsw, times_mapped)
            # Format time based on selected time_format
            if time_format == 'utc':
                utc_mapped = gps_to_utc(gpsw, times_mapped)
                time_data = [pd.to_datetime(u) for u in utc_mapped]
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            elif time_format == 'gsow':
                time_data = times_mapped
            elif time_format == 'rel':
                _, eq_time_sow = utc_to_gps(utc_times=[eq_time])
                time_data = times_mapped - eq_time_sow
            pcm = ax3.pcolormesh(time_data, freqs, 10 * np.log10(Sxx), cmap='jet')
            ax3.set_ylabel('Frequency (Hz)', fontsize=fnt_size)
            ax3.set_title('v3D ', fontsize=fnt_size + 2)
            ax3.grid(True)
            cbar = fig3.colorbar(pcm, ax=ax3)
            cbar.set_label('Power/Frequency (dB/Hz)', fontsize=fnt_size - 4)
            ax3.tick_params(axis='both', which='major', labelsize=fnt_size - 2)

            if time_format =='utc':
                ax3.set_xlabel('UTC Time', fontsize=fnt_size)
            elif time_format=='gsow':
                ax3.set_xlabel('GSOW [sec]', fontsize=fnt_size)
            elif time_format == 'rel':
                ax3.set_xlabel('Relative O.T [sec]', fontsize=fnt_size)

            plt.suptitle(f'GNSS Instantaneous Velocity Spectrogram (3D), {sta_name}, ({utc[0].date()})', fontsize=fnt_size + 2)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(f'{output_path}/{sta_name}_freq_3D_PRE.png')
            plt.savefig(f'{output_path}/{sta_name}_freq_3D_PRE.svg')
            plt.close()

    except ValueError as e:
        print(f"Error: {e}")
