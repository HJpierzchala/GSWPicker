import numpy as np
from .prep import sigma_filter, moving_average_with_regression_interpolation, butter_bandpass_filter, check_bandpass_conditions, gaussian_filter1d_modified
import pandas as pd
from .config import CFG
from .time import gps_to_utc, utc_to_gps
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from  datetime import datetime
import copy
from copy import deepcopy
import concurrent.futures
from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view
from .detect import find_swave_indices, BLUE, findLast
import os
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression
import traceback
from pathlib import Path
import warnings

def process_mad_component(time, grad_mavg_vele, vel_org, WS, WS_short, shaking_length, fre, method, comp_name, nvel_com, input_data, gps_week):
    """
    Processes the given component data to calculate S-wave characteristics using a median absolute deviation (MAD) approach.

    Parameters
    ----------
    time : numpy.ndarray
        The array of time values for the data.
    grad_mavg_vele : numpy.ndarray
        The array of gradient moving averages for velocity.
    vel_org : numpy.ndarray
        The original velocity data.
    WS : int
        The window size for the sliding window approach.
    WS_short : int
        The short window size for the sliding window approach.
    shaking_length : float
        The length of the shaking in seconds.
    fre : float
        The sampling frequency in Hz.
    method : str
        The method used for the analysis.
    comp_name : str
        The component name (e.g., 'e', 'n', 'u' for East, North, Up).
    nvel_com : float
        The noise level for the velocity component.
    input_data: dict
        Input parameters with necessary info for generating report
    gps_week: int
        GPS Week number

    Returns
    -------
    swave_time : float
        The time at which the S-wave is detected.
    comp_name : str
        The name of the component (input argument).
    df : pandas.DataFrame
        A DataFrame containing key calculated parameters (S-wave detected time, shaking duration, etc.).
    data2 : pandas.DataFrame
        A DataFrame containing more detailed information on the S-wave characteristics.
    to_plot : tuple
        A tuple containing data required for plotting, including S-wave info and velocities.

    Notes
    -----
    This function computes the median absolute deviation (MAD) for each sliding window of data and
    uses it to identify and process S-wave characteristics such as detection time, shaking duration, and peak ground velocity.
    """
    chuncks_grad_mavg_vele = sliding_window_view(grad_mavg_vele, WS)
    gs_vel = np.zeros(len(chuncks_grad_mavg_vele))
    for ii in range(np.shape(chuncks_grad_mavg_vele)[0]):
        abs_dev = np.absolute(chuncks_grad_mavg_vele[ii, :] - np.nanmedian(chuncks_grad_mavg_vele[ii, :]))
        gs_vel[ii] = np.nanmedian(abs_dev)

    nan_values = np.full(len(grad_mavg_vele) - len(gs_vel), np.nan)
    num_nan = len(grad_mavg_vele) - len(gs_vel)
    num_nan_beginning = num_nan // 2
    num_nan_end = num_nan // 2
    if num_nan % 2 != 0:
        num_nan_beginning += 1

    nan_values_beginning = np.full(num_nan_beginning, np.nan)
    nan_values_end = np.full(num_nan_end, np.nan)
    gs_vel = np.concatenate((nan_values_beginning, gs_vel, nan_values_end))
    (noise_thr_pos, noise_gs_vel, ind_sw, grouped_indices, sorted_grouped_indices, sorted_pgvs, first_index_sw,
     last_index_sw, swave_info_sorted, sorted_idx_pgv_list, indices_mainshock, indices_secondaryshocks) = find_swave_indices(
        arr=gs_vel, arr_org=vel_org, time=time, sampf_freq=fre, window_size=WS, method=method,
        shaking_length=shaking_length)

    if swave_info_sorted.any():
        if input_data['time'] =='gsow':
            array = np.array(swave_info_sorted)
            data2 = pd.DataFrame(data={
                'S-wave detected at (sec)': array[:, 2],
                'End of shaking at (sec)': array[:, 3],
                'Shaking duration (sec)': array[:, 4],
                'Peak Ground Velocity (m/s)': array[:, 5].round(2),
                'Integrated energy (m)': array[:, 6],
                'Weighted energy (m)': array[:, 7],
                'Sampling frequency (Hz)': fre,
                'Velocity noise (m/s)': nvel_com,
                'Ground shaking noise (m/s2)': noise_gs_vel})

            data = {
                'S-wave detected at (sec)': [swave_info_sorted[0][2]],
                'End of shaking at (sec)': [swave_info_sorted[0][3]],
                'Shaking duration (sec)': [swave_info_sorted[0][4]],
                'Peak Ground Velocity': [round(swave_info_sorted[0][5], 2)],
                'Integrated energy': [swave_info_sorted[0][6]],
                'Weighted energy': [swave_info_sorted[0][6]],
                'Sampling frequency': fre,
                'Velocity noise (m/s)': nvel_com,
                'Ground shaking noise (m/s2)': noise_gs_vel}

            df = pd.DataFrame(data)
            to_plot = (swave_info_sorted, time, grad_mavg_vele, gs_vel,
                       indices_secondaryshocks, indices_mainshock, noise_thr_pos, vel_org)
            return swave_info_sorted[0, 2], comp_name, df, data2, to_plot
        elif input_data['time'] =='rel':
            T0 = datetime.strptime(input_data['eq_time'], "%Y-%m-%d %H:%M:%S.%f")
            _, T0_SOW = utc_to_gps([T0])
            swave_info_sorted_copy = np.copy(swave_info_sorted)
            swave_info_sorted_copy[:, 2] -= T0_SOW
            swave_info_sorted_copy[:, 3] -= T0_SOW

            array = np.array(swave_info_sorted_copy)
            data2 = pd.DataFrame(data={
                'S-wave detected at (sec)': array[:, 2].round(3),
                'End of shaking at (sec)': array[:, 3].round(3),
                'Shaking duration (sec)': array[:, 4].round(3),
                'Peak Ground Velocity (m/s)': array[:, 5].round(3),
                'Integrated energy (m)': array[:, 6],
                'Weighted energy (m)': array[:, 7],
                'Sampling frequency (Hz)': fre,
                'Velocity noise (m/s)': nvel_com,
                'Ground shaking noise (m/s2)': noise_gs_vel})

            data = {
                'S-wave detected at (sec)': [swave_info_sorted_copy[0][2].round(3)],
                'End of shaking at (sec)': [swave_info_sorted_copy[0][3].round(3)],
                'Shaking duration (sec)': [swave_info_sorted_copy[0][4].round(3)],
                'Peak Ground Velocity': [round(swave_info_sorted_copy[0][5], 3)],
                'Integrated energy': [swave_info_sorted_copy[0][6]],
                'Weighted energy': [swave_info_sorted_copy[0][6]],
                'Sampling frequency': fre,
                'Velocity noise (m/s)': nvel_com,
                'Ground shaking noise (m/s2)': noise_gs_vel}

            df = pd.DataFrame(data)
            to_plot = (swave_info_sorted, time, grad_mavg_vele, gs_vel,
                       indices_secondaryshocks, indices_mainshock, noise_thr_pos, vel_org)
            return swave_info_sorted[0, 2], comp_name, df, data2, to_plot
        elif input_data['time'] =='utc':
            array = np.array(swave_info_sorted)
            detection_time_utc = gps_to_utc(gps_week=gps_week, gps_sow_array=array[:, 2])
            end_of_shaking_utc = gps_to_utc(gps_week=gps_week, gps_sow_array=array[:, 3])
            data2 = pd.DataFrame(data={
                                       'S-wave detected at': detection_time_utc,
                                       'End of shaking at': end_of_shaking_utc,
                                       'Shaking duration': array[:, 4],
                                       'Peak Ground Velocity (m/s)': array[:, 5].round(2),
                                       'Integrated energy (m)': array[:, 6],
                                       'Weighted energy (m)': array[:, 7],
                                       'Sampling frequency (Hz)': fre,
                                       'Velocity noise (m/s)': nvel_com,
                                       'Ground shaking noise (m/s2)': noise_gs_vel})

            detection_utc = gps_to_utc(gps_week=gps_week, gps_sow_array=[swave_info_sorted[0][2]])
            shaking_utc = gps_to_utc(gps_week=gps_week, gps_sow_array=[swave_info_sorted[0][3]])
            data = {
                'S-wave detected at': detection_utc,
                'End of shaking at': shaking_utc,
                'Shaking duration': [swave_info_sorted[0][4]],
                'Peak Ground Velocity (m/s)': [round(swave_info_sorted[0][5], 2)],
                'Integrated energy (m)': [swave_info_sorted[0][6]],
                'Weighted energy (m)': [swave_info_sorted[0][7]],
                'Sampling frequency (Hz)': fre,
                'Velocity noise (m/s)': nvel_com,
                'Ground shaking noise (m/s2)': noise_gs_vel
            }
            to_plot = (swave_info_sorted, time, grad_mavg_vele, gs_vel,
                       indices_secondaryshocks, indices_mainshock, noise_thr_pos, vel_org)
            df = pd.DataFrame(data)
            return detection_utc, comp_name, df, data2, to_plot
    else:
        data = pd.DataFrame(data={
            'S-wave detected at (sec)': [np.nan],
            'End of shaking at (sec)': [np.nan],
            'Shaking duration (sec)': [np.nan],
            'Peak Ground Velocity (m/s)': [np.nan],
            'Integrated energy (m)':[np.nan],
            'Weighted energy (m)': [np.nan],
            'Sampling frequency (Hz)': [np.nan],
            'Velocity noise (m/s)': [np.nan],
            'Ground shaking noise (m/s2)': [np.nan]})
        return data, comp_name

def capture_errors(func):
    """
    Decorator that captures errors and logs them to a `.err` file.

    The error log file is stored in the `errors` directory with a timestamped filename.

    Parameters
    ----------
    func : function
        The function to be wrapped and monitored for errors.

    Returns
    -------
    function
        A wrapped function that logs errors if an exception occurs.

    Notes
    -----
    - If an error occurs, it is logged to a file named `errors_<timestamp>.err`.
    - The log contains the function name, timestamp, and full traceback.
    - If `rich` is installed, the error message is displayed in the terminal with formatting.
    - If `rich` is not available, a simple print statement is used.

    """

    def wrapper(*args, **kwargs):
        t = datetime.now().strftime("%Y%H%M%S")
        err_name = f"errors_{t}.err"

        error_output_path = Path(__file__).resolve().parent.parent / "errors"
        error_output_path.mkdir(exist_ok=True)  # utwórz katalog, jeśli nie ma

        err_file = error_output_path / err_name

        try:
            return func(*args, **kwargs)
        except Exception as e:
            with open(err_file, "a") as f:
                f.write("=" * 80 + "\n")
                f.write(f"ERROR in function {func.__name__} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(traceback.format_exc() + "\n")
                f.write("=" * 80 + "\n")

            try:
                from rich.console import Console
                console = Console()
                console.print(f"[red]Error occurred, see .err file: {err_file}[/]")
            except ImportError:
                print(f"Error occurred, see .err file: {err_file}")
    return wrapper

class CustomDateFormatter(mdates.DateFormatter):
    """
    A custom date formatter class that inherits from `mdates.DateFormatter` to format
    the datetime values to a specific string format with limited decimal places for microseconds.

    Parameters
    ----------
    fmt : str
        The format string used for the datetime conversion. This is passed to
        the parent `DateFormatter` class.

    Methods
    -------
    __call__(x, pos=None)
        Converts a given value to a formatted datetime string with microsecond precision.
    """

    def __call__(self, x, pos=None):
        """
        Converts the given value `x` (in matplotlib's internal date format) to a formatted string.

        Parameters
        ----------
        x : float
            The date value to format, in matplotlib's internal date format (matplotlib's
            representation of the number of days since 0001-01-01 UTC).
        pos : int, optional
            The tick position. Default is None.

        Returns
        -------
        str
            A formatted string representing the time with the format `HH:MM:SS.microsecond`.
        """
        dt = mdates.num2date(x)
        formatted_time = dt.strftime('%H:%M:%S') + f".{int(dt.microsecond / 10000):01d}"
        return formatted_time

class Station:
    """
    A class to handle station-wise s-wave detection in seismology.

    It processes station data, computes noise levels, and provides utilities for analyzing
    s-wave behavior, including the calculation of `mag_V` and `grad_mavg_V`.

    Attributes
    ----------
    name : str
        Name of the station.
    data : dict
        A dictionary containing station data, with keys: 'times', 'vele', 'veln', 'velu'.
    input_data : dict
        A dictionary containing input configuration for the station.
    fre : float
        Sampling frequency of the station's receiver.
    compute_noise : bool or numpy.ndarray
        If True, compute the noise level; can also accept a numpy array for custom noise levels.
    eq_time : float or None
        Time of the earthquake, or None if no earthquake time is specified.
    window_size : int
        The size of the time window for calculation.
    mode : str
        Calculation mode, one of '1D', '2D', or '3D'.
    fig_path : str
        Path to save result figures.
    report_path : str
        Path to save result CSV files.
    method : str
        The method of calculation being used.
    wcase : str
        The working case.
    shaking_length : float
        The length of the shaking observed.
    gps_week : int
        GPS week number.
    CFG: configuration dict
    """

    def __init__(self, sta_name, data_dict, input_data, sampling_freq, compute_noise=True, eq_time=None, CFG=CFG):
        """
        Initializes the Station object with the given data and configuration.

        Parameters
        ----------
        sta_name : str
            Name of the station.
        data_dict : dict
            A dictionary containing station data, with keys: 'times', 'vele', 'veln', 'velu'.
        input_data : dict
            A dictionary containing input configuration for the station.
        sampling_freq : float
            Sampling frequency of the station's receiver.
        compute_noise : bool or numpy.ndarray, optional
            If True, compute the noise level; if a numpy array is provided, it is used as the noise level for each ENU component.
        eq_time : float or None, optional
            The time of the earthquake, or None if no earthquake time is specified.
        """
        self.name = sta_name
        self.data = data_dict
        self.input_data = input_data
        self.fre = sampling_freq
        self.compute_noise = compute_noise
        self.eq_time = eq_time
        self.window_size = self.input_data['window_size']
        self.mode = self.input_data['CALCULATION_MODE']
        self.fig_path = self.input_data['result_figures']
        self.report_path = self.input_data['result_csv']
        self.method = self.input_data['MODE']
        self.wcase = self.input_data['wcase']
        self.shaking_length = self.input_data['shaking_length']
        self.gps_week = self.input_data['GPS_WEEK']
        self.CFG = CFG
        assert self.mode in ['1D', '2D', '3D']

    def get_noise(self):
        """
        Computes the noise level of the station's velocity components (ENU: east, north, up)
        using a sigma filter and calculates the standard deviation and mean of the noise.

        If no earthquake time is provided, the noise is calculated from the entire dataset.
        If an earthquake time is provided, the noise is calculated using only the data before
        the earthquake event.

        Results
        -------
        Updates the `data` dictionary with the following noise statistics:
            - 'nvele' : Standard deviation of east velocity (e).
            - 'nveln' : Standard deviation of north velocity (n).
            - 'nvelu' : Standard deviation of up velocity (u).
            - 'mnvele' : Mean of east velocity (e).
            - 'mnveln' : Mean of north velocity (n).
            - 'mnvelu' : Mean of up velocity (u).

        Notes
        -----
        The noise is calculated using the `sigma_filter` function on the ENU velocity components.
        The threshold for noise is set to 3 standard deviations (3 * std).
        """
        time, e, n, u = self.data['times'], self.data['vele'], self.data['veln'], self.data['velu']
        arr = np.column_stack((e, n, u))

        if self.eq_time is None:
            vele_filt, idx_vele_filt = sigma_filter(arr=arr[:, 0])
            veln_filt, idx_velveln_filt = sigma_filter(arr=arr[:, 1])
            velu_filt, idx_velvelu_filt = sigma_filter(arr=arr[:, 2])
            velenu = np.column_stack((vele_filt, veln_filt, velu_filt))
            std_velenu = 3 * np.nanstd(velenu, axis=0)
            mean_noise_velenu = np.nanmean(velenu, axis=0)
            self.data['nvele'] = std_velenu[0]
            self.data['nveln'] = std_velenu[1]
            self.data['nvelu'] = std_velenu[2]
            self.data['mnvele'] = mean_noise_velenu[0]
            self.data['mnveln'] = mean_noise_velenu[1]
            self.data['mnvelu'] = mean_noise_velenu[2]

        elif self.eq_time is not None:
            indices = np.where(time < self.eq_time)
            time_before = time[indices]
            vel_enu_before = arr[indices]
            vele_filt, idx_vele_filt = sigma_filter(arr=vel_enu_before[:, 0])
            veln_filt, idx_velveln_filt = sigma_filter(arr=vel_enu_before[:, 1])
            velu_filt, idx_velvelu_filt = sigma_filter(arr=vel_enu_before[:, 2])
            velenu = np.column_stack((vele_filt, veln_filt, velu_filt))
            std_velenu = 3 * np.nanstd(velenu, axis=0)
            mean_noise_velenu = np.nanmean(velenu, axis=0)
            self.data['nvele'] = std_velenu[0]
            self.data['nveln'] = std_velenu[1]
            self.data['nvelu'] = std_velenu[2]
            self.data['mnvele'] = mean_noise_velenu[0]
            self.data['mnveln'] = mean_noise_velenu[1]
            self.data['mnvelu'] = mean_noise_velenu[2]

    def input_prep(self):
        """
        Prepares the input data for further processing.

        This method calculates the window size and applies filtering and noise handling for the
        velocity components (VELE, VELN, VELU) using a moving average and Gaussian filters.

        It also prepares the components for different calculation modes (1D, 2D, 3D) and returns
        the processed data.

        Returns
        -------
        tuple
            A tuple containing:
            - vel_com : numpy.ndarray
                The processed velocity component data (based on mode).
            - vel_org : numpy.ndarray
                The original velocity data (based on mode).
            - nvel_com : numpy.ndarray
                The processed noise level for the velocity components.
            - WS : int
                The window size for the moving average.
            - WS_short : int
                The shortened window size.
        """
        WS = int(self.window_size * self.fre)  # Window size: 5 (UI) * frequency (station dependent)
        if WS < 3:
            raise ValueError(
                f'Station: {self.name} \n Sampling frequency {self.fre}, corresponding to {WS} samples < 3, select larger window size.')
        WS_short = WS // 2

        vele = self.data['vele'] * self.fre  # m/s
        veln = self.data['veln'] * self.fre  # m/s
        velu = self.data['velu'] * self.fre  # m/s

        nvele = self.data['nvele']
        nveln = self.data['nveln']
        nvelu = self.data['nvelu']
        avg_nvele = self.data['mnvele']  # m/s
        avg_nveln = self.data['mnveln']  # m/s
        avg_nvelu = self.data['mnvelu']  # m/s
        char_timescale = 1 / self.fre

        # Calculating the sigma values for Gaussian filter
        sigma_te = ((nvele + nvele * 0.5) / abs(avg_nvele)) * char_timescale
        sigma_tn = ((nveln + nveln * 0.5) / abs(avg_nveln)) * char_timescale
        sigma_tu = ((nvelu + nvelu * 0.5) / abs(avg_nvelu)) * char_timescale

        def prepare_components():
            """
            Prepares the components for different calculation modes (1D, 2D, 3D).

            This method applies filtering to the velocity components (VELE, VELN, VELU) and computes
            the gradient of the moving average. The noise is then handled accordingly.

            Returns
            -------
            tuple
                A tuple containing:
                - vel_com : numpy.ndarray
                    The processed velocity component data (based on mode).
                - vel_org : numpy.ndarray
                    The original velocity data (based on mode).
                - nvel_com : numpy.ndarray
                    The processed noise level for the velocity components.
            """
            r = defaultdict()
            for comp_name, component, noise, sigma_t in zip(('vele', 'veln', 'velu'),
                                                            (vele, veln, velu),
                                                            (nvele, nveln, nvelu),
                                                            (sigma_te, sigma_tn, sigma_tu)):
                vel_org = component
                mavg_vele_n = moving_average_with_regression_interpolation(vel_org, WS)

                # Apply Gaussian filter
                mavg_vele_g = gaussian_filter1d_modified(vel_org, sigma_t, WS, mode="valid")
                mavg_vele = mavg_vele_g - mavg_vele_n

                grad_mavg_vele = mavg_vele

                r[comp_name] = grad_mavg_vele
                r[f'vel_org_{comp_name}'] = vel_org
                r[f'nvel_org_{comp_name}'] = noise

            if self.mode == '1D':
                vel_com_tup = (r['vele'], r['veln'], r['velu'])
                vel_org_tup = (r['vel_org_vele'], r['vel_org_veln'], r['vel_org_velu'])
                nvel_com_tup = (r['nvel_org_vele'], r['nvel_org_veln'], r['nvel_org_velu'])
                return vel_com_tup, vel_org_tup, nvel_com_tup
            elif self.mode == '2D':
                vel_com = np.sqrt(r['vele'] ** 2 + r['veln'] ** 2)
                vel_org = np.sqrt(r['vel_org_vele'] ** 2 + r['vel_org_veln'] ** 2)
                nvel_com = np.sqrt(r['nvel_org_vele'] ** 2 + r['nvel_org_veln'] ** 2)
                return vel_com, vel_org, nvel_com
            elif self.mode == '3D':
                vel_com = np.sqrt(r['vele'] ** 2 + r['veln'] ** 2 + r['velu'] ** 2)
                vel_org = np.sqrt(r['vel_org_vele'] ** 2 + r['vel_org_veln'] ** 2 + r['vel_org_velu'] ** 2)
                nvel_com = np.sqrt(r['nvel_org_vele'] ** 2 + r['nvel_org_veln'] ** 2 + r['nvel_org_velu'] ** 2)
                return vel_com, vel_org, nvel_com

        vel_com, vel_org, nvel_com = prepare_components()
        return vel_com, vel_org, nvel_com, WS, WS_short

    def prep_report(self, swave_info_sorted, nvel_com, noise_gs_vel, name):
        """
        Prepares a report based on the S-wave information and the noise data.

        This method generates a pandas DataFrame containing relevant station information,
        including detected S-wave times, peak ground velocity, energy, noise levels, and other
        calculated attributes.

        Parameters
        ----------
        swave_info_sorted : list
            A sorted list containing S-wave detection information.
        nvel_com : numpy.ndarray
            Processed velocity component data.
        noise_gs_vel : numpy.ndarray
            Noise levels for the ground shaking velocity.
        name : str
            The name of the station.

        Returns
        -------
        tuple
            A tuple containing:
            - df : pandas.DataFrame
                The main report DataFrame with the relevant station information.
            - data2 : pandas.DataFrame
                An alternative report DataFrame with more detailed information.
        """

        if self.input_data['time'] == 'gsow':
            array = np.array(swave_info_sorted)
            data2 = pd.DataFrame(data={'Station': self.name,
                                       'S-wave detected at (sec)': array[:, 2].round(3),
                                       'End of shaking at (sec)': array[:, 3].round(3),
                                       'Shaking duration (sec)': array[:, 4].round(3),
                                       'Peak Ground Velocity (m/s)': array[:, 5].round(3),
                                       'Integrated energy (m)': array[:, 6],
                                       'Weighted energy (m)': array[:, 7],
                                       'Sampling frequency (Hz)': self.fre,
                                       'Velocity noise (m/s)': nvel_com,
                                       'Ground shaking noise (m/s2)': noise_gs_vel}).set_index(['Station'])

            data = {
                'Station': self.name,
                'S-wave detected at (sec)': [swave_info_sorted[0][2].round(3)],
                'End of shaking at (sec)': [swave_info_sorted[0][3].round(3)],
                'Shaking duration (sec)': [swave_info_sorted[0][4].round(3)],
                'Peak Ground Velocity (m/s)': [round(swave_info_sorted[0][5], 3)],
                'Integrated energy (m)': [swave_info_sorted[0][6]],
                'Weighted energy (m)': [swave_info_sorted[0][7]],
                'Sampling frequency (Hz)': self.fre,
                'Velocity noise (m/s)': nvel_com,
                'Ground shaking noise (m/s2)': noise_gs_vel
            }

            df = pd.DataFrame(data).set_index(['Station'])

            return df, data2
        elif self.input_data['time'] == 'rel':
            T0 = datetime.strptime(self.input_data['eq_time'], "%Y-%m-%d %H:%M:%S.%f")
            _, T0_SOW = utc_to_gps([T0])
            swave_info_sorted[:, 2] -= T0_SOW
            swave_info_sorted[:, 3] -= T0_SOW
            array = np.array(swave_info_sorted)
            data2 = pd.DataFrame(data={'Station': self.name,
                                       'S-wave detected at (sec)': array[:, 2].round(3),
                                       'End of shaking at (sec)': array[:, 3].round(3),
                                       'Shaking duration (sec)': array[:, 4].round(3),
                                       'Peak Ground Velocity (m/s)': array[:, 5].round(3),
                                       'Integrated energy (m)': array[:, 6],
                                       'Weighted energy (m)': array[:, 7],
                                       'Sampling frequency (Hz)': self.fre,
                                       'Velocity noise (m/s)': nvel_com,
                                       'Ground shaking noise (m/s2)': noise_gs_vel}).set_index(['Station'])

            data = {
                'Station': self.name,
                'S-wave detected at (sec)': [swave_info_sorted[0][2].round(3)],
                'End of shaking at (sec)': [swave_info_sorted[0][3].round(3)],
                'Shaking duration (sec)': [swave_info_sorted[0][4].round(3)],
                'Peak Ground Velocity (m/s)': [round(swave_info_sorted[0][5], 3)],
                'Integrated energy (m)': [swave_info_sorted[0][6]],
                'Weighted energy (m)': [swave_info_sorted[0][7]],
                'Sampling frequency (Hz)': self.fre,
                'Velocity noise (m/s)': nvel_com,
                'Ground shaking noise (m/s2)': noise_gs_vel
            }

            df = pd.DataFrame(data).set_index(['Station'])

            return df, data2
        elif self.input_data['time'] == 'utc':
            array = np.array(swave_info_sorted)
            detection_time_utc = gps_to_utc(gps_week=self.gps_week, gps_sow_array=array[:,2])
            end_of_shaking_utc  = gps_to_utc(gps_week=self.gps_week, gps_sow_array=array[:,3])
            data2 = pd.DataFrame(data={'Station': self.name,
                                       'S-wave detected at (sec)': detection_time_utc,
                                       'End of shaking at (sec)': end_of_shaking_utc,
                                       'Shaking duration (sec)': array[:, 4],
                                       'Peak Ground Velocity (m/s)': array[:, 5].round(2),
                                       'Integrated energy (m)': array[:, 6],
                                       'Weighted energy (m)': array[:, 7],
                                       'Sampling frequency (Hz)': self.fre,
                                       'Velocity noise (m/s)': nvel_com,
                                       'Ground shaking noise (m/s2)': noise_gs_vel}).set_index(['Station'])

            detection_utc = gps_to_utc(gps_week=self.gps_week, gps_sow_array=[swave_info_sorted[0][2]])
            shaking_utc = gps_to_utc(gps_week=self.gps_week, gps_sow_array=[swave_info_sorted[0][3]])
            data = {
                'Station': self.name,
                'S-wave detected at (sec)': detection_utc,
                'End of shaking at (sec)': shaking_utc,
                'Shaking duration (sec)': [swave_info_sorted[0][4]],
                'Peak Ground Velocity (m/s)': [round(swave_info_sorted[0][5], 2)],
                'Integrated energy (m)': [swave_info_sorted[0][6]],
                'Weighted energy (m)': [swave_info_sorted[0][7]],
                'Sampling frequency (Hz)': self.fre,
                'Velocity noise (m/s)': nvel_com,
                'Ground shaking noise (m/s2)': noise_gs_vel
            }

            df = pd.DataFrame(data).set_index(['Station'])

            return df, data2


    def plot_data(self, swave_info_sorted, time, grad_mavg_vele, gs_vel,
                  indices_secondaryshocks, indices_mainshock, noise_thr_pos, vel_org, utc, component_name):
        """
        Plots the data for the S-wave detection and related analysis.

        This method generates various plots based on the S-wave detection information and the
        associated velocity components.

        Parameters
        ----------
        swave_info_sorted : list
            A sorted list containing S-wave detection information.
        time : numpy.ndarray
            Array of time values.
        grad_mavg_vele : numpy.ndarray
            Array of the gradient of the moving average for the velocity components.
        gs_vel : numpy.ndarray
            Ground shaking velocity data.
        indices_secondaryshocks : numpy.ndarray
            Indices identifying the secondary shocks in the data.
        indices_mainshock : numpy.ndarray
            Indices identifying the main shock in the data.
        noise_thr_pos : float
            Noise threshold for the detection.
        vel_org : numpy.ndarray
            The original velocity data.
        utc : Bool
            If use UTC time scale
        component_name : str or None
            The component name (e.g., 'vele', 'veln', 'velu').

        Notes
        -----
        This method handles the plot generation by selecting the appropriate plotting function
        based on the input configuration.
        """
        option_methods = {
            'swa': lambda: self.custom_plot(swave_info_sorted, time, grad_mavg_vele, gs_vel,
                                            indices_secondaryshocks, indices_mainshock, noise_thr_pos,
                                            component_name),
            'zswa': lambda swave_copy=copy.deepcopy(swave_info_sorted),
                           time_copy=copy.deepcopy(time):
            self.zoomed_custom_plot(swave_copy, time_copy,
                                    grad_mavg_vele, gs_vel,
                                    indices_secondaryshocks, indices_mainshock,
                                    vel_org, component_name)
        }

        for key in ['swa', 'zswa']:
            value = self.input_data[key]
            if isinstance(value, str):
                bool_value = (value.lower() == "true")
            else:
                bool_value = bool(value)

            if bool_value:
                option_methods[key]()

    def zoomed_custom_plot(self, swave_info_sorted, time, grad_mavg_vele, gs_vel,
                           indices_secondaryshocks, indices_mainshock, vel_org, component_name):
        """
        Plots a zoomed-in view of seismic velocity data with multiple subplots for different velocity components.

        This method generates a figure with three subplots:
        - The first subplot compares the gradient of the moving average velocity and the GS velocity with labeled main and secondary shocks.
        - The second subplot shows the original velocity signal along with an S-wave arrival marker.
        - The third subplot shows the bandpass-filtered velocity signal, with additional markers for S-wave arrival.

        Parameters
        ----------
        swave_info_sorted : numpy.ndarray
            A sorted array containing information about the seismic waves. Expected shape is (n, 6), where columns
            represent different information such as S-wave start time, end time, etc.

        time : numpy.ndarray
            Array of time points corresponding to the velocity data.

        grad_mavg_vele : numpy.ndarray
            Array representing the gradient of the moving average of the velocity data.

        gs_vel : numpy.ndarray
            Array representing the GS velocity data.

        indices_secondaryshocks : numpy.ndarray
            Indices in the `gs_vel` array where secondary shocks are detected.

        indices_mainshock : numpy.ndarray
            Indices in the `gs_vel` array where the main shock is detected.

        vel_org : numpy.ndarray
            Original velocity data before any filtering.

        component_name : str
            The name of the component (e.g., 'e', 'n', 'u') for the specific velocity data to be plotted.

        Notes
        -----
        This method takes into account whether the time is in UTC, GSOW, or relative format to adjust for time
        conversion accordingly. The method also checks if a bandpass filter is applied to the velocity data and
        adjusts the plot accordingly. The `utc` flag ensures that the plot's x-axis uses UTC time if set.

        """

        # Check if time is in UTC, GSOW, or relative format
        utc = self.input_data['time'] == 'utc'
        gsow = self.input_data['time'] == 'gsow'
        rel = self.input_data['time'] == 'rel'
        fs = self.fre

        cutoff_duration = float(self.input_data['time_cutoff']) * 60
        num_samples = int(cutoff_duration * fs)  # Number of samples for each side

        if bool(self.input_data['bandpass_cutoffs']):
            lowcut = np.float16(self.input_data['lowcut'])
            substr = np.float16(self.input_data['substr'])
            lowcut, substr = check_bandpass_conditions(fs, lowcut, substr)
            highcut = (fs / 2) - substr
        else:
            lowcut = 0.1
            highcut = (fs / 2) - 0.1
        fdn = butter_bandpass_filter(vel_org, lowcut, highcut, fs, order=2)

        fnt_size = 16

        if utc:
            time = gps_to_utc(gps_week=self.gps_week, gps_sow_array=list(time))
            main_shock_time = gps_to_utc(gps_week=self.gps_week, gps_sow_array=swave_info_sorted[0, 2])[0]
            main_shock_time_end = gps_to_utc(self.gps_week, swave_info_sorted[0, 3])[0]
            sis2 = gps_to_utc(gps_week=self.gps_week, gps_sow_array=swave_info_sorted[:, 2])
            sis3 = gps_to_utc(gps_week=self.gps_week, gps_sow_array=swave_info_sorted[:, 3])
        elif gsow:
            # Use time in GPS format
            main_shock_time = swave_info_sorted[0, 2]
            main_shock_time_end = swave_info_sorted[0, 3]
        elif rel:
            # Calculate indices based on the original (absolute) time array
            main_shock_time_abs = swave_info_sorted[0, 2]
            main_shock_time_end_abs = swave_info_sorted[0, 3]
            start_idx = np.searchsorted(time, main_shock_time_abs) - num_samples
            end_idx = np.searchsorted(time, main_shock_time_end_abs) + num_samples
            start_index = max(start_idx, 0)
            end_index = min(end_idx, len(time) - 1)

            # Convert time to relative scale based on T0
            T0 = datetime.strptime(self.input_data['eq_time'], "%Y-%m-%d %H:%M:%S.%f")
            _, T0_SOW = utc_to_gps([T0])

            time_rel = time - T0_SOW
            time = np.copy(time_rel)

            swave_info_sorted[:, 2] -= T0_SOW
            swave_info_sorted[:, 3] -= T0_SOW

            main_shock_time = swave_info_sorted[0, 2]
            main_shock_time_end = swave_info_sorted[0, 3]

        if not rel:
            start_idx = np.searchsorted(time, main_shock_time) - num_samples
            end_idx = np.searchsorted(time, main_shock_time_end) + num_samples
            start_index = max(start_idx, 0)
            end_index = min(end_idx, len(time) - 1)

        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 15))

        ax[0].plot(time, grad_mavg_vele, color='orange', label=fr'$res_{{mavg_{{V(t)}}}}$', alpha=0.5)
        ax[0].plot(time, gs_vel, color='green', lw=2, label=fr'$gs_{{V(t)}}$')
        ax[0].scatter(np.array(time)[indices_mainshock], gs_vel[indices_mainshock], color='m', s=40, label='main shock')
        ax[0].scatter(np.array(time)[indices_secondaryshocks], gs_vel[indices_secondaryshocks], color='red', s=40,
                      label='secondary shocks')

        if utc:
            ax[0].axvspan(main_shock_time, sis3[0], color='lavender', alpha=0.5)
            ax[0].axvline(x=sis2[0], color='m', linestyle='--', linewidth=3, label='S-wave arrival')
            ax[0].set_xlim(time[start_index], time[end_index])
            ax[0].set_ylabel(fr'$res_{{mavg_{{V(t)}}}}$', fontsize=fnt_size)
            ax[0].tick_params(axis='x', labelbottom=False, labelsize=fnt_size)

            ax[1].plot(time, vel_org, color='grey')
            ax[1].axvspan(sis2[0], sis3[0], color='lavender', alpha=0.5)
            ax[1].axvline(x=sis2[0], color='m', linestyle='--', linewidth=3, label='S-wave arrival')
            ax[1].set_xlim(time[start_index], time[end_index])
            ax[1].set_ylabel(f'vel', fontsize=fnt_size)
            ax[1].tick_params(axis='x', labelbottom=False, labelsize=fnt_size)

            ax[2].plot(time, fdn, label=f'Bandpass: {lowcut:.2f} - {highcut:.2f} Hz', color='black')
            ax[2].axvspan(sis2[0], sis3[0], color='lavender', alpha=0.5)
            ax[2].axvline(x=sis2[0], color='m', linestyle='--', linewidth=3)
            ax[2].set_xlim(time[start_index], time[end_index])
            ax[2].set_ylabel(r'$vel_{\mathrm{bfil}}$', fontsize=fnt_size)
            ax[2].set_xlabel('UTC Time [HH:MM:SS]' if utc else 'GPS SOW [sec]', fontsize=fnt_size)
            ax[2].tick_params(axis='both', labelsize=fnt_size)

            ax[2].xaxis.set_major_formatter(CustomDateFormatter('%H:%M:%S.%f'))
            ax[2].xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()
            plt.xlabel('UTC Time [HH:MM:SS]', fontsize=fnt_size)
        elif gsow or rel:
            ax[0].axvspan(swave_info_sorted[0, 2], swave_info_sorted[0, 3], color='lavender', alpha=0.5)
            ax[0].axvline(x=swave_info_sorted[0, 2], color='m', linestyle='--', linewidth=3, label='S-wave arrival')
            ax[0].set_xlim(time[start_index], time[end_index])
            ax[0].set_ylabel(fr'$res_{{mavg_{{V(t)}}}}$', fontsize=fnt_size)
            ax[0].tick_params(axis='x', labelbottom=False, labelsize=fnt_size)

            ax[1].plot(time, vel_org, color='grey')
            ax[1].axvspan(swave_info_sorted[0, 2], swave_info_sorted[0, 3], color='lavender', alpha=0.5)
            ax[1].axvline(x=swave_info_sorted[0, 2], color='m', linestyle='--', linewidth=3, label='S-wave arrival')
            ax[1].set_xlim(time[start_index], time[end_index])
            ax[1].set_ylabel(f'vel', fontsize=fnt_size)
            ax[1].tick_params(axis='x', labelbottom=False, labelsize=fnt_size)

            ax[2].plot(time, fdn, color='black',label=f'Bandpass: {lowcut:.2f} - {highcut:.2f} Hz')
            ax[2].axvspan(swave_info_sorted[0, 2], swave_info_sorted[0, 3], color='lavender', alpha=0.5)
            ax[2].axvline(x=swave_info_sorted[0, 2], color='m', linestyle='--', linewidth=3)
            ax[2].set_xlim(time[start_index], time[end_index])
            ax[2].set_ylabel(r'$vel_{\mathrm{bfil}}$', fontsize=fnt_size)
            ax[2].set_xlabel('UTC Time [HH:MM:SS]' if utc else 'GPS SOW [sec]' if gsow else 'Relative eq time [sec]',
                             fontsize=fnt_size)
            ax[2].tick_params(axis='both', labelsize=fnt_size)

        for a in ax:
            for spine in a.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

        for i in range(3):
            ax[i].grid(True)
            ax[i].tick_params(axis='both', which='major', labelsize=fnt_size - 2)
            if i != 1:
                ax[i].legend(fontsize=fnt_size - 2, loc='upper left')

        if self.mode == '1D':
            comp_map = {'e': 'East', 'n': 'North', 'u': 'Up'}
            plt.suptitle(
                self.name + f": {self.mode} {self.method} {comp_map[component_name]} component - Comparison with original and bandpass filtered vel \n"
                            f"{self.input_data['DATE_STR']}",
                fontsize=fnt_size + 2)

            plt.savefig(f'{self.fig_path}/{self.method}_BPS_{self.name}_{self.mode}_{component_name}.png')
            plt.close()
        else:
            plt.suptitle(
                self.name + f": {self.mode} {self.method}  - Comparison with original and bandpass filtered vel \n"
                            f"{self.input_data['DATE_STR']}",
                fontsize=fnt_size + 2)
            plt.tight_layout()
            plt.savefig(f'{self.fig_path}/{self.method}_BPS_{self.name}_{self.mode}.png')
            plt.close()

    def custom_plot(self, swave_info_sorted, time, grad_mavg_vele, gs_vel,
                    indices_secondaryshocks, indices_mainshock, noise_thr_pos, component_name=None):
        """
        Generates and saves a plot of the seismic wave data, with different modes of time representation.

        Parameters
        ----------
        swave_info_sorted : ndarray
            Sorted seismic wave information with columns representing various timestamps
            such as detection time and end time of shaking.
        time : ndarray
            Array of time points, either in UTC, GPS SOW, or relative earthquake time.
        grad_mavg_vele : ndarray
            Gradient of the moving average of velocity.
        gs_vel : ndarray
            Ground speed velocity data.
        indices_secondaryshocks : ndarray
            Indices of secondary shock events.
        indices_mainshock : ndarray
            Indices of the main shock event.
        noise_thr_pos : float
            Position of the noise threshold to be marked on the plot.
        component_name : str, optional
            The component of the seismic wave ('e', 'n', 'u'), default is None.

        Notes
        -----
        This function generates a plot and saves it as both PNG and SVG files. The plot includes information
        about secondary and main shocks, seismic wave durations, and peak ground velocity. The time axis
        may be represented in UTC, GPS SOW, or relative earthquake time depending on the selected mode.
        """

        utc = self.input_data['time'] == 'utc'
        gsow = self.input_data['time'] == 'gsow'
        rel = self.input_data['time'] == 'rel'

        if self.mode in ['2D', '3D']:
            if utc:
                time = gps_to_utc(gps_week=self.gps_week, gps_sow_array=time)
            if rel:
                T0 = datetime.strptime(self.input_data['eq_time'], "%Y-%m-%d %H:%M:%S.%f")
                _, T0_SOW = utc_to_gps([T0])
                time -= T0_SOW
                swave_info_sorted[:, 2] -= T0_SOW
                swave_info_sorted[:, 3] -= T0_SOW

            fnt_size = 16
            plt.figure(figsize=(15, 9))
            plt.title(self.name + f" {self.mode} \n {self.input_data['DATE_STR']}", fontsize=fnt_size)
            plt.plot(time, grad_mavg_vele, color='orange', label=fr'$res_{{mavg_{{V(t)}}}}$', alpha=0.5)
            plt.plot(time, gs_vel, color='green', lw=2, label=fr'$gs_{{V(t)}}$')

            plt.scatter(np.array(time)[indices_secondaryshocks], gs_vel[indices_secondaryshocks], color='red', s=40,
                        label='secondary shocks')
            plt.scatter(np.array(time)[indices_mainshock], gs_vel[indices_mainshock], color='m', s=40,
                        label='main shock')
            plt.axhline(y=noise_thr_pos, color='red', linestyle='--')

            if utc:
                x1 = gps_to_utc(self.gps_week, swave_info_sorted[0, 2])
                x2 = gps_to_utc(self.gps_week, swave_info_sorted[0, 3])
                plt.axvline(x=x1[0], color='m', linestyle='--')
                plt.axvline(x=x2[0], color='m', linestyle='--')
            elif rel or gsow:
                x1 = swave_info_sorted[0, 2]
                x2 = swave_info_sorted[0, 3]
                plt.axvline(x=x1, color='m', linestyle='--')
                plt.axvline(x=x2, color='m', linestyle='--')

            plt.ylabel(fr'$res_{{mavg_{{V(t)}}}}$', fontsize=fnt_size)
            if utc:
                ax = plt.gca()
                ax.xaxis.set_major_formatter(CustomDateFormatter('%H:%M:%S.%f'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.gcf().autofmt_xdate()
                plt.xlabel('UTC Time [HH:MM:SS]', fontsize=fnt_size)
            elif gsow:
                plt.xlabel('GPS SOW [sec]', fontsize=fnt_size)
            else:
                plt.xlabel('Relative eq time [sec]', fontsize=fnt_size)

            if utc:
                t1 = gps_to_utc(self.gps_week, [swave_info_sorted[0, 2]])[0]
                t2 = gps_to_utc(self.gps_week, [swave_info_sorted[0, 3]])[0]
                t3 = swave_info_sorted[0, 4]
                t4 = round(swave_info_sorted[0, 5], 2)
                info_text = (
                    f"S-wave detected at: {t1.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]} sec\n"
                    f"End of shaking at: {t2.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]} sec\n"
                    f"Shaking duration: {t3} sec\n"
                    f"Peak Ground Velocity: {t4}"
                )
            else:
                info_text = (
                    f"S-wave detected at: {round(swave_info_sorted[0, 2],2)} sec\n"
                    f"End of shaking at: {round(swave_info_sorted[0, 3],2)} sec\n"
                    f"Shaking duration: {round(swave_info_sorted[0, 4],2)} sec\n"
                    f"Peak Ground Velocity: {round(swave_info_sorted[0, 5], 2)}"
                )

            plt.grid(True)
            plt.legend(fontsize=fnt_size - 2, loc='upper left')  # Show legend with reduced font size
            plt.tick_params(axis='both', which='major', labelsize=fnt_size - 2)
            plt.subplots_adjust(bottom=0.3)
            plt.figtext(0.95, 0.05, info_text, wrap=True, horizontalalignment='right', fontsize=fnt_size - 2,
                        bbox=dict(facecolor='white', alpha=0.8))

            plt.savefig(f'{self.fig_path}/{self.method}_{self.name}_{self.mode}.png')
            plt.close()

        elif self.mode == '1D':
            if utc:
                time = gps_to_utc(gps_week=self.gps_week, gps_sow_array=time)
            if rel:
                T0 = datetime.strptime(self.input_data['eq_time'], "%Y-%m-%d %H:%M:%S.%f")
                _, T0_SOW = utc_to_gps([T0])
                time -= T0_SOW
                swave_info_sorted[:, 2] -= T0_SOW
                swave_info_sorted[:, 3] -= T0_SOW
            fnt_size = 16
            plt.figure(figsize=(15, 9))
            comp_map = {'e': 'East', 'n': 'North', 'u': 'Up'}

            plt.title(self.name + f" {self.mode} {comp_map[component_name]} component \n {self.input_data['DATE_STR']}",
                      fontsize=fnt_size)
            plt.plot(time, grad_mavg_vele, color='orange', label=fr'$res_{{mavg_{{V(t)}}}}$', alpha=0.5)
            plt.plot(time, gs_vel, color='green', lw=2, label=fr'$gs_{{V(t)}}$')

            plt.scatter(np.array(time)[indices_secondaryshocks], gs_vel[indices_secondaryshocks], color='red', s=40,
                        label='secondary shocks')
            plt.scatter(np.array(time)[indices_mainshock], gs_vel[indices_mainshock], color='m', s=40,
                        label='main shock')
            plt.axhline(y=noise_thr_pos, color='red', linestyle='--')

            if utc:
                x1 = gps_to_utc(self.gps_week, [swave_info_sorted[0, 2]])
                x2 = gps_to_utc(self.gps_week, [swave_info_sorted[0, 3]])
                plt.axvline(x=x1, color='m', linestyle='--')
                plt.axvline(x=x2, color='m', linestyle='--')
            else:
                x1 = swave_info_sorted[0, 2]
                x2 = swave_info_sorted[0, 3]
                plt.axvline(x=x1, color='m', linestyle='--')
                plt.axvline(x=x2, color='m', linestyle='--')

            plt.ylabel(fr'$res_{{mavg_{{V(t)}}}}$', fontsize=fnt_size)

            if utc:
                ax = plt.gca()
                ax.xaxis.set_major_formatter(CustomDateFormatter('%H:%M:%S.%f'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.gcf().autofmt_xdate()
                plt.xlabel('UTC Time [HH:MM:SS]', fontsize=fnt_size)
            elif gsow:
                plt.xlabel('GPS SOW [sec]', fontsize=fnt_size)
            else:
                plt.xlabel('Relative eq time [sec]', fontsize=fnt_size)

            if utc:
                t1 = gps_to_utc(self.gps_week, [swave_info_sorted[0, 2]])[0]
                t2 = gps_to_utc(self.gps_week, [swave_info_sorted[0, 3]])[0]
                t3 = swave_info_sorted[0, 4]
                t4 = round(swave_info_sorted[0, 5], 2)
                info_text = (
                    f"S-wave detected at: {t1.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]} sec\n"
                    f"End of shaking at: {t2.strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]} sec\n"
                    f"Shaking duration: {t3} sec\n"
                    f"Peak Ground Velocity: {t4}"
                )
            else:
                info_text = (
                    f"S-wave detected at: {swave_info_sorted[0, 2].round(2)} sec\n"
                    f"End of shaking at: {swave_info_sorted[0, 3].round(2)} sec\n"
                    f"Shaking duration: {swave_info_sorted[0, 4]} sec\n"
                    f"Peak Ground Velocity: {round(swave_info_sorted[0, 5], 2)}"
                )

            ax = plt.gca()  # Gets the current Axes object
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

            plt.grid(True)
            plt.legend(fontsize=fnt_size - 2, loc='upper left')  # Show legend with reduced font size
            plt.tick_params(axis='both', which='major', labelsize=fnt_size - 2)
            plt.subplots_adjust(bottom=0.3)
            plt.figtext(0.95, 0.05, info_text, wrap=True, horizontalalignment='right', fontsize=fnt_size - 2,
                        bbox=dict(facecolor='white', alpha=0.8))
            plt.savefig(f'{self.fig_path}/{self.method}_{self.name}_{self.mode}_{component_name}.png')
            plt.close()

    def run(self):
        """
        Executes the noise removal and input preparation process.

        Returns
        -------
        times : ndarray
            Array of times for the data.
        vel_com : ndarray
            Computed velocity data.
        vel_org : ndarray
            Original velocity data.
        nvel_com : ndarray
            Computed noise-removed velocity data.
        WS : ndarray
            Wind speed data.
        WS_short : ndarray
            Short-term wind speed data.
        """
        self.get_noise()
        vel_com, vel_org, nvel_com, WS, WS_short = self.input_prep()
        times = self.data['times'].copy()
        return times, vel_com, vel_org, nvel_com, WS, WS_short


class MADDetector(Station):
    """
    MAD (Median Absolute Deviation) Detector class for processing seismic data.

    Parameters
    ----------
    sta_name : str
        The name of the station.
    data_dict : dict
        A dictionary containing the data to process.
    input_data : dict
        The input data for analysis.
    sampling_freq : float
        The sampling frequency in Hz.
    compute_noise : bool, optional
        Whether to compute noise (default is True).
    eq_time : float, optional
        The time of the earthquake (default is None).
    """

    def __init__(
            self,
            sta_name,
            data_dict,
            input_data,
            sampling_freq,
            compute_noise=True,
            eq_time=None
    ):
        super().__init__(sta_name, data_dict, input_data, sampling_freq, compute_noise, eq_time)

    @capture_errors
    def detect(self):
        """
        Detects the S-wave based on the given data. For '2D' and '3D' modes, it processes the data and
        identifies the S-wave, generating a report and plotting results. For '1D' mode, it processes each
        component (East, North, Up) using parallel processing.

        Returns
        -------
        detection_time : float
            The time of the detected S-wave.
        df : pandas.DataFrame
            A DataFrame containing the detected event's details.
        data2 : pandas.DataFrame
            A more detailed DataFrame with additional information.
        or
        -------
        res : list
            A list of results from each component processed in '1D' mode.
        full_result : pandas.DataFrame
            A DataFrame containing the full results for each component.
        full_result2 : pandas.DataFrame
            A DataFrame containing additional results for each component.
        """

        if self.mode in ['2D', '3D']:
            result = self.run()
            time, grad_mavg_vele, vel_org, nvel_com, WS, WS_short = result
            chuncks_grad_mavg_vele = sliding_window_view(grad_mavg_vele, WS)
            gs_vel = np.zeros(len(chuncks_grad_mavg_vele))
            for ii in range(np.shape(chuncks_grad_mavg_vele)[0]):
                abs_dev = np.absolute(chuncks_grad_mavg_vele[ii, :] - np.nanmedian(chuncks_grad_mavg_vele[ii, :]))
                gs_vel[ii] = np.nanmedian(abs_dev)

            nan_values = np.full(len(grad_mavg_vele) - len(gs_vel), np.nan)
            num_nan = len(grad_mavg_vele) - len(gs_vel)
            num_nan_beginning = num_nan // 2
            num_nan_end = num_nan // 2
            if num_nan % 2 != 0:
                num_nan_beginning += 1

            nan_values_beginning = np.full(num_nan_beginning, np.nan)
            nan_values_end = np.full(num_nan_end, np.nan)
            gs_vel = np.concatenate((nan_values_beginning, gs_vel, nan_values_end))

            (noise_thr_pos, noise_gs_vel, ind_sw, grouped_indices, sorted_grouped_indices, sorted_pgvs, first_index_sw,
             last_index_sw, swave_info_sorted, sorted_idx_pgv_list, indices_mainshock,
             indices_secondaryshocks) = find_swave_indices(
                arr=gs_vel, arr_org=vel_org, time=time.copy(), sampf_freq=self.fre, window_size=WS, method=self.method,
                shaking_length=self.shaking_length)

            if swave_info_sorted.any():
                df, data2 = self.prep_report(swave_info_sorted, nvel_com, noise_gs_vel, self.name)
                self.plot_data(swave_info_sorted, time.copy(), grad_mavg_vele, gs_vel,
                               indices_secondaryshocks, indices_mainshock, noise_thr_pos, vel_org=vel_org, utc=True,
                               component_name=None)
                return swave_info_sorted[0, 2], df, data2

        if self.mode == '1D':
            time, grad_mavg_vele_tup, vel_org_tup, nvel_com_tup, WS, WS_short = self.run()
            res = []

            shaking_length = self.shaking_length
            fre = self.fre
            method = self.method
            result_dict = {}
            full_result_dict = {}
            try:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = [
                        executor.submit(process_mad_component, time.copy(), grad_mavg_vele, vel_org, WS, WS_short,
                                        shaking_length, fre, method, cname, nvel_com, self.input_data, self.gps_week)
                        for grad_mavg_vele, vel_org, cname, nvel_com in
                        zip(grad_mavg_vele_tup, vel_org_tup, ('e', 'n', 'u'), nvel_com_tup)
                    ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if len(result) == 2:
                            report, name = result

                            result_dict[(name, self.name)] = report
                            full_result_dict[(name, self.name)] = report
                            res.append((np.nan, name))
                            t = datetime.now()
                            timestamp = t.strftime("%Y%m%d_%H%M%S")
                            error_path = os.path.join(self.input_data['logdir'], f'{self.name}_err_{timestamp}')
                            print(f'Station: {self.name} not picked for {name}.')
                            with open(error_path, 'w') as file:
                                file.write(f'Error while processing station {self.name}: \n\n')
                                file.write(f'Station: {self.name} not picked for {name}.'
                                )

                        else:
                            detection_time, name, report, report2, to_plot = result
                            (swave_info_sorted, time, grad_mavg_vele, gs_vel, indices_secondaryshocks, indices_mainshock,
                             noise_thr_pos, vel_org) = to_plot
                            self.plot_data(swave_info_sorted, time.copy(), grad_mavg_vele, gs_vel,
                                           indices_secondaryshocks, indices_mainshock, noise_thr_pos, vel_org=vel_org,
                                           utc=True, component_name=name)
                            result_dict[(name, self.name)] = report
                            full_result_dict[(name, self.name)] = report2
                            res.append((detection_time, name))
                    except Exception as e:

                        print(f"Error while processing component: {self.name} - check error file: {error_path}")
                        continue

            except Exception as e:
                t = datetime.now()
                timestamp = t.strftime("%Y%m%d_%H%M%S")
                error_path = os.path.join(self.input_data['logdir'], f'{self.name}_err_{timestamp}')
                with open(error_path, 'w') as file:
                    file.write(f'Error while processing station {self.name}: \n')
                    file.write('\n')
                    file.write(f'{traceback.print_exc()}')
                print(f"Error while running parallel process: {self.name} \ check error file: {error_path}")

            full_result = pd.concat(result_dict, keys=result_dict.keys(), names=['component', 'Station']).swaplevel(0,
                                                                                                                    1)
            full_result2 = pd.concat(full_result_dict, keys=full_result_dict.keys(),
                                     names=['component', 'Station']).swaplevel(0, 1)
            return res, full_result, full_result2


class SLOPEDetector(Station):
    """
    A class that detects slopes in seismic data using the Best Linear Unbiased Estimation (BLUE) method and provides slope-based analysis.

    Parameters
    ----------
    sta_name : str
        The name of the station.
    data_dict : dict
        A dictionary containing data for the station.
    input_data : any
        Input data for the station.
    sampling_freq : float
        The sampling frequency of the data.
    compute_noise : bool, optional
        Flag to compute noise, by default True.
    eq_time : float, optional
        The time of the earthquake, by default None.
    """

    def __init__(self, sta_name, data_dict, input_data, sampling_freq, compute_noise=True, eq_time=None):
        """
        Initializes the SLOPEDetector class.

        Parameters
        ----------
        sta_name : str
            The name of the station.
        data_dict : dict
            A dictionary containing data for the station.
        input_data : any
            Input data for the station.
        sampling_freq : float
            The sampling frequency of the data.
        compute_noise : bool, optional
            Flag to compute noise, by default True.
        eq_time : float, optional
            The time of the earthquake, by default None.
        """
        super().__init__(sta_name, data_dict, input_data, sampling_freq, compute_noise, eq_time, CFG)

    def fill_slopes_with_regression(self, slopes, nan_indices):
        """
        Fills missing slope values using linear regression.

        This method interpolates the NaN values in the given `slopes` array by using linear regression
        based on the valid slope values at the beginning of the array.

        Parameters
        ----------
        slopes : numpy.ndarray
            Array of slope values, some of which may be NaN.
        nan_indices : numpy.ndarray
            Boolean array indicating where NaN values are located in `slopes`.

        Returns
        -------
        numpy.ndarray
            The slopes array with missing values filled using linear regression.
        """
        # Determine the number of NaNs to add at the beginning
        num_nan_beginning = np.sum(nan_indices)  # Number of NaNs at the beginning

        # Interpolate missing values at the beginning using linear regression
        if num_nan_beginning > 0:
            # Use the first few valid slope values for interpolation
            num_valid_values_for_interp = min(len(slopes), num_nan_beginning)
            X = np.arange(num_valid_values_for_interp).reshape(-1, 1)
            y = slopes[:num_valid_values_for_interp]

            # Fit a linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predict values for the missing beginning points
            interp_X = np.arange(-num_nan_beginning, 0).reshape(-1, 1)
            interpolated_values = model.predict(interp_X)
        else:
            interpolated_values = np.array([])

        # Combine interpolated values with the original slopes
        slopes_filled = np.concatenate((interpolated_values, slopes))

        return slopes_filled

    def detect(self):
        """
        Detects seismic events by computing slopes and applying S-wave picking.

        Depending on the mode (1D, 2D, or 3D), this method processes the seismic data to detect S-waves,
        estimates slopes, and identifies seismic events.

        Returns
        -------
        tuple
            If mode is '2D' or '3D', returns the first detected S-wave time, a DataFrame containing the report,
            and the full data for S-wave picking.
            If mode is '1D', returns the pick times, full result, and the second full result.
        """

        if self.mode in ['2D', '3D']:
            result = self.run()
            time, grad_mavg_vele, vel_org, nvel_com, WS, WS_short = result
            nan_indices = np.isnan(time)
            chuncks_grad_mavg_vele = grad_mavg_vele[~nan_indices]
            chuncks_grad_mavg_vele = sliding_window_view(chuncks_grad_mavg_vele, WS_short)
            chuncks_time = np.array(time)[~nan_indices]
            chuncks_time = sliding_window_view(chuncks_time, WS_short)
            chuncks_window_indices = sliding_window_view(np.arange(len(grad_mavg_vele))[~nan_indices],
                                                         WS_short)

            slopes = np.zeros(len(chuncks_grad_mavg_vele))
            time_slopes = np.zeros(len(chuncks_grad_mavg_vele))

            for ii in range(
                    len(chuncks_grad_mavg_vele)):
                chunck = chuncks_grad_mavg_vele[ii]
                chunck_time = chuncks_time[ii]
                chunck_time = chunck_time - chunck_time[-1]
                chunck_idx = chuncks_window_indices[ii]

                time_slopes[ii] = findLast(chunck_time)

                # Initialize estimation process
                nr_epochs = len(chunck)  # Estimate the nr of epochs of the observations in the chunk
                A_mat = np.ones((nr_epochs, 2))  # Create the functional A matrix
                A_mat[:, 1] = chunck_time
                yobs = chunck  # Get the y vector with the observations
                Qyy = np.identity(nr_epochs) * self.CFG['SLOPE_METHOD']['Qyy_scaler']  # Define identity variance covariance matrix
                wq = np.full(nr_epochs, self.CFG['SLOPE_METHOD']['Qyy_scaler'])
                if self.wcase == 'boxcar':
                    # Case boxcart (segment: half 0.4, half 1)
                    w1 = np.full(int(nr_epochs / 2), self.CFG['SLOPE_METHOD']['Qyy_scaler'])  # Create arrays filled with the desired values
                    w2 = np.full(int(nr_epochs / 2), 1)
                    weights_diag = np.concatenate((w1, w2))  # Concatenate the arrays
                    np.fill_diagonal(Qyy, weights_diag)  # Place the values from weights_diag on the diagonal of Qyy

                elif self.wcase == 'sigmoid':
                    # Case sigmoid (segment: flat 0.4, sigmoid, flat 1)
                    segment_length = nr_epochs // 3  # Calculate the lengths of each segment (flat 0.4, sigmoid, flat 1)
                    middle_segment_length = nr_epochs - 2 * segment_length  # Calculate the length of the middle segment and the remaining values
                    lengths = [segment_length, middle_segment_length,
                               segment_length]  # Calculate the lengths of w1, w2, and w3
                    remaining_values = nr_epochs - np.sum(lengths)

                    if remaining_values > 0:  # Adjust the length of the middle segment by distributing remaining values
                        lengths[1] += remaining_values // 2
                        lengths[2] += remaining_values - remaining_values // 2

                    w1 = np.full(lengths[0], self.CFG['SLOPE_METHOD']['Qyy_scaler'])  # Create the first flat segment with values of 0.4
                    sigmoid_values = np.linspace(self.CFG['SLOPE_METHOD']['Qyy_scaler'], 1, lengths[
                        1])  # Create the middle sigmoid segment with values increasing from 0.4 to 1
                    w2 = sigmoid_values
                    w3 = np.full(lengths[2], 1)  # Create the last flat segment with values of 1
                    weights_diag = np.concatenate((w1, w2, w3))  # Concatenate the arrays
                    np.fill_diagonal(Qyy, weights_diag)  # Place the values from weights_diag on the diagonal of Qyy

                elif self.wcase == 'trapez':
                    # Case trapezoid (segment: ramp 0.4 - 1, flat 1, downward ramp 1 to 0.4)
                    segment_length = nr_epochs // 3  # Calculate the lengths of each segment (ramp 0.4 - 1, flat 1, downward ramp 1 to 0.4))
                    middle_segment_length = nr_epochs - 2 * segment_length  # Calculate the length of the middle segment and the remaining values
                    lengths = [segment_length, middle_segment_length, 0]  # Calculate the lengths of w1, w2, and w3
                    remaining_values = nr_epochs - np.sum(lengths)

                    if remaining_values > 0:  # Adjust the length of the middle segment by distributing remaining values
                        lengths[1] += remaining_values - 4
                        lengths[2] += 4

                    w1 = np.linspace(self.CFG['SLOPE_METHOD']['Qyy_scaler'], 1, lengths[0])  # Create the ramp segment with values ramping from 0.4 to 1
                    w2 = np.full(lengths[1], 1)  # Create the flat segment with values flat at 1
                    w3 = np.linspace(1, self.CFG['SLOPE_METHOD']['Qyy_scaler'],
                                     lengths[2])  # Create the downward ramp segment with values ramping from 1 to 0.4

                    weights_diag = np.concatenate((w1, w2, w3))  # Concatenate the arrays
                    np.fill_diagonal(Qyy, weights_diag)  # Place the values from weights_diag on the diagonal of Qyy
                    np.fill_diagonal(Qyy, weights_diag)  # Place the values from weights_diag on the diagonal of Qyy

                # Estimate the unknown parameters using Best Linear Unbiased Estiation (BLUE)
                x_hat, Qx_hat = BLUE(A_mat, yobs, Qyy)  # x_hat  contains intercept, slope
                yobs_hat = x_hat[0] + x_hat[1] * chunck_time

                slopes[ii] = x_hat[1]
            num_nan_beginning = np.sum(nan_indices)  # Number of NaNs at the beginning
            num_nan_end = len(grad_mavg_vele) - len(slopes) - num_nan_beginning
            slopes = self.fill_slopes_with_regression(slopes, num_nan_beginning + num_nan_end)
            neg_idx = np.where(slopes <= 0)[0]
            pos_idx = np.where(slopes > 0)[0]
            posm_gs_vel = slopes.copy()
            posm_gs_vel[neg_idx] = np.absolute(slopes[neg_idx])
            chuncks_posm_gs_vel = sliding_window_view(posm_gs_vel, WS)
            gs_vel = np.nanstd(chuncks_posm_gs_vel, -1)
            nan_values = np.full(len(posm_gs_vel) - len(gs_vel), np.nan)

            num_nan = len(grad_mavg_vele) - len(gs_vel)
            num_nan_beginning = num_nan // 2
            num_nan_end = num_nan // 2
            if num_nan % 2 != 0:
                num_nan_beginning += 1

            nan_values_beginning = np.full(num_nan_beginning, np.nan)
            nan_values_end = np.full(num_nan_end, np.nan)

            gs_vel = np.concatenate((nan_values_beginning, gs_vel, nan_values_end))

            (noise_thr_pos, noise_gs_vel, ind_sw, grouped_indices, sorted_grouped_indices, sorted_pgvs, first_index_sw,
             last_index_sw,
             swave_info_sorted, sorted_idx_pgv_list, indices_mainshock, indices_secondaryshocks) = find_swave_indices(
                arr=gs_vel.copy(), arr_org=vel_org.copy(), time=time.copy(), sampf_freq=self.fre, window_size=WS, method=self.method,
                shaking_length=self.shaking_length)

            if swave_info_sorted.any():
                df, data2 = self.prep_report(swave_info_sorted, nvel_com, noise_gs_vel, self.name)
                self.plot_data(swave_info_sorted, time, grad_mavg_vele, gs_vel,
                               indices_secondaryshocks, indices_mainshock, noise_thr_pos, vel_org=vel_org, utc=True,
                               component_name=None)
                return swave_info_sorted[0, 2], df, data2


        elif self.mode == '1D':
            time, grad_mavg_vele_tup, vel_org_tup, nvel_com_tup, WS, WS_short = self.run()
            result_dict = {}
            full_result_dict = {}
            pick_times = []
            for grad_mavg_vele, vel_org, comp_name, nvel_com in zip(grad_mavg_vele_tup, vel_org_tup, ('e', 'n', 'u'),
                                                                    nvel_com_tup):
                nan_indices = np.isnan(time)
                chuncks_grad_mavg_vele = grad_mavg_vele[~nan_indices]
                chuncks_grad_mavg_vele = sliding_window_view(chuncks_grad_mavg_vele, WS_short)
                chuncks_time = np.array(time)[~nan_indices]
                chuncks_time = sliding_window_view(chuncks_time, WS_short)
                chuncks_window_indices = sliding_window_view(np.arange(len(grad_mavg_vele))[~nan_indices],
                                                             WS_short)  # Generate sliding windows with indices of the grad_mavg_vele (without nans)

                #### SLOPE COMPUTING USING LS-BLUE ####
                slopes = np.zeros(len(chuncks_grad_mavg_vele))
                time_slopes = np.zeros(len(chuncks_grad_mavg_vele))

                for ii in range(
                        len(chuncks_grad_mavg_vele)):  # loop over each chunk from norm_mavg_vele (their time stamp and indices)
                    chunck = chuncks_grad_mavg_vele[ii]
                    chunck_time = chuncks_time[ii]
                    chunck_time = chunck_time - chunck_time[-1]
                    chunck_idx = chuncks_window_indices[ii]

                    time_slopes[ii] = findLast(chunck_time)

                    # Initialize estimation process
                    nr_epochs = len(chunck)  # Estimate the nr of epochs of the observations in the chunk
                    A_mat = np.ones((nr_epochs, 2))  # Create the functional A matrix
                    A_mat[:, 1] = chunck_time
                    yobs = chunck  # Get the y vector with the observations
                    Qyy = np.identity(nr_epochs) * self.CFG['SLOPE_METHOD']['Qyy_scaler']  # Define identity variance covariance matrix

                    # Weighting scheme
                    if self.wcase == 'boxcar':
                        # Case boxcart (segment: half 0.4, half 1)
                        w1 = np.full(int(nr_epochs / 2), self.CFG['SLOPE_METHOD']['Qyy_scaler'])  # Create arrays filled with the desired values
                        w2 = np.full(int(nr_epochs / 2), 1)
                        weights_diag = np.concatenate((w1, w2))  # Concatenate the arrays
                        np.fill_diagonal(Qyy, weights_diag)  # Place the values from weights_diag on the diagonal of Qyy

                    elif self.wcase == 'sigmoid':
                        # Case sigmoid (segment: flat 0.4, sigmoid, flat 1)
                        segment_length = nr_epochs // 3  # Calculate the lengths of each segment (flat 0.4, sigmoid, flat 1)
                        middle_segment_length = nr_epochs - 2 * segment_length  # Calculate the length of the middle segment and the remaining values
                        lengths = [segment_length, middle_segment_length,
                                   segment_length]  # Calculate the lengths of w1, w2, and w3
                        remaining_values = nr_epochs - np.sum(lengths)

                        if remaining_values > 0:  # Adjust the length of the middle segment by distributing remaining values
                            lengths[1] += remaining_values // 2
                            lengths[2] += remaining_values - remaining_values // 2

                        w1 = np.full(lengths[0], self.CFG['SLOPE_METHOD']['Qyy_scaler'])  # Create the first flat segment with values of 0.4
                        sigmoid_values = np.linspace(self.CFG['SLOPE_METHOD']['Qyy_scaler'], 1, lengths[
                            1])  # Create the middle sigmoid segment with values increasing from 0.4 to 1
                        w2 = sigmoid_values
                        w3 = np.full(lengths[2], 1)  # Create the last flat segment with values of 1
                        weights_diag = np.concatenate((w1, w2, w3))  # Concatenate the arrays
                        np.fill_diagonal(Qyy, weights_diag)  # Place the values from weights_diag on the diagonal of Qyy

                    elif self.wcase == 'trapez':
                        # Case trapezoid (segment: ramp 0.4 - 1, flat 1, downward ramp 1 to 0.4)
                        segment_length = nr_epochs // 3  # Calculate the lengths of each segment (ramp 0.4 - 1, flat 1, downward ramp 1 to 0.4))
                        middle_segment_length = nr_epochs - 2 * segment_length  # Calculate the length of the middle segment and the remaining values
                        lengths = [segment_length, middle_segment_length, 0]  # Calculate the lengths of w1, w2, and w3
                        remaining_values = nr_epochs - np.sum(lengths)

                        if remaining_values > 0:  # Adjust the length of the middle segment by distributing remaining values
                            lengths[1] += remaining_values - 4
                            lengths[2] += 4

                        w1 = np.linspace(self.CFG['SLOPE_METHOD']['Qyy_scaler'], 1,
                                         lengths[0])  # Create the ramp segment with values ramping from 0.4 to 1
                        w2 = np.full(lengths[1], 1)  # Create the flat segment with values flat at 1
                        w3 = np.linspace(1, self.CFG['SLOPE_METHOD']['Qyy_scaler'], lengths[
                            2])  # Create the downward ramp segment with values ramping from 1 to 0.4

                        weights_diag = np.concatenate((w1, w2, w3))  # Concatenate the arrays
                        np.fill_diagonal(Qyy, weights_diag)  # Place the values from weights_diag on the diagonal of Qyy
                        np.fill_diagonal(Qyy, weights_diag)  # Place the values from weights_diag on the diagonal of Qyy

                    # Estimate the unknown parameters using Best Linear Unbiased Estiation (BLUE)
                    x_hat, Qx_hat = BLUE(A_mat, yobs, Qyy)  # x_hat  contains intercept, slope
                    yobs_hat = x_hat[0] + x_hat[1] * chunck_time

                    slopes[ii] = x_hat[1]

                #### FILL SLOPES WITH INTERPOLATED VALUES TO BRING SLOPES AT INITIAL LENGTH ####

                num_nan_beginning = np.sum(nan_indices)  # Number of NaNs at the beginning
                num_nan_end = len(grad_mavg_vele) - len(slopes) - num_nan_beginning  # Remaining NaNs needed at the end
                slopes = self.fill_slopes_with_regression(slopes, num_nan_beginning + num_nan_end)
                #### ABSOLUTE ENVELOPE - FLIP NEGATIVE SLOPES ####
                neg_idx = np.where(slopes <= 0)[0]
                pos_idx = np.where(slopes > 0)[0]
                posm_gs_vel = slopes.copy()
                posm_gs_vel[neg_idx] = np.absolute(slopes[neg_idx])

                #### sliding window STD ####
                chuncks_posm_gs_vel = sliding_window_view(posm_gs_vel, WS)
                gs_vel = np.nanstd(chuncks_posm_gs_vel, -1)

                nan_values = np.full(len(posm_gs_vel) - len(gs_vel), np.nan)

                num_nan = len(grad_mavg_vele) - len(gs_vel)
                num_nan_beginning = num_nan // 2
                num_nan_end = num_nan // 2
                if num_nan % 2 != 0:
                    num_nan_beginning += 1

                nan_values_beginning = np.full(num_nan_beginning, np.nan)
                nan_values_end = np.full(num_nan_end, np.nan)

                gs_vel = np.concatenate((nan_values_beginning, gs_vel, nan_values_end))

                #### S-WAVE PICKING #### (repeats within SLOPE method but is different from MAD and W-TEST)

                (noise_thr_pos, noise_gs_vel, ind_sw, grouped_indices, sorted_grouped_indices, sorted_pgvs,
                 first_index_sw, last_index_sw,
                 swave_info_sorted, sorted_idx_pgv_list, indices_mainshock,
                 indices_secondaryshocks) = find_swave_indices(
                    arr=gs_vel.copy(), arr_org=vel_org.copy(), time=time.copy(), sampf_freq=self.fre, window_size=WS, method=self.method,
                    shaking_length=self.shaking_length)

                if swave_info_sorted.any():
                    df, data2 = self.prep_report(swave_info_sorted.copy(), nvel_com.copy(), noise_gs_vel, self.name)
                    result_dict[comp_name] = df
                    full_result_dict[comp_name] = data2
                    pick_times.append((comp_name, swave_info_sorted[0, 2]))
                    time_deepcopy = deepcopy(time)
                    sis_deepcopy = deepcopy(swave_info_sorted)
                    self.plot_data(sis_deepcopy, time_deepcopy, grad_mavg_vele, gs_vel,
                                   indices_secondaryshocks, indices_mainshock, noise_thr_pos, vel_org=vel_org,
                                   utc=True, component_name=comp_name)
                else:
                    if self.input_data['time']!='utc':
                        data = pd.DataFrame(data={
                            'S-wave detected at': [np.nan],
                            'End of shaking at': [np.nan],
                            'Shaking duration (sec)': [np.nan],
                            'Peak Ground Velocity (m/s)': [np.nan],
                            'Integrated energy (m)': [np.nan],
                            'Weighted energy (m)': [np.nan],
                            'Sampling frequency (Hz)': [np.nan],
                            'Velocity noise (m/s)': [np.nan],
                            'Ground shaking noise (m/s2)': [np.nan]})
                    else:
                        data = pd.DataFrame(data={
                            'S-wave detected at (sec)': [np.nan],
                            'End of shaking at (sec)': [np.nan],
                            'Shaking duration (sec)': [np.nan],
                            'Peak Ground Velocity (m/s)': [np.nan],
                            'Integrated energy (m)': [np.nan],
                            'Weighted energy (m)': [np.nan],
                            'Sampling frequency (Hz)': [np.nan],
                            'Velocity noise (m/s)': [np.nan],
                            'Ground shaking noise (m/s2)': [np.nan]})
                    result_dict[comp_name] = data
                    full_result_dict[comp_name] = data
                    pick_times.append((comp_name, np.nan))

            full_result = pd.concat(result_dict, keys=result_dict.keys(), names=['component', 'Station']).swaplevel(0,
                                                                                                                    1)
            full_result2 = pd.concat(full_result_dict, keys=full_result_dict.keys(),
                                     names=['component', 'Station']).swaplevel(0, 1)
            return pick_times, full_result, full_result2


class WTESTDetector(Station):
    """
    A class for detecting and processing seismic data using a W-test_varout statistical method.

    Inherits from the Station class and adds functionality for detecting seismic events by applying a statistical model to the given data.

    Attributes:
    - sta_name: str, station name.
    - data_dict: dict, dictionary containing station data.
    - input_data: array-like, input seismic data.
    - sampling_freq: float, the sampling frequency of the data.
    - compute_noise: bool, flag to compute noise (default True).
    - eq_time: float or None, the time of the earthquake (default None).
    """

    def __init__(self, sta_name, data_dict, input_data, sampling_freq, compute_noise=True, eq_time=None):
        """
        Initialize the WTESTDetector class.

        Parameters:
        - sta_name: str, the name of the station.
        - data_dict: dict, the dictionary containing station data.
        - input_data: array-like, input seismic data.
        - sampling_freq: float, the sampling frequency of the data.
        - compute_noise: bool, flag to compute noise (default True).
        - eq_time: float or None, the time of the earthquake (default None).
        """
        super().__init__(sta_name, data_dict, input_data, sampling_freq, compute_noise, eq_time, CFG)

    def wtesting(self, data, sigma, alpha, time, plot=False):
        """
        Performs Overall Model Test and Data Snooping (computing the w-test_varout statistic) on 1D time series.

        Parameters:
        - data: numpy array, the input data (e.g., grad_mavg_vele).
        - sigma: float, the standard deviation of the noise.
        - alpha: float, the significance level for hypothesis testing (e.g., 0.05 for a 5% significance level).
        - time: numpy array, the time variable for plotting or further analysis.
        - plot: bool, whether to plot the results (default False).

        Returns:
        - obs_removed: numpy array of indices of removed observations.
        - w_removed: list of w values corresponding to the removed observations.
        - w_removed_indices: list of the original indices of the removed observations.
        - w_values: numpy array of all w values, corresponding to the original data.
        """

        # Initialize data variables
        NOrig = len(data)
        N = NOrig
        yOrig = data
        y = yOrig.copy()

        # Setup the model
        AOrig = np.ones((N, 1))
        A = AOrig.copy()

        # Initialize w_values array with NaN
        w_values = np.full(N, np.nan)  # Store all w values, default to NaN

        # Test loop
        kb = 0
        obs = np.arange(0, N)  # Index array to keep track of observations
        obs_removed = np.array([], dtype=int)
        w_removed = []  # List to store w values of removed observations
        w_removed_indices = []  # List to store original indices of removed observations
        OMT = np.inf

        while OMT > kb and N > 5:
            # Compute Qxhat using the simplified inverse
            Qxhat = sigma ** 2 * np.linalg.inv(A.T @ A)

            xhat = (1 / sigma ** 2) * (Qxhat @ A.T) @ y.T

            # Predicted values and residuals
            yhat = A @ xhat
            ehat = y.T - yhat

            # Overall Model Test (OMT) value
            OMT = (1 / sigma ** 2) * (ehat.T @ ehat)
            kb = chi2.ppf(1 - alpha, N - 1)

            if OMT > kb:
                # Compute Qyhat as a scalar (diagonal values)
                Qyhat_diag = np.full(N, Qxhat[0, 0])

                # Compute Qehat (diagonal elements only)
                Qehat_diag = (sigma ** 2) - Qyhat_diag

                # Compute standard deviation of residuals (sigma_ehat)
                sigma_ehat = np.sqrt(Qehat_diag)

                # Compute w values for all observations
                w_value = ehat.reshape((len(ehat), 1)) / sigma_ehat.reshape((len(ehat), 1))

                # Update the full array of w values
                w_values[obs] = w_value.flatten()  # Store w values in the correct index

                # Find the observation with the largest w value
                idx = np.where(abs(w_value) == np.max(abs(w_value)))
                idx = idx[0][0]

                # Save the removed observation's w value and its original index
                obs_removed = np.concatenate((obs_removed, np.array([obs[idx]])))
                w_removed.append(w_value[idx][0])  # Append the w value (as a scalar)
                w_removed_indices.append(obs[idx])  # Append the original index

                # Remove the observation from data
                y = np.delete(y, idx)
                A = np.delete(A, idx, axis=0)
                obs = np.delete(obs, idx)
                N = len(y)

            if N <= 5:
                raise Exception(
                    'The minimum number of 5 observations is reached (all other removed) and the testing scheme is aborted.')

        # Return the removed indices, their corresponding w values, and the vector of all w values
        return obs_removed, w_removed, w_removed_indices, w_values
    @capture_errors
    def detect(self):
        """
        Detect seismic events based on seismic data and apply the W-test_varout and other statistical methods.

        This method uses the W-test_varout to identify significant seismic events and processes the data accordingly.
        It also performs filtering, computes sliding window statistics, and identifies S-wave arrivals.

        Returns:
        - pick_times: list of tuples containing component names and pick times.
        - full_result: pandas DataFrame containing the processed results for each component.
        - full_result2: pandas DataFrame containing additional processed data for each component.
        """

        if self.mode in ['2D', '3D']:
            result = self.run()
            time, grad_mavg_vele, vel_org, nvel_com, WS, WS_short = result
            if self.eq_time is None:
                noise_grad_mavg_vele, idx_noise_grad_mavg_vele = sigma_filter(arr=grad_mavg_vele)
            else:
                indices = np.where(time < self.eq_time)
                time = time[indices]
                noise_grad_mavg_vele, idx_noise_grad_mavg_vele = sigma_filter(arr=grad_mavg_vele[indices])
            sigma_wtest = np.nanstd(noise_grad_mavg_vele)
            level_significance = self.CFG['SLOPE_METHOD']['alpha_significance']  # 5%
            obs_removed, w_removed, w_removed_indices, w_values = self.wtesting(grad_mavg_vele, sigma_wtest,
                                                                                level_significance, time, plot=False)

            nidx_wvalues = np.where(w_values <= 0)[0]
            pidx_wvalues = np.where(w_values > 0)[0]
            pos_wvalues = w_values.copy()
            pos_wvalues[nidx_wvalues] = np.absolute(w_values[nidx_wvalues])
            chuncks_grad_mavg_vele = sliding_window_view(pos_wvalues,
                                                         WS)
            # sliding window MAD
            gs_vel = np.zeros(len(chuncks_grad_mavg_vele))
            for ii in range(np.shape(chuncks_grad_mavg_vele)[0]):
                abs_dev = np.absolute(chuncks_grad_mavg_vele[ii, :] - np.nanmedian(chuncks_grad_mavg_vele[ii, :]))
                gs_vel[ii] = np.nanmedian(abs_dev)

            num_nan = len(grad_mavg_vele) - len(gs_vel)
            num_nan_beginning = num_nan // 2
            num_nan_end = num_nan // 2
            if num_nan % 2 != 0:
                num_nan_beginning += 1

            nan_values_beginning = np.full(num_nan_beginning, np.nan)
            nan_values_end = np.full(num_nan_end, np.nan)
            gs_vel = np.concatenate((nan_values_beginning, gs_vel, nan_values_end))

            (noise_thr_pos, noise_gs_vel, ind_sw, grouped_indices, sorted_grouped_indices, sorted_pgvs, first_index_sw,
             last_index_sw,
             swave_info_sorted, sorted_idx_pgv_list, indices_mainshock, indices_secondaryshocks) = find_swave_indices(
                arr=gs_vel.copy(), arr_org=vel_org.copy(), time=time.copy(), sampf_freq=self.fre, window_size=WS, method=self.method,
                shaking_length=self.shaking_length)
            if swave_info_sorted.any():
                df, data2 = self.prep_report(swave_info_sorted.copy(), nvel_com.copy(), noise_gs_vel.copy(), self.name)
                self.plot_data(swave_info_sorted.copy(), time.copy(), grad_mavg_vele.copy(), gs_vel.copy(),
                               indices_secondaryshocks, indices_mainshock, noise_thr_pos, vel_org=vel_org, utc=True,
                               component_name=None)

                return swave_info_sorted[0, 2], df, data2

        if self.mode == '1D':
            time, grad_mavg_vele_tup, vel_org_tup, nvel_com_tup, WS, WS_short = self.run()
            result_dict = {}
            full_result_dict = {}
            pick_times = []
            for grad_mavg_vele, vel_org, comp_name, nvel_com in zip(grad_mavg_vele_tup, vel_org_tup, ('e', 'n', 'u'),
                                                                    nvel_com_tup):
                if self.eq_time is None:
                    noise_grad_mavg_vele, idx_noise_grad_mavg_vele = sigma_filter(arr=grad_mavg_vele)
                else:
                    indices = np.where(time < self.eq_time)
                    time_before = time[indices]
                    noise_grad_mavg_vele, idx_noise_grad_mavg_vele = sigma_filter(arr=grad_mavg_vele[indices])
                sigma_wtest = np.nanstd(noise_grad_mavg_vele)
                level_significance = 0.05  # 5%
                obs_removed, w_removed, w_removed_indices, w_values = self.wtesting(grad_mavg_vele, sigma_wtest,
                                                                                    self.CFG['SLOPE_METHOD']
                                                                                    ['alpha_significance'], time,
                                                                                    plot=False)

                # chuncks_grad_mavg_vele =  sliding_window_view(grad_mavg_vele, WS)  # creates a collection of mvg window arrays
                nidx_wvalues = np.where(w_values <= 0)[0]
                pidx_wvalues = np.where(w_values > 0)[0]
                pos_wvalues = w_values.copy()
                pos_wvalues[nidx_wvalues] = np.absolute(w_values[nidx_wvalues])
                chuncks_grad_mavg_vele = sliding_window_view(pos_wvalues,
                                                             WS)  # creates a collection of mvg window arrays
                # sliding window MAD
                gs_vel = np.zeros(len(chuncks_grad_mavg_vele))
                for ii in range(np.shape(chuncks_grad_mavg_vele)[0]):
                    abs_dev = np.absolute(chuncks_grad_mavg_vele[ii, :] - np.nanmedian(chuncks_grad_mavg_vele[ii, :]))
                    gs_vel[ii] = np.nanmedian(abs_dev)

                nan_values = np.full(len(grad_mavg_vele) - len(gs_vel), np.nan)

                num_nan = len(grad_mavg_vele) - len(gs_vel)
                num_nan_beginning = num_nan // 2
                num_nan_end = num_nan // 2
                if num_nan % 2 != 0:
                    num_nan_beginning += 1

                nan_values_beginning = np.full(num_nan_beginning, np.nan)
                nan_values_end = np.full(num_nan_end, np.nan)
                gs_vel = np.concatenate((nan_values_beginning, gs_vel, nan_values_end))

                # ↓↓↓ wywołanie zostaje niezmienione
                (noise_thr_pos, noise_gs_vel, ind_sw, grouped_indices, sorted_grouped_indices,
                 sorted_pgvs, first_index_sw, last_index_sw, swave_info_sorted,
                 sorted_idx_pgv_list, indices_mainshock, indices_secondaryshocks) = find_swave_indices(
                    arr=gs_vel, arr_org=vel_org, time=time,
                    sampf_freq=self.fre, window_size=WS,
                    method=self.method, shaking_length=self.shaking_length
                )


                if swave_info_sorted.any():
                    df, data2 = self.prep_report(swave_info_sorted.copy(), nvel_com.copy(), noise_gs_vel.copy(), self.name)
                    result_dict[comp_name] = df
                    full_result_dict[comp_name] = data2
                    pick_times.append((comp_name, swave_info_sorted[0, 2]))
                    time_deepcopy = deepcopy(time)
                    sis_deepcopy = deepcopy(swave_info_sorted)
                    self.plot_data(sis_deepcopy, time_deepcopy, grad_mavg_vele, gs_vel,
                                   indices_secondaryshocks, indices_mainshock, noise_thr_pos, vel_org=vel_org,
                                   utc=True, component_name=comp_name)
                else:
                    if self.input_data['time']!='utc':
                        data = pd.DataFrame(data={
                            'S-wave detected at': [np.nan],
                            'End of shaking at': [np.nan],
                            'Shaking duration (sec)': [np.nan],
                            'Peak Ground Velocity (m/s)': [np.nan],
                            'Integrated energy (m)': [np.nan],
                            'Weighted energy (m)': [np.nan],
                            'Sampling frequency (Hz)': [np.nan],
                            'Velocity noise (m/s)': [np.nan],
                            'Ground shaking noise (m/s2)': [np.nan]})
                    else:
                        data = pd.DataFrame(data={
                            'S-wave detected at (sec)': [np.nan],
                            'End of shaking at (sec)': [np.nan],
                            'Shaking duration (sec)': [np.nan],
                            'Peak Ground Velocity (m/s)': [np.nan],
                            'Integrated energy (m)': [np.nan],
                            'Weighted energy (m)': [np.nan],
                            'Sampling frequency (Hz)': [np.nan],
                            'Velocity noise (m/s)': [np.nan],
                            'Ground shaking noise (m/s2)': [np.nan]})
                    result_dict[comp_name] = data
                    full_result_dict[comp_name] = data
                    pick_times.append((comp_name, np.nan))
            full_result = pd.concat(result_dict, keys=result_dict.keys(), names=['component', 'Station']).swaplevel(0,
                                                                                                                    1)
            full_result2 = pd.concat(full_result_dict, keys=full_result_dict.keys(),
                                     names=['component', 'Station']).swaplevel(0, 1)
            return pick_times, full_result, full_result2
