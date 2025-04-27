import numpy as np
from .prep import sigma_filter
import functools
from datetime import datetime
import traceback
from .config import CFG
from pathlib import Path
import os
import traceback
import warnings
from datetime import datetime
from pathlib import Path

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

        error_output_path = Path(__file__).resolve().parent.parent/ "errors"
        error_output_path.mkdir(exist_ok=True)

        err_file = error_output_path / err_name
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Capture the traceback and write it to a file
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


def safe_log(x, y, threshold=1e-10):
    """
    Safely compute logarithm by ensuring values are above a threshold.

    Parameters
    ----------
    x : float
        First input value.
    y : float
        Second input value.
    threshold : float, optional
        Minimum allowed absolute value for x and y to safely compute log (default is 1e-10).

    Returns
    -------
    bool :
        True if both values are above the threshold, False otherwise.
    """

    if abs(x) < threshold or abs(y) < threshold:
        first_index_sw = np.nan
        last_index_sw = np.nan

        print(
            f"One of the variables is too close to zero for log computation: integ_value={x}, pgv={y} \n Skipping component...")
        return False
    else:
        return True


@capture_errors
def find_swave_indices(arr, arr_org, time, sampf_freq, window_size, method='MAD', beta=CFG['PICKING']['beta_weight'],
                       gamma=CFG['PICKING']['gamma_weight'],
                       shaking_length=3, CFG=CFG):
    """
    Identifies S-wave arrival indices in time series data using thresholding methods.

    Parameters
    ----------
    arr : ndarray
        Preprocessed time series (1D/2D/3D).
    arr_org : ndarray
        Original signal-to-noise time series.
    time : ndarray
        Time array corresponding to the time series data.
    sampf_freq : float
        Sampling frequency of the data.
    window_size : int
        Window size for extending grouped indices.
    method : str, optional
        Method for defining threshold ('MAD', 'W-TEST', 'SLOPE', 'OTHER'). Default is 'MAD'.
    beta : float, optional
        Weight given to integrated energy. Default is 0.7.
    gamma : float, optional
        Weight given to peak ground velocity (PGV). Default is 0.3.
    shaking_length : int, optional
        Minimum length of shaking to be considered. Default is 3.
    CFG: dict
        configuration dictionary
    Returns
    -------
    noise_thr_pos : float
        Threshold for noise level.
    noise_level_3 : float
        Noise level based on standard deviation or median absolute deviation.
    ind_sw : ndarray
        Indices where the preprocessed time series exceeds the threshold.
    grouped_indices : list of ndarray
        Grouped indices of detected earthquake events.
    sorted_grouped_indices : list of ndarray
        Grouped indices sorted by weighted energy.
    sorted_pgvs : list
        Peak ground velocity values sorted by weighted energy.
    first_index_sw : int or np.nan
        Index of first detected S-wave arrival.
    last_index_sw : int or np.nan
        Index of last detected S-wave arrival.
    swave_info_sorted : ndarray
        Information array for detected events, where each row represents an event:
        [first_sw, last_sw, time_f_sw, time_l_sw, shaking duration, pgv, integrated energy, weighted energy].
    sorted_idx_pgv_list : list
        Indices of PGV values corresponding to sorted PGVs.
    indices_mainshock : list
        Indices of the main earthquake event.
    indices_secondaryshocks : list
        List of indices for all other detected earthquakes.

    """

    # Calculate threshold for MAD, W-TEST, or SLOPE method
    if method in ['MAD', 'W-TEST', 'SLOPE']:
        med_mvg_std = np.nanmedian(arr)
        mad_mvg_std = np.nanmedian(np.absolute(arr - med_mvg_std))
        noise_level_3 = 3 * mad_mvg_std
        noise_thr_pos = med_mvg_std + CFG['PICKING']['rareevent_t'] * mad_mvg_std
        pos_indices = np.where(arr >= noise_thr_pos)[0]
        ind_sw = pos_indices
        ind_sw.sort()

    # Use sigma_filter for 'OTHER' method to identify noise thresholds
    elif method == 'OTHER':
        e_noise, ei_noise = sigma_filter(arr=arr, sigma=5)
        noise_arr = e_noise
        mean_noise_arr = np.nanmean(noise_arr)
        std_noise_arr = np.nanstd(noise_arr)
        noise_level_3 = 3 * std_noise_arr
        noise_thr_pos = mean_noise_arr + CFG['PICKING']['rareevent_t'] * std_noise_arr
        noise_thr_neg = mean_noise_arr - CFG['PICKING']['rareevent_t'] * std_noise_arr
        neg_indices = np.where(arr <= noise_thr_neg)[0]
        pos_indices = np.where(arr >= noise_thr_pos)[0]
        ind_sw = np.concatenate((neg_indices, pos_indices))

    # Function to group indices based on threshold and minimum length
    def group_indices(indices, threshold, min_length):
        if len(indices) == 0:
            return []
        diffs = np.diff(indices)
        gap_indices = np.where(diffs > threshold)[0]
        groups = np.split(indices, gap_indices + 1)
        filtered_groups = [group for group in groups if len(group) >= min_length]
        return filtered_groups if filtered_groups else [indices]

    # Define threshold_gap based on sampling frequency
    samp_in_onesec = sampf_freq
    if samp_in_onesec >= 1 and samp_in_onesec < 10:  # for data lower than 10Hz
        threshold_gap = 10 * sampf_freq
        min_length = shaking_length
    elif samp_in_onesec >= CFG['PICKING']['shock_separation_len_hr']:  # for data higher than 10Hz
        threshold_gap = CFG['PICKING']['shock_separation_len'] * sampf_freq
        min_length = shaking_length * sampf_freq

    # Group indices based on threshold and minimum shaking length
    grouped_indices = group_indices(ind_sw, threshold_gap, min_length)
    if not grouped_indices:
        return noise_thr_pos, ind_sw, [], [], [], [], np.nan, np.nan, np.array([]), [], [], []

    # Extend grouped indices by window_size samples at both ends
    grouped_indices_pgv = []
    for i, group in enumerate(grouped_indices):
        start_idx = max(0, group[0] - window_size)
        end_idx = min(len(arr_org), group[-1] + window_size + 1)
        grouped_indices_pgv.append(np.arange(start_idx, end_idx))

    # Compute PGV for each group
    def compute_pgv(group, data):
        """
        Computes the Peak Ground Velocity (PGV) and its corresponding index for a given group of data.

        This function calculates the maximum absolute value of the data within the specified group and
        returns both the maximum value (PGV) and the index of the data point where this maximum occurs.

        Parameters
        ----------
        group : array-like
            A sequence of indices or labels used to select the data from the `data` array.

        data : array-like
            An array or list containing the data values. The function calculates the maximum absolute value
            from the specified `group` of data points.

        Returns
        -------
        max_value : float
            The maximum absolute value (Peak Ground Velocity, PGV) within the selected group of data points.

        max_index : int or label
            The index or label corresponding to the data point with the maximum absolute value.

        """
        max_value = np.max(np.abs(data[group]))
        max_index = group[np.argmax(np.abs(data[group]))]
        return max_value, max_index

    pgv_list, idx_pgv_list = zip(*[compute_pgv(group, arr_org) for group in grouped_indices_pgv])

    # Function to compute shaking duration based on time array
    def compute_shaking_duration(group, time_array):
        """
        Computes the shaking duration based on the first and last indices in the given group.

        This function calculates the difference between the time values at the first and last indices
        within the specified `group` in `time_array`, representing the total shaking duration. If the group
        is empty, the function returns a duration of 0.

        Parameters
        ----------
        group : array-like
            A sequence of indices used to select the relevant data points from `time_array`.

        time_array : array-like
            An array of time values corresponding to the data points. The function calculates the duration
            by finding the time difference between the first and last indices in the `group`.

        Returns
        -------
        shaking_duration : float
            The calculated shaking duration, which is the time difference between the first and last
            indices in the `group`. The result is rounded to two decimal places.

        """
        if len(group) == 0:
            return 0
        shaking_duration = round(time_array[group[-1]] - time_array[group[0]], 2)
        return shaking_duration

    # Compute shaking duration for each group
    shaking_durations = [compute_shaking_duration(group, time) for group in grouped_indices]
    shake_pgv = np.array(pgv_list) * shaking_durations

    # Combine indices with shaking durations and PGVs
    grouped_indices_with_duration = list(zip(grouped_indices, shaking_durations, pgv_list, idx_pgv_list))

    # Calculate integrated energy and weighted energy for each group
    swave_info = []
    integ_energy_pgv = []
    weighted_energy_pgv = []
    for group, duration, pgv, idx_pgv in grouped_indices_with_duration:
        first_index_sw = group[0]
        last_index_sw = group[-1]
        times_first_idx_sw = time[first_index_sw]
        times_last_idx_sw = time[last_index_sw]
        shaking_durations = round(times_last_idx_sw - times_first_idx_sw, 2)
        # Integrate over arr from (first_index_sw ) to (last_index_sw )
        start_integ_idx = max(0, first_index_sw)  # - window_size)
        end_integ_idx = min(len(arr), last_index_sw + 1)  # + window_size + 1)

        # integ_value = np.trapz(np.abs(arr[start_integ_idx:end_integ_idx]), dx=1 / sampf_freq)
        integ_value = np.trapezoid(np.abs(arr[start_integ_idx:end_integ_idx]), dx=1 / sampf_freq)
        # integ_value = np.trapz(np.abs(arr[first_index_sw:last_index_sw + 1]), dx=1 / sampf_freq)
        integ_energy_pgv.append(integ_value)
        flag = safe_log(x=integ_value, y=pgv)
        if not flag:
            return [], [], [], [], [], [], np.nan, np.nan, np.array([]), [], [], []

        weighted_value = np.exp(beta * np.log(integ_value) + gamma * np.log(pgv))
        weighted_energy_pgv.append(weighted_value)
        swave_info.append(
            [first_index_sw, last_index_sw, time[first_index_sw], time[last_index_sw], duration, pgv, integ_value,
             weighted_value])

    # Combine indices with energy values
    grouped_indices_with_duration_and_energy = [(group, duration, pgv, idx_pgv, energy) for
                                                (group, duration, pgv, idx_pgv), energy in
                                                zip(grouped_indices_with_duration, integ_energy_pgv)]
    grouped_indices_with_duration_and_wenergy = [(group, duration, pgv, idx_pgv, energy, weighted_energy) for
                                                 (group, duration, pgv, idx_pgv, energy), weighted_energy in
                                                 zip(grouped_indices_with_duration_and_energy, weighted_energy_pgv)]

    # Sort groups based on shaking duration
    sorted_grouped_indices_with_duration_and_wenergy = sorted(grouped_indices_with_duration_and_wenergy,
                                                              key=lambda x: x[1], reverse=True)

    # Extract sorted values
    sorted_grouped_indices = [group for group, duration, pgv, idx_pgv, integ_energy_pgv, weighted_energy_pgv in
                              sorted_grouped_indices_with_duration_and_wenergy]
    sorted_durations = [duration for group, duration, pgv, idx_pgv, integ_energy_pgv, weighted_energy_pgv in
                        sorted_grouped_indices_with_duration_and_wenergy]
    sorted_pgvs = [pgv for group, duration, pgv, idx_pgv, integ_energy_pgv, weighted_energy_pgv in
                   sorted_grouped_indices_with_duration_and_wenergy]
    sorted_idx_pgv_list = [idx_pgv for group, duration, pgv, idx_pgv, integ_energy_pgv, weighted_energy_pgv in
                           sorted_grouped_indices_with_duration_and_wenergy]
    sorted_integ_energy_pgv = [integ_energy_pgv for group, duration, pgv, idx_pgv, integ_energy_pgv, weighted_energy_pgv
                               in sorted_grouped_indices_with_duration_and_wenergy]
    sorted_weighted_energy_pgv = [weighted_energy_pgv for
                                  group, duration, pgv, idx_pgv, integ_energy_pgv, weighted_energy_pgv in
                                  sorted_grouped_indices_with_duration_and_wenergy]

    # Sort swave_info based on weighted energy
    swave_info_sorted = sorted(swave_info, key=lambda x: x[-1], reverse=True)

    # Extract mainshock and secondary shock indices
    if len(sorted_grouped_indices) > 0:
        indices_mainshock = sorted_grouped_indices[0]
        first_index_sw = indices_mainshock[0]
        last_index_sw = indices_mainshock[-1]
    else:
        indices_mainshock = []
        first_index_sw = np.nan
        last_index_sw = np.nan

    # Get secondary shocks indices
    if len(sorted_grouped_indices) > 1:
        indices_secondaryshocks = np.concatenate(sorted_grouped_indices[1:])
        indices_secondaryshocks.sort()
    else:
        indices_secondaryshocks = []

    return noise_thr_pos, noise_level_3, ind_sw, grouped_indices, sorted_grouped_indices, sorted_pgvs, first_index_sw, last_index_sw, np.array(
        swave_info_sorted), sorted_idx_pgv_list, indices_mainshock, indices_secondaryshocks


def BLUE(A, y, Qyy):
    """
    Compute the Best Linear Unbiased Estimator (BLUE).

    Parameters
    ----------
    A : ndarray of shape (m, n)
        Projection matrix of the 3D displacements to the Line of Sight (LoS).
    y : ndarray of shape (m, 1)
        Vector with observations.
    Qyy : ndarray of shape (m, m)
        Variance-covariance matrix of the observations.

    Returns
    -------
    x_hat : ndarray of shape (n, 1)
        Vector with the estimated parameters.
    Qx_hat : ndarray of shape (n, n)
        Variance-covariance matrix of the estimated parameters.
    """

    x_hat = np.linalg.inv(A.T @ np.linalg.inv(Qyy) @ A) @ A.T @ np.linalg.inv(Qyy) @ y
    Qx_hat = np.linalg.inv(A.T @ np.linalg.inv(Qyy) @ A)
    return x_hat, Qx_hat


def findMiddle(input_list):
    """
    Find the middle element(s) of a list.

    Parameters
    ----------
    input_list : list
        The list for which to find the middle element(s).

    Returns
    -------
    mixed :
        The middle element if the list length is odd, or a tuple containing the two middle elements if the list length is even.
    """
    middle = float(len(input_list)) / 2
    if middle % 2 != 0:
        return input_list[int(middle - .5)]
    else:
        return (input_list[int(middle)], input_list[int(middle - 1)])


def findLast(input_list):
    """
    Find the last element of a list.

    Parameters
    ----------
    input_list : list or ndarray
        The list for which to find the last element.

    Returns
    -------
    mixed :
        The last element of the list.
    """
    # if len(input_list) == 1:
    return input_list[-1]
    # else:
    #    return (input_list[-1], input_list[-2])
