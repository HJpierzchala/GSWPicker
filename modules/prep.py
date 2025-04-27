import numpy as np
from scipy.signal import butter, lfilter
import traceback
from datetime import datetime
import pandas as pd
from .time import mjd2datetime, utc_to_gps, get_gps_utc_difference, get_leap_seconds_table2
from sklearn.linear_model import LinearRegression
import sys
import re
import os
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

        error_output_path = Path(__file__).resolve().parent.parent / "errors"
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

def read_file(station_path):
    """
    Reads the station rnx.varout file and returns numpy (N,3) array with dN, dE, dU displacements
    :param station_path: path to rnx.varout file
    :return: time vector (N,1), numpy array (N,3) with dN, dE, dU displacements
    """

    obs = pd.read_csv(station_path, header=None)
    obs.set_index(obs.iloc[:, 1], inplace=True)
    return np.array(obs.index.tolist()), obs.iloc[:, [7, 8, 9]].to_numpy()


def read_sys_params():
    """
    Reads parameters from command line arguments and returns them as a dictionary.

    Parameters
    ----------
    None

    Returns
    -------
    dict :
        A dictionary with the following parameters:
        - CALCULATION_MODE : str
            The mode of calculation, e.g., '1D', '2D', '3D'.
        - MODE : str
            The mode of the process, e.g., 'MAD', 'SLOPE', 'RIGID'.
        - wcase : str
            The case type, e.g., 'sigmoid', 'trapez', 'method2', or 'None'.
        - vadase_path : str
            Path to the VADASE directory.
        - DATE_STR : str
            Date in the 'YYYY-MM-DD' format.
        - GPS_DATE : datetime
            The GPS date in datetime format.
        - GPS_WEEK : int
            The GPS week.
        - COMPUTE_NOISE : bool
            Whether to compute noise or not (True/False).
        - shaking_length : int
            Length of shaking in seconds.
        - window_size : int
            The size of the window.
        - INCLUDE_STATIONS : mixed
            List of stations or None/False.
        - result_csv : str
            Path to save CSV result files.
        - result_figures : str
            Path to save figure result files.
        - storage_option : str
            Data storage option.
        - PROJECT_ID : str
            Project identifier.
        - eq_time : str
            Earthquake time.
        - swa : str
            S-wave arrivals.
        - zswa : str
            Zoomed S-wave arrivals.
        - time_cutoff : str
            Time cutoff value.
        - bandpass_cutoffs : str
            Bandpass cutoff frequencies.
        - lowcut : float
            Low cut frequency for filtering.
        - substr : str
            Substring used for filtering.
        - logdir : str
            Log directory path.
        - time : str
            Time scale.
        - leap_sec : float
            Leap seconds difference for the specific year.

    Raises
    ------
    ValueError :
        If an invalid number of arguments is passed or if there is an issue with type conversion.
    IndexError :
        If the required parameters are not provided in the correct number.
    """
    try:
        CALCULATION_MODE     = sys.argv[1]   # '1D', '2D', '3D'
        MODE                 = sys.argv[2]   # 'MAD', 'SLOPE', 'RIGID'
        wcase               = sys.argv[3]   # 'sigmoid', 'trapez', 'method2' or 'None'
        vadase_path         = sys.argv[4]   # Path to the VADASE directory
        DATE_STR            = sys.argv[5]   # Date in the format 'YYYY-MM-DD'
        # Processing the date
        DATE                = datetime.strptime(DATE_STR, '%Y-%m-%d')  # Convert date string to datetime object
        YEAR                = DATE.year
        try:
            # take local table from oyeah
            # try to connect to net
            # update if needed
            table = get_leap_seconds_table2()
            leap_table = table[table['date'].dt.year<=YEAR]
            if not leap_table.empty:
                row = leap_table.loc[leap_table['date'].idxmax()]
                leap_sec = row['leap']
        except:
            leap_sec = get_gps_utc_difference(YEAR)

        GPS_WEEK, GPS_DATE = utc_to_gps([DATE])  # If function doesn't work well, apply leap seconds

        COMPUTE_NOISE      = sys.argv[6].lower() == 'true'  # Convert to boolean
        shaking_length     = int(sys.argv[7])  # Length of shaking in seconds
        window_size        = float(sys.argv[8])  # Window size
        INCLUDE_STATIONS   = sys.argv[9]

        if INCLUDE_STATIONS.lower() == 'false':
            INCLUDE_STATIONS = False
        elif INCLUDE_STATIONS.lower() == 'none' or INCLUDE_STATIONS.strip() == '':
            INCLUDE_STATIONS = None
        else:
            # Assume user entered station names separated by commas or spaces
            INCLUDE_STATIONS = re.split(r'[,;\s]+', INCLUDE_STATIONS.strip())
            # Remove any empty strings from the list
            INCLUDE_STATIONS = [station for station in INCLUDE_STATIONS if station]

        result_csv        = sys.argv[10]  # Path to save CSV result files
        result_figures    = sys.argv[11]  # Path to save figure result files
        storage_option    = sys.argv[12]  # Data storage option
        project_id        = sys.argv[13]
        eq_time           = sys.argv[14].strip()
        if eq_time != 'None':
            eq_datetime = f'{DATE_STR} {eq_time}'
        else:
            eq_datetime = eq_time
        SWAVE_ARRIVALS    = sys.argv[15]
        ZOOMED_SWAVE_ARRIVALS = sys.argv[16]
        time_cutoff       = sys.argv[17]
        bandpass_cutoffs  = sys.argv[18]
        lowcut            = sys.argv[19]
        substr            = sys.argv[20]
        time_scale        = sys.argv[21]

        # Create required directories if they do not exist
        rcsv   = os.path.join(result_csv, f'{project_id}', CALCULATION_MODE, 'reports')
        rfig   = os.path.join(result_csv, f'{project_id}', CALCULATION_MODE, 'figures')
        log_dir = os.path.join(result_csv, f'{project_id}', CALCULATION_MODE, 'logs')
        os.makedirs(rcsv, exist_ok=True)
        os.makedirs(rfig, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Return parameters as a dictionary
        return {
            "CALCULATION_MODE"     : CALCULATION_MODE,
            "MODE"                 : MODE,
            "wcase"                : wcase,
            "vadase_path"          : vadase_path,
            "DATE_STR"             : DATE_STR,
            "GPS_DATE"             : GPS_DATE,
            "GPS_WEEK"             : GPS_WEEK,
            "COMPUTE_NOISE"        : COMPUTE_NOISE,
            "shaking_length"       : shaking_length,
            "window_size"          : window_size,
            "INCLUDE_STATIONS"     : INCLUDE_STATIONS,
            "result_csv"           : rcsv,
            "result_figures"       : rfig,
            "storage_option"       : storage_option,
            "PROJECT_ID"           : project_id,
            "eq_time"              : eq_datetime,
            "swa"                  : SWAVE_ARRIVALS,
            "zswa"                 : ZOOMED_SWAVE_ARRIVALS,
            "time_cutoff"          : time_cutoff,
            "bandpass_cutoffs"     : bandpass_cutoffs,
            "lowcut"               : lowcut,
            "substr"               : substr,
            "logdir"               : log_dir,
            "time"                 : time_scale,
            "leap_sec"             : leap_sec
        }

    except IndexError:
        raise ValueError("Incorrect number of arguments. Check if all required parameters are passed.")
    except ValueError as e:
        raise ValueError(f"Error during parameter conversion: {e}")


@capture_errors
def read_input_files(vadase_path, include_stations):
    """
    Reads GNSS data from input files and returns the data as a dictionary.

    This function processes files in the provided directory (`vadase_path`), including files with
    extensions `.varout`, `.kin`, and `.oy`. It reads the data, extracts velocity components in
    East, North, and Up directions, and stores them in a dictionary for each station. The sampling
    frequency for each station is also computed and stored.

    Parameters
    ----------
    vadase_path : str
        Path to the directory containing the GNSS data files.

    include_stations : list of str, optional
        List of station names to include for processing. If empty or None, all stations will be
        processed.

    Returns
    -------
    gnss_data_dict : dict
        A dictionary where keys are station names and values are dictionaries containing:
        - 'times' : array
            Array of timestamps for the GNSS measurements.
        - 'vele' : array
            Array of Eastward velocity components.
        - 'veln' : array
            Array of Northward velocity components.
        - 'velu' : array
            Array of Upward velocity components.

    SAMP_FREQ_DICT : dict
        A dictionary where keys are station names and values are the corresponding sampling frequencies.

    Raises
    ------
    FileNotFoundError
        If no files with supported formats (.varout, .kin, .oy) are found in the directory.
    """
    SAMP_FREQ_DICT = {}
    gnss_data_dict = {}
    stations = []

    for file in os.listdir(vadase_path):  # iterate over vadase path
        try:
            if file.endswith('.varout'):  # Process .varout files
                NAME = file.split('_')[0][:4]
                if include_stations and any(station.strip() for station in include_stations):
                    if NAME in include_stations:
                        gnss_data_dict[NAME] = {}
                        time, arr = read_file(os.path.join(vadase_path, file))
                        gnss_data_dict[NAME]['times'] = time
                        gnss_data_dict[NAME]['vele'] = arr[:, 0]
                        gnss_data_dict[NAME]['veln'] = arr[:, 1]
                        gnss_data_dict[NAME]['velu'] = arr[:, 2]
                        SAMP_FREQ_DICT[NAME] = sampfreq(time)[1]
                else:
                    gnss_data_dict[NAME] = {}
                    time, arr = read_file(os.path.join(vadase_path, file))
                    gnss_data_dict[NAME]['times'] = time
                    gnss_data_dict[NAME]['vele'] = arr[:, 0]
                    gnss_data_dict[NAME]['veln'] = arr[:, 1]
                    gnss_data_dict[NAME]['velu'] = arr[:, 2]
                    SAMP_FREQ_DICT[NAME] = sampfreq(time)[1]
            if 'kin' in file:  # Process .kin files
                NAME = file.split('_')[-1]
                if include_stations and any(station.strip() for station in include_stations):
                    if NAME in include_stations:
                        data = kin2oyeah(kin_path=os.path.join(vadase_path, file))
                        time = data['t_gsow'].to_numpy()
                        arr = data[['e', 'n', 'u']].to_numpy()
                        gnss_data_dict[NAME] = {}
                        gnss_data_dict[NAME]['times'] = time
                        gnss_data_dict[NAME]['vele'] = arr[:, 0]
                        gnss_data_dict[NAME]['veln'] = arr[:, 1]
                        gnss_data_dict[NAME]['velu'] = arr[:, 2]
                        SAMP_FREQ_DICT[NAME] = sampfreq(time)[1]
                else:
                    data = kin2oyeah(kin_path=os.path.join(vadase_path, file))
                    time = data['t_gsow'].to_numpy()
                    arr = data[['e', 'n', 'u']].to_numpy()
                    gnss_data_dict[NAME] = {}
                    gnss_data_dict[NAME]['times'] = time
                    gnss_data_dict[NAME]['vele'] = arr[:, 0]
                    gnss_data_dict[NAME]['veln'] = arr[:, 1]
                    gnss_data_dict[NAME]['velu'] = arr[:, 2]
                    SAMP_FREQ_DICT[NAME] = sampfreq(time)[1]
            if file.endswith('.oy'):  # Process .oy files
                NAME = file.split('.')[0]  # [:4]
                if include_stations and any(station.strip() for station in include_stations):
                    if NAME in include_stations:
                        data = pd.read_csv(os.path.join(vadase_path, file))
                        time = data['t_gsow'].to_numpy()
                        arr = data[['e', 'n', 'u']].to_numpy()
                        gnss_data_dict[NAME] = {}
                        gnss_data_dict[NAME]['times'] = time
                        gnss_data_dict[NAME]['vele'] = arr[:, 0]
                        gnss_data_dict[NAME]['veln'] = arr[:, 1]
                        gnss_data_dict[NAME]['velu'] = arr[:, 2]
                        SAMP_FREQ_DICT[NAME] = sampfreq(time)[1]
                else:
                    data = pd.read_csv(os.path.join(vadase_path, file))
                    time = data['t_gsow'].to_numpy()
                    arr = data[['e', 'n', 'u']].to_numpy()
                    gnss_data_dict[NAME] = {}
                    gnss_data_dict[NAME]['times'] = time
                    gnss_data_dict[NAME]['vele'] = arr[:, 0]
                    gnss_data_dict[NAME]['veln'] = arr[:, 1]
                    gnss_data_dict[NAME]['velu'] = arr[:, 2]
                    SAMP_FREQ_DICT[NAME] = sampfreq(time)[1]
        except Exception as e:
            print(f'Error at reading: {file}')
            print('Continuing to next file...')
            continue

    if not gnss_data_dict:
        raise FileNotFoundError("Unsupported input file format \n Provide .varout or .kin or .oy")

    return gnss_data_dict, SAMP_FREQ_DICT

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Design a Butterworth bandpass filter.

    This function creates a digital bandpass filter using the Butterworth design.
    The filter allows frequencies within the specified range (`lowcut` to `highcut`) to pass while attenuating others.

    Parameters
    ----------
    lowcut : float
        The lower cutoff frequency of the bandpass filter (Hz).
    highcut : float
        The upper cutoff frequency of the bandpass filter (Hz).
    fs : float
        The sampling frequency of the signal (Hz).
    order : int, optional
        The order of the Butterworth filter (default is 5).

    Returns
    -------
    tuple
        A tuple `(b, a)` containing the filter coefficients.

    """

    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to a 1D signal.

    This function filters the input signal using a bandpass filter designed with the Butterworth method.
    It allows frequencies within the range `[lowcut, highcut]` to pass while attenuating others.

    Parameters
    ----------
    data : numpy.ndarray
        A 1D numpy array representing the input signal.
    lowcut : float
        The lower cutoff frequency of the bandpass filter (Hz).
    highcut : float
        The upper cutoff frequency of the bandpass filter (Hz).
    fs : float
        The sampling frequency of the signal (Hz).
    order : int, optional
        The order of the Butterworth filter (default is 5).

    Returns
    -------
    numpy.ndarray
        The filtered signal as a 1D numpy array.

    """

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y    = lfilter(b, a, data)

    return y


@capture_errors
def sampfreq(arr):
    """
    Calculate the sample frequency from an array of values.

    This function computes the differences between consecutive values in `arr`,
    finds the most frequently occurring difference (mode), and determines the
    corresponding sampling frequency.

    Parameters
    ----------
    arr : array-like
        A sequence of numerical values, which may contain NaN values.

    Returns
    -------
    tuple
        - mode_value (float): The most frequently occurring difference in `arr`, rounded to 2 decimal places.
        - smpf (int): The corresponding sampling frequency (Hz).

    Raises
    ------
    ValueError
        If the detected sampling frequency is below 1 Hz or the mode value
        is not within the supported set {0.01, 0.02, 0.05, 0.1, 0.2, 1}.

    """

    # Compute differences between consecutive values
    arr_diff = np.diff(arr)

    # Remove NaN values from the differences array
    arr_diff_without_nan = arr_diff[~np.isnan(arr_diff)]

    # Find unique values and their counts
    unique_values, counts = np.unique(arr_diff_without_nan, return_counts=True)

    # Identify the mode value (most frequently occurring difference)
    max_count_index = np.argmax(counts)
    mode_value = round(unique_values[counts == counts[max_count_index]][0], 2)

    def validate_sampling(mode_value):
        """
        Validate the calculated sampling frequency.

        Ensures the sampling frequency is at least 1 Hz and that the mode
        value belongs to a predefined set of supported values.

        Parameters
        ----------
        mode_value : float
            The detected mode value from the input array.

        Returns
        -------
        tuple
            - mode_value (float): The validated mode value.
            - smpf (int): The corresponding sampling frequency.

        Raises
        ------
        ValueError
            If the sampling frequency is below 1 Hz or the mode value is not supported.
        """
        smpf = int(1 / mode_value)

        if smpf < 1:
            raise ValueError(f"Error: Sampling frequency must be at least 1 Hz. Detected: {smpf} Hz")

        if mode_value in {0.01, 0.02, 0.05, 0.1, 0.2, 1}:
            return mode_value, smpf
        else:
            raise ValueError(f"Error: Unsupported mode value {mode_value}. Corresponding sampling frequency: {smpf} Hz")

    mode_value, smpf = validate_sampling(mode_value)

    return mode_value, smpf

def sigma_filter(arr, sigma=3):
    """
    Detect and filter outliers using the standard deviation method.

    This function identifies outliers in an array based on the specified number of
    standard deviations (`sigma`). Values that exceed `sigma * std` are replaced with NaN.

    Parameters
    ----------
    arr : numpy.ndarray or pandas.Series
        The dataset to be filtered, which may contain NaN values.
    sigma : int, optional
        The number of standard deviations used to detect outliers. Default is 3.

    Returns
    -------
    tuple
        - filtered_arr (numpy.ndarray): The filtered array with outliers replaced by NaN.
        - valid_indices (numpy.ndarray): Indices of values that are not considered outliers.

    """

    std = np.nanstd(arr)

    # Remove NaN values from the array
    arr_without_nan      = arr[~np.isnan(arr)]
    abs_arr_without_nan  = np.abs(arr_without_nan)

    # Create an array of NaN values with the same shape as the original array
    abs_arr              = np.full_like(arr, np.nan)
    abs_arr[~np.isnan(arr)] = abs_arr_without_nan

    # Identify values within the acceptable range
    condition            = abs_arr < sigma * std

    # Create a filtered array with NaN values replacing outliers
    filtered_arr         = np.full_like(arr, np.nan)
    filtered_arr[condition] = arr[condition]

    # Get indices of valid (non-outlier) values
    valid_indices        = np.squeeze(np.argwhere(condition))

    return filtered_arr, valid_indices

@capture_errors
def ecef2enu(dx, dy, dz, lat, lon, h):
    """
    Convert ECEF displacements (dx, dy, dz) to the local ENU (East, North, Up) coordinate system.

    This function transforms displacement values in the ECEF (Earth-Centered, Earth-Fixed) coordinate system
    to the ENU (East, North, Up) local coordinate system using the provided geodetic latitude and longitude.

    Parameters
    ----------
    dx : numpy.ndarray or float
        The displacement in the ECEF X direction (in meters).

    dy : numpy.ndarray or float
        The displacement in the ECEF Y direction (in meters).

    dz : numpy.ndarray or float
        The displacement in the ECEF Z direction (in meters).

    lat : numpy.ndarray or float
        The geodetic latitude (in degrees).

    lon : numpy.ndarray or float
        The geodetic longitude (in degrees).

    h : numpy.ndarray or float
        The height (in meters) – this parameter is not used in this conversion, but is included for interface compatibility.

    Returns
    -------
    dN : numpy.ndarray or float
        The displacement in the North direction (in meters).

    dE : numpy.ndarray or float
        The displacement in the East direction (in meters).

    dU : numpy.ndarray or float
        The displacement in the Up direction (in meters).

    """
    # Convert latitude and longitude from degrees to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    # Calculate displacement in the East direction
    dE = -np.sin(lon_rad) * dx + np.cos(lon_rad) * dy

    # Calculate displacement in the North direction
    dN = (-np.sin(lat_rad) * np.cos(lon_rad) * dx
          - np.sin(lat_rad) * np.sin(lon_rad) * dy
          + np.cos(lat_rad) * dz)

    # Calculate displacement in the Up direction
    dU = (np.cos(lat_rad) * np.cos(lon_rad) * dx
          + np.cos(lat_rad) * np.sin(lon_rad) * dy
          + np.sin(lat_rad) * dz)

    return dN, dE, dU


def kin2oyeah(kin_path, stde=None, stdn=None, stdu=None):
    """
    Parses a PRIDE-PPP kinematic file (kin_*) to extract and outputs following the structure of the GSW format (.oy)
    the position and displacement information in Earth-Centered Earth-Fixed (ECEF)
    and East-North-Up (ENU) coordinates, along with optional standard deviations.

    Parameters
    ----------
    kin_path : str
        Path to the kinematic file to be processed.
    stde : float, optional
        Standard deviation of the east coordinate, if available.
    stdn : float, optional
        Standard deviation of the north coordinate, if available.
    stdu : float, optional
        Standard deviation of the up coordinate, if available.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the following columns:
        - 'week': GPS week number
        - 't': Time of week in seconds, corrected for leap seconds
        - 'n': North displacement in meters
        - 'e': East displacement in meters
        - 'u': Up displacement in meters
        - 'std_e': Standard deviation of east coordinate
        - 'std_n': Standard deviation of north coordinate
        - 'std_u': Standard deviation of up coordinate
        - 'std_en': Placeholder value (9999.0, since no noise info is available)
        - 'std_eu': Placeholder value (9999.0)
        - 'std_nu': Placeholder value (9999.0)

    Raises
    ------
    ValueError
        If the input file does not contain the 'END OF HEADER' line, indicating a missing or incorrect header.
    """
    start_reading = False
    data = []
    with open(kin_path, 'r') as kin:
        lines = kin.readlines()

        for line in lines:

            if 'INTERVAL' in line or 'OBS INTERVAL' in line:
                dT = float(line.split()[0])
            if 'END OF HEADER' in line:
                start_reading = True
                continue
            if start_reading:
                # print(line)
                data.append(line)
    if not start_reading:
        raise ValueError(f"File '{kin_path}' does not contain 'END OF HEADER' line. \n"
                         f"Possibly header is missing")
    data_list = []
    columns = None
    for num, line in enumerate(data):
        if line.strip():
            if num == 0:
                columns = line.split()[1:9]
            else:
                data_list.append(line.split()[:8])
    df = pd.DataFrame(data_list, columns=columns).apply(lambda row: pd.to_numeric(row), axis=1)

    df['utc'] = df.apply(lambda row: mjd2datetime(mjd=row['Mjd'], seconds_of_day=row['Sod']), axis=1)

    df['week'], df['t'] = utc_to_gps(df['utc'].tolist())
    leap_sec = get_gps_utc_difference(df['utc'].loc[0].year)

    df['t_gsow'] = df['t'].astype(np.float64) - leap_sec
    first_latlon = df[['Latitude', 'Longitude', 'Height']].median(axis=0).to_numpy()
    df['dX'] = df['X'].diff().fillna(0.0)
    df['dY'] = df['Y'].diff().fillna(0.0)
    df['dZ'] = df['Z'].diff().fillna(0.0)
    dneu = df[['dX', 'dY', 'dZ']].apply(lambda row: ecef2enu(dx=row['dX'], dy=row['dY'], dz=row['dZ'],
                                                             lat=first_latlon[0], lon=first_latlon[1],
                                                             h=first_latlon[2]),
                                        axis=1)

    df[['n', 'e', 'u']] = dneu.apply(pd.Series)
    df[['std_e', 'std_n', 'std_u', 'std_en', 'std_eu',
        'std_nu']] = 9999.0  # no noise info in kin files, so I'm marking these columns with 9999
    if stde is not None:
        df['std_e'] = stde
    if stdn is not None:
        df['std_n'] = stdn
    if stdu is not None:
        df['std_u'] = stdu

    return df[
        ['week', 't_gsow', 'n', 'e', 'u', 'std_e', 'std_n', 'std_u', 'std_en', 'std_eu', 'std_nu']]  # .shift(-1).dropna()


@capture_errors
def varout2oyeah(varout_path, stde=None, stdn=None, stdu=None,date=None):
    """
    Parses a VarioPy variometric output file (.varout) to extract and outputs following the structure of the GSW format (.oy)
    east, north, and up displacement data, along with optional standard deviation values.

    Parameters
    ----------
    varout_path : str
        Path to the variometric output file to be processed.
    stde : float, optional
        Standard deviation of the east coordinate, if available.
    stdn : float, optional
        Standard deviation of the north coordinate, if available.
    stdu : float, optional
        Standard deviation of the up coordinate, if available.
    date: datetime
        UTC datetime object needed for computing gps week. If None then GPS week will not be calculated.
        None by default

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the following columns:
        - 't_gsow': Time index from the input file
        - 'gweek' GPS week number
        - 'e': East displacement in meters
        - 'n': North displacement in meters
        - 'u': Up displacement in meters
        - 'std_e': Standard deviation of east coordinate
        - 'std_n': Standard deviation of north coordinate
        - 'std_u': Standard deviation of up coordinate
        - 'std_en': Placeholder value (9999.0, since no noise info is available)
        - 'std_eu': Placeholder value (9999.0)
        - 'std_nu': Placeholder value (9999.0)
    """
    obs = pd.read_csv(varout_path, header=None)
    obs.set_index(obs.iloc[:, 1], inplace=True)
    df = obs.rename(columns={1: 't_gsow', 7: 'e', 8: 'n', 9: 'u'})

    df = df[['t_gsow', 'e', 'n', 'u']]
    if date is not None:
        week, _ = utc_to_gps(utc_times=[date])
        # week = datetime_to_gps_week(dt=date)
        df['gweek'] = week
    df[['std_e', 'std_n', 'std_u', 'std_en', 'std_eu',
        'std_nu']] = 9999.0  # no noise info in kin files, so I'm marking these columns with 9999
    if stde is not None:
        df['std_e'] = stde
    if stdn is not None:
        df['std_n'] = stdn
    if stdu is not None:
        df['std_u'] = stdu
    return df


def moving_average_with_regression_interpolation(arr_org, window_size):
    """
    Compute the moving average of a given array and perform linear regression
    interpolation to fill missing values at the beginning of the array.

    The function first computes the moving average using a sliding window of
    the specified size, then applies linear regression to interpolate the
    missing values at the beginning of the array based on the computed moving
    average.

    Parameters
    ----------
    arr_org : numpy.ndarray
        1D array of numeric values for which the moving average and regression
        interpolation will be computed.

    window_size : int
        The size of the sliding window used to compute the moving average. The
        window size must be a positive integer.

    Returns
    -------
    numpy.ndarray
        A 1D array of the same length as `arr_org`, with the moving average
        applied to valid entries, and interpolated values at the beginning
        using linear regression.

    Notes
    -----
    - The function uses the 'valid' mode of `np.convolve` to compute the moving
      average, which discards the boundary values that don't fully fit the
      window.
    - Linear regression interpolation is applied only to the missing values
      at the beginning of the array. The interpolation is based on the
      valid moving average values computed at the beginning of the series.
    - The function assumes that `window_size` is smaller than or equal to the
      length of `arr_org`.

    """
    # Compute the moving average with mode='valid'
    valid_avg = np.convolve(arr_org, np.ones(window_size) / window_size, mode='valid')

    # Initialize the result array with NaNs
    result_length = len(arr_org)
    result = np.full(result_length, np.nan)

    # Calculate the start index where valid_avg will be placed
    start_idx = (window_size - 1)  # Align with the start of the series
    result[start_idx:start_idx + len(valid_avg)] = valid_avg

    # Determine the gap length at the beginning
    gap_length = start_idx

    # Interpolate the missing values at the beginning using linear regression
    if gap_length > 0:
        # Use valid values equal to the window size for trend estimation
        num_valid_values_for_interp = min(window_size, len(valid_avg))
        X = np.arange(gap_length, gap_length + num_valid_values_for_interp).reshape(-1, 1)
        y = result[gap_length:gap_length + num_valid_values_for_interp]

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict values for the missing beginning points
        interp_X = np.arange(gap_length).reshape(-1, 1)
        result[:gap_length] = model.predict(interp_X)

    return result


def gaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Half-Gaussian convolution kernel.

    This function generates a 1-dimensional Gaussian kernel, centered at zero,
    which is truncated based on the specified radius. The kernel is then
    normalized so that its sum is equal to 1. It can be used for convolution
    operations in image processing or signal smoothing.

    Parameters
    ----------
    sigma : float
        The standard deviation of the Gaussian distribution. It controls the
        width of the Gaussian kernel. Larger values of `sigma` will result in
        a smoother kernel with a wider spread.

    radius : int
        The radius of the kernel, determining the extent of the kernel. The
        kernel will have a range of `[-radius, radius-1]` and will be truncated
        to this range, ensuring the kernel is centered at zero.

    Returns
    -------
    numpy.ndarray
        A 1D array representing the normalized Gaussian kernel. The length of
        the array is `2 * radius` and the sum of all values in the array is 1.

    Notes
    -----
    - The kernel is computed using the standard Gaussian formula:
      \(\phi(x) = \exp\left(-\frac{x^2}{2\sigma^2}\right)\), where `x` ranges
      from `-radius` to `radius-1`.
    - The kernel is then normalized so that the sum of all its values is equal to 1.
    - This kernel can be used for smoothing or blurring operations in 1D or as part
      of a larger 2D convolution.

    """
    sigma2 = sigma * sigma  # Computes the variance of the Gaussian distribution.
    x = np.arange(-radius,
                  radius)  # +1) # Generates a range of values centered around zero, covering the desired extent (determined by radius) of the kernel.
    # x = np.arange(0, radius*2)#+1)
    # x = np.arange(-radius, 0)
    phi_x = np.exp(
        -0.5 / sigma2 * x ** 2)  # Calculates the Gaussian function values for each point in x, resulting in the unnormalized kernel.
    # phi_x = 1/(np.sqrt(2*np.pi)sigma)*np.exp(-1(x)*2/(2*sigma*2))
    # phi_x = np.flip(phi_x)
    phi_x = phi_x / phi_x.sum()  # normalization of the Gaussian filter

    return phi_x


def custom_convolve1d(input, weights, window_size, mode='nearest'):
    """
    Custom 1D convolution function that aligns the result to the middle of the window.
    This mimics the behavior of a moving average with middle alignment.

    This function performs a 1D convolution operation, aligning the result to the middle
    of the convolution window. It also handles edge effects with customizable modes,
    including nearest neighbor replication and linear regression-based interpolation.

    Parameters
    ----------
    input : numpy.ndarray
        The 1D input array to which the convolution will be applied.

    weights : numpy.ndarray
        The 1D array of weights or kernel used for convolution. It must have a length
        equal to or smaller than the input array.

    window_size : int
        The size of the window used for convolution, which also determines the
        number of values to be handled for edge effects.

    mode : str, optional, default='nearest'
        The mode to handle the edge effects. Possible values are:
        - 'nearest': Replicates the first and last values of the convolution result
          to fill the edges.
        - 'valid': Only returns the valid part of the convolution (i.e., after
          applying the window). Missing values at the beginning are filled using
          linear regression interpolation.

    Returns
    -------
    numpy.ndarray
        A 1D array representing the convolved result with appropriate handling
        of edge effects according to the specified mode.

    Notes
    -----
    - The convolution is performed using `np.convolve` with mode='valid' to
      get the valid part of the result first.
    - In 'nearest' mode, the first and last values of the result are replicated
      to handle the edges of the array.
    - In 'valid' mode, a linear regression model is used to interpolate missing
      values at the beginning of the array, based on the valid part of the convolution.
    - The result is aligned with the middle of the window, making it similar to a
      moving average operation.

    """
    # Perform convolution using np.convolve with mode 'full' to get all elements
    full_conv = np.convolve(input, weights, mode='valid')

    # Calculate the start and end indices to center the result
    half_len = len(weights) // 2
    if window_size % 2 != 0:
        start_idx = half_len + 1
    else:
        start_idx = half_len
    end_idx = start_idx + len(input)

    # # Extract the part of the convolution that aligns with the input length
    conv_result = full_conv  # [start_idx:end_idx]

    # # Handle edge effects depending on the mode
    if mode == 'nearest':
        # Nearest mode: replicate the first and last values to the edges
        pad_left = np.full(half_len, conv_result[0])
        pad_right = np.full(half_len - 1, conv_result[-1])
        conv_result = np.concatenate([pad_left, pad_right, conv_result])

    elif mode == 'valid':
        # Initialize the result array with NaNs
        result_length = len(input)
        conv_result = np.full(result_length, np.nan)

        # Calculate the start index where valid_avg will be placed
        start_idx = (window_size - 1)  # Align with the start of the series
        conv_result[start_idx:start_idx + len(full_conv)] = full_conv

        # Determine the gap length at the beginning
        gap_length = start_idx

        # Interpolate the missing values at the beginning using linear regression
        if gap_length > 0:
            # Use valid values equal to the window size for trend estimation
            num_valid_values_for_interp = min(window_size, len(full_conv))
            X = np.arange(gap_length, gap_length + num_valid_values_for_interp).reshape(-1, 1)
            y = conv_result[gap_length:gap_length + num_valid_values_for_interp]

            # Fit a linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predict values for the missing beginning points
            interp_X = np.arange(gap_length).reshape(-1, 1)
            conv_result[:gap_length] = model.predict(interp_X)

    return conv_result


def gaussian_filter1d_modified(input, sigma, window_size, mode="nearest"):
    """
    Apply a modified 1D Gaussian filter to the input array with adjustable window size.

    This function generates a 1D Gaussian kernel based on the given standard deviation
    (sigma) and window size. It applies the generated kernel to the input array using
    the custom convolution function, allowing various edge-handling modes. If the
    window size is even, an additional central value is inserted to make it odd-length.

    Parameters
    ----------
    input : numpy.ndarray
        The 1D array to which the Gaussian filter will be applied.

    sigma : float
        The standard deviation of the Gaussian distribution. It controls the width
        of the Gaussian kernel. Larger values of `sigma` result in a smoother filter.

    window_size : int
        The size of the window used for the filter. The kernel size is derived
        from this value, and if it's even, an extra value is inserted to make it odd.

    mode : str, optional, default='nearest'
        The mode to handle edge effects during convolution. Possible values are:
        - 'nearest': Replicates the first and last values of the convolution result
          to handle the edges.
        - 'valid': Only returns the valid part of the convolution result, with
          missing values at the beginning filled using linear regression interpolation.

    Returns
    -------
    numpy.ndarray
        The filtered 1D array, where the Gaussian kernel has been convolved with the input
        array, and edge effects have been handled according to the specified mode.

    Notes
    -----
    - The kernel is generated using a Gaussian distribution with a standard deviation
      (`sigma`), and the window size controls the extent of the filter.
    - The kernel is adjusted if the `window_size` is even, adding an additional central
      value to make the kernel have an odd length.
    - The function uses `custom_convolve1d` for the actual convolution, which provides
      various edge-handling options.

    """
    sd = float(sigma)
    lw = window_size // 2  # Half of the window size

    # Generate the Gaussian kernel
    weights = gaussian_kernel1d(sd, lw)

    # Handle odd window sizes
    if window_size % 2 != 0:
        # Add an additional central value to make the kernel have an odd length
        insert_weight = (weights[-1] + weights[-2]) / 2
        weights = np.append(weights, insert_weight)

    return custom_convolve1d(input, weights, window_size=window_size, mode=mode)

def check_bandpass_conditions(fs, a, b):
    """
    Validates UI parameters provided for bandpass filtering.

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz).
    a  : float
        Lowcut frequency (Hz).
    b  : float
        Subtraction parameter used to calculate the highcut frequency.

    Returns
    -------
    tuple of float
        If parameters are valid, returns a, b; otherwise returns default values (0.1, 0.1).
    """
    nyquist   = fs / 2
    lowcut    = a
    highcut   = nyquist - b

    if lowcut <= 0:
        print("Lowcut must be greater than 0")
        return 0.1, 0.1
    if highcut >= nyquist:
        print("Highcut must be lower than Nyquist frequency")
        return 0.1, 0.1
    if lowcut >= highcut:
        print("Lowcut must be lower than highcut")
        return 0.1, 0.1
    return a, b