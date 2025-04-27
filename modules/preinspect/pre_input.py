import sys
from datetime import datetime
from ..time import utc_to_gps, get_gps_utc_difference, get_leap_seconds_table2
from ..prep import sigma_filter, sampfreq
import os
from astropy.utils.iers import IERS_Auto
from astropy.time import Time
import re
import numpy as np
import pandas as pd


def read_file(station_path):
    """
    Reads a station RNX `.varout` file and extracts time and displacement data.

    This function reads the `.varout` file and returns:
    - Seconds of Week (SOW)
    - Displacement values (North, East, Up)
    - Uncertainty (sigma) values

    Parameters
    ----------
    station_path : str
        Path to the `.varout` file containing GNSS displacement data.

    Returns
    -------
    sow : numpy.ndarray
        1D array of seconds of the week (time index).
    displacements : numpy.ndarray
        2D array of shape (N, 3) containing displacements in North, East, and Up directions.
    sigma : numpy.ndarray
        2D array of uncertainty (sigma) values for the displacement measurements.
    """
    obs = pd.read_csv(station_path, header=None)
    obs.set_index(obs.iloc[:, 1], inplace=True)

    sow          = np.array(obs.index.tolist())
    displacements = obs.iloc[:, [7, 8, 9]].to_numpy()
    sigma        = obs.iloc[:, [-2]].to_numpy()

    # If variopy provides additional uncertainties and correlations, include them:
    # extra_sigma = obs.iloc[:, [19, 20, 21, 22, 23, 24]].to_numpy()

    return sow, displacements, sigma



def read_sys_params():
    """
    Reads command-line parameters and returns them as a dictionary.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        A dictionary containing the parsed parameters from the command line input.
        The dictionary keys and their descriptions:
            - CALCULATION_MODE: '1D', '2D', '3D'
            - MODE: 'MAD', 'SLOPE', 'RIGID'
            - wcase: 'sigmoid', 'trapez', 'method2' or 'None'
            - vadase_path: Path to the VADASE directory
            - DATE_STR: Date in the format 'YYYY-MM-DD'
            - GPS_DATE: GPS date corresponding to DATE_STR
            - GPS_WEEK: GPS week number
            - COMPUTE_NOISE: Boolean flag to compute noise
            - shaking_length: Length of shaking
            - window_size: Size of the window for analysis
            - INCLUDE_STATIONS: List of station names or None
            - result_csv: Path to save CSV results
            - result_figures: Path to save figure results
            - storage_option: Option for data storage
            - PROJECT_ID: Project ID
            - eq_time: Earthquake time
            - swa: S-wave arrivals
            - zswa: Zoomed S-wave arrivals
            - time_cutoff: Time cutoff for analysis
            - bandpass_cutoffs: Bandpass filter cutoffs
            - lowcut: Low cutoff frequency
            - substr: Substring for filtering data
            - time: Time scale for processing
            - leap_sec: Number of leap seconds
            - domain: Processing domain

    Raises
    ------
    ValueError
        If there is an issue with the command-line arguments or conversion of parameters.
    """
    try:
        # Read parameters from command-line arguments
        CALCULATION_MODE   = sys.argv[1]         # '1D', '2D', '3D'
        MODE               = sys.argv[2]         # 'MAD', 'SLOPE', 'RIGID'
        wcase              = sys.argv[3]         # 'sigmoid', 'trapez', 'method2' or 'None'
        vadase_path        = sys.argv[4]         # Path to VADASE directory
        DATE_STR           = sys.argv[5]         # Date in format 'YYYY-MM-DD'
        
        # Date processing
        DATE               = datetime.strptime(DATE_STR, '%Y-%m-%d')  # Convert to datetime object
        YEAR               = DATE.year

        # Get leap seconds and GPS time
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
        GPS_WEEK, GPS_DATE = utc_to_gps([DATE])  # Convert UTC to GPS date and week
        # Additional system parameters
        COMPUTE_NOISE      = sys.argv[6].lower() == 'true'  # Convert to boolean
        shaking_length     = int(sys.argv[7])                # Length of shaking
        window_size        = int(sys.argv[8])                # Size of analysis window
        INCLUDE_STATIONS   = sys.argv[9]

        # Process station names input
        if INCLUDE_STATIONS.lower() == 'false':
            INCLUDE_STATIONS = False
        elif INCLUDE_STATIONS.lower() == 'none' or INCLUDE_STATIONS.strip() == '':
            INCLUDE_STATIONS = None
        else:
            INCLUDE_STATIONS = re.split(r'[,;\s]+', INCLUDE_STATIONS.strip())
            INCLUDE_STATIONS = [station for station in INCLUDE_STATIONS if station]  # Clean up empty strings
        
        # File paths for results
        result_csv         = sys.argv[10]  # Path for saving CSV results
        result_figures     = sys.argv[11]  # Path for saving figure results
        storage_option     = sys.argv[12]  # Data storage option
        project_id         = sys.argv[13]  # Project ID
        eq_time            = sys.argv[14].strip()  # Earthquake time

        eq_time = sys.argv[14].strip()
        if eq_time != 'None':
            eq_datetime = f'{DATE_STR} {eq_time}'
        else:
            eq_datetime = eq_time
        SWAVE_ARRIVALS     = sys.argv[15]  # S-wave arrivals
        ZOOMED_SWAVE_ARRIVALS = sys.argv[16]  # Zoomed S-wave arrivals
        time_cutoff        = sys.argv[17]  # Time cutoff for analysis
        bandpass_cutoffs   = sys.argv[18]  # Bandpass filter cutoffs
        lowcut             = sys.argv[19]  # Low cutoff frequency
        substr             = sys.argv[20]  # Substring for filtering data
        time_scale         = sys.argv[21]  # Time scale for processing
        domain             = sys.argv[-1]  # Processing domain
        # Create required directories if they don't exist
        rcsv    = os.path.join(result_csv, f'{project_id}', CALCULATION_MODE, 'reports')
        rfig    = os.path.join(result_csv, f'{project_id}', CALCULATION_MODE, 'figures')
        log_dir = os.path.join(result_csv, f'{project_id}', CALCULATION_MODE, 'logs')
        
        os.makedirs(rcsv, exist_ok=True)
        os.makedirs(rfig, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Return parameters as a dictionary
        return {
            "CALCULATION_MODE"    : CALCULATION_MODE,
            "MODE"                : MODE,
            "wcase"               : wcase,
            "vadase_path"         : vadase_path,
            "DATE_STR"            : DATE_STR,
            "GPS_DATE"            : GPS_DATE,
            "GPS_WEEK"            : GPS_WEEK,
            "COMPUTE_NOISE"       : COMPUTE_NOISE,
            "shaking_length"      : shaking_length,
            "window_size"         : window_size,
            "INCLUDE_STATIONS"    : INCLUDE_STATIONS,
            "result_csv"          : rcsv,
            "result_figures"      : rfig,
            "storage_option"      : storage_option,
            'PROJECT_ID'          : project_id,
            'eq_time'             : eq_datetime,
            'swa'                 : SWAVE_ARRIVALS,
            'zswa'                : ZOOMED_SWAVE_ARRIVALS,
            'time_cutoff'         : time_cutoff,
            'bandpass_cutoffs'    : bandpass_cutoffs,
            'lowcut'              : lowcut,
            'substr'              : substr,
            'logdir'              : log_dir,
            'time'                : time_scale,
            'leap_sec'            : leap_sec,
            'domain'              : domain
        }

    except IndexError:
        raise ValueError("Incorrect number of arguments. Please check if all required parameters are provided.")
    except ValueError as e:
        raise ValueError(f"Error while converting parameters: {e}")