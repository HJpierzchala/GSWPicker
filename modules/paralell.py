import traceback
from .classes import MADDetector, SLOPEDetector, WTESTDetector
import numpy as np
from .detect import find_swave_indices
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from datetime import datetime
from pathlib import Path
import os

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

def process_batch(batch, samp_fre_dict, d):
    """
    Processes a batch of data based on the specified mode in the input dictionary.

    This function processes the data based on different modes ('MAD', 'SLOPE', or 'W-TEST') specified
    in the input dictionary `d`. For each mode, it creates an appropriate detector object, performs
    the detection process, and stores the results. The results are returned as a list of tuples containing
    the station name and the detection pick.

    Parameters
    ----------
    batch : iterable of tuples
        A list of tuples where each tuple contains the station name and its corresponding data dictionary.

    samp_fre_dict : dict
        A dictionary containing the sampling frequency for each station.

    d : dict
        A dictionary containing configuration options, including the mode of operation ('MAD', 'SLOPE', or 'W-TEST').

    Returns
    -------
    results : list of tuples
        A list of tuples where each tuple contains the station name and the detection pick for that station.

    Raises
    ------
    Exception
        If an error occurs during the detection process for any station, the exception is caught and logged.

    """
    results = []
    if d['MODE'] == 'MAD':
        for k, v in batch:
            try:
                station_obj = MADDetector(
                    sta_name=k,
                    data_dict=v,
                    input_data=d,
                    sampling_freq=samp_fre_dict[k]
                )
                pick = station_obj.detect()
                results.append((k, pick))
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing station {k}: {e}")
        return results
    elif d['MODE'] == 'SLOPE':
        for k, v in batch:
            try:
                station_obj = SLOPEDetector(
                    sta_name=k,
                    data_dict=v,
                    sampling_freq=samp_fre_dict[k],
                    input_data=d
                )
                pick = station_obj.detect()
                results.append((k, pick))
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing station {k}: {e}")
        return results
    elif d['MODE'] == 'W-TEST':
        for k, v in batch:
            try:
                station_obj = WTESTDetector(
                    sta_name=k,
                    data_dict=v,
                    sampling_freq=samp_fre_dict[k],
                    input_data=d
                )
                pick = station_obj.detect()
                results.append((k, pick))
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing station {k}: {e}")
        return results

