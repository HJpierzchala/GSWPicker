import numpy as np
import json
import os
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from concurrent.futures import ProcessPoolExecutor, as_completed
from .noise_plot import plot_gnssvel_autocorrelation, plot_gnssvel_periodogram
from .vel_plot import plot_gnssvel_data
from .spect_plot import plot_gnssvel_spectrum
import shutil
from time import sleep
import traceback
import linecache

def convert_for_json(obj):
    """
    Recursively converts NumPy arrays (and other non-serializable types)
    to standard Python types that can be written to JSON.

    Parameters:
    obj : any
        The object to be converted, which can be a NumPy array, dictionary,
        list, or any other type.

    Returns:
    obj : any
        The converted object, which will be a JSON serializable version
        of the input (e.g., NumPy arrays converted to lists).
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    else:
        return obj


def save_data_to_json(data, filename):
    """
    Saves data (a list or dictionary) to a JSON file.

    Parameters:
    data : list or dict
        The data to be saved, which can be a list or dictionary.

    filename : str
        The name of the file where the data will be saved.
    """
    serializable_data = convert_for_json(data)
    with open(filename, 'w') as f:
        json.dump(serializable_data, f, indent=4)


def process_file_time(f, vadase_path, ev_date, result_figures, mode, time_scale, eq_time, log_dir):
    """
    Processes a single file for the 'time' domain.

    The function determines the station name, constructs the full file path,
    and calls the `plot_gnssvel_data` function to generate a plot.

    Parameters
    ----------
    f : str
        The filename to be processed.

    vadase_path : str
        The base path where the file is located.

    ev_date : datetime.datetime
        The event date to be passed to the plotting function.

    result_figures : str
        The path where the result figures will be saved.

    mode : str
        The mode used for the plotting function.

    Returns
    -------
    None
        The function doesn't return any value. It directly generates the plot and saves the output.

    """
    if 'kin' in f:
        NAME = f.split('_')[-1]
    else:
        NAME = f.split('_')[0][:4]
    file_path = os.path.join(vadase_path, f)

    plot_gnssvel_data(
        file_path=file_path,
        event_date=ev_date,
        output_path=result_figures,
        time_format=time_scale,
        sta_name=NAME,
        mode=mode,
        eq_time=eq_time,
        log_dir=log_dir
    )



def process_file_noise(f, vadase_path, ev_date, result_figures, log_path):
    """
    Processes a single file for the 'noise' domain.

    The function determines the station name, constructs the full file path,
    and calls the periodogram and autocorelation plotting functions.

    Parameters
    ----------
    f : str
        The filename to be processed.

    vadase_path : str
        The base path where the file is located.

    ev_date : datetime.datetime
        The event date to be passed to the plotting function.

    result_figures : str
        The path where the result figures will be saved.

    log_path : str
        The mode used for the plotting function.

    Returns
    -------
    None
        The function doesn't return any value. It directly generates the frequency spectrum plot and saves the output.

    """
    if 'kin' in f:
        NAME = f.split('_')[-1]
    else:
        NAME = f.split('_')[0][:4]
    file_path = os.path.join(vadase_path, f)
    plot_gnssvel_periodogram(
        file_path=file_path,
        event_date=ev_date,
        output_path=result_figures,
        sta_name=NAME
    )
    result_dict = plot_gnssvel_autocorrelation(
        file_path=file_path,
        event_date=ev_date,
        sta_name=NAME,
        output_path=result_figures,
        log_path=log_path    )
    return result_dict

def process_file_freq(f, vadase_path, ev_date, result_figures, mode, time_scale, eq_time=None):
    """
    Processes a single file for the 'freq' domain.

    The function determines the station name, constructs the full file path,
    and calls the `plot_gnssvel_spectrum` function to generate the frequency spectrum plot.

    Parameters
    ----------
    f : str
        The filename to be processed.

    vadase_path : str
        The base path where the file is located.

    ev_date : datetime.datetime
        The event date to be passed to the plotting function.

    result_figures : str
        The path where the result figures will be saved.

    mode : str
        The mode used for the plotting function.

    Returns
    -------
    None
        The function doesn't return any value. It directly generates the frequency spectrum plot and saves the output.

    """
    if 'kin' in f:
        NAME = f.split('_')[-1]
    else:
        NAME = f.split('_')[0][:4]
    file_path = os.path.join(vadase_path, f)
    plot_gnssvel_spectrum(
        file_path=file_path,
        event_date=ev_date,
        output_path=result_figures,
        time_format=time_scale,
        sta_name=NAME,
        mode=mode,
        eq_time=eq_time

    )


def main(d, console):
    """
    Main function to process files based on the specified domain and options.

    The function performs file processing in one of three domains: 'time', 'freq', or 'noise'.
    It reads the configuration from the input dictionary `d`, processes the files in batches,
    and uses a progress bar to show the status of the processing.

    Parameters
    ----------
    d : dict
        A dictionary containing the configuration for the processing, including:
        - `vadase_path`: The path to the folder containing the files.
        - `DATE_STR`: The event date as a string.
        - `eq_time`: The event time (optional).
        - `INCLUDE_STATIONS`: A list of stations to include (optional).
        - `domain`: The domain for processing ('time', 'freq', or 'noise').
        - `result_figures`: The path where the results will be saved.
        - `CALCULATION_MODE`: The mode used for the calculation.
        - `logdir`: Directory for logging (only for 'noise' domain).

    console : rich.console.Console
        A `Console` object from the `rich` library to display the progress bar and error messages.

    Returns
    -------
    list or None
        - For the 'noise' domain, a list of results is returned.
        - For the 'time' and 'freq' domains, no results are returned, but the files are processed.

    """
    vadase_path = d['vadase_path']
    event_date_str = d['DATE_STR'] # YYYY MM DD
    event_time_str = '00:00:00'
    ev_date = datetime.strptime(f"{event_date_str} {event_time_str}", "%Y-%m-%d %H:%M:%S")
    if d['eq_time'] != 'None':
        # If eq_time is provided, use the format with microseconds
        eq_time = datetime.strptime(d['eq_time'], "%Y-%m-%d %H:%M:%S.%f")
    else:
        eq_time=None
    assert  isinstance(ev_date, datetime)
    # Get the list of files
    files = [f for f in os.listdir(vadase_path) if f.endswith(('varout','oy','kin')) or 'kin' in f]
    include_stations = d['INCLUDE_STATIONS']
    if include_stations and any(station.strip() for station in include_stations):
        files = [f for f in files if f.split('_')[0][:4] in include_stations]

    # Determine the batch size based on the domain
    if d['domain'] in ['time', 'freq']:
        batch_size = 4
    elif d['domain'] == 'noise':
        batch_size = 2
    else:
        batch_size = 4
    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

    # Processing depending on the domain with progress bar
    if d['domain'] == 'time':
        with Progress("[progress.description]{task.description}",
                      BarColumn(),
                      "[progress.percentage]{task.percentage:>3.0f}%",
                      TimeRemainingColumn(),
                      console=console) as progress:
            task = progress.add_task("[cyan]Processing files (time domain)...", total=len(batches))
            with ProcessPoolExecutor() as executor:
                for batch in batches:
                    futures = {
                        executor.submit(process_file_time, f, vadase_path, ev_date, d['result_figures'],
                                        d['CALCULATION_MODE'],d['time'],eq_time, d['logdir']): f
                        for f in batch
                    }
                    for future in as_completed(futures):
                        try:
                            _ = future.result()
                        except Exception as exc:
                            tb = exc.__traceback__
                            while tb.tb_next:
                                tb = tb.tb_next

                            filename = tb.tb_frame.f_code.co_filename
                            lineno = tb.tb_lineno
                            code_line = linecache.getline(filename, lineno).strip()

                            console.print(
                                f"[red]Error in file  {filename}, line {lineno}:[/red]\n"
                                f"[red]    -> {code_line}[/red]\n"
                                f"[red]Exception: {exc}[/red]"
                            )
                    progress.update(task, advance=1)

    elif d['domain'] == 'freq':
        with Progress("[progress.description]{task.description}",
                      BarColumn(),
                      "[progress.percentage]{task.percentage:>3.0f}%",
                      TimeRemainingColumn(),
                      console=console) as progress:
            task = progress.add_task("[cyan]Processing files (freq domain)...", total=len(batches))
            with ProcessPoolExecutor() as executor:
                for batch in batches:
                    futures = {
                        executor.submit(process_file_freq, f, vadase_path, ev_date, d['result_figures'],
                                        d['CALCULATION_MODE'], d['time'], eq_time): f
                        for f in batch
                    }
                    for future in as_completed(futures):
                        try:
                            _ = future.result()
                        except Exception as exc:
                            console.print(f"[red]Error during processing:  {futures[future]}: {exc}[/red]")
                    progress.update(task, advance=1)

    elif d['domain'] == 'noise':
        noise_results = []
        with Progress("[progress.description]{task.description}",
                      BarColumn(),
                      "[progress.percentage]{task.percentage:>3.0f}%",
                      TimeRemainingColumn(),
                      console=console) as progress:
            task = progress.add_task("[cyan]Processing files (noise domain)...", total=len(batches))
            with ProcessPoolExecutor() as executor:
                for batch in batches:
                    futures = {
                        executor.submit(process_file_noise, f, vadase_path, ev_date, d['result_figures'],
                                        d['logdir']): f
                        for f in batch
                    }
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            noise_results.append(result)
                        except Exception as exc:
                            console.print(f"[red]Error during processing:  {futures[future]}: {exc}[/red]")
                    progress.update(task, advance=1)
        return noise_results


def clear_console():
    """
    Clears the console screen.

    This function clears the console screen based on the operating system:
    - For Windows (nt), it uses the `cls` command.
    - For other operating systems (e.g., Linux, macOS), it uses the `clear` command.

    Returns
    -------
    None
        This function does not return any value. It directly clears the console screen.

    """
    os.system('cls' if os.name == 'nt' else 'clear')


def display_intro(console):
    """
    Displays an introduction and initialization animation for the program.

    This function prints a title and subtitle centered in the console, along with a separator line.
    It also displays the current date and time when the program is run, followed by an animated progress bar
    indicating the initialization process.

    Parameters
    ----------
    console : rich.console.Console
        A `Console` object from the `rich` library used to print formatted text and display progress bars.

    Returns
    -------
    None
        This function does not return any value. It only prints information to the console and shows a progress bar.

    Notes
    -----
    - The function retrieves the terminal width using `shutil.get_terminal_size()` to ensure that the separator line spans the entire width of the terminal.
    - It uses the `rich.console.Console` for colored and styled text output, including a progress bar.
    - The progress bar animates for 100 steps, simulating the initialization process.
    - The current date and time are displayed in the format "MM/DD/YYYY, HH:MM:SS".
    """
    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns
    # Text to be centered
    title = "GSWPicker v1.0.0"
    subtitle = "Preprocessing mode"
    # Determine the number of '=' characters for the top and bottom line
    separator = "=" * terminal_width

    # Display animation and text
    console.print(f"[bold cyan]{separator}[/]")
    console.print(f"[bold cyan]{title}[/]", justify="center")
    console.print(f"[bold cyan]{subtitle}[/]", justify="center")
    console.print(f"[bold cyan]{separator}[/]")
    console.print(f"[green bold]Program run {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} \n")

    with Progress("[progress.description]{task.description}",
                  BarColumn(),
                  "[progress.percentage]{task.percentage:>3.0f}%",
                  TimeRemainingColumn(),
                  console=console) as progress:
        task = progress.add_task("[cyan]Initializing program...", total=100)
        for _ in range(100):
            sleep(0.02)
            progress.update(task, advance=1)
    console.print("\n[green bold]Initialization complete! Ready to process data.\n")


def display_closing(console):
    """
    Displays a closing message and contact information in the console.

    This function prints a thank-you message, followed by contact information for feedback or questions,
    and a separator line to format the closing message in the console.

    Parameters
    ----------
    console : rich.console.Console
        A `Console` object from the `rich` library used to print formatted text.

    Returns
    -------
    None
        This function does not return any value. It only prints information to the console.

    """
    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns
    # Text to be centered
    thank_you = "Thank you for using GSWPicker Software!"
    message = "Questions or feedback, feel free to reach out to: \n hpierzchala@cbk.waw.pl, a.m.lapadat@tudelft.nl."
    # Determine the number of '=' characters for the top and bottom line
    separator = "=" * terminal_width

    # Display closing message and text
    console.print(f"[bold cyan]{separator}[/]")
    console.print(f"[green bold]{thank_you}[/]", justify="center")
    console.print(f"[cyan]{message}[/]", justify="center")
    console.print(f"[bold cyan]{separator}[/]\n")

