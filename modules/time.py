from astropy.utils.iers import IERS_Auto
import numpy as np
from astropy.time import Time
import pandas as pd
import os
import requests
import traceback
from .config import CFG
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Union, Sequence, Tuple

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

@capture_errors
def datetime_to_gps_week(dt,CFG=CFG):
    """
    Convert a datetime object to the corresponding GPS week number.

    This function calculates the GPS week number based on the given `datetime` object.
    The GPS week starts on January 6, 1980, which is the GPS epoch. The difference in
    days between the input date and the GPS epoch is used to determine the week number.

    Parameters
    ----------
    dt : datetime
        The `datetime` object representing the date to convert to GPS week number.
    CFG: dict
        configuration dict
    Returns
    -------
    int
        The GPS week number corresponding to the input date.


    """

    # GPS epoch start date
    iso = CFG['TIME']['GPST_start']
    gps_epoch = datetime.fromisoformat(iso)

    # Calculate the difference in days between the given date and the GPS epoch
    delta = dt - gps_epoch

    # Calculate the GPS week number
    gps_week = delta.days // 7

    return gps_week


@capture_errors
def get_gps_utc_difference(year):
    """
    Calculate the number of leap seconds between the GPS start date and the given year.

    This function loads the IERS table, calculates the difference in time between
    the GPS system and UTC, and counts the leap seconds that have occurred between
    the GPS start date (January 6, 1980) and the end of the given year.

    Parameters
    ----------
    year : int
        The year until which leap seconds are counted. The function calculates leap
        seconds from the GPS epoch (January 6, 1980) to the end of the specified year.

    Returns
    -------
    int
        The total number of leap seconds between the GPS epoch and the end of the given year.

    """

    # Load the IERS table (International Earth Rotation and Reference Systems Service)
    iers_table = IERS_Auto.open()

    # Extract the MJD (Modified Julian Date) and UT1-UTC columns
    mjd = iers_table['MJD']
    ut1_utc = iers_table['UT1_UTC'].to_value()  # Remove unit to get a NumPy array

    # Find the indices of the jumps (where UT1_UTC changes by more than a typical step)
    leap_indices = np.where(np.diff(ut1_utc) > 0.5)[0]
    leap_mjd = mjd[leap_indices + 1]  # Indexes of jumps (+1 because diff shifts)

    # Convert MJD to datetime
    leap_dates = Time(leap_mjd, format='mjd').datetime

    # GPS start date (January 6, 1980)
    gps_start_date = Time("1980-01-06").datetime

    # Count the number of leap seconds from the GPS start date to the end of the given year
    leap_seconds_count = sum(1 for date in leap_dates if gps_start_date <= date <= Time(f"{year}-12-31").datetime)

    return leap_seconds_count


@capture_errors
def gps_to_utc(
    gps_week: Union[np.ndarray, list, int],
    gps_sow_array: Union[np.ndarray, list[float]],
    cfg=CFG,
    leap_sec: Optional[int] = None,
) -> np.ndarray:
    """
    Convert GPS week and seconds-of-week array to UTC datetimes.

    Logic:
      1) Try get_gps_utc_difference()
      2) Fallback to get_leap_seconds_table2()
      3) Finally fallback to Astropy Time

    Parameters
    ----------
    gps_week : int
        GPS week number.
    gps_sow_array : array-like of float
        Seconds of week.
    cfg : dict
        Must contain cfg['TIME']['GPST_start'] ISO string (e.g. "1980-01-06T00:00:00").
    leap_sec : int, optional
        If provided, używane bez obliczeń.

    Returns
    -------
    numpy.ndarray
        Array of timezone-aware UTC datetime objects.
    """
    if isinstance(gps_week, list):
        gps_week = np.array(gps_week)
    if isinstance(gps_sow_array, list):
        gps_sow_array=np.array(gps_sow_array)
    # 1. przygotowanie epochy jako UTC-aware
    epoch_iso = cfg['TIME']['GPST_start']
    gps_epoch = (
        datetime.fromisoformat(epoch_iso)

    )

    sow = np.asarray(gps_sow_array, dtype=float)
    gps_seconds = gps_week * 7 * 24 * 3600 + sow
    utc_base = np.array([
        gps_epoch + timedelta(seconds=sec)
        for sec in gps_seconds
    ], dtype=object)

    if leap_sec is not None:
        return utc_base - np.array([
            timedelta(seconds=leap_sec)
            for _ in utc_base
        ], dtype=object)

    try:
        year = utc_base[0].year
        # IERS Table approach
        leap = get_gps_utc_difference(year=year)
        return utc_base - np.array([
            timedelta(seconds=leap)
            for _ in utc_base
        ], dtype=object)
    except (ValueError, KeyError, Exception):
        pass

    try:
        # Web scraping approach / local table approach
        table = get_leap_seconds_table2()
        year = utc_base[0].year
        recent = table[table['date'].dt.year <= year]
        if recent.empty:
            raise KeyError("Brak wpisu w tabeli leap seconds")
        leap = int(recent.sort_values('date').iloc[-1]['leap'])
        return utc_base - np.array([
            timedelta(seconds=leap)
            for _ in utc_base
        ], dtype=object)
    except (KeyError, AttributeError, ValueError, Exception):
        pass
    #Astropy approach
    t = Time(gps_seconds, format='gps', scale='utc')
    return t.utc.to_datetime()




@capture_errors
def utc_to_gps(
    utc_times: Union[Sequence[datetime], np.ndarray],
    cfg=CFG,
    leap_sec: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert UTC datetime(s) to GPS week number and seconds-of-week array.

    Logic:
      1) Compute for each dt: t_sec = (dt - gps_epoch).total_seconds()
      2) Add leap seconds (either provided or via get_gps_utc_difference / table)
      3) week = floor(gps_sec / 604800), sow = gps_sec % 604800
      4) Fallback: Astropy Time

    Parameters
    ----------
    utc_times : sequence of datetime.datetime
        Input UTC datetimes. Naive are assumed UTC; aware dt.tzinfo kept.
    cfg : dict
        Must contain cfg['TIME']['GPST_start'] ISO string (e.g. "1980-01-06T00:00:00").
    leap_sec : int, optional
        If provided, use this many leap seconds for all times.

    Returns
    -------
    weeks : numpy.ndarray of int
        GPS week numbers.
    sow  : numpy.ndarray of float
        Seconds-of-week corresponding to each input datetime.
    """
    epoch_iso = cfg['TIME']['GPST_start']
    gps_epoch = datetime.fromisoformat(epoch_iso)
    if not isinstance(utc_times, list):
        dt_list = [utc_times]
    else:
        dt_list = utc_times
    sec_since = np.array([
        (dt- gps_epoch).total_seconds()
        for dt in dt_list
    ], dtype=float)

    if leap_sec is not None:
        gps_seconds = sec_since + leap_sec
    else:
        gps_seconds = np.empty_like(sec_since)
        for i, dt in enumerate(dt_list):
            try:
                ls = get_gps_utc_difference(year=dt.year)
            except Exception:
                try:
                    table = get_leap_seconds_table2()
                    recent = table[table['date'].dt.year <= dt.year]
                    ls = int(recent.sort_values('date').iloc[-1]['leap'])
                except Exception:
                    t = Time(dt, scale='utc')
                    gps_seconds[i] = t.gps
                    continue
            gps_seconds[i] = sec_since[i] + ls

    WEEK_SEC = 7 * 24 * 3600
    weeks = (gps_seconds // WEEK_SEC).astype(int)
    sow   = gps_seconds - weeks * WEEK_SEC

    return weeks, sow


@capture_errors
def mjd2datetime(mjd, seconds_of_day, CFG=CFG):
    """
    Convert a Modified Julian Date (MJD) and the number of seconds since the start of the day to a datetime object.

    This function takes the MJD (number of days since November 17, 1858) and the number of seconds from midnight
    (00:00:00) to compute the corresponding datetime.

    Parameters
    ----------
    mjd : int or float
        The Modified Julian Date (MJD), which is the number of days since November 17, 1858.

    seconds_of_day : int or float
        The number of seconds elapsed since the start of the day (00:00:00).
    CFG: dict
        configuration dict
    Returns
    -------
    datetime
        A datetime object corresponding to the given MJD and seconds of the day.

    Examples
    --------
    # >>> dt = mjd2datetime(59580, 45000)
    # >>> print(dt)
    2022-01-01 12:30:00
    """
    # Base date for MJD: November 17, 1858
    iso = CFG['TIME']['MJD_base_date']
    base_date = datetime.fromisoformat(iso)

    # Add the number of days corresponding to MJD
    dt = base_date + timedelta(days=mjd)

    # Add the seconds of the day
    dt = dt + timedelta(seconds=seconds_of_day)

    return dt


@capture_errors
def get_leap_seconds_table(url="https://data.iana.org/time-zones/data/leap-seconds.list"):
    """
    Download and parse the leap-seconds.list file from the given URL,
    returning a DataFrame with the following columns:
        - NTP_time: The NTP time (number of seconds since 1900-01-01).
        - DTAI: The correction value (TAI-UTC).
        - date: The date in datetime format (extracted from the comment, e.g., "1 Jan 1972").

    Parameters
    ----------
    url : str, optional
        The URL from which to fetch the leap-seconds.list file. Default is
        "https://data.iana.org/time-zones/data/leap-seconds.list".

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the following columns:
        - NTP_time: The NTP time in seconds.
        - DTAI: The correction value (TAI-UTC).
        - date: The date corresponding to the leap second event.
        - leap: The leap seconds relative to January 1, 1980.

    Raises
    ------
    Exception
        If there is an error in fetching the data from the provided URL.

    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error with reading the website. Check your internet connection: {response.status_code}")

    content = response.text
    data = []

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Check if the line contains a comment (part after '#' contains the date)
        if '#' in line:
            data_part, comment_part = line.split('#', 1)
            comment_part = comment_part.strip()
        else:
            data_part = line
            comment_part = ""

        # Parse the main part of the line (expect at least 2 elements: NTP_time and DTAI)
        parts = data_part.split()
        if len(parts) < 2:
            continue

        try:
            ntp_time = int(parts[0])
            dtai = int(parts[1])
        except ValueError:
            continue

        # Try to extract the date from the comment part, assuming the format "1 Jan 1972"
        date_val = None
        if comment_part:
            comment_tokens = comment_part.split()
            if len(comment_tokens) >= 3:
                date_str = ' '.join(comment_tokens[:3])
                try:
                    date_val = pd.to_datetime(date_str, format='%d %b %Y')
                except Exception as e:
                    # If the date parsing fails, leave it as None
                    date_val = None

        data.append({
            "NTP_time": ntp_time,
            "DTAI": dtai,
            "date": date_val
        })

    df = pd.DataFrame(data)
    df['leap'] = 0
    gps_year = df[df['date'] == datetime(1980, 1, 1)]
    leap_seconds_1980 = gps_year['DTAI'].values
    df['leap'] = df['DTAI'].values - leap_seconds_1980
    return df

_DEFAULT_LST = (
    Path(__file__).resolve().parent
        .parent
        / "config" / "leap_seconds_table.csv"
)
@capture_errors
def get_leap_seconds_table2(url="https://data.iana.org/time-zones/data/leap-seconds.list",
                            local_csv_file=_DEFAULT_LST):
    """
    Download and parse the leap-seconds.list file from the given URL,
    returning a DataFrame with the following columns:
        - NTP_time: The NTP time (number of seconds since 1900-01-01).
        - DTAI: The correction value (TAI-UTC).
        - date: The date extracted from the comment (e.g., "1 Jan 1972").
        - leap: The difference between DTAI and the value for January 1, 1980.

    If the data fetched from the internet differs from the local CSV table,
    the local file will be updated. In offline mode, if the data cannot be fetched,
    the local table is returned.

    Parameters
    ----------
    url : str, optional
        The URL from which to fetch the leap-seconds.list file. Default is
        "https://data.iana.org/time-zones/data/leap-seconds.list".

    local_csv_file : str, optional
        The path to the local CSV file where the leap seconds table is stored.
        If the local file is outdated or not present, it will be updated or created.
        Default is "./GSWPicker/config/leap_seconds_table.csv".

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the following columns:
        - NTP_time: The NTP time in seconds.
        - DTAI: The correction value (TAI-UTC).
        - date: The date corresponding to the leap second event.
        - leap: The leap seconds relative to January 1, 1980.

    Raises
    ------
    Exception
        If neither internet data nor the local CSV file is available.

    Notes
    -----
    This function attempts to download the leap-seconds.list file from the given URL.
    If the download is successful, it compares the data with the local CSV file.
    If there is a difference, the local file will be overwritten with the updated data.
    If the download fails, the function will try to read from the local CSV file.

    """
    remote_content = None
    try:
        response = requests.get(url)
        if response.status_code == 200:
            remote_content = response.text
        else:
            print("Failed to fetch data from the internet: status_code", response.status_code)
    except Exception as e:
        print("Error fetching data from the internet:", e)

    df_remote = None
    if remote_content is not None:
        data = []
        for line in remote_content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Split the data part from the comment (with the date)
            if '#' in line:
                data_part, comment_part = line.split('#', 1)
                comment_part = comment_part.strip()
            else:
                data_part = line
                comment_part = ""

            parts = data_part.split()
            if len(parts) < 2:
                continue

            try:
                ntp_time = int(parts[0])
                dtai = int(parts[1])
            except ValueError:
                continue

            date_val = None
            if comment_part:
                comment_tokens = comment_part.split()
                if len(comment_tokens) >= 3:
                    date_str = ' '.join(comment_tokens[:3])
                    try:
                        date_val = pd.to_datetime(date_str, format='%d %b %Y')
                    except Exception:
                        date_val = None

            data.append({
                "NTP_time": ntp_time,
                "DTAI": dtai,
                "date": date_val
            })

        df_remote = pd.DataFrame(data)
        # Calculate the "leap" column
        df_remote['leap'] = 0
        gps_year = df_remote[df_remote['date'] == datetime(1980, 1, 1)]
        if not gps_year.empty:
            leap_seconds_1980 = gps_year.iloc[0]['DTAI']
        else:
            leap_seconds_1980 = 0
        df_remote['leap'] = df_remote['DTAI'] - leap_seconds_1980

    # If data is fetched, compare with the local CSV table
    if df_remote is not None:
        if os.path.exists(local_csv_file):
            try:
                df_local = pd.read_csv(local_csv_file, parse_dates=["date"])
            except Exception as e:
                print("Error reading the local leap seconds file:", e)
                df_local = None

            if df_local is not None:
                # Compare DataFrame; equals method considers row order
                if not df_local.equals(df_remote):

                    print("Local table differs from remote data. Overwriting local leap seconds file...")
                    df_remote.to_csv(local_csv_file, index=False)
            else:
                print("Failed to load the local table. Saving new data.")
                df_remote.to_csv(local_csv_file, index=False)
        else:
            print("No local leap seconds file found. Saving fetched data.")
            df_remote.to_csv(local_csv_file, index=False)
        return df_remote
    else:
        # If data cannot be fetched, try to use the local CSV file
        if os.path.exists(local_csv_file):
            try:
                df_local = pd.read_csv(local_csv_file, parse_dates=["date"])
                print("Working in offline mode, using the local leap seconds table.")
                return df_local
            except Exception as e:
                raise Exception("Failed to read the local leap seconds file: " + str(e))
        else:
            raise Exception("No data: failed to fetch from the internet and no local leap seconds file exists.")


