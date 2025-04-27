import pandas as pd



def save_station_dependent_results(station_name, report2, d):
    """
    Saves station-dependent report data into CSV files.

    Depending on the calculation mode:
    - For '2D'/'3D': saves one CSV file for the entire station.
    - Otherwise (commonly '1D'): saves three files, one per component ('n', 'e', 'u').

    Parameters
    ----------
    station_name : str
        Name of the station.
    report2 : pandas.DataFrame
        The DataFrame containing report data for the station.
    d : dict
        Configuration dictionary with keys such as 'result_csv', 'PROJECT_ID', 'MODE', 'CALCULATION_MODE'.
    """
    if d['CALCULATION_MODE'] in ['2D', '3D']:
        report2.to_csv(
            f"{d['result_csv']}/{d['PROJECT_ID']}_{d['MODE']}_{d['CALCULATION_MODE']}_{station_name}.csv"
        )
    else:
        # Typically the 1D case, we handle separate 'n', 'e', 'u' components
        for comp in ['n', 'e', 'u']:
            try:
                report2.sort_index(inplace=True)
                r = report2.loc[(station_name, comp), :].reset_index(drop=True)
                r['Station'] = station_name
                r['component'] = comp

                cols = list(r)
                cols.insert(0, cols.pop(cols.index('Station')))
                cols.insert(1, cols.pop(cols.index('component')))
                r = r.loc[:, cols]

                r.to_csv(
                    f"{d['result_csv']}/{d['PROJECT_ID']}_{d['MODE']}_{d['CALCULATION_MODE']}_{comp}_{station_name}.csv"
                )
            except KeyError:
                print(f"Unable to obtain results for station: {station_name}, component: {comp}.")
                continue

def finalize_all_s_wave_arrivals(reports, d):
    """
    Finalizes and saves the report for the 'All S-wave arrivals' storage option.

    Parameters
    ----------
    reports : list of pandas.DataFrame
        A list of DataFrames (each is a station's 'report').
    d : dict
        Configuration dictionary with keys such as 'CALCULATION_MODE', 'result_csv', etc.

    Notes
    -----
    - For 1D mode, sorts data by the earliest S-wave arrival time per station.
    - For 2D/3D modes, sorts data simply by 'S-wave detected at (sec)'.
    - Only the 'S-wave detected at (sec)' column is exported to the final CSV.
    """

    final_report = pd.concat(reports)
    if d['time'] != 'utc':
        col = 'S-wave detected at (sec)'
    elif d['time'] == 'utc':
        col = [col for col in final_report.columns if col.startswith('S-wave')][0]

    if d['CALCULATION_MODE'] == '1D':

        min_s_wave_per_station = final_report.groupby(level='Station')[f'{col}'].min()
        final_report['station_min_s_wave'] = final_report.index.get_level_values('Station').map(min_s_wave_per_station)

        sorted_df = final_report.sort_values(
            ['station_min_s_wave', f'{col}']
        ).drop(columns=['station_min_s_wave'])#.droplevel(level=2)
        if sorted_df.index.nlevels > 2:
            sorted_df=sorted_df.droplevel(level=2)
        sorted_df[[f'{col}']].to_csv(
            f"{d['result_csv']}/{d['PROJECT_ID']}_{d['MODE']}_{d['CALCULATION_MODE']}.csv"
        )

    elif d['CALCULATION_MODE'] in ['2D', '3D']:
        final_report = final_report.sort_values(by=f'{col}')
        final_report[[f'{col}']].to_csv(
            f"{d['result_csv']}/{d['PROJECT_ID']}_{d['MODE']}_{d['CALCULATION_MODE']}.csv"
        )

def finalize_component_wise_s_wave_arrivals(reports, d):
    """
    Finalizes and saves the report for the 'Component wise S-wave arrivals' storage option.

    Parameters
    ----------
    reports : list of pandas.DataFrame
        A list of DataFrames (each is a station's 'report').
    d : dict
        Configuration dictionary with keys such as 'CALCULATION_MODE', 'result_csv', etc.

    Notes
    -----
    - For 1D mode, creates one CSV file per component (e.g. 'n', 'e', 'u').
    - For 2D/3D modes, merges everything into a single CSV file.
    """
    processed = []
    for r in reports:
        tmp = r.reset_index()
        tmp = tmp.rename(columns={'index': 'station'})
        if 'level_2' in tmp.columns:
            tmp = tmp.drop(columns='level_2')
        processed.append(tmp)
    final_report = pd.concat(processed, ignore_index=True)

    if d['CALCULATION_MODE'] == '1D':
        for comp, group in final_report.groupby('component'):
            group = group.sort_values(by='S-wave detected at (sec)')
            group.to_csv(
                f"{d['result_csv']}/"
                f"{d['PROJECT_ID']}_{d['MODE']}_{d['CALCULATION_MODE']}_{comp}.csv",
                index=False
            )
    else:
        final_report = final_report.sort_values(by='S-wave detected at (sec)')
        final_report.to_csv(
            f"{d['result_csv']}/"
            f"{d['PROJECT_ID']}_{d['MODE']}_{d['CALCULATION_MODE']}.csv",
            index=False
        )


def generate_report(all_results, d, console):
    """
    Generates final reports based on 'all_results' and the provided configuration.

    This function iterates through results from all stations, saves station-dependent
    data (if applicable), and collects data for final summary reports.

    Parameters
    ----------
    all_results : list of tuples
        A list of (station_name, run_results) tuples, where run_results = (pick, report, report2).
    d : dict
        Configuration dictionary with keys such as 'storage_option', 'CALCULATION_MODE', etc.
    console : rich.console.Console
        A Rich Console for printing messages in a styled way.

    Returns
    -------
    None
    """
    console.print("[bold green]Generating reports...[/]")
    reports = []
    full_reports = []

    for station_name, run_results in all_results:
        if run_results is None:
            print(f"No results for station {station_name} due to an error.")
            continue

        pick, report, report2 = run_results

        if d['storage_option'] == 'Station dependent':
            save_station_dependent_results(station_name, report2, d)

        if d['storage_option'] in ['All S-wave arrivals', 'Component wise S-wave arrivals']:
            reports.append(report)
            full_reports.append(report2)

    if d['storage_option'] == 'All S-wave arrivals':
        try:
            finalize_all_s_wave_arrivals(reports, d)
        except ValueError:
            print('No data to store')
    if d['storage_option'] == 'Component wise S-wave arrivals':
        try:
            finalize_component_wise_s_wave_arrivals(reports, d)
        except ValueError:
            print('No data to store')