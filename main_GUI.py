from modules.ui import clear_console, display_intro, display_closing, write_header
from modules.prep import read_sys_params, read_input_files
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from datetime import datetime
import os
from modules.paralell import process_batch
from modules.output import generate_report
import concurrent.futures
from modules.config import CFG
if __name__ =='__main__':
    console=Console()
    clear_console()
    display_intro(console)

    d = read_sys_params()
    gnss_data_dict, samp_fre_dict = read_input_files(vadase_path=d['vadase_path'],
                                                     include_stations=d['INCLUDE_STATIONS'])

    t = datetime.now()
    timestamp = t.strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_{timestamp}.txt"
    with open(os.path.join(d['logdir'], log_filename), 'w') as log:
        write_header(file=log, t=t)
        log.write('\n')
        for k, v in d.items():
            log.write(f"{k}: {v}\n")
            log.write("\n")

    items = list(gnss_data_dict.items())
    val = CFG['CPUS'].get('cpu_counter', '')
    default_workers = os.cpu_count()//2
    try:
        max_workers = int(val)
    except (ValueError, TypeError):
        max_workers = default_workers

    batch_size = max_workers
    console.print(f"[green bold]Workers: {max_workers} Batch size: {batch_size}[/]")
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    all_results = []
    with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Processing data...", total=len(batches))

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_batch, batch, samp_fre_dict, d): i for i, batch in
                       enumerate(batches, start=1)}

            for future in concurrent.futures.as_completed(futures):
                batch_index = futures[future]
                try:
                    batch_result = future.result()
                    all_results.extend(batch_result)
                    progress.update(task, advance=1)
                except Exception as e:
                    console.print(f"[red]Error processing batch: {e}[/]")
    # console.print("[bold green]Generating reports...[/]")
    generate_report(all_results=all_results,d=d,console=console)
    console.print(f"[bold green]Log file saved in {d['logdir']}")
    console.print(f"[bold green]Reports saved in {d['result_csv']}")
    console.print(f"[bold green]Figures saved in {d['result_figures']}")
    console.print('\n')
    console.print(f"[green bold]Program run end: {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}[/]")
    display_closing(console)