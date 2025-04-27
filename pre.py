from datetime import datetime
from rich.console import Console
from modules.preinspect import read_sys_params
from modules.preinspect.functions import clear_console, display_intro, main, display_closing


if __name__ == '__main__':
    console = Console()
    clear_console()
    display_intro(console)

    # Odczyt parametrów systemowych
    d = read_sys_params()

    result = main(d, console)
    # main(d, console)

    console.print(f"[bold green]Program end: {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}[/]")
    display_closing(console)
