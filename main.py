import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box

console = Console()
ROOT = Path(__file__).resolve().parent


def run_lecture(lecture: str):
    lecture_path = ROOT / lecture
    if not lecture_path.exists():
        console.print(f"[bold red]Error:[/bold red] Folder not found: {lecture_path}")
        return

    scripts = sorted(lecture_path.glob("*.py"))
    if not scripts:
        console.print(f"[bold yellow]Warning:[/bold yellow] No Python scripts found in [bold]{lecture}[/bold]")
        return

    script_path = scripts[0]

    table = Table(title="Torchure CLI Execution", box=box.ROUNDED, highlight=True)
    table.add_column("Lecture", style="cyan", no_wrap=True)
    table.add_column("Script", style="magenta")
    table.add_row(lecture, script_path.name)
    console.print(table)

    console.print(f"[bold green]→ Running:[/bold green] {script_path}\n")
    console.print(f"\n[bold blue]📦 Output of {lecture}:[/bold blue]\n")
    subprocess.run([sys.executable, str(script_path)])

    


def main():
    console.print(Panel("[bold cyan]Welcome to Torchure[/bold cyan]\n[white]A beautiful PyTorch adventure begins...[/white]", style="bold blue"))

    while True:
        user_input = Prompt.ask("\n[bold magenta]Enter lecture number (1-16) or type 'exit' to quit[/bold magenta]")

        if user_input.lower() == "exit":
            console.print("\n[bold green]👋 Exiting Torchure. Happy learning![/bold green]")
            break

        if not user_input.isdigit() or not (1 <= int(user_input) <= 16):
            console.print("[bold red]Invalid input.[/bold red] Please enter a number between 1 and 16.")
            continue

        lecture = f"Lecture{int(user_input):02d}"
        run_lecture(lecture)


if __name__ == "__main__":
    main()
