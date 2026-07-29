"""Rich console wrapper with configurable output modes."""

import json
import sys
from dataclasses import dataclass
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class ConsoleConfig:
    """Console configuration."""
    quiet: bool = False
    json_output: bool = False
    force_terminal: Optional[bool] = None
    non_interactive: bool = False


class AtlasConsole:
    """Wrapper around Rich console with structured output support."""

    def __init__(self, config: ConsoleConfig = None):
        self.config = config or ConsoleConfig()
        self._console = Console(
            force_terminal=self.config.force_terminal,
            file=sys.stdout,
            highlight=False,
            markup=True,
        )
        self._json_buffer = []

    # --- Output modes ---

    def _output(self, *args, **kwargs):
        if self.config.quiet and not self.config.json_output:
            return
        if self.config.json_output:
            self._json_buffer.append({"type": "log", "args": args, "kwargs": kwargs})
        else:
            self._console.print(*args, **kwargs)

    def flush_json(self):
        """Flush buffered JSON output."""
        if self.config.json_output and self._json_buffer:
            print(json.dumps(self._json_buffer))
            self._json_buffer = []

    # --- Levels ---

    def debug(self, msg: str):
        self._output(f"[dim]{msg}[/]")

    def info(self, msg: str):
        self._output(f"[cyan]{msg}[/]")

    def success(self, msg: str):
        self._output(f"[green]✓[/] {msg}")

    def warning(self, msg: str):
        self._output(f"[yellow]⚠[/] {msg}")

    def error(self, msg: str):
        self._output(f"[red]✗[/] {msg}")

    def step(self, msg: str):
        self._output(f"[bold blue]▶[/] {msg}")

    def rule(self, title: str = ""):
        if not self.config.quiet:
            self._console.rule(title)

    # --- Interactive ---

    def prompt(self, prompt: str, default: str = "", password: bool = False) -> str:
        if self.config.non_interactive:
            return default
        from rich.prompt import Prompt
        return Prompt.ask(prompt, default=default, console=self._console, password=password)

    def confirm(self, prompt: str, default: bool = True) -> bool:
        if self.config.non_interactive:
            return default
        from rich.prompt import Confirm
        return Confirm.ask(prompt, default=default, console=self._console)

    # --- Structured ---

    def table(self, title: str, columns: list[str], rows: list[list[str]]):
        if self.config.json_output:
            self._json_buffer.append({
                "type": "table",
                "title": title,
                "columns": columns,
                "rows": rows,
            })
        else:
            table = Table(title=title, show_header=True, header_style="bold magenta")
            for col in columns:
                table.add_column(col)
            for row in rows:
                table.add_row(*row)
            self._console.print(table)

    def panel(self, content: str, title: str = "", style: str = "blue"):
        if self.config.json_output:
            self._json_buffer.append({
                "type": "panel",
                "title": title,
                "content": content,
                "style": style,
            })
        else:
            self._console.print(Panel(content, title=title, border_style=style))

    def key_value(self, key: str, value: Any):
        if self.config.json_output:
            self._json_buffer.append({"type": "kv", "key": key, "value": str(value)})
        else:
            self._console.print(f"  [cyan]{key}[/]: {value}")

    def banner(self, title: str, subtitle: str = ""):
        """Print a banner panel."""
        content = title
        if subtitle:
            content += f"\n[dim]{subtitle}[/]"
        self._console.print(Panel(content, title="Atlas Setup", border_style="magenta"))


_console: Optional[AtlasConsole] = None


def get_console(
    quiet: bool = False,
    json_output: bool = False,
    force_terminal: Optional[bool] = None,
    non_interactive: bool = False,
) -> AtlasConsole:
    """Get or create the global console instance."""
    global _console
    if _console is None:
        _console = AtlasConsole(ConsoleConfig(
            quiet=quiet,
            json_output=json_output,
            force_terminal=force_terminal,
            non_interactive=non_interactive,
        ))
    return _console


def set_console(console: AtlasConsole) -> None:
    """Set the global console (for testing)."""
    global _console
    _console = console