"""Interactive `atlas switch` command — provider + model picker.

Flow:
  1. Provider menu (OpenRouter / NVIDIA / Hugging Face)
  2. Fetch live models for the chosen provider (cached, 6h TTL)
  3. Numbered model list with search-by-substring
  4. Save the selection via the existing runtime_provider.json mechanism

Works in any TTY (including piped stdin) — no special keypress handling.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Type-only import for the _ctx_str helper.
if TYPE_CHECKING:
    from bin.models import Model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path("/root/atlas_proxy")
RUNTIME_PROVIDER_FILE = REPO_ROOT / "data" / "proxy_data" / "runtime_provider.json"
PREFS_FILE = REPO_ROOT / ".atlas_preferences.json"

CONSOLE = Console()


# ---------------------------------------------------------------------------
# Favourites
# ---------------------------------------------------------------------------

def _load_favourites() -> set[str]:
    if not PREFS_FILE.exists():
        return set()
    try:
        return set(json.loads(PREFS_FILE.read_text()).get("favourites", []))
    except Exception:
        return set()


def _save_favourites(favs: set[str]) -> None:
    PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREFS_FILE.write_text(json.dumps(
        {"favourites": sorted(favs), "updated_at": int(time.time())}, indent=2
    ))


# ---------------------------------------------------------------------------
# Save selection
# ---------------------------------------------------------------------------

def _save_selected(provider: str, model: str) -> None:
    """Write the chosen provider + model to the existing runtime_provider.json."""
    RUNTIME_PROVIDER_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if RUNTIME_PROVIDER_FILE.exists():
        try:
            existing = json.loads(RUNTIME_PROVIDER_FILE.read_text())
        except Exception:
            existing = {}
    existing.update({
        "provider": provider,
        "model": model,
        "selected_at": int(time.time()),
    })
    RUNTIME_PROVIDER_FILE.write_text(json.dumps(existing, indent=2))


# ---------------------------------------------------------------------------
# Model picker (numbered, searchable, with favourites)
# ---------------------------------------------------------------------------

def _short_name(model_id: str) -> str:
    """Return a short, friendly name for display."""
    # Strip provider prefix: 'z-ai/glm-5.2:free' -> 'glm-5.2:free'
    return model_id.split("/")[-1] if "/" in model_id else model_id


def _ctx_str(m: Model) -> str:
    """Context length display: '128k', '1M', or '—' if unknown."""
    cl = m.context_length
    if cl <= 0:
        return "—"
    if cl >= 1_000_000:
        return f"{cl // 1_000_000}M"
    if cl >= 1000:
        return f"{cl // 1000}k"
    return str(cl)


def _render_model_table(models, favs, page=0, page_size=20, query="") -> tuple[Table, int, int]:
    """Render a paginated table of models. Returns (table, total_pages, total_filtered)."""
    # Filter
    if query:
        q = query.lower()
        filtered = [
            m for m in models
            if q in m.id.lower() or q in _short_name(m.id).lower()
            or any(q in t for t in m.tasks)
        ]
    else:
        filtered = list(models)

    total = len(filtered)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(0, min(page, total_pages - 1))
    start = page * page_size
    end = start + page_size
    page_items = filtered[start:end]

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("ID", style="bold", overflow="fold")
    table.add_column("Tasks", style="dim", overflow="ellipsis")
    table.add_column("Ctx", style="dim", justify="right")
    table.add_column("Price", style="dim")
    table.add_column("", style="yellow", width=3)

    for i, m in enumerate(page_items, start=1):
        is_fav = m.id in favs
        tasks_str = ",".join(t for t in m.tasks if t != "conversational") or "chat"
        ctx = _ctx_str(m)
        if m.is_free:
            price = "[green]FREE[/green]"
        elif m.prompt_price or m.completion_price:
            price = f"${m.prompt_price:g}/${m.completion_price:g}"
        else:
            price = "—"
        star = "★" if is_fav else ""
        table.add_row(str(i), m.id, tasks_str, ctx, price, star)

    return table, total_pages, total


def _filter_models(models, query: str) -> list[Model]:
    """Return models matching the search query (substring on id/name/tasks)."""
    if not query:
        return list(models)
    q = query.lower()
    return [
        m for m in models
        if q in m.id.lower()
        or q in _short_name(m.id).lower()
        or any(q in t for t in m.tasks)
    ]


def _pick_model_interactive(models, provider_label: str) -> str | None | tuple:
    """Interactive picker.  Returns chosen model id, None (back), or
    tuple ("__refresh__",) to signal cache bust + refetch.

    Single-prompt model: type a number to pick, n/p to page,
    /text to search, fN to favourite row N, q to quit, r to refresh.
    """
    favs = _load_favourites()
    page = 0
    page_size = 20
    query = ""

    while True:
        filtered = _filter_models(models, query)
        total = len(filtered)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(0, min(page, total_pages - 1))
        start = page * page_size
        page_items = filtered[start:start + page_size]

        CONSOLE.print(f"\n[bold cyan]{provider_label}[/bold cyan] — {len(models)} models total")
        if query:
            CONSOLE.print(f"[dim]search: [bold]{query}[/bold][/dim]")
        if not page_items:
            CONSOLE.print("[yellow]No models match — press / to change search, q to back out.[/yellow]")
        else:
            table, _, _ = _render_model_table_paginated(
                page_items, favs, start_offset=start,
            )
            CONSOLE.print(table)
            CONSOLE.print(f"\n[dim]page {page + 1}/{total_pages} • {total} match[/dim]")
        CONSOLE.print("[dim]n/p next/prev • /term search • f<N> ★ row N • r refresh • q back • <number> pick[/dim]")

        # Single prompt — accept any of the above.
        raw = Prompt.ask(">").strip()
        if not raw:
            # Blank enter = pick the first item on this page (if any)
            if page_items:
                return page_items[0].id
            continue

        low = raw.lower()

        # Quit
        if low in ("q", "quit", "back", "exit", "b"):
            return None

        # Refresh
        if low in ("r", "refresh"):
            cache_slug = _cache_slug_for_provider(provider_label)
            cache = REPO_ROOT / ".cache" / f"{cache_slug}_models.json"
            meta = REPO_ROOT / ".cache" / f"{cache_slug}_models.meta.json"
            for p in (cache, meta):
                if p.exists():
                    p.unlink()
            CONSOLE.print("[yellow]Cache cleared.[/yellow]")
            return ("__refresh__",)

        # Page nav
        if low in ("n", "next", "p", "prev"):
            if low.startswith("n"):
                page = min(page + 1, total_pages - 1)
            else:
                page = max(page - 1, 0)
            continue

        # Search
        if low.startswith("/") or low.startswith("s "):
            new_query = raw[1:].strip() if low.startswith("/") else raw[2:].strip()
            query = new_query
            page = 0
            continue

        # Favourite
        if low.startswith("f") and low[1:].isdigit():
            num = int(low[1:])
            if 1 <= num <= len(page_items):
                mid = page_items[num - 1].id
                if mid in favs:
                    favs.discard(mid)
                    CONSOLE.print(f"[dim]unfavourited {mid}[/dim]")
                else:
                    favs.add(mid)
                    CONSOLE.print(f"[yellow]★ {mid}[/yellow]")
                _save_favourites(favs)
            else:
                CONSOLE.print(f"[red]row {num} not on this page (1-{len(page_items)})[/red]")
            continue

        # Number → pick
        if raw.isdigit():
            num = int(raw)
            if 1 <= num <= len(page_items):
                return page_items[num - 1].id
            CONSOLE.print(f"[red]row {num} not on this page (1-{len(page_items)})[/red]")
            continue

        # Otherwise: try to match by id or short-name
        match = next(
            (m for m in filtered if m.id == raw or _short_name(m.id) == raw),
            None,
        )
        if match:
            return match.id
        CONSOLE.print(f"[red]don't understand: {raw!r}.  type a number, n/p, /term, f<N>, r, or q[/red]")


def _render_model_table_paginated(page_items, favs, start_offset=0) -> tuple[Table, int, int]:
    """Render a single page.  start_offset is the absolute index of the first
    item on this page in the full filtered list (used for display numbering
    so /<n> page-nav picks the right model)."""
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("ID", style="bold", overflow="fold")
    table.add_column("Tasks", style="dim", overflow="ellipsis")
    table.add_column("Ctx", style="dim", justify="right", width=6)
    table.add_column("Price", style="dim", width=10)
    table.add_column("", style="yellow", width=3)

    for i, m in enumerate(page_items, start=start_offset + 1):
        is_fav = m.id in favs
        tasks_str = ",".join(t for t in m.tasks if t != "conversational") or "chat"
        ctx = _ctx_str(m)
        if m.is_free:
            price = "[green]FREE[/green]"
        elif m.prompt_price or m.completion_price:
            price = f"${m.prompt_price:g}/${m.completion_price:g}"
        else:
            price = "—"
        star = "★" if is_fav else ""
        table.add_row(str(i), m.id, tasks_str, ctx, price, star)

    return table, 1, len(page_items)


def _cache_slug_for_provider(provider_label: str) -> str:
    return {"OpenRouter": "openrouter", "NVIDIA": "nvidia", "Hugging Face": "huggingface"}.get(
        provider_label, provider_label.lower()
    )


# ---------------------------------------------------------------------------
# Provider menu
# ---------------------------------------------------------------------------

def _pick_provider() -> str | None:
    from bin.models import PROVIDER_MENU
    CONSOLE.print("[bold]Atlas Model Switch[/bold]\n")
    for i, p in enumerate(PROVIDER_MENU, start=1):
        CONSOLE.print(f"  {i}. [bold]{p.label}[/bold] — {p.description}")
    CONSOLE.print()
    choice = Prompt.ask(
        "Choose a provider",
        choices=[str(i) for i in range(1, len(PROVIDER_MENU) + 1)],
        default="1",
        show_choices=False,
    )
    return PROVIDER_MENU[int(choice) - 1].label


# ---------------------------------------------------------------------------
# Top-level: fetch + pick + save + restart
# ---------------------------------------------------------------------------

async def _fetch_for_provider(provider_slug: str):
    from bin import models as models_mod
    if provider_slug == "openrouter":
        return await models_mod.openrouter_models(REPO_ROOT, free_only=True)
    if provider_slug == "nvidia":
        return await models_mod.nvidia_models(REPO_ROOT)
    if provider_slug == "huggingface":
        return await models_mod.huggingface_models(REPO_ROOT)
    return []


def _slug(label: str) -> str:
    return {"OpenRouter": "openrouter", "NVIDIA": "nvidia", "Hugging Face": "huggingface"}.get(label, label.lower())


def _do_restart() -> None:
    """Restart the proxy to pick up the new model."""
    from atlas.bin.atlas import service_action
    service_action("stop")
    time.sleep(1)
    service_action("start")


def cmd_switch(args) -> int:
    """Interactive provider + model switcher.

    Usage: atlas switch
    """
    sys.path.insert(0, str(REPO_ROOT / "atlas"))

    label = _pick_provider()
    if not label:
        return 0
    provider_slug = _slug(label)

    # Fetch + show progress
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[progress.description]{{task.description}}"),
        transient=True,
        console=CONSOLE,
    ) as progress:
        progress.add_task(f"Fetching {label} models...", total=None)
        try:
            models = asyncio.run(_fetch_for_provider(provider_slug))
        except KeyboardInterrupt:
            CONSOLE.print("\n[yellow]Cancelled.[/yellow]")
            return 0

    if not models:
        CONSOLE.print(f"[red]No models available for {label}.[/red]")
        CONSOLE.print("[dim]Tip: run `atlas switch` again — caching may have caught a transient error.[/dim]")
        return 1

    CONSOLE.print(f"\n[green]{len(models)} models for {label}[/green]")

    # Interactive picker (handles its own loop, including refresh)
    chosen: str | None = None
    while True:
        result = _pick_model_interactive(models, label)
        if result is None:
            return 0
        if isinstance(result, tuple) and result and result[0] == "__refresh__":
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[progress.description]{{task.description}}"),
                transient=True,
                console=CONSOLE,
            ) as progress:
                progress.add_task(f"Refetching {label} models...", total=None)
                models = asyncio.run(_fetch_for_provider(provider_slug))
            if not models:
                CONSOLE.print(f"[red]No models available for {label} after refresh.[/red]")
                return 1
            CONSOLE.print(f"\n[green]{len(models)} models for {label}[/green]")
            continue
        chosen = result
        break

    # Save selection + restart
    _save_selected(provider_slug, chosen)
    CONSOLE.print(f"\n[green]✓ Selected {chosen}[/green] on [bold]{label}[/bold]")
    CONSOLE.print(f"[dim]  Saved to {RUNTIME_PROVIDER_FILE}[/dim]")

    if Confirm.ask("Restart proxy now?", default=True):
        CONSOLE.print("[cyan]Restarting proxy...[/cyan]")
        _do_restart()
        CONSOLE.print("[green]✓ Proxy restarted[/green]")
    else:
        CONSOLE.print("[dim]Restart later with: atlas restart[/dim]")

    return 0
