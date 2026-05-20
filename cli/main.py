"""
cli/main.py

Rich terminal interface for the Autonomous Incident Response System (AIRS).

Workflow (ImplementationPlan.md Section 9):
  1. User runs:  python cli/main.py run --incident bad-db-connection.json
     (or omits --incident to use the built-in fixture alert)
  2. Terminal streams a live Rich panel showing:
       [cyan]  Tool calls  (get_metrics, get_logs)
       [green] LLM thoughts (AI message tokens)
       [yellow] Node transitions
       [red]   Errors / high-risk flags
  3. Graph pauses at the HITL approval node → interrupt payload rendered as
     a full Markdown panel showing the RemediationPlan.
  4. User types 'approve' or 'reject' → CLI sends Command(resume={...}).
  5. Graph resumes → execution & postmortem → written to postmortem.md.

Crash-safe demo:
  Run with the same --thread-id after a crash to resume from the exact
  checkpoint without re-running triage or investigation.

  python cli/main.py run --thread-id <previous-thread-id>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Load .env before anything else so GOOGLE_API_KEY etc. are available
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.orchestrator import build_graph
from agent.state import make_initial_state
from langgraph.types import Command

# ---------------------------------------------------------------------------
# Rich console theme
# ---------------------------------------------------------------------------

_THEME = Theme(
    {
        "node":        "bold bright_blue",
        "tool_call":   "bold cyan",
        "tool_result": "dim cyan",
        "llm_thought": "green",
        "interrupt":   "bold yellow",
        "error":       "bold red",
        "success":     "bold green",
        "info":        "dim white",
        "severity_p0": "bold red on dark_red",
        "severity_p1": "bold yellow",
        "severity_p2": "bold blue",
        "severity_p3": "dim white",
    }
)

console = Console(theme=_THEME, highlight=False)

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="airs",
    help="Autonomous Incident Response System — terminal interface.",
    add_completion=False,
    pretty_exceptions_enable=False,
)

# ---------------------------------------------------------------------------
# Embedded demo alert (used when no --incident file is supplied)
# ---------------------------------------------------------------------------

_DEMO_ALERT: dict[str, Any] = {
    "id": "PD-20240519-0042",
    "title": "CRITICAL: Database connection pool exhausted — payments-service",
    "service": "payments-service",
    "severity": "P0",
    "triggered_at": datetime.utcnow().isoformat() + "Z",
    "source": "PagerDuty",
    "description": (
        "Connection pool for payments-db has reached 100% utilization. "
        "New requests are queuing and timing out. Error rate has spiked to 47%."
    ),
}

# ---------------------------------------------------------------------------
# Rich rendering helpers
# ---------------------------------------------------------------------------


def _severity_style(severity: str) -> str:
    return {
        "P0": "severity_p0",
        "P1": "severity_p1",
        "P2": "severity_p2",
        "P3": "severity_p3",
    }.get(severity.upper(), "white")


def _print_header(alert: dict[str, Any]) -> None:
    """Render the incident banner at startup."""
    severity = alert.get("severity", "??")
    service  = alert.get("service",  "unknown")
    title    = alert.get("title",    "Incident")
    alert_id = alert.get("id",       "N/A")

    sev_style = _severity_style(severity)

    table = Table.grid(expand=True, padding=(0, 2))
    table.add_column(ratio=1)
    table.add_column(ratio=3)
    table.add_row(
        Text("ALERT ID",  style="info"),
        Text(alert_id, style="bold white"),
    )
    table.add_row(
        Text("SERVICE",   style="info"),
        Text(service,  style="bold bright_white"),
    )
    table.add_row(
        Text("SEVERITY",  style="info"),
        Text(f" {severity} ", style=sev_style),
    )
    table.add_row(
        Text("TRIGGERED", style="info"),
        Text(alert.get("triggered_at", "N/A"), style="dim white"),
    )

    console.print()
    console.print(
        Panel(
            table,
            title="[bold bright_red]⚡ AIRS — INCIDENT DETECTED[/]",
            subtitle=f"[dim]{title}[/]",
            border_style="bright_red",
            padding=(1, 2),
        )
    )
    console.print()


def _print_node_transition(node_name: str) -> None:
    """Print a visual separator when a new node begins."""
    icons = {
        "triage":      "🔍",
        "escalate":    "🚨",
        "investigate": "🔎",
        "plan":        "📋",
        "approval":    "⏸️ ",
        "execute":     "⚙️ ",
        "reject":      "🚫",
    }
    icon = icons.get(node_name, "▶")
    console.print(Rule(f"{icon}  [node]{node_name.upper()} NODE[/]", style="bright_blue"))


def _print_tool_call(tool_name: str, args: dict[str, Any]) -> None:
    """Print a tool invocation in cyan."""
    args_str = "  ".join(f"[dim]{k}[/]=[bold]{v!r}[/]" for k, v in args.items())
    console.print(
        f"  [tool_call]→ TOOL CALL:[/] [bold cyan]{tool_name}[/]  {args_str}"
    )


def _print_tool_result(tool_name: str, content: str) -> None:
    """Print a summarised tool result (first 300 chars to keep terminal clean)."""
    preview = content[:300].replace("\n", " ").strip()
    if len(content) > 300:
        preview += " …"
    console.print(
        f"  [tool_result]← TOOL RESULT:[/] [dim cyan]{tool_name}[/]  {preview}"
    )


def _print_ai_chunk(text: str) -> None:
    """Print an AI message token stream inline (no newline)."""
    console.print(f"  [llm_thought]{text}[/]", end="")


def _print_ai_message(content: str) -> None:
    """Print a complete AI message on its own line."""
    if not content.strip():
        return
    console.print(f"  [llm_thought]{content}[/]")


def _render_interrupt_panel(payload: dict[str, Any]) -> None:
    """Render the HITL approval panel using Rich Markdown."""
    console.print()
    console.print(Rule("[interrupt]⏸  EXECUTION PAUSED — HUMAN APPROVAL REQUIRED[/]", style="yellow"))
    console.print()

    plan_md: str = payload.get("plan", "No plan available.")
    risk: str    = payload.get("risk", "Unknown")
    message: str = payload.get("message", "Review and approve.")

    # Render the full remediation plan as Markdown
    console.print(
        Panel(
            Markdown(plan_md),
            title="[bold yellow]📋 Remediation Plan[/]",
            subtitle=f"[bold]Risk: {risk}[/]",
            border_style="yellow",
            padding=(1, 2),
        )
    )
    console.print()
    console.print(f"  [interrupt]{message}[/]")
    console.print()


def _render_postmortem_panel(postmortem: str, thread_id: str) -> None:
    """Render the postmortem and write it to disk."""
    console.print()
    console.print(Rule("[success]✅  INCIDENT RESOLVED — POSTMORTEM GENERATED[/]", style="green"))
    console.print()
    console.print(
        Panel(
            Markdown(postmortem),
            title="[bold green]📄 Postmortem[/]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Write to disk
    out_path = Path(f"postmortem_{thread_id[:8]}.md")
    out_path.write_text(postmortem, encoding="utf-8")
    console.print(f"\n  [success]Postmortem saved to:[/] [bold]{out_path}[/]\n")


# ---------------------------------------------------------------------------
# Core streaming runner
# ---------------------------------------------------------------------------


async def _stream_graph(
    alert: dict[str, Any],
    thread_id: str,
    resume_command: Command | None = None,
) -> bool:
    """
    Drive the LangGraph execution loop, streaming all events to the console.

    Returns True if the graph paused at an interrupt (approval needed),
    False if it ran to completion or was rejected.
    """
    config = {"configurable": {"thread_id": thread_id}}

    async with build_graph() as graph:
        # Determine input: fresh start vs resume after interrupt
        if resume_command is not None:
            graph_input: Any = resume_command
        else:
            graph_input = make_initial_state(alert)

        _current_node: list[str] = [""]  # mutable container for closure

        async for event in graph.astream_events(graph_input, config, version="v2"):
            event_type: str = event.get("event", "")
            event_name: str = event.get("name", "")
            data: dict = event.get("data", {})
            tags: list[str] = event.get("tags", [])
            metadata: dict = event.get("metadata", {})

            # ----------------------------------------------------------------
            # Node transition
            # ----------------------------------------------------------------
            if event_type == "on_chain_start" and event_name in (
                "triage", "escalate", "investigate", "plan",
                "approval", "execute", "reject",
            ):
                if event_name != _current_node[0]:
                    _current_node[0] = event_name
                    _print_node_transition(event_name)

            # ----------------------------------------------------------------
            # Tool call dispatched by the LLM
            # ----------------------------------------------------------------
            elif event_type == "on_tool_start":
                tool_name = event_name
                tool_input = data.get("input", {})
                if isinstance(tool_input, str):
                    try:
                        tool_input = json.loads(tool_input)
                    except Exception:
                        tool_input = {"input": tool_input}
                _print_tool_call(tool_name, tool_input)

            # ----------------------------------------------------------------
            # Tool result returned
            # ----------------------------------------------------------------
            elif event_type == "on_tool_end":
                tool_name = event_name
                output = data.get("output", "")
                content = output if isinstance(output, str) else str(output)
                _print_tool_result(tool_name, content)

            # ----------------------------------------------------------------
            # LLM streaming tokens
            # ----------------------------------------------------------------
            elif event_type == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    # Don't stream raw tokens for structured output calls —
                    # they produce JSON fragments that confuse the terminal.
                    # Only stream plain-text AI messages.
                    if isinstance(chunk.content, str) and not chunk.content.startswith("{"):
                        _print_ai_chunk(chunk.content)

            # ----------------------------------------------------------------
            # Complete AI message (non-streaming)
            # ----------------------------------------------------------------
            elif event_type == "on_chat_model_end":
                output = data.get("output")
                if output and hasattr(output, "content"):
                    content = output.content
                    if isinstance(content, str) and content.strip():
                        # Only print if we didn't already stream it token-by-token
                        if not any(
                            kw in content for kw in ("{", "severity", "root_cause")
                        ):
                            _print_ai_message(content)

            # ----------------------------------------------------------------
            # INTERRUPT — graph paused for HITL approval
            # ----------------------------------------------------------------
            elif event_type == "on_chain_end" and event_name == "__interrupt__":
                # In LangGraph 1.x, the interrupt payload is in data["output"]
                # It's a tuple of Interrupt objects
                output = data.get("output", ())
                interrupt_payload: dict = {}

                if output:
                    # output is typically a tuple[Interrupt, ...]
                    first = output[0] if isinstance(output, (list, tuple)) else output
                    if hasattr(first, "value"):
                        interrupt_payload = first.value
                    elif isinstance(first, dict):
                        interrupt_payload = first

                _render_interrupt_panel(interrupt_payload)
                return True  # Signal: graph is paused, caller handles HITL

            # ----------------------------------------------------------------
            # Graph completed (reached __end__)
            # ----------------------------------------------------------------
            elif event_type == "on_chain_end" and event_name == "LangGraph":
                output = data.get("output", {})
                postmortem = output.get("postmortem", "")
                if postmortem:
                    _render_postmortem_panel(postmortem, thread_id)
                elif output.get("is_approved") is False:
                    console.print(
                        "\n  [error]Incident response aborted by operator.[/]\n"
                    )

    return False  # Graph ran to completion without pause


# ---------------------------------------------------------------------------
# HITL approval loop
# ---------------------------------------------------------------------------


async def _run_with_hitl(alert: dict[str, Any], thread_id: str) -> None:
    """
    Full execution loop including the HITL approval interaction.
    Handles the crash-safe resume pattern transparently.
    """
    console.print(f"\n  [info]Thread ID:[/] [bold]{thread_id}[/]")
    console.print(
        "  [info](Save this ID to resume after a crash)[/]\n"
    )

    # Phase 1: Run until interrupt or completion
    paused = await _stream_graph(alert, thread_id)

    if not paused:
        return  # Graph ran to completion (rejected plan, etc.)

    # Phase 2: HITL interaction
    while True:
        raw = Prompt.ask(
            "  [interrupt]Decision[/]",
            choices=["approve", "reject", "view", "quit"],
            default="approve",
            console=console,
        )
        decision = raw.strip().lower()

        if decision == "view":
            # Re-display the plan from the checkpoint
            async with build_graph() as graph:
                config = {"configurable": {"thread_id": thread_id}}
                state = await graph.aget_state(config)
                plan_md = state.values.get("plan", "No plan in state.")
                console.print(
                    Panel(
                        Markdown(plan_md),
                        title="[yellow]📋 Remediation Plan (re-display)[/]",
                        border_style="yellow",
                    )
                )
            continue

        if decision == "quit":
            console.print("\n  [error]Aborting. Graph state preserved. Resume later with same thread-id.[/]\n")
            raise typer.Exit(0)

        approved = decision == "approve"
        resume_cmd = Command(resume={"approved": approved})

        console.print()
        if approved:
            console.print(Rule("[success]▶  RESUMING — EXECUTING REMEDIATION[/]", style="green"))
        else:
            console.print(Rule("[error]✖  RESUMING — PLAN REJECTED[/]", style="red"))
        console.print()

        # Phase 3: Resume graph from checkpoint
        await _stream_graph(alert, thread_id, resume_command=resume_cmd)
        break


# ---------------------------------------------------------------------------
# Typer commands
# ---------------------------------------------------------------------------


@app.command()
def run(
    incident: Path = typer.Option(
        None,
        "--incident", "-i",
        help="Path to a JSON file containing the PagerDuty-style alert payload. "
             "If omitted, the built-in demo alert (DB connection exhaustion) is used.",
        exists=False,
    ),
    thread_id: str = typer.Option(
        None,
        "--thread-id", "-t",
        help="LangGraph thread ID for crash-safe resume. "
             "If omitted, a new UUID is generated.",
    ),
    mock_api_url: str = typer.Option(
        "http://localhost:8000",
        "--mock-api-url",
        help="Base URL of the mock enterprise API server.",
        envvar="MOCK_API_BASE_URL",
    ),
) -> None:
    """
    Run the AIRS incident response pipeline.

    Start the mock enterprise server first:

        uvicorn mock_enterprise.api:app --port 8000 --reload

    Then in a second terminal:

        python cli/main.py run
        python cli/main.py run --incident my-alert.json
        python cli/main.py run --thread-id <previous-id>   # resume after crash
    """
    # Inject config into environment before any agent modules spin up
    os.environ["MOCK_API_BASE_URL"] = mock_api_url

    # Resolve alert payload
    if incident is not None:
        if not incident.exists():
            console.print(f"[error]Alert file not found: {incident}[/]")
            raise typer.Exit(1)
        try:
            alert: dict[str, Any] = json.loads(incident.read_text())
        except json.JSONDecodeError as exc:
            console.print(f"[error]Invalid JSON in {incident}: {exc}[/]")
            raise typer.Exit(1)
    else:
        alert = _DEMO_ALERT
        console.print("\n  [info]No --incident file specified. Using built-in demo alert.[/]")

    # Resolve thread ID
    tid = thread_id or str(uuid.uuid4())

    _print_header(alert)
    asyncio.run(_run_with_hitl(alert, tid))


@app.command()
def resume(
    thread_id: str = typer.Argument(..., help="Thread ID of the paused graph to resume."),
    approve: bool = typer.Option(
        True,
        "--approve/--reject",
        help="Approve or reject the remediation plan.",
    ),
    mock_api_url: str = typer.Option(
        "http://localhost:8000",
        "--mock-api-url",
        envvar="MOCK_API_BASE_URL",
    ),
) -> None:
    """
    Resume a paused graph by thread ID (crash-safe demo command).

    This is used in the 'Server Crash Test' demo:

        1. Run: python cli/main.py run
        2. Kill the terminal (Ctrl+C) after the graph pauses at approval.
        3. Resume: python cli/main.py resume <thread-id> --approve
    """
    os.environ["MOCK_API_BASE_URL"] = mock_api_url

    resume_cmd = Command(resume={"approved": approve})
    decision_label = "APPROVE" if approve else "REJECT"

    console.print()
    console.print(
        Panel(
            f"[bold]Resuming thread:[/] {thread_id}\n"
            f"[bold]Decision:[/] {decision_label}",
            title="[bold yellow]⚡ AIRS — CRASH-SAFE RESUME[/]",
            border_style="yellow",
        )
    )
    console.print()

    async def _do_resume() -> None:
        await _stream_graph({}, thread_id, resume_command=resume_cmd)

    asyncio.run(_do_resume())


@app.command()
def status(
    thread_id: str = typer.Argument(..., help="Thread ID to inspect."),
) -> None:
    """
    Show the current checkpoint state for a given thread ID.
    Useful for debugging or verifying what the graph has persisted.
    """
    async def _show_status() -> None:
        async with build_graph() as graph:
            config = {"configurable": {"thread_id": thread_id}}
            state = await graph.aget_state(config)
            if state is None or not state.values:
                console.print(f"\n  [error]No checkpoint found for thread_id={thread_id!r}[/]\n")
                return

            vals = state.values
            table = Table(
                title=f"Graph State — {thread_id[:16]}…",
                box=box.ROUNDED,
                border_style="bright_blue",
                show_header=True,
            )
            table.add_column("Field", style="bold cyan", no_wrap=True)
            table.add_column("Value", style="white")

            for key in ("severity", "retry_count", "is_approved", "postmortem"):
                val = vals.get(key, "—")
                if isinstance(val, bool):
                    val = "✅ True" if val else "❌ False"
                table.add_row(key, str(val)[:120])

            if tr := vals.get("triage_result"):
                table.add_row(
                    "triage_result",
                    f"severity={tr.severity} service={tr.service} confidence={tr.confidence:.0%}",
                )

            telemetry = vals.get("telemetry", "")
            table.add_row("telemetry", f"[{len(telemetry)} chars]")

            plan = vals.get("plan", "")
            table.add_row("plan", f"[{len(plan)} chars]")

            next_nodes = list(state.next) if state.next else ["(completed)"]
            table.add_row("next nodes", ", ".join(next_nodes))

            console.print()
            console.print(table)
            console.print()

    asyncio.run(_show_status())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
