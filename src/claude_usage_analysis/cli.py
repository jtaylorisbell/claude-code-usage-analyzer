"""CLI entry point for claude-usage-analysis."""

import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from .analyzer import AggregateKPIs, ConversationKPIs, analyze_conversation, compute_aggregate
from .parser import find_project_dirs, load_conversations

app = typer.Typer(
    name="claude-usage",
    help="Analyze Claude Code conversation histories and quantify usage KPIs.",
    no_args_is_help=False,
)
console = Console()


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def fmt_latency(sec: float | None) -> str:
    return f"{sec:.1f}s" if sec is not None else "N/A"


def _load_and_analyze(
    all_projects: bool,
    project: str | None,
    since: str | None,
    human_only: bool,
) -> tuple[list[ConversationKPIs], list[ConversationKPIs]]:
    """Load conversations and return (all_kpis, filtered_kpis)."""
    if project:
        dirs = find_project_dirs(project_filter=project)
    elif all_projects:
        dirs = find_project_dirs()
    else:
        dirs = find_project_dirs(cwd=os.getcwd())

    if not dirs:
        if not all_projects and not project:
            console.print(
                f"[yellow]No conversations found for current directory.[/yellow]\n"
                f"[dim]cwd: {os.getcwd()}[/dim]\n"
                f"Try [bold]--all[/bold] for all projects, or [bold]--project NAME[/bold] to filter."
            )
        else:
            console.print("[yellow]No matching project directories found.[/yellow]")
        raise typer.Exit(0)

    conversations = load_conversations(project_dirs=dirs)
    if not conversations:
        console.print("[yellow]No conversations found.[/yellow]")
        raise typer.Exit(0)

    all_kpis = [analyze_conversation(conv) for conv in conversations]

    # Date filter
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            console.print(f"[red]Invalid date format: {since}. Use YYYY-MM-DD.[/red]")
            raise typer.Exit(1)
        since_str = since_dt.isoformat()
        all_kpis = [k for k in all_kpis if k.start_time and k.start_time >= since_str]

    filtered = [k for k in all_kpis if not k.is_subagent] if human_only else all_kpis
    filtered.sort(key=lambda k: k.start_time or "", reverse=True)
    all_kpis.sort(key=lambda k: k.start_time or "", reverse=True)
    return all_kpis, filtered


def _print_conversation_summary(kpi: ConversationKPIs) -> None:
    tag = " [dim]\\[subagent][/dim]" if kpi.is_subagent else ""
    status = "[green]completed[/green]" if kpi.completed_naturally else "[dim]incomplete[/dim]"

    console.print(f"\n[bold]{'─' * 70}[/bold]")
    console.print(f"[bold]{kpi.session_id[:12]}...[/bold]  {kpi.project}{tag}")

    if kpi.start_time:
        parts = [f"Started: {kpi.start_time[:19]}"]
        if kpi.active_minutes:
            parts.append(f"Active: {kpi.active_minutes:.1f}m")
        if kpi.wall_clock_minutes:
            parts.append(f"Wall: {kpi.wall_clock_minutes:.1f}m")
        console.print("  ".join(parts))

    console.print(f"Turns: {kpi.user_turns} user / {kpi.assistant_turns} assistant  {status}")
    console.print(
        f"Tokens: {fmt_tokens(kpi.total_input_tokens)} in / "
        f"{fmt_tokens(kpi.total_output_tokens)} out / "
        f"{fmt_tokens(kpi.total_cache_read_tokens)} cache-read  "
        f"(hit rate: {kpi.cache_hit_rate:.0%})"
    )
    console.print(f"Tools: {kpi.tool_call_count} calls ({kpi.unique_tools_used} unique)")
    if kpi.tool_breakdown:
        top = ", ".join(f"{n}({c})" for n, c in list(kpi.tool_breakdown.items())[:5])
        console.print(f"  Top: {top}")
    if kpi.skill_call_count > 0:
        skills = ", ".join(f"{s}({c})" for s, c in kpi.skill_breakdown.items())
        console.print(f"Skills: {kpi.skill_call_count} calls — {skills}")
    console.print(f"Code: +{kpi.lines_written} / -{kpi.lines_deleted} (net {kpi.net_lines_changed:+d})")
    console.print(f"Corrections: {kpi.correction_count}  Denials: {kpi.denial_count}  API errors: {kpi.api_error_count}  Tool errors: {kpi.tool_error_count}  Tool success: {kpi.tool_success_rate:.0%}")
    if kpi.correction_messages:
        for msg in kpi.correction_messages[:3]:
            console.print(f'  [yellow]> "{msg[:80]}{"..." if len(msg) > 80 else ""}"[/yellow]')
    console.print(
        f"Latency: avg={fmt_latency(kpi.avg_response_latency_sec)} "
        f"med={fmt_latency(kpi.median_response_latency_sec)} "
        f"max={fmt_latency(kpi.max_response_latency_sec)}"
    )
    if kpi.primary_model:
        console.print(f"Model: {kpi.primary_model}")
    console.print(f"[bold]API list price: ${kpi.estimated_cost_usd:.2f}[/bold] [dim](not actual spend)[/dim]")


def _print_aggregate(agg: AggregateKPIs, label: str) -> None:
    console.print(f"\n[bold]{'═' * 70}[/bold]")
    console.print(f"  [bold]{label}[/bold]")
    console.print(f"[bold]{'═' * 70}[/bold]")

    console.print(f"\nConversations: {agg.total_conversations} ({agg.human_conversations} human, {agg.subagent_conversations} subagent)")
    console.print(f"Total active time: {agg.total_active_minutes:.0f} min ({agg.total_active_minutes / 60:.1f} hrs)")
    console.print(f"Avg active time: {agg.avg_active_minutes:.1f} min")
    console.print(f"Completion rate: {agg.completion_rate:.0%}")

    # Tokens
    console.print(f"\n[bold]--- Tokens ---[/bold]")
    console.print(f"Input:          {fmt_tokens(agg.total_input_tokens)}")
    console.print(f"Output:         {fmt_tokens(agg.total_output_tokens)}")
    console.print(f"Cache read:     {fmt_tokens(agg.total_cache_read_tokens)}")
    console.print(f"Cache creation: {fmt_tokens(agg.total_cache_creation_tokens)}")
    console.print(f"Cache hit rate: {agg.overall_cache_hit_rate:.1%}")

    # Interactions
    console.print(f"\n[bold]--- Interactions ---[/bold]")
    console.print(f"User turns:      {agg.total_user_turns}")
    console.print(f"Assistant turns:  {agg.total_assistant_turns}")
    console.print(f"Avg turns/conv:   {agg.avg_turns_per_conversation:.1f}")
    console.print(f"Total tool calls: {agg.total_tool_calls}")

    # Quality
    console.print(f"\n[bold]--- Quality ---[/bold]")
    console.print(f"Total corrections:    {agg.total_corrections}")
    console.print(f"Avg corrections/conv: {agg.avg_corrections_per_conversation:.2f}")
    console.print(f"Correction rate:      {agg.correction_rate:.1%} of user turns")
    console.print(f"Total denials:        {agg.total_denials}")
    console.print(f"API errors:           {agg.total_api_errors}")
    console.print(f"Tool call errors:     {agg.total_tool_errors}")
    console.print(f"Tool success rate:    {agg.overall_tool_success_rate:.1%}")

    # Code changes
    console.print(f"\n[bold]--- Code Changes ---[/bold]")
    console.print(f"Lines written:  {agg.total_lines_written:,}")
    console.print(f"Lines deleted:  {agg.total_lines_deleted:,}")
    console.print(f"Net change:     {agg.total_net_lines_changed:+,}")

    # Latency
    console.print(f"\n[bold]--- Response Latency ---[/bold]")
    console.print(f"Avg:    {fmt_latency(agg.avg_response_latency_sec)}")
    console.print(f"Median: {fmt_latency(agg.median_response_latency_sec)}")

    # Skills
    console.print(f"\n[bold]--- Skills ---[/bold]")
    if agg.skill_breakdown:
        console.print(f"Total skill calls: {agg.total_skill_calls}")
        for skill, count in agg.skill_breakdown.items():
            console.print(f"  {skill:<45} {count:>4}")
    else:
        console.print("  No skill calls recorded")

    # Tools
    console.print(f"\n[bold]--- Tool Usage (Top 15) ---[/bold]")
    max_count = max(agg.tool_breakdown.values()) if agg.tool_breakdown else 1
    for tool, count in list(agg.tool_breakdown.items())[:15]:
        bar_len = int(count / max_count * 30)
        bar = "█" * bar_len
        console.print(f"  {tool:<30} {count:>5}  {bar}")

    # Projects
    console.print(f"\n[bold]--- Projects (Top 10) ---[/bold]")
    for proj, count in list(agg.project_breakdown.items())[:10]:
        console.print(f"  {proj:<55} {count:>3} convs")

    # Models
    console.print(f"\n[bold]--- Models ---[/bold]")
    for model, count in agg.model_breakdown.items():
        console.print(f"  {model:<35} {count:>3} convs")

    # Cost
    console.print(f"\n[bold]--- Cost (API List Price — not actual spend) ---[/bold]")
    console.print(f"[dim]Based on per-model token pricing. Does not reflect subscription plans.[/dim]")
    console.print(f"Total:              ${agg.estimated_total_cost_usd:.2f}")
    if agg.total_conversations > 0:
        console.print(f"Avg per conversation: ${agg.estimated_total_cost_usd / agg.total_conversations:.2f}")

    # Weekly trends
    if agg.weekly_conversations:
        console.print(f"\n[bold]--- Weekly Trends ---[/bold]")
        console.print(f"  {'Week':<10} {'Convs':>6} {'Cost':>10}")
        for week in list(agg.weekly_conversations.keys())[-12:]:
            convs = agg.weekly_conversations[week]
            cost = agg.weekly_cost.get(week, 0)
            bar = "▓" * min(convs, 40)
            console.print(f"  {week:<10} {convs:>6} ${cost:>8.2f}  {bar}")

    # Daily trends
    if agg.daily_conversations:
        console.print(f"\n[bold]--- Daily Trends (Last 14 Days) ---[/bold]")
        console.print(f"  {'Date':<12} {'Convs':>6} {'Tokens Out':>12} {'Cost':>10}")
        for day in list(agg.daily_conversations.keys())[-14:]:
            convs = agg.daily_conversations[day]
            toks = agg.daily_tokens_out.get(day, 0)
            cost = agg.daily_cost.get(day, 0)
            console.print(f"  {day:<12} {convs:>6} {fmt_tokens(toks):>12} ${cost:>8.2f}")

    console.print()


@app.command()
def report(
    all_projects: Annotated[bool, typer.Option("--all", "-a", help="Analyze all projects (not just current directory)")] = False,
    project: Annotated[Optional[str], typer.Option("--project", "-p", help="Filter by project name (substring match)")] = None,
    since: Annotated[Optional[str], typer.Option("--since", "-s", help="Only include conversations after this date (YYYY-MM-DD)")] = None,
    human_only: Annotated[bool, typer.Option("--human-only", "-h", help="Exclude subagent conversations")] = False,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of recent conversations to show")] = 5,
    output_json: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
    output_csv: Annotated[bool, typer.Option("--csv", help="Output as CSV")] = False,
):
    """Show usage report for Claude Code conversations."""
    all_kpis, filtered = _load_and_analyze(all_projects, project, since, human_only)

    if output_json:
        _output_json(all_kpis, filtered, human_only)
        return

    if output_csv:
        _output_csv(filtered)
        return

    scope = "all projects" if all_projects else (project or os.path.basename(os.getcwd()))
    console.print(f"[dim]Scope: {scope}  |  {len(filtered)} conversations loaded[/dim]")

    # Recent conversations
    console.print(f"\n[bold]--- Recent Conversations (Last {limit}) ---[/bold]")
    for kpi in filtered[:limit]:
        _print_conversation_summary(kpi)

    # Aggregate
    if human_only:
        agg = compute_aggregate(all_kpis, human_only=True)
        _print_aggregate(agg, "HUMAN-INTERACTIVE USAGE REPORT")
    else:
        human_agg = compute_aggregate(all_kpis, human_only=True)
        _print_aggregate(human_agg, "HUMAN-INTERACTIVE USAGE REPORT")
        if any(k.is_subagent for k in all_kpis):
            full_agg = compute_aggregate(all_kpis, human_only=False)
            _print_aggregate(full_agg, "FULL REPORT (incl. subagents)")


def _output_json(all_kpis: list[ConversationKPIs], filtered: list[ConversationKPIs], human_only: bool) -> None:
    agg = compute_aggregate(all_kpis, human_only=human_only)
    output = {
        "aggregate": _kpi_to_dict(agg),
        "conversations": [_kpi_to_dict(k) for k in filtered],
    }
    print(json.dumps(output, indent=2, default=str))


def _output_csv(filtered: list[ConversationKPIs]) -> None:
    headers = [
        "session_id", "project", "is_subagent", "start_time", "active_minutes",
        "user_turns", "assistant_turns", "input_tokens", "output_tokens",
        "cache_read_tokens", "cache_hit_rate", "tool_calls", "skill_calls",
        "corrections", "denials", "api_errors", "tool_errors", "lines_written", "lines_deleted",
        "net_lines", "avg_latency_sec", "completed", "model", "est_cost_usd",
    ]
    print(",".join(headers))
    for k in filtered:
        row = [
            k.session_id, k.project, k.is_subagent, k.start_time or "",
            f"{k.active_minutes:.1f}" if k.active_minutes else "",
            k.user_turns, k.assistant_turns, k.total_input_tokens, k.total_output_tokens,
            k.total_cache_read_tokens, f"{k.cache_hit_rate:.3f}", k.tool_call_count,
            k.skill_call_count, k.correction_count, k.denial_count, k.api_error_count, k.tool_error_count,
            k.lines_written, k.lines_deleted, k.net_lines_changed,
            f"{k.avg_response_latency_sec:.1f}" if k.avg_response_latency_sec else "",
            k.completed_naturally, k.primary_model or "", f"{k.estimated_cost_usd:.2f}",
        ]
        print(",".join(str(v) for v in row))


def _kpi_to_dict(obj: object) -> dict:
    """Convert a dataclass to dict, handling non-serializable types."""
    d = asdict(obj) if hasattr(obj, "__dataclass_fields__") else {}
    return d


def main():
    app()
