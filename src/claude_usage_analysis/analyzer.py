"""Analyze parsed conversations to produce KPIs."""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from .parser import Conversation, Turn

# Patterns that suggest the user is correcting Claude
CORRECTION_PATTERNS = [
    r"\bno[,.]?\s",
    r"\bthat'?s not\b",
    r"\bthat'?s wrong\b",
    r"\bwrong\b",
    r"\bincorrect\b",
    r"\bactually[,.]?\s",
    r"\binstead[,.]?\s",
    r"\bdon'?t do\b",
    r"\bstop\b",
    r"\bundo\b",
    r"\brevert\b",
    r"\brollback\b",
    r"\bthat broke\b",
    r"\bfix (that|this|it)\b",
    r"\bnot what I\b",
    r"\bI said\b",
    r"\bI meant\b",
    r"\bI asked\b",
    r"\bplease don'?t\b",
    r"\bwhy did you\b",
]

CORRECTION_RE = re.compile("|".join(CORRECTION_PATTERNS), re.IGNORECASE)

DENIAL_PATTERNS = [
    "user denied",
    "permission denied",
    "not allowed",
    "user rejected",
    "blocked by user",
    "The user declined",
]

# Model pricing: (input $/M, output $/M, cache_read $/M, cache_write $/M)
MODEL_PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-4-5-20251101": (15.0, 75.0, 1.50, 18.75),
    "claude-opus-4-6": (15.0, 75.0, 1.50, 18.75),
    "claude-sonnet-4-5-20250929": (3.0, 15.0, 0.30, 3.75),
    "claude-sonnet-4-6": (3.0, 15.0, 0.30, 3.75),
    "claude-haiku-4-5-20251001": (0.80, 4.0, 0.08, 1.0),
}
DEFAULT_PRICING = (15.0, 75.0, 1.50, 18.75)


def _parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _compute_active_time(timestamps: list[str], gap_threshold_minutes: float = 30.0) -> float:
    parsed = []
    for ts in timestamps:
        dt = _parse_timestamp(ts)
        if dt:
            parsed.append(dt)
    if len(parsed) < 2:
        return 0.0
    parsed.sort()
    active = 0.0
    for i in range(1, len(parsed)):
        gap = (parsed[i] - parsed[i - 1]).total_seconds() / 60
        if gap < gap_threshold_minutes:
            active += gap
    return active


def _compute_turn_latencies(turns: list[Turn]) -> list[float]:
    latencies = []
    for i, turn in enumerate(turns):
        if turn.type == "user" and turn.timestamp:
            for j in range(i + 1, len(turns)):
                if turns[j].type == "assistant" and turns[j].timestamp:
                    user_dt = _parse_timestamp(turn.timestamp)
                    asst_dt = _parse_timestamp(turns[j].timestamp)
                    if user_dt and asst_dt:
                        latency = (asst_dt - user_dt).total_seconds()
                        if 0 < latency < 600:
                            latencies.append(latency)
                    break
    return latencies


@dataclass
class ConversationKPIs:
    session_id: str
    project: str
    is_subagent: bool = False
    start_time: str | None = None
    end_time: str | None = None
    wall_clock_minutes: float | None = None
    active_minutes: float | None = None

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    cache_hit_rate: float = 0.0

    total_turns: int = 0
    user_turns: int = 0
    assistant_turns: int = 0
    system_turns: int = 0

    tool_call_count: int = 0
    tool_breakdown: dict[str, int] = field(default_factory=dict)
    unique_tools_used: int = 0
    tool_success_rate: float = 0.0

    correction_count: int = 0
    correction_messages: list[str] = field(default_factory=list)
    api_error_count: int = 0
    tool_error_count: int = 0
    denial_count: int = 0
    denial_messages: list[str] = field(default_factory=list)

    lines_written: int = 0
    lines_deleted: int = 0
    net_lines_changed: int = 0

    skill_call_count: int = 0
    skill_breakdown: dict[str, int] = field(default_factory=dict)

    avg_response_latency_sec: float | None = None
    median_response_latency_sec: float | None = None
    max_response_latency_sec: float | None = None

    models_used: list[str] = field(default_factory=list)
    primary_model: str | None = None
    version: str | None = None

    tokens_per_user_turn: float = 0.0
    tool_calls_per_assistant_turn: float = 0.0

    completed_naturally: bool = False

    estimated_cost_usd: float = 0.0


def _compute_turn_cost(turn: Turn) -> float:
    """Compute cost for a single turn based on its model."""
    model = turn.model
    if model and model in MODEL_PRICING:
        pricing = MODEL_PRICING[model]
    else:
        pricing = DEFAULT_PRICING
    inp_rate, out_rate, cache_read_rate, cache_write_rate = pricing
    u = turn.usage
    return (
        u.input_tokens * inp_rate / 1_000_000
        + u.output_tokens * out_rate / 1_000_000
        + u.cache_read_input_tokens * cache_read_rate / 1_000_000
        + u.cache_creation_input_tokens * cache_write_rate / 1_000_000
    )


def _detect_corrections(user_turns: list[Turn]) -> tuple[int, list[str]]:
    corrections = []
    for turn in user_turns:
        text = turn.content_text.strip()
        if not text or len(text) > 500:
            continue
        if text.startswith("<command") or text.startswith("/"):
            continue
        if CORRECTION_RE.search(text):
            corrections.append(text[:200])
    return len(corrections), corrections


def _detect_denials(conv: Conversation) -> tuple[int, list[str]]:
    denials = []
    for turn in conv.turns:
        for result in turn.tool_results:
            if result.is_error:
                text = result.content_text.lower()
                for pattern in DENIAL_PATTERNS:
                    if pattern.lower() in text:
                        denials.append(result.content_text[:200])
                        break
    return len(denials), denials


def _compute_code_changes(conv: Conversation) -> tuple[int, int]:
    written = 0
    deleted = 0
    for tc in conv.all_tool_calls:
        written += tc.lines_written
        deleted += tc.lines_deleted
    return written, deleted


def analyze_conversation(conv: Conversation) -> ConversationKPIs:
    kpis = ConversationKPIs(
        session_id=conv.session_id,
        project=conv.project,
        is_subagent=conv.is_subagent,
        start_time=conv.start_time,
        end_time=conv.end_time,
        version=conv.version,
    )

    start = _parse_timestamp(conv.start_time)
    end = _parse_timestamp(conv.end_time)
    if start and end:
        kpis.wall_clock_minutes = (end - start).total_seconds() / 60
    kpis.active_minutes = _compute_active_time(conv.timestamps)

    kpis.total_input_tokens = conv.total_input_tokens
    kpis.total_output_tokens = conv.total_output_tokens
    kpis.total_cache_read_tokens = conv.total_cache_read_tokens
    kpis.total_cache_creation_tokens = conv.total_cache_creation_tokens

    total_input_context = (
        kpis.total_input_tokens + kpis.total_cache_read_tokens + kpis.total_cache_creation_tokens
    )
    if total_input_context > 0:
        kpis.cache_hit_rate = kpis.total_cache_read_tokens / total_input_context

    kpis.total_turns = len(conv.turns)
    kpis.user_turns = len(conv.user_turns)
    kpis.assistant_turns = len(conv.assistant_turns)
    kpis.system_turns = len([t for t in conv.turns if t.type == "system"])

    tool_counts: Counter[str] = Counter()
    for tc in conv.all_tool_calls:
        tool_counts[tc.name] += 1
    kpis.tool_call_count = sum(tool_counts.values())
    kpis.tool_breakdown = dict(tool_counts.most_common())
    kpis.unique_tools_used = len(tool_counts)

    kpis.correction_count, kpis.correction_messages = _detect_corrections(conv.user_turns)
    kpis.denial_count, kpis.denial_messages = _detect_denials(conv)
    kpis.api_error_count = len([t for t in conv.turns if t.is_error])
    total_tool_results = sum(len(t.tool_results) for t in conv.turns)
    tool_errors = sum(
        1 for t in conv.turns for r in t.tool_results
        if r.is_error and not any(p.lower() in r.content_text.lower() for p in DENIAL_PATTERNS)
    )
    kpis.tool_error_count = tool_errors
    if total_tool_results > 0:
        kpis.tool_success_rate = (total_tool_results - tool_errors) / total_tool_results

    kpis.lines_written, kpis.lines_deleted = _compute_code_changes(conv)
    kpis.net_lines_changed = kpis.lines_written - kpis.lines_deleted

    skill_counts: Counter[str] = Counter()
    for tc in conv.all_tool_calls:
        if tc.skill_name:
            skill_counts[tc.skill_name] += 1
    kpis.skill_call_count = sum(skill_counts.values())
    kpis.skill_breakdown = dict(skill_counts.most_common())

    latencies = _compute_turn_latencies(conv.turns)
    if latencies:
        kpis.avg_response_latency_sec = sum(latencies) / len(latencies)
        sorted_lat = sorted(latencies)
        mid = len(sorted_lat) // 2
        kpis.median_response_latency_sec = (
            sorted_lat[mid]
            if len(sorted_lat) % 2 == 1
            else (sorted_lat[mid - 1] + sorted_lat[mid]) / 2
        )
        kpis.max_response_latency_sec = max(latencies)

    model_counts: Counter[str] = Counter()
    for t in conv.turns:
        if t.model and t.model != "<synthetic>":
            model_counts[t.model] += 1
    kpis.models_used = sorted(model_counts.keys())
    if model_counts:
        kpis.primary_model = model_counts.most_common(1)[0][0]

    assistant_turns = conv.assistant_turns
    if assistant_turns:
        last = assistant_turns[-1]
        kpis.completed_naturally = last.stop_reason in ("end_turn", "stop_sequence")

    if kpis.user_turns > 0:
        kpis.tokens_per_user_turn = kpis.total_output_tokens / kpis.user_turns
    if kpis.assistant_turns > 0:
        kpis.tool_calls_per_assistant_turn = kpis.tool_call_count / kpis.assistant_turns

    # Cost — computed per-turn using each turn's actual model
    kpis.estimated_cost_usd = sum(_compute_turn_cost(t) for t in conv.turns)

    return kpis


# ── Aggregate ──────────────────────────────────────────────────────


@dataclass
class AggregateKPIs:
    total_conversations: int = 0
    human_conversations: int = 0
    subagent_conversations: int = 0

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    overall_cache_hit_rate: float = 0.0

    total_user_turns: int = 0
    total_assistant_turns: int = 0
    total_tool_calls: int = 0
    total_corrections: int = 0
    total_denials: int = 0
    total_api_errors: int = 0
    total_tool_errors: int = 0
    overall_tool_success_rate: float = 0.0

    total_lines_written: int = 0
    total_lines_deleted: int = 0
    total_net_lines_changed: int = 0

    total_skill_calls: int = 0

    total_wall_clock_minutes: float = 0.0
    total_active_minutes: float = 0.0
    estimated_total_cost_usd: float = 0.0

    avg_response_latency_sec: float | None = None
    median_response_latency_sec: float | None = None

    completion_rate: float = 0.0

    tool_breakdown: dict[str, int] = field(default_factory=dict)
    skill_breakdown: dict[str, int] = field(default_factory=dict)
    project_breakdown: dict[str, int] = field(default_factory=dict)
    model_breakdown: dict[str, int] = field(default_factory=dict)

    avg_turns_per_conversation: float = 0.0
    avg_corrections_per_conversation: float = 0.0
    avg_active_minutes: float = 0.0
    correction_rate: float = 0.0

    daily_conversations: dict[str, int] = field(default_factory=dict)
    daily_tokens_out: dict[str, int] = field(default_factory=dict)
    daily_cost: dict[str, float] = field(default_factory=dict)
    weekly_conversations: dict[str, int] = field(default_factory=dict)
    weekly_cost: dict[str, float] = field(default_factory=dict)


def _iso_to_date(ts: str | None) -> str | None:
    dt = _parse_timestamp(ts)
    return dt.strftime("%Y-%m-%d") if dt else None


def _iso_to_week(ts: str | None) -> str | None:
    dt = _parse_timestamp(ts)
    if dt:
        year, week, _ = dt.isocalendar()
        return f"{year}-W{week:02d}"
    return None


def compute_aggregate(
    kpis_list: list[ConversationKPIs],
    human_only: bool = False,
) -> AggregateKPIs:
    agg = AggregateKPIs()

    filtered = [k for k in kpis_list if not k.is_subagent] if human_only else kpis_list
    agg.total_conversations = len(filtered)
    agg.human_conversations = len([k for k in kpis_list if not k.is_subagent])
    agg.subagent_conversations = len([k for k in kpis_list if k.is_subagent])

    tool_counter: Counter[str] = Counter()
    skill_counter: Counter[str] = Counter()
    project_counter: Counter[str] = Counter()
    model_counter: Counter[str] = Counter()
    daily_convs: Counter[str] = Counter()
    daily_toks: Counter[str] = Counter()
    daily_costs: defaultdict[str, float] = defaultdict(float)
    weekly_convs: Counter[str] = Counter()
    weekly_costs: defaultdict[str, float] = defaultdict(float)
    all_latencies: list[float] = []
    completed = 0

    for k in filtered:
        agg.total_input_tokens += k.total_input_tokens
        agg.total_output_tokens += k.total_output_tokens
        agg.total_cache_read_tokens += k.total_cache_read_tokens
        agg.total_cache_creation_tokens += k.total_cache_creation_tokens
        agg.total_user_turns += k.user_turns
        agg.total_assistant_turns += k.assistant_turns
        agg.total_tool_calls += k.tool_call_count
        agg.total_corrections += k.correction_count
        agg.total_denials += k.denial_count
        agg.total_api_errors += k.api_error_count
        agg.total_tool_errors += k.tool_error_count
        agg.total_lines_written += k.lines_written
        agg.total_lines_deleted += k.lines_deleted
        agg.total_net_lines_changed += k.net_lines_changed
        agg.total_skill_calls += k.skill_call_count
        agg.estimated_total_cost_usd += k.estimated_cost_usd

        if k.wall_clock_minutes:
            agg.total_wall_clock_minutes += k.wall_clock_minutes
        if k.active_minutes:
            agg.total_active_minutes += k.active_minutes
        if k.completed_naturally:
            completed += 1
        if k.avg_response_latency_sec is not None:
            all_latencies.append(k.avg_response_latency_sec)

        for tool, count in k.tool_breakdown.items():
            tool_counter[tool] += count
        for skill, count in k.skill_breakdown.items():
            skill_counter[skill] += count
        project_counter[k.project] += 1
        for model in k.models_used:
            model_counter[model] += 1

        day = _iso_to_date(k.start_time)
        week = _iso_to_week(k.start_time)
        if day:
            daily_convs[day] += 1
            daily_toks[day] += k.total_output_tokens
            daily_costs[day] += k.estimated_cost_usd
        if week:
            weekly_convs[week] += 1
            weekly_costs[week] += k.estimated_cost_usd

    agg.tool_breakdown = dict(tool_counter.most_common())
    agg.skill_breakdown = dict(skill_counter.most_common())
    agg.project_breakdown = dict(project_counter.most_common())
    agg.model_breakdown = dict(model_counter.most_common())
    agg.daily_conversations = dict(sorted(daily_convs.items()))
    agg.daily_tokens_out = dict(sorted(daily_toks.items()))
    agg.daily_cost = dict(sorted(daily_costs.items()))
    agg.weekly_conversations = dict(sorted(weekly_convs.items()))
    agg.weekly_cost = dict(sorted(weekly_costs.items()))

    total_ctx = agg.total_input_tokens + agg.total_cache_read_tokens + agg.total_cache_creation_tokens
    if total_ctx > 0:
        agg.overall_cache_hit_rate = agg.total_cache_read_tokens / total_ctx

    if agg.total_conversations > 0:
        agg.completion_rate = completed / agg.total_conversations

    if all_latencies:
        agg.avg_response_latency_sec = sum(all_latencies) / len(all_latencies)
        sl = sorted(all_latencies)
        mid = len(sl) // 2
        agg.median_response_latency_sec = (
            sl[mid] if len(sl) % 2 == 1 else (sl[mid - 1] + sl[mid]) / 2
        )

    n = agg.total_conversations
    if n > 0:
        agg.avg_turns_per_conversation = (agg.total_user_turns + agg.total_assistant_turns) / n
        agg.avg_corrections_per_conversation = agg.total_corrections / n
        convs_with_active = len([k for k in filtered if k.active_minutes and k.active_minutes > 0])
        if convs_with_active > 0:
            agg.avg_active_minutes = agg.total_active_minutes / convs_with_active

    if agg.total_user_turns > 0:
        agg.correction_rate = agg.total_corrections / agg.total_user_turns

    if agg.total_tool_calls > 0:
        agg.overall_tool_success_rate = (agg.total_tool_calls - agg.total_tool_errors) / agg.total_tool_calls

    return agg
