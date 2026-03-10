"""Parse Claude Code conversation JSONL files into structured data."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


CLAUDE_PROJECTS_DIR = Path("~/.claude/projects").expanduser()


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class ToolCall:
    name: str
    tool_use_id: str
    timestamp: str
    lines_written: int = 0
    lines_deleted: int = 0
    skill_name: str | None = None


@dataclass
class ToolResult:
    tool_use_id: str
    is_error: bool = False
    content_text: str = ""


@dataclass
class Turn:
    """A single message in a conversation."""

    type: str  # user, assistant, system
    timestamp: str | None = None
    content_text: str = ""
    model: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    stop_reason: str | None = None
    is_error: bool = False
    error: str | None = None
    uuid: str | None = None
    parent_uuid: str | None = None


@dataclass
class Conversation:
    """A full conversation session."""

    session_id: str
    project: str
    project_dir: str
    file_path: str
    turns: list[Turn] = field(default_factory=list)
    version: str | None = None
    git_branch: str | None = None
    is_subagent: bool = False

    @property
    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.type == "user" and t.content_text.strip()]

    @property
    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.type == "assistant"]

    @property
    def total_input_tokens(self) -> int:
        return sum(t.usage.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.usage.output_tokens for t in self.turns)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(t.usage.cache_read_input_tokens for t in self.turns)

    @property
    def total_cache_creation_tokens(self) -> int:
        return sum(t.usage.cache_creation_input_tokens for t in self.turns)

    @property
    def all_tool_calls(self) -> list[ToolCall]:
        calls = []
        for t in self.turns:
            calls.extend(t.tool_calls)
        return calls

    @property
    def all_tool_results(self) -> list[ToolResult]:
        results = []
        for t in self.turns:
            results.extend(t.tool_results)
        return results

    @property
    def start_time(self) -> str | None:
        for t in self.turns:
            if t.timestamp:
                return t.timestamp
        return None

    @property
    def end_time(self) -> str | None:
        for t in reversed(self.turns):
            if t.timestamp:
                return t.timestamp
        return None

    @property
    def timestamps(self) -> list[str]:
        return [t.timestamp for t in self.turns if t.timestamp]


# ── Internal helpers ──────────────────────────────────────────────


def _extract_text(content: list | str) -> str:
    if isinstance(content, str):
        return content
    text_parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    return "\n".join(text_parts)


def _count_lines(s: str) -> int:
    if not s:
        return 0
    return s.count("\n") + 1


def _extract_tool_calls(content: list | str, timestamp: str | None) -> list[ToolCall]:
    if isinstance(content, str):
        return []
    calls = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name", "unknown")
            inp = block.get("input", {})
            lines_written = 0
            lines_deleted = 0
            skill_name = None
            if name == "Write":
                lines_written = _count_lines(inp.get("content", ""))
            elif name == "Edit":
                lines_deleted = _count_lines(inp.get("old_string", ""))
                lines_written = _count_lines(inp.get("new_string", ""))
            elif name == "Skill":
                skill_name = inp.get("skill")
            calls.append(
                ToolCall(
                    name=name,
                    tool_use_id=block.get("id", ""),
                    timestamp=timestamp or "",
                    lines_written=lines_written,
                    lines_deleted=lines_deleted,
                    skill_name=skill_name,
                )
            )
    return calls


def _extract_tool_results(content: list | str) -> list[ToolResult]:
    if isinstance(content, str):
        return []
    results = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            result_content = block.get("content", "")
            if isinstance(result_content, list):
                text_parts = []
                for sub in result_content:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        text_parts.append(sub.get("text", ""))
                result_content = "\n".join(text_parts)
            results.append(
                ToolResult(
                    tool_use_id=block.get("tool_use_id", ""),
                    is_error=block.get("is_error", False),
                    content_text=str(result_content)[:500],
                )
            )
    return results


def _parse_usage(usage_dict: dict) -> TokenUsage:
    if not usage_dict:
        return TokenUsage()
    return TokenUsage(
        input_tokens=usage_dict.get("input_tokens", 0),
        output_tokens=usage_dict.get("output_tokens", 0),
        cache_creation_input_tokens=usage_dict.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=usage_dict.get("cache_read_input_tokens", 0),
    )


def _parse_line(raw: dict) -> Turn | None:
    msg_type = raw.get("type")
    if msg_type in ("file-history-snapshot", "progress", "queue-operation"):
        return None
    if msg_type not in ("user", "assistant", "system"):
        return None

    message = raw.get("message", {})
    content = message.get("content", "")
    timestamp = raw.get("timestamp")
    usage = _parse_usage(message.get("usage", {}))
    tool_calls = _extract_tool_calls(content, timestamp)
    tool_results = _extract_tool_results(content)

    return Turn(
        type=msg_type,
        timestamp=timestamp,
        content_text=_extract_text(content),
        model=message.get("model"),
        usage=usage,
        tool_calls=tool_calls,
        tool_results=tool_results,
        stop_reason=message.get("stop_reason"),
        is_error=raw.get("isApiErrorMessage", False),
        error=raw.get("error"),
        uuid=raw.get("uuid"),
        parent_uuid=raw.get("parentUuid"),
    )


# ── Public API ────────────────────────────────────────────────────


def cwd_to_project_dir(cwd: str | None = None) -> str:
    """Convert a working directory path to the Claude project directory name.

    Claude Code replaces both / and . with - in the directory name.
    e.g. /Users/taylor.isbell/projects/foo -> -Users-taylor-isbell-projects-foo
    """
    cwd = cwd or os.getcwd()
    return cwd.replace("/", "-").replace(".", "-")


def project_dir_to_display_name(project_dir: str) -> str:
    """Convert a Claude project dir name to a human-friendly display name.

    Strips the user's home path prefix for brevity.
    Claude replaces both / and . with - in dir names.
    """
    home = str(Path.home()).replace("/", "-").replace(".", "-")
    name = project_dir
    # Strip home-projects- prefix
    projects_prefix = home + "-projects-"
    if name.startswith(projects_prefix):
        return name[len(projects_prefix):]
    # Strip just home prefix
    if name.startswith(home + "-"):
        return "~/" + name[len(home) + 1:]
    return name.lstrip("-")


def parse_conversation(filepath: Path) -> Conversation:
    """Parse a JSONL conversation file."""
    session_id = filepath.stem
    # Walk up from the file to find the actual project directory.
    # Possible structures:
    #   project-dir/session.jsonl
    #   project-dir/session-uuid/subagents/agent-xxx.jsonl
    #   project-dir/session-uuid/session.jsonl  (worktree copies)
    # The project dir is always a direct child of ~/.claude/projects/
    parent = filepath.parent
    while parent.parent != CLAUDE_PROJECTS_DIR and parent.parent != parent:
        parent = parent.parent
    project_dir = parent.name

    display_name = project_dir_to_display_name(project_dir)
    is_subagent = session_id.startswith("agent-")

    conv = Conversation(
        session_id=session_id,
        project=display_name,
        project_dir=project_dir,
        file_path=str(filepath),
        is_subagent=is_subagent,
    )

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not conv.version and raw.get("version"):
                conv.version = raw["version"]
            if not conv.git_branch and raw.get("gitBranch"):
                conv.git_branch = raw["gitBranch"]

            turn = _parse_line(raw)
            if turn:
                conv.turns.append(turn)

    return conv


def find_project_dirs(
    base_dir: Path | None = None,
    project_filter: str | None = None,
    cwd: str | None = None,
) -> list[Path]:
    """Find project directories to scan.

    - If project_filter is given, match by name substring.
    - If cwd is given (and no project_filter), match the cwd's project dir.
    - Otherwise return all project dirs.
    """
    base = base_dir or CLAUDE_PROJECTS_DIR
    if not base.exists():
        return []

    all_dirs = sorted(d for d in base.iterdir() if d.is_dir())

    if project_filter:
        return [d for d in all_dirs if project_filter.lower() in d.name.lower()]

    if cwd:
        target = cwd_to_project_dir(cwd)
        return [d for d in all_dirs if d.name == target]

    return all_dirs


def load_conversations(
    project_dirs: list[Path] | None = None,
    base_dir: Path | None = None,
) -> list[Conversation]:
    """Load conversations from the given project directories (or all)."""
    if project_dirs is None:
        base = base_dir or CLAUDE_PROJECTS_DIR
        project_dirs = sorted(d for d in base.iterdir() if d.is_dir())

    conversations = []
    seen_sessions: set[str] = set()
    for proj_dir in project_dirs:
        for jsonl_file in sorted(proj_dir.rglob("*.jsonl")):
            try:
                conv = parse_conversation(jsonl_file)
                if conv.turns and conv.session_id not in seen_sessions:
                    seen_sessions.add(conv.session_id)
                    conversations.append(conv)
            except Exception as e:
                print(f"Warning: Failed to parse {jsonl_file}: {e}")
    return conversations
