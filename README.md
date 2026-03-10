# claude-code-usage-analyzer

CLI tool that parses Claude Code conversation histories from `~/.claude/projects/` and produces detailed usage reports with KPIs.

## Install

```bash
# From GitHub
uvx --from git+https://github.com/jtaylorisbell/claude-code-usage-analyzer claude-usage

# Or clone and install locally
git clone https://github.com/jtaylorisbell/claude-code-usage-analyzer.git
cd claude-code-usage-analyzer
uv sync
uv run claude-usage
```

## Usage

```bash
# Analyze current project (cwd-aware)
claude-usage

# Analyze all projects
claude-usage --all

# Filter by project name
claude-usage --project lakebase

# Human conversations only (exclude subagents)
claude-usage --human-only

# Date filter
claude-usage --since 2026-03-01

# Export formats
claude-usage --json
claude-usage --csv

# Combine flags
claude-usage --all --human-only --since 2026-02-01 -n 10
```

## KPIs

### Per Conversation

| Category | Metrics |
|---|---|
| **Session** | Project, subagent detection, wall-clock duration, active duration (excludes idle gaps > 30 min) |
| **Tokens** | Input, output, cache read, cache creation, cache hit rate |
| **Turns** | User, assistant, system turn counts |
| **Tools** | Call count, breakdown by tool, unique tools used, tool success rate |
| **Skills** | Call count, breakdown by skill name |
| **Code** | Lines written, lines deleted, net lines changed |
| **Quality** | Correction count (NLP-detected), denials, API errors, tool call errors |
| **Latency** | Avg, median, max response time (user message to assistant response) |
| **Model** | Models used, primary model |
| **Completion** | Whether conversation ended naturally |
| **Cost** | API list price estimate (per-turn, per-model pricing) |

### Aggregate

All per-conversation metrics summed and averaged, plus:

- Human vs. subagent conversation split
- Daily and weekly trends (conversations, output tokens, cost)
- Tool usage breakdown (ranked)
- Skill usage breakdown
- Project breakdown
- Model breakdown
- Overall cache hit rate, correction rate, tool success rate

## How It Works

Claude Code stores conversation histories as JSONL files under `~/.claude/projects/`. Each line is a JSON object representing a message, tool call, tool result, or system event.

The tool:

1. **Discovers** project directories by mapping your cwd to Claude's naming convention (`/path/to/project` -> `-path-to-project`)
2. **Parses** JSONL files, extracting turns, token usage, tool calls (with code change metrics from Write/Edit inputs), tool results (with error detection), and skill invocations
3. **Deduplicates** by session ID (handles worktree copies)
4. **Analyzes** each conversation to compute KPIs
5. **Aggregates** across conversations with breakdowns and trends

### Correction Detection

User messages are scanned for patterns suggesting a correction (e.g., "no", "that's wrong", "undo", "revert", "not what I asked"). Messages over 500 characters and command invocations are excluded.

### Cost Estimation

Costs are computed per-turn using each turn's actual model and Anthropic's published API list prices. These represent API-equivalent costs, **not** actual spend on subscription plans (Max, etc.).

## Example Output

```
Scope: lakebase  |  19 conversations loaded

--- Recent Conversations (Last 2) ---

──────────────────────────────────────────────────────────────────────
ec243ff3-98b...  lakebase-todo-app
Started: 2026-03-05T18:21:27  Active: 117.9m  Wall: 117.9m
Turns: 51 user / 348 assistant  completed
Tokens: 669 in / 39.9K out / 31.0M cache-read  (hit rate: 98%)
Tools: 210 calls (17 unique)
  Top: Bash(115), Read(27), Edit(27), ToolSearch(12), Write(7)
Code: +695 / -199 (net +496)
Corrections: 2  Denials: 0  API errors: 0  Tool errors: 21  Tool success: 90%
Latency: avg=6.9s med=4.0s max=243.3s
Model: claude-opus-4-6
API list price: $59.78 (not actual spend)

══════════════════════════════════════════════════════════════════════
  HUMAN-INTERACTIVE USAGE REPORT
══════════════════════════════════════════════════════════════════════

Conversations: 19 (19 human, 23 subagent)
Total active time: 1549 min (25.8 hrs)
Avg active time: 81.5 min
Completion rate: 5%

--- Tokens ---
Input:          42.9K
Output:         300.6K
Cache read:     326.7M
Cache hit rate: 96.8%

--- Quality ---
Total corrections:    24
Correction rate:      7.4% of user turns
Tool success rate:    93.2%

--- Code Changes ---
Lines written:  15,717
Lines deleted:  3,498
Net change:     +12,219

--- Weekly Trends ---
  Week        Convs       Cost
  2026-W08        8 $  306.38  ▓▓▓▓▓▓▓▓
  2026-W09        5 $  224.63  ▓▓▓▓▓
  2026-W10        6 $  180.29  ▓▓▓▓▓▓
```

## Requirements

- Python 3.12+
- Claude Code conversation histories in `~/.claude/projects/`
