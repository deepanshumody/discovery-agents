# MCP server

`discovery-agents` ships a [Model Context Protocol](https://modelcontextprotocol.io)
server so an MCP client (Claude Desktop, Claude Code, or any MCP-aware app) can drive
the discovery workflow as tools.

## Install & run

```bash
pip install "discovery-agents[mcp]"
discovery-agents-mcp            # serves over stdio
```

The server runs **keyless by default** (the deterministic mock provider). To back the
tools with a real model, set the provider's key and pass `provider` in the tool call,
e.g. `discovery_run(provider="anthropic")` with `ANTHROPIC_API_KEY` set.

## Tools

| Tool | Arguments | Returns |
|------|-----------|---------|
| `discovery_run` | `provider?` | Selected direction, all directions (+ citations), evals, trace totals |
| `evidence_search` | `query`, `k?` | Top-k evidence snippets with citation ids and scores |
| `eval_run` | `provider?` | Eval scorecard + regression-gate verdict |

The tool logic lives in plain functions in `discovery_agents/mcp_server.py`
(`run_discovery`, `evidence_search`, `run_eval`) and is unit-tested without the MCP
SDK; `build_server()` is the thin FastMCP wiring layer.

## Register with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "discovery-agents": {
      "command": "discovery-agents-mcp"
    }
  }
}
```

## Register with Claude Code

```bash
claude mcp add discovery-agents -- discovery-agents-mcp
```

Then ask Claude to "run the discovery workflow" or "search the evidence for the handoff gap".
