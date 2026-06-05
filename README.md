# Local Micro Agent

Ultra-light local coding-agent skeleton for M2 Max 64GB class machines.

The design goal is to avoid the prompt bloat of full agent frameworks by
running a small finite-state workflow:

1. `PLAN`: produce a compact file/action plan.
2. `READ`: load only planned source files through MCP.
3. `CODE`: generate strict patch/write operations from the plan and source.
4. `TEST`: run configured commands through MCP and loop back on failure.

The implementation is intentionally small. It is a scaffold for experiments
with local models such as Qwen 3.6 35B/27B, Ollama, llama-server, vLLM, or
commercial APIs behind an OpenAI-compatible endpoint.

## Files

- `config.example.yaml`: provider and MCP server configuration.
- `src/local_micro_agent/orchestrator.py`: FSM runner.
- `src/local_micro_agent/state.py`: single global state bag.
- `src/local_micro_agent/models.py`: model-manager abstraction.
- `src/local_micro_agent/mcp_client.py`: async stdio MCP adapter skeleton.
- `src/local_micro_agent/prompts.py`: micro system prompts per state.
- `src/local_micro_agent/validators.py`: JSON validation/retry helpers.

## Smoke

```bash
python3 -m compileall projects/local-micro-agent/src
```

## Next Practical Step

Wire the MCP server commands in `config.example.yaml` to the installed
filesystem and command-executor MCP servers on the target host, then point
`models.default` at the preferred local OpenAI-compatible server.
