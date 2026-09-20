# Running an Ollama server for the pipelines (SLURM / OSC)

Rules for every job that starts its own Ollama server (multi-agent pipeline,
simple-prompt, skills). Reference implementation:
`jobs/job_skills_qwen36.sh` and `jobs/job_pipeline_qwen36.sh` in the benchmark
working folders.

## 1. Use a random port derived from the SLURM job id

Never hardcode `11434`. Several jobs can land on the same node, and a fixed port
makes them collide: one server dies with "address already in use" while the other
jobs silently talk to a neighbour's server.

A job id cannot be used as-is (7 digits, ports max out at 65535), so fold it into a
port range:

```bash
PORT_BASE=$(( 20000 + ${SLURM_JOB_ID:-$$} % 12000 ))   # 20000-31999
```

- `12000` keeps the port below the Linux ephemeral range (starts at 32768), so an
  outgoing connection can't already own it. Older scripts use `% 20000`, which
  reaches 39999 and can overlap that range.
- Jobs with nearby ids (everything running at the same time) get different ports,
  deterministically.
- If the port is still busy (a foreign process, or two ids that differ by a
  multiple of 12000), step to the next free one:

```bash
while ss -tln 2>/dev/null | grep -q ":${OLLAMA_PORT} "; do
  OLLAMA_PORT=$(( OLLAMA_PORT + 1 ))
done
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
```

- Wait for readiness with `curl http://127.0.0.1:${OLLAMA_PORT}/api/tags`, and use
  `kill -0 "$OLLAMA_PID"` in the loop so the job fails fast if **its own** server
  crashed instead of adopting a neighbour's.
- Hand the same port to the client: `OLLAMA_BASE_URL` for the LangChain pipeline,
  `ANTHROPIC_BASE_URL` for Claude Code (skills).

## 2. Set the server context length to 131072

Start the server with:

```bash
apptainer exec --nv \
  --env OLLAMA_MODELS=/users/PCON0100/feng1426/ollama/models \
  --env OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}" \
  --env OLLAMA_CONTEXT_LENGTH=131072 \
  /users/PCON0100/feng1426/ollama/ollama-old.sif ollama serve &
```

Keep the value in one variable (`OLLAMA_CTX`, default `131072`) so it is easy to
override per job: `sbatch --export=ALL,PMID=...,OLLAMA_CTX=65536 ...`.

Where the context length actually comes from depends on the client:

| Client | Context in effect |
|---|---|
| **Skills** (Claude Code, `claude -p`) | The server's `OLLAMA_CONTEXT_LENGTH`. Also export `CLAUDE_CODE_MAX_CONTEXT_TOKENS` with the **same** value: Claude Code doesn't recognize the model and would otherwise assume 200k, overrunning the server's window and silently triggering llama-server's context shift. |
| **Simple-prompt** (`scripts/simple_prompt/`) | The server's `OLLAMA_CONTEXT_LENGTH` (the client sets no `num_ctx`). |
| **Multi-agent pipeline** (`app_script_pmids.py`) | The **client's** `num_ctx`, not the server's. The pipeline agents send their own `num_ctx` with every request, and a per-request value overrides the server default. It is set in `extractor/agents/agent_factory.py` (`MAX_PIPELINE_AGENT_CONTENT_NUM`, 48k = 49152; override with `PIPELINE_AGENT_NUM_CTX`). The server's 131072 only serves as the default for requests that don't send one. |

## 3. Always use `ollama-old.sif` for qwen3.6

`ollama.sif` (v0.33.0) has a CUDA illegal-memory-access bug with qwen3.6
(`ollama/ollama#17434`, `ggml-org/llama.cpp#21383`). `ollama-old.sif` (v0.20.0) does
not. Use it for every qwen3.6 job.

## 4. Quick check inside a job

Log the values at the start of the job so a run can be audited later:

```
[INFO] pipeline qwen36  PMID=...  node=...  job=...  port=38326
[INFO] server context=131072; pipeline client num_ctx=code default (49152)
```
