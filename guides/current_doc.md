## Setup

1. Set `.env` (model slots, ports, GPU IDs - reference .env.example) and `src/configs/model.yaml` (inference params)
2. Upload to compute cluster (need to update with the new RTX 6000 cluster):
   ```bash
   bash scripts/sync_cluster.sh
   bash scripts/sync_cluster.sh --host mlat_spark_01 --env-file .env.dgx_spark
   ```

---

## Scripts

### `start_eval.sh` — run a single benchmark on one slot

Starts the LLM service for a slot, waits for it to load, then runs one benchmark. Reads `MODEL_N` / `PORT_N` from `.env`.

```bash
# Smoke test — 2 questions, tail logs
bash scripts/start_eval.sh --benchmark gpqa --limit 2 --follow

# Full run on slot 2
bash scripts/start_eval.sh --slot 2 --benchmark aime24

# Key flags
--slot        slot number (default: 1)
--benchmark   benchmark name (required) — see list below
--limit N     only run first N problems
--follow      tail docker logs after launch
--timeout     seconds to wait for model ready (default: 600)
```

Available benchmarks:
```
aime24  aime25  gpqa  
lcb  lcb_pro  piqa_global
scicode  aa_omniscience  matharena_apex  
global_mmlu_lite  mmlu
```

---

### `run_all_benchmarks.sh` — run all benchmarks on one slot, sequentially

Starts the slot's LLM service once, then runs every benchmark one after another. Only works with a single model.

```bash
bash scripts/run_all_benchmarks.sh --slot 3
bash scripts/run_all_benchmarks.sh --slot 4 --skip aime24,mmlu
bash scripts/run_all_benchmarks.sh --slot 1 --limit 2   # smoke test

# Key flags
--slot N      slot to use (default: 1)
--skip list   comma-separated benchmarks to skip
--limit N     only run first N problems per benchmark
--timeout     seconds to wait for model ready (default: 900)
```

---

### `run_batch_job.sh` — run all slots in parallel screen sessions

Reads every `MODEL_N` defined in `.env` and launches a `screen` session per slot, each running `run_all_benchmarks.sh`. All models run simultaneously.

```bash
bash scripts/run_batch_job.sh
bash scripts/run_batch_job.sh --skip lcb_pro,scicode
bash scripts/run_batch_job.sh --slots 1,3        # specific slots only

# Monitor
screen -ls                  # list sessions
screen -x <session-name>    # attach
# Ctrl-A D to detach
```

---

### `run_multiple_prompts.sh` — system prompt sweep across all prompt variants

Runs all benchmarks specified for each prompt in `src/configs/prompts.yaml`. Launches one `screen` session per model slot so all models run in parallel.

Will probably change this so that system prompts override? the pre and post prompts and gauge results again?

```bash
bash scripts/run_multiple_prompts.sh
bash scripts/run_multiple_prompts.sh --limit 2                 # smoke test
bash scripts/run_multiple_prompts.sh --slots 1,3               # specific slots
bash scripts/run_multiple_prompts.sh --benchmarks aime24,aime25,mmlu_lite,gpqa  # subset of benchmarks
bash scripts/run_multiple_prompts.sh --prompts src/configs/prompts.yaml

# Key flags
--slots        comma-separated slot numbers (default: all slots with MODEL_N set)
--benchmarks   comma-separated benchmarks (default: aime24,aime25,gpqa,global_mmlu_lite)
--prompts      path to prompts YAML (default: src/configs/prompts.yaml)
--limit N      only run first N problems per benchmark
--timeout      seconds to wait for model ready (default: 900)
```

Logs land at `logs/{benchmark}/{model}/{prompt_key}/summary.json`.

---
<!-- 
### `run_model_list.sh` — iterate over a `model_list.json`

Runs all benchmarks for each model defined in a JSON list, either sequentially (default) or in parallel. Each model gets a temp `.env` and model config generated automatically.

```bash
bash scripts/run_model_list.sh                         # sequential, uses model_list.json
bash scripts/run_model_list.sh --list my_list.json
bash scripts/run_model_list.sh --limit 2 --dry-run     # preview without running
bash scripts/run_model_list.sh --parallel              # all models simultaneously
```

`model_list.json` format:
```json
{
  "slot": 1,
  "skip": "lcb_pro,scicode",
  "env_file": ".env",
  "models": [
    { "hf_repo": "Qwen/Qwen3-8B-GGUF", "weights": "Qwen3-8B-Q8_0.gguf" },
    { "hf_repo": "Qwen/Qwen3.5-9B-GGUF", "weights": "Qwen3.5-9B-Q8_0.gguf" }
  ]
}
```

--- -->

### `convert_to_gguf.sh` — convert a HuggingFace model to GGUF

Downloads a model from HuggingFace, converts it with `llama.cpp`, saves the GGUF to `models/`, and cleans the HF cache. 

Clones `llama.cpp` into `~/llama.cpp` automatically if not present.

Defaults to Q8_0 weight quantization. 

```bash
bash scripts/convert_to_gguf.sh --model Qwen/Qwen3.5-9B-Base --family qwen3.5
bash scripts/convert_to_gguf.sh --model google/gemma-4-E4B-it --family gemma4 --quant f16

# Key flags
--model    HuggingFace repo ID (required)
--family   subdir under models/ (qwen3.5 results in models/qwen3.5/<name>.gguf)
--quant    q8_0 or f16 (default: q8_0)
--llama    path to llama.cpp repo (default: ~/llama.cpp)
```

After conversion, add to `.env`:
```
LOCAL_MODEL_1=/app/models/<family>/<model-name>-q8_0.gguf
```

<!-- ---

### `smoke_test.sh` — verify a model slot loads and responds

Pulls the llama.cpp Docker image, starts a slot, waits for it to be ready, then sends a single "Reply with one word: hello" request to confirm the model is responding.

```bash
bash scripts/smoke_test.sh --slot 1
bash scripts/smoke_test.sh --slot 2 --stop-after   # stop container after test passes
bash scripts/smoke_test.sh --slot 1 --timeout 1200

# Key flags
--slot        slot to test (default: 1)
--stop-after  stop the LLM container once the test passes
--timeout     seconds to wait for model to load (default: 900)
```

--- -->

### `sync_cluster.sh` — rsync project to compute cluster

Uploads `src/`, `scripts/`, `docker-compose.yml`, and `requirements.txt` to the remote host. Syncs the specified `.env` as `.env` on the remote.

```bash
bash scripts/sync_cluster.sh # defaults to MLAT 07 cluster
bash scripts/sync_cluster.sh --host mlat_spark_02
bash scripts/sync_cluster.sh --host mlat_spark_02 --env-file .env.dgx_spark
```

<!-- ---

### `extract_metrics.py` — extract accuracy + latency from `logs_new/`

Walks `logs_new/{benchmark}/{model}/{system_prompt}/summary.json` and writes a structured JSON summary. Also prints a quick table to stdout.

```bash
python extract_metrics.py
# → writes metrics_summary.json
```

Output structure:
```json
{
  "model-name": {
    "benchmark": {
      "SystemPrompt": {
        "accuracy": 0.43,
        "questions_passed": 13,
        "total_questions": 30,
        "avg_tokens_per_sec": 75.5,
        "avg_elapsed_s": 40.5,
        "avg_completion_tokens": 10995,
        "avg_total_tokens": 11160,
        "avg_thinking_tokens": 0.0
      }
    }
  }
}
``` -->

<!-- ---

### `test_benchmark_load.py` — verify a benchmark dataset loads correctly

Loads a benchmark locally and prints the first question. Optionally runs a few questions against a running local server.

```bash
# Just test dataset loading (no server needed)
python scripts/test_benchmark_load.py --benchmark lcb_pro

# Load + run 2 questions against a local server on port 20003
python scripts/test_benchmark_load.py --benchmark aa_omniscience --limit 2 --port 20003
```

--- -->

---

### `delete_all_screens.sh` — kill all screen sessions

Quits every active `screen` session and wipes dead ones.

```bash
bash scripts/delete_all_screens.sh
```
