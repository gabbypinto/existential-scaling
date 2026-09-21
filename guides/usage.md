## Setup

1. Set `.env` (model slots, ports, GPU IDs - reference .env.example) and `src/configs/model.yaml` (inference params)
2. Upload to compute cluster: (also possibly change to just leverage vscode ssh and use git instead of this janky upload script)

```bash
# defaults to MLAT Cluster 07 (the one with 8x 32GB V100s)
bash scripts/sync_cluster.sh
bash scripts/sync_cluster.sh --host mlat_spark_01 --env-file .env.dgx_spark
# need to add support for DGX0 - 8x A100s

# need to add support for new ML2 Cluster - 4x RTX 6000 PROs

```

---

## Scripts

### `start_eval.sh`

#### Description

Run a single, specific benchmark

#### Arguments

```
--slot        slot number, corresponds to slot number in the .env file
--benchmark   benchmark name (required)
--limit N     only run first N problems
--follow      tail docker logs after launch
--timeout     seconds to wait for model ready (default: 600)
```

#### Benchmarks

```
aime24  aime25  gpqa  
lcb  lcb_pro  piqa_global
scicode  aa_omniscience  matharena_apex  
global_mmlu_lite  mmlu  social_iqa
```

#### Usage

```bash
# Smoke test of 2 questions on GPQA Diamond
bash scripts/start_eval.sh --benchmark gpqa --limit 2

# Full run on slot 2
bash scripts/start_eval.sh --slot 2 --benchmark aime24
```

---

### `run_all_benchmarks.sh`

#### Description

Run all benchmarks on one specified slot (defaults to slot 1) sequentially

#### Arguments

```
--slot N      slot to use from .env file
--skip list   benchmarks to skip
--limit N     only run first N problems per benchmark
--timeout     seconds to wait for model ready (default: 900)
```

#### Usage

```bash
# run all benchmarks on slot 3
bash scripts/run_all_benchmarks.sh --slot 3

# run all benchmarks on slot 1 but skip AIME 2024 and MMLU
bash scripts/run_all_benchmarks.sh --skip aime24,mmlu
```

---

### `run_batch_job.sh`

#### Description

Launch a screen session for every slot/model in .env and then run every benchmark on that respective model in parallel

#### Arguments

```
--skip    benchmarks to skip
--slots   slot numbers to run (default: all slots with MODEL_N set)
```

#### Usage

```bash
# Runs all models specified in .env with all benchmarks
bash scripts/run_batch_job.sh

# Runs slots 1 and 3 while skipping LCB Pro and SciCode
bash scripts/run_batch_job.sh --slots 1,3 --skip lcb_pro,scicode
```

---

### `run_multiple_prompts.sh`

#### Description

Runs all benchmarks specified for each prompt in `prompts.yaml` via separate screen sessions.

Logs land at `logs/{benchmark}/{model}/{prompt_key}/summary.json`.

#### Arguments

```
--slots       slots to run
--benchmarks  benchmarks to run
--prompts     path to prompts YAML
--limit       only run first N problems per benchmark
--timeout     seconds to wait for LLM service ready (default: 900)
--env-file    path to env file
--model       model config yaml
```

#### Usage

```bash
# Run all slots in .env with all benchmarks with each unique model/prompt combination
bash scripts/run_multiple_prompts.sh

# Run slots 1 and 3 using a subset of the benchmarks using different model parameters
bash scripts/run_multiple_prompts.sh --slots 1,3 --benchmarks aime24,aime25,mmlu_lite,gpqa --model config/model_2.yaml
```

---

### `convert_to_gguf.sh`

#### Description

Downloads a .safetensors format model from HuggingFace and converts it ot GGUF using llama.cpp, defaults ot Q8_0 weight quantization.

Clones `llama.cpp` into `~/llama.cpp` automatically if not present.

After conversion, add to `.env` via:
```
LOCAL_MODEL_1=/app/models/<family>/<model-name>-q8_0.gguf
```

#### Arguments

```
--model    HuggingFace repo ID (required)
--family   subdir under models/ (qwen3.5 results in models/qwen3.5/<name>.gguf)
--quant    weight quant (default: q8_0)
--llama    path to llama.cpp repo (default: ~/llama.cpp)
```

#### Usage

```bash
# convert Qwen3.5 9B Base from safetensors to GGUF
bash scripts/convert_to_gguf.sh --model Qwen/Qwen3.5-9B-Base --family qwen3.5

# convert Gemma 4 E4B to GGUF at F16 quant
bash scripts/convert_to_gguf.sh --model google/gemma-4-E4B-it --family gemma4 --quant f16
```

---

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

### `delete_all_screens.sh`

#### Description

Quits every active screen session and wipes dead ones.

#### Usage

```bash
bash scripts/delete_all_screens.sh
```

---

### `src/analysis/prompt_stats.py`

#### Description

Statistical tests for system-prompt effects (the Phase 1 method from the 07/17 slides). Two modes:

- `aggregate`: one value per model × benchmark × prompt. Friedman test across prompts (blocks = model × benchmark) for output tokens and accuracy. Also sign tests of the reference prompt against its cell average, and sign and Wilcoxon tests of the reference against each other prompt, raw and Holm-corrected. Also Pearson r between tokens and accuracy per cell. It can write the pgfplots `.dat` files used by the Phase 1 figures.
- `bullshit`: per-question BullshitBench judge scores. Per model it reports the mean score, clear-pushback rate, and no-answer count per prompt. It runs a Friedman test across prompts on per-question scores. It compares each prompt with a reference prompt using Wilcoxon on scores and exact McNemar on clear pushback, raw and Holm-corrected.

#### Usage

```bash
# on your own runs: first roll up the logs, then test
python scripts/extract_metrics.py --logs-dir logs
python src/analysis/prompt_stats.py aggregate --metrics metrics_summary.json --benchmarks aime24,aime25,gpqa,global_mmlu_lite
python src/analysis/prompt_stats.py aggregate --metrics metrics_summary.json --dat-dir plots/phase1   # + pgfplots tables

# reproduce Emma's 07/17 numbers from the results-sheet export
python src/analysis/prompt_stats.py aggregate --csv src/analysis/data/phase1_emma.csv

# BullshitBench (after grading)
python src/analysis/prompt_stats.py bullshit --logs-dir logs/bullshit_bench --by domain_group
python src/analysis/prompt_stats.py bullshit --logs-dir logs/bullshit_bench --reference none   # vs no system prompt
```

#### Key flags

```
--reference    prompt compared against the others (aggregate default: Collapse; bullshit default: Baseline; 'none' = no system prompt)
--acc-key      metrics_summary field used as accuracy (e.g. mean_score for bullshit_bench)
--benchmarks / --models / --prompts   filters (comma-separated)
--include-none include the no-system-prompt run as a condition in aggregate mode
--by           bullshit mode: break mean score down by domain_group or technique
```
