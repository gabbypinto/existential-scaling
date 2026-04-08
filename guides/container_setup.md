## .env File

Create an `.env` file in your root project directory

Change out MODEL, GPU_IDS/GPU_LABEL, and PORT fields 

```.env
# Model to serve (pick one)
MODEL=deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

# HuggingFace token (required for gated models)
HF_TOKEN=hf_...

# Which GPU pair to use (0,1 | 2,3 | 4,5 | 6,7)
GPU_IDS=0,1
GPU_LABEL=0.1

# Port to expose the OpenAI-compatible API on
PORT=10001

# Max context length (V100 32GB x2 = 64GB — keep <=32768 for 7-9B models)
MAX_MODEL_LEN=32768

# Fraction of GPU memory to use (0.90 is safe, lower if OOM)
GPU_MEM_UTIL=0.90
```


## Upload to Cluster
use `script/sync_cluster.sh` to upload files to cluster.

Something like `bash sync_cluster.sh` or `./sync_cluster.sh` works.


## Docker Container Setup

<!-- Run `docker compose up -d` to create a container on there -->

1. Start vLLM first, wait for it to load the model:  
`docker compose up vllm -d`  
`docker compose logs -f vllm`
 - wait until you see "Application startup complete

2. Create container that will run python code:  
`docker compose up eval -d`

3. You can watch logs/progress of your test with:  
`docker logs -f spencer_eval`

    Probably do this in a screen session