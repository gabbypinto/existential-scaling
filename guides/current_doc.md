## Current Testing Commands

- Need to redo documentation and update

0. Set .env and .yaml values

1. Upload to Compute Cluster

    `bash scripts/sync_cluster.sh`

2. Smoke Test with n Questions

    `bash scripts/start_eval.sh --benchmark gpqa --limit 2 --follow`

    ```bash
    --benchmark options

    aime25
    lcb
    gpqa
    piqa_global
    scicode
    ```

3. Actual Full Run

    slot 1, 2, 3, and 4 correspond to the .env setup for that respective slot

    `bash scripts/start_eval.sh --slot 1 --benchmark gpqa`

    `bash scripts/start_eval.sh --slot 2 --benchmark lcb`


