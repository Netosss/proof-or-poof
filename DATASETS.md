# Datasets (Local Only)

This repo intentionally does **not** include the benchmark datasets (images/videos).

They are large and should live **outside the git repo** (e.g. on your Desktop).

## Recommended location

Put your datasets here:

- `/Users/netanel.ossi/Desktop/ai-detector-datasets`

Example structure (same as the old `tests/data/`):

```
ai-detector-datasets/
  benchmark_50/
  benchmark_hf_50/
  hf_200_benchmark/
  kaggle_benchmark/
  screenshots/
  screen_records/
  ai/
  original/
```

## Configure scripts

Set this env var so scripts use your external dataset folder:

```bash
export AI_DETECTOR_DATASETS_ROOT="/Users/netanel.ossi/Desktop/ai-detector-datasets"
```

If the env var is not set, scripts fall back to `tests/data` (which should remain empty/untracked).


