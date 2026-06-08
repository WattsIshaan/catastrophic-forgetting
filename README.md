# Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting

**Authors:** Ishaan Watts*, Catherine Li*, Sachin Goyal, Jacob Mitchell Springer, Aditi Raghunathan

**[Paper](https://arxiv.org/abs/2605.02105)** | **[Code](https://github.com/WattsIshaan/sharpness-aware-pretraining)**  | **[Website]()**

## Installation

The codebase uses two conda environments: `forgetting` for training and launching experiments, and `olmes`
for downstream task evaluation.

### Prerequisites

- Python 3.13
- CUDA-capable GPU with NVIDIA driver ≥ 12.8 (PyTorch is pinned to CUDA 12.8)
- [gcloud SDK](https://cloud.google.com/sdk/docs/install) with `gsutil` — data and checkpoints use Google Cloud Storage

### 1. Training and launch environment (`forgetting`)

```bash
conda create -n forgetting python=3.13 -y
conda activate forgetting

git clone https://github.com/WattsIshaan/sharpness-aware-pretraining.git
cd sharpness-aware-pretraining

pip install -U pip setuptools wheel
pip install -e ".[all,launch]" --extra-index-url https://download.pytorch.org/whl/cu128

pip install flash-attn==2.8.3 --no-build-isolation
```

### 2. Downstream evaluation environment (`olmes`)

Downstream benchmarks (MMLU, core MCQA, generative tasks, etc.) run via
[OLMES](https://github.com/allenai/olmes). Launch scripts activate this environment automatically.

```bash
conda create -n olmes python=3.10 -y
conda activate olmes

pip install git+https://github.com/allenai/olmes.git
```

### 3. Google Cloud Storage

Authenticate before running pipelines that read/write checkpoints and data:

```bash
gcloud auth login
```

Configure bucket names and paths in `launch/globals.py` (via the `experiments` project config).

## Running experiments

Each `launch/run_*.py` script builds a set of artifacts and submits them. Pass `launch` to submit jobs
or `printlines` for a dry run that prints the planned commands.

```bash
conda activate forgetting

# Submit pipelines
python launch/run_adamw.py launch        # AdamW small-scale pretraining + sft + eval
python launch/run_sam.py launch          # SAM small-scale pretraining + sft + eval
python launch/run_midtrain.py launch     # 1B midtraining + sft + eval
```

### Launching an individual stage

Each pipeline is split into named **stages** via the `executor.stage('<name>', artifacts)` calls at the
bottom of each `launch/run_*.py`. Pass one or more stage names to `launch` (or `drylaunch`) to run only
those; omit them to run everything.

```bash
# Run only the pretraining stage
python launch/run_adamw.py launch pretrain
```

## Where artifacts are stored

All artifacts are written to **Google Cloud Storage**, under the project's bucket root (`GS_PATH`, set in
the `experiments` project config and read in `launch/globals.py`). The project is selected by the
`Project.init(...)` call at the top of each runner (e.g. `60m-experiments`, `1b-experiments`), so paths
look like:

```
gs://<bucket>/outputs/<project>/<ArtifactType>/<run_name>/...
# e.g. gs://cmu-gpucloud-iwatts/outputs/60m-experiments/PretrainedModel/<run_name>/final-unsharded
```

Each artifact type lives in its own subfolder (the `relpath` property on each class in
`launch/artifacts.py`).

Evaluation results can be pulled down with `gsutil`, e.g.:

```bash
gsutil cp -r gs://<bucket>/outputs/<project>/ModelEvaluation results/
```

