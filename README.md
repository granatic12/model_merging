# Model merging

This repository contains the code, prepared data, raw measurements, aggregated results, and task-vector analyses accompanying the manuscript **“What Survives a Merge? Amplitude, Coverage, and an Adapter-Geometry Confound in Data-Free Composition of Domain and Task Updates.”** The study considers a decoupled-update setting in which a domain update obtained by continued pre-training and task-specific LoRA updates must be composed without retraining on the original task data. All update artifacts are represented relative to a common base-model anchor before merging.

The experiments use `unsloth/Qwen3-4B-Instruct-2507` as the common base model. The official model card describes it as a 4B-parameter Qwen3 instruction model; the repository uses its 4-bit Unsloth loading path for training and evaluation.

The artifact covers:

- continued pre-training with LoRA;
- task-specific LoRA training for HDFS log anomaly detection and technical QA;
- CPT → task fine-tuning references;
- multitask baselines;
- Task Arithmetic, Model Soup, TIES, DARE, and their compositions;
- sequential adapter merging;
- hyperparameter sweeps;
- task-vector cosine similarity, sign conflicts, norms, and sparsity diagnostics;
- per-seed raw metrics and aggregated tables.

The main experiment uses seeds `42`, `123`, and `777`. The technical QA branch is retained as a **generation-quality probe rather than evidence for successful task composition**, because the QA adapter did not learn a valid specialist in the reported experiment; this limitation is part of the manuscript's analysis rather than being hidden from the artifact.

## Repository structure

The repository has the following structure:

```text
├── 00_setup_common.ipynb
├── 01_prepare_data.ipynb
├── 02_train_cpt_lora.ipynb
├── 03_train_sft_lora_base.ipynb
├── 04_train_sft_lora_on_cpt.ipynb
├── 05_eval_baselines.ipynb
├── 06_task_vector_analysis.ipynb
├── 07_merge_adapters.ipynb
├── 08_eval_merged_and_grid.ipynb
├── 09_aggregate_results.ipynb
├── exp_common.py
├── data/
│   └── .gitkeep
└── artifacts/
    ├── prepared_data/
    │   ├── cpt_train.jsonl
    │   ├── cpt_val.jsonl
    │   ├── logs_train_sft.jsonl
    │   ├── logs_val_sft.jsonl
    │   ├── logs_test_eval.jsonl
    │   ├── qa_train_sft.jsonl
    │   ├── qa_val_sft.jsonl
    │   ├── qa_test_eval.jsonl
    │   ├── multitask_train_sft.jsonl
    │   └── data_manifest.json
    ├── adapters/               # generated LoRA adapters
    │   └── .gitkeep
    ├── merged_adapters/        # generated merged adapters
    │   └── .gitkeep
    ├── analysis/
    │   ├── task_vectors_all.csv
    │   ├── task_vectors_summary.csv
    │   ├── task_vectors_seed_*_per_tensor.csv
    │   ├── task_vectors_seed_*_summary.json
    │   ├── zeroing_after_methods.csv
    │   ├── cosine_hist.png
    │   ├── conflict_frac_hist.png
    │   ├── logs_l2_hist.png
    │   └── qa_l2_hist.png
    └── results/
        ├── all_metrics_raw.csv
        ├── all_metrics_mean_std.csv
        ├── main_report_mean_std.csv
        ├── main_report_mean_std.md
        ├── baselines/
        │   ├── baseline_metrics_raw.csv
        │   └── baseline_metrics_mean_std.csv
        └── merged/
            ├── merged_metrics_raw.csv
            ├── merged_metrics_mean_std.csv
            ├── grid_metrics_raw_42.csv
            ├── grid_metrics_raw_123.csv
            ├── pareto_candidates_42.csv
            └── pareto_candidates_123.csv
```

- **Notebooks**: Each numbered notebook contains code for a stage of the experiment (see below for details).
- **exp_common.py**: Shared Python code for model loading, LoRA setup, training, evaluation, merging, and analysis.
- **data/**: Contains a `.gitkeep` placeholder (no raw data files are provided here).
- **artifacts/prepared_data/**: Contains all pre-processed JSONL datasets. Use these directly for reproducibility (see below).
- **artifacts/adapters/** and **artifacts/merged_adapters/**: Contain `.gitkeep` placeholders. These directories will be filled with generated LoRA checkpoint files when you run the training notebooks. (No pre-trained checkpoints are included in the repository.)
- **analysis/** and **results/**: Contain CSVs, PNGs, and aggregated results produced by the notebooks.

## Notebook descriptions

- **`00_setup_common.ipynb`**: Verify the environment, install dependencies, and set up common directories.  
- **`01_prepare_data.ipynb`**: Sample and format raw data. **Not needed if you use `artifacts/prepared_data/` directly.**  
- **`02_train_cpt_lora.ipynb`**: Train the LoRA adapter for continued pre-training (CPT) for each seed.  
- **`03_train_sft_lora_base.ipynb`**: Train LoRA adapters on the base model for Logs, QA, and Multitask.  
- **`04_train_sft_lora_on_cpt.ipynb`**: Fine-tune new task adapters (Logs and QA) on top of the merged CPT adapter.  
- **`05_eval_baselines.ipynb`**: Evaluate baseline models (base, CPT, fine-tuned, multitask) on the test sets.  
- **`06_task_vector_analysis.ipynb`**: Analyze the geometric properties of the LoRA adapter vectors (cosine similarity, norms, sparsity).  
- **`07_merge_adapters.ipynb`**: Create merged adapters using various methods (Task Arithmetic, Model Soup, TIES, DARE, and their combinations).  
- **`08_eval_merged_and_grid.ipynb`**: Evaluate merged adapters and perform a hyperparameter sweep over merge weights and sparsity.  
- **`09_aggregate_results.ipynb`**: Aggregate raw metrics and produce summary tables.

## Environment and installation

Use the following environment to reproduce the experiments. In summary:

| Component              | Version/Value                      |
|------------------------|------------------------------------|
| Python                 | 3.10.11                            |
| CUDA (runtime)         | 12.8                               |
| PyTorch                | 2.10.0+cu128                       |
| Transformers           | 5.5.0                              |
| Unsloth                | 2026.7.1                           |
| xFormers               | 0.0.35                             |
| Training precision     | bfloat16                           |

To set up a matching environment, you can use the provided pip commands. Example (from repository README):

```bash
python3.10 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools

# Install core dependencies
pip install torch==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

pip install \
    "unsloth==2026.7.1" \
    "transformers==5.5.0" \
    "xformers==0.0.35"

pip install \
    trl \
    peft \
    accelerate \
    bitsandbytes

pip install \
    datasets \
    pandas \
    numpy

pip install \
    safetensors \
    sentence-transformers \
    scikit-learn

pip install \
    matplotlib \
    tqdm
```

Run the following to verify your installation:

```python 
import torch
import transformers
import unsloth
import trl
import peft
import accelerate
import datasets

print(f"PyTorch:           {torch.__version__}")
print(f"CUDA runtime:      {torch.version.cuda}")
print(f"CUDA available:    {torch.cuda.is_available()}")
print(f"BFloat16 support:  {torch.cuda.is_bf16_supported()}")
print(f"Transformers:      {transformers.__version__}")
print(f"Unsloth:           {unsloth.__version__}")
print(f"TRL:               {trl.__version__}")
print(f"PEFT:              {peft.__version__}")
print(f"Accelerate:        {accelerate.__version__}")
print(f"Datasets:          {datasets.__version__}")
```

Expected output:  
```
PyTorch:           2.10.0+cu128
CUDA runtime:      12.8
CUDA available:    True
BFloat16 support:  True
Transformers:      5.5.0
Unsloth:           2026.7.1
TRL:               0.24.0
PEFT:              0.19.1
Accelerate:        1.14.0
Datasets:          4.3.0
```


## Common model and training configuration

Unless overridden in a notebook, the shared default configuration (`exp_common.py`) is:

```text
Base model:        unsloth/Qwen3-4B-Instruct-2507
Seeds:             42, 123, 777
Maximum sequence:  2048
Quantization:      4-bit model loading
Training dtype:    bfloat16

LoRA rank:         16
LoRA alpha:        32
LoRA dropout:      0.05
LoRA bias:         none

Target modules:
  q_proj
  k_proj
  v_proj
  o_proj
  gate_proj
  up_proj
  down_proj

Optimizer:         AdamW 8-bit
LR scheduler:      cosine
Epochs:            1
Default batch:     2
Gradient accum:    8  (effective batch size = 16 per GPU)
Checkpoint criterion: lowest validation loss
```

## Data and checksums

All downstream experiments use the pre-processed JSONL files in `artifacts/prepared_data/`. Do **not** rerun `01_prepare_data.ipynb` unless you have the original raw data; instead, use these files directly.

The prepared datasets are:

| File                       | Rows  | Purpose                                |
|----------------------------|------:|----------------------------------------|
| `cpt_train.jsonl`          | 9750  | CPT (continued pre-training) train     |
| `cpt_val.jsonl`            | 250   | CPT validation                         |
| `logs_train_sft.jsonl`     | 39000 | HDFS anomaly train (fine-tuning logs)  |
| `logs_val_sft.jsonl`       | 1000  | HDFS anomaly validation                |
| `logs_test_eval.jsonl`     | 5000  | HDFS anomaly test pool                 |
| `qa_train_sft.jsonl`       | 8000  | QA fine-tuning train                   |
| `qa_val_sft.jsonl`         | 500   | QA validation                          |
| `qa_test_eval.jsonl`       | 500   | QA test                               |
| `multitask_train_sft.jsonl`| 47000 | Combined Logs+QA train                 |

The main evaluation uses **first 2000** examples from `logs_test_eval.jsonl` and **all 500** examples from `qa_test_eval.jsonl`.

### Checksum verification

To ensure data integrity, the repository provides SHA-256 checksums for each prepared file. Before running experiments, you can verify them:

```bash
sha256sum artifacts/prepared_data/*
```

Expected SHA-256 sums:

```text
503b69563b6ea24ba42bfa3b96d3cb27f8f3d0c89124919b04fd22eb5ab8237f *artifacts/prepared_data/cpt_train.jsonl
75097f0fe1bd7687992792c3c6cd2af29602e38c7bab77cf71fb18dd74d151b5 *artifacts/prepared_data/cpt_val.jsonl
64a18e23b9de899e2f27d5f72abdecd5efba0210d7492f03f51967a78fd4172e *artifacts/prepared_data/data_manifest.json
a89890497dd8d11b4cd0245665fb14d119aead4f678eeb7eac2485c98c0352e5 *artifacts/prepared_data/logs_test_eval.jsonl
a2e91f670b258add5f3f65f69ff53638188eed8133ae4ea03b2f26d353fcc540 *artifacts/prepared_data/logs_train_sft.jsonl
c85b2c17ea1768a200bf3ecb960020dc99b22c8acfe2b14d82344bd340ba9252 *artifacts/prepared_data/logs_val_sft.jsonl
69c465dae6be0587de265ec525ba8fbf83f2c24ebf01844cc99b0b6303788e78 *artifacts/prepared_data/multitask_train_sft.jsonl
f26aeea8600551187f6743e0358b7053d9e05952382186953e1ca7ba2b965e17 *artifacts/prepared_data/qa_test_eval.jsonl
78b8211ca1f6aebcbe26b44b70ff7eaf62f2e6ff5ce8177640d35389cf56f5a1 *artifacts/prepared_data/qa_train_sft.jsonl
1866308940063a4740bf398bb427a9bc52dab17f0552d90aa558c69261ee407f *artifacts/prepared_data/qa_val_sft.jsonl
```

You can also verify the number of lines:

```bash
wc -l artifacts/prepared_data/*.jsonl
```

## Running the experiments

Make sure to use the **repository root as the current working directory** when running notebooks, since paths in `exp_common.py` are relative to `ROOT = Path.cwd()`.

A typical workflow:

1. **Start Jupyter:**  
   ```
   cd /path/to/repo
   source .venv/bin/activate
   jupyter lab
   ```
2. **Set GPU device:**  
   Some notebooks set `CUDA_VISIBLE_DEVICES` manually (e.g., to `"5"` or `"6"`). Before running, edit those cells to use a valid GPU index on your machine (e.g., `"0"` if you have one GPU). Do this **before** any cell imports `exp_common` or PyTorch.
3. **Run notebooks in order (skip 01):**  
   Execute the notebooks in the following sequence:

   ```
   00_setup_common.ipynb
           ↓
   02_train_cpt_lora.ipynb
           ↓
   03_train_sft_lora_base.ipynb
           ↓
   04_train_sft_lora_on_cpt.ipynb
           ↓
   05_eval_baselines.ipynb
           ↓
   06_task_vector_analysis.ipynb
           ↓
   07_merge_adapters.ipynb
           ↓
   08_eval_merged_and_grid.ipynb
           ↓
   09_aggregate_results.ipynb
   ```

   - **Skip `01_prepare_data.ipynb`** if you are using the provided `artifacts/prepared_data/` files (recommended).  
   - After training notebooks (02–04), checkpoint files will be saved under `artifacts/adapters/` (e.g., `artifacts/adapters/cpt/seed_42/`, etc.).  
   - `05_eval_baselines.ipynb` evaluates all baseline models.  
   - `06_task_vector_analysis.ipynb` computes vector geometry diagnostics.  
   - `07_merge_adapters.ipynb` generates merged adapters (Task Arithmetic, Model Soup, TIES, DARE, etc.).  
   - `08_eval_merged_and_grid.ipynb` evaluates merged models and runs a sweep.  
   - `09_aggregate_results.ipynb` combines results into CSV/MD summary tables.

## Evaluation settings

- **Evaluation set sizes:**  
  - Logs (anomaly detection): `EVAL_LOGS_N = 2000`  
  - QA: `EVAL_QA_N = 500`  

- **Batch sizes:**  
  - Logs evaluation: `BATCH_LOGS = 16`  
  - QA evaluation: `BATCH_QA = 4`  

- **Log evaluation:** Decoding is greedy with up to 8 tokens. Outputs containing anomaly-related keywords are labeled as `Anomaly`; others as `OK` (see `05_eval_baselines.ipynb` code).  
- **QA evaluation:** Semantic similarity is computed using the `sentence-transformers/all-MiniLM-L6-v2` model. QA responses are additionally evaluated using an LLM-as-Judge approach with the `unsloth/Llama-3.1-8B-Instruct` model configured in the current evaluation code.

## Hyperparameter sweep

Notebook `08_eval_merged_and_grid.ipynb` runs a parameter sweep with:

```python
ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.2]
TOP_KS = [0.1, 0.2, 0.3, 0.5]
DROP_RATES = [0.2, 0.4, 0.6]
```

Sweep is done for seeds `42` and `123`. Output CSVs (grid metrics and Pareto candidates) are in `artifacts/results/merged/`.

## Files intentionally omitted

- The repo **does not include** trained LoRA checkpoint files in `artifacts/adapters/` or `artifacts/merged_adapters/`. These will be generated when you run the training notebooks.
- Directories `data/`, `artifacts/adapters/`, and `artifacts/merged_adapters/` contain `.gitkeep` placeholders to ensure they appear in the repo even if empty.

## Troubleshooting & notes
  
- **ALPHA parameter:** The `07_merge_adapters.ipynb` default cell sets `ALPHA = 0.8`.  

- **Reproducing archived results:**  
  The files in `artifacts/results/` contain the saved CSVs used in the manuscript. For example, `artifacts/results/all_metrics_raw.csv` has the full set of raw metrics for each model and seed. You can load these CSVs directly (no training needed) to recompute tables or plots. For instance:

## Reproducibility checklist

1. **Verify data**: Check SHA-256 sums and row counts of `artifacts/prepared_data/*`.  
2. **Set up environment**: Create Python 3.10 venv and install dependencies. Verify versions.  
3. **Run notebooks in order** (skipping 01): 00 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09.  
4. **Monitor outputs**: Training notebooks will write LoRA adapter files to `artifacts/adapters/`; merged adapters go to `artifacts/merged_adapters/`.  
6. **Compute metrics**: Evaluation notebooks will write CSVs under `artifacts/results/`.  
7. **Check merged results**: Inspect `artifacts/results/merged/merged_metrics_mean_std.csv` for final metrics.  

Following the above steps using the provided code and data will reproduce the experiment results without requiring additional data or external resources.