# Model merging

This repository contains the code, prepared data, raw measurements, aggregated results, and task-vector analyses accompanying the manuscript **“What Survives a Merge? Auditing Data-Free Composition of Domain and Task Updates.”** The study considers a decoupled-update setting in which a domain update obtained by continued pre-training and task-specific LoRA updates must be composed without retraining on the original task data. All update artifacts are represented relative to a common base-model anchor before merging.

The experiments use `unsloth/Qwen3-4B-Instruct-2507` as the common base model. The repository uses its 4-bit loading path; evaluation applies adapters on top of the frozen quantized base rather than performing a full dequantized model merg

The artifact covers:

- continued pre-training with LoRA;
- task-specific LoRA training for HDFS log anomaly detection and technical QA;
- CPT → task fine-tuning references;
- multitask baselines;
- factor-space and Delta-W merging experiments;
- Task Arithmetic, Model Soup, TIES, DARE, and their compositions;
- sequential adapter merging;
- coefficient sweeps over log-task update scaling;
- task-vector geometry and audit diagnostics;
- per-seed raw metrics and aggregated tables.

The main experiment uses seeds `42`, `123`, and `777`. The QA branch is treated as a generation-quality probe and is not used as evidence of successful task-task interference. The artifact also preserves unstable stochastic outcomes (including collapsed sparse-merge runs) instead of removing them.

## Repository structure

The repository has the following structure:

```text
├── artifacts
│   ├── adapters
│   │   ├── cpt
│   │   ├── logs
│   │   ├── logs_on_cpt
│   │   ├── multitask
│   │   ├── multitask_on_cpt
│   │   ├── qa
│   │   └── qa_on_cpt
│   ├── analysis
│   │   ├── coefficient_sweep
│   │   │   ├── coefficient_sweep_agg.csv
│   │   │   ├── coefficient_sweep_all.csv
│   │   │   ├── coefficient_sweep_seed123.csv
│   │   │   ├── coefficient_sweep_seed42.csv
│   │   │   └── coefficient_sweep_seed777.csv
│   │   ├── a_l2_hist.png
│   │   ├── audit_base_task_pairs.csv
│   │   ├── b_l2_hist.png
│   │   ├── conflict_frac_hist.png
│   │   ├── cosine_hist.png
│   │   ├── merge_operator_audit.csv
│   │   ├── task_vectors_all_per_tensor.csv
│   │   ├── task_vectors_seed_123_per_tensor.csv
│   │   ├── task_vectors_seed_123_summary.json
│   │   ├── task_vectors_seed_42_per_tensor.csv
│   │   ├── task_vectors_seed_42_summary.json
│   │   ├── task_vectors_seed_777_per_tensor.csv
│   │   ├── task_vectors_seed_777_summary.json
│   │   └── task_vectors_summary.csv
│   ├── merged_adapters
│   │   ├── base_logs_plus_qa
│   │   │   ├── delta_w
│   │   │   │   ├── dare_plus_model_soup
│   │   │   │   ├── dare_plus_task_arithmetic
│   │   │   │   ├── dare_plus_ties
│   │   │   │   ├── model_soup
│   │   │   │   ├── task_arithmetic
│   │   │   │   └── ties
│   │   │   └── factor
│   │   │       ├── dare_plus_model_soup
│   │   │       ├── dare_plus_task_arithmetic
│   │   │       ├── dare_plus_ties
│   │   │       ├── model_soup
│   │   │       ├── task_arithmetic
│   │   │       └── ties
│   │   ├── coefficient_sweep
│   │   │   └── delta_w
│   │   │       ├── lambda_0.25
│   │   │       ├── lambda_0.33
│   │   │       ├── lambda_0.5
│   │   │       ├── lambda_0.75
│   │   │       └── lambda_1.0
│   │   ├── cpt_anchor_logs_plus_qa
│   │   │   ├── delta_w
│   │   │   │   ├── dare_plus_model_soup
│   │   │   │   ├── dare_plus_task_arithmetic
│   │   │   │   ├── dare_plus_ties
│   │   │   │   ├── model_soup
│   │   │   │   ├── task_arithmetic
│   │   │   │   └── ties
│   │   │   └── factor
│   │   │       ├── dare_plus_model_soup
│   │   │       ├── dare_plus_task_arithmetic
│   │   │       ├── dare_plus_ties
│   │   │       ├── model_soup
│   │   │       ├── task_arithmetic
│   │   │       └── ties
│   │   └── sequential_cpt_anchor
│   │       └── factor
│   │           └── task_arithmetic
│   ├── prepared_data
│   │   ├── cpt_train.jsonl
│   │   ├── cpt_val.jsonl
│   │   ├── data_manifest.json
│   │   ├── logs_test_eval.jsonl
│   │   ├── logs_train_sft.jsonl
│   │   ├── logs_val_sft.jsonl
│   │   ├── multitask_train_sft.jsonl
│   │   ├── qa_test_eval.jsonl
│   │   ├── qa_train_sft.jsonl
│   │   └── qa_val_sft.jsonl
│   └── results
│       ├── baselines
│       │   ├── baseline_metrics_mean_std.csv
│       │   └── baseline_metrics_raw.csv
│       ├── merged
│       │   ├── dare_plus_ties_seed_diagnostic.csv
│       │   ├── merged_metrics_mean_std.csv
│       │   └── merged_metrics_raw.csv
│       ├── all_metrics_mean_std.csv
│       ├── all_metrics_raw.csv
│       └── main_report_camera_ready.csv
├── data
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
├── 10_coefficient_sweep.ipynb
├── exp_common.py
├── README.md
└── requirements.txt
```

- **Notebooks**: Each numbered notebook contains code for a stage of the experiment (see below for details).
- **exp_common.py**: Shared Python code for model loading, LoRA setup, training, evaluation, merging, and analysis.
- **data/**: Contains a `.gitkeep` placeholder (no raw data files are provided here).
- **artifacts/prepared_data/**: Contains all pre-processed JSONL datasets. Use these directly for reproducibility (see below).
- **artifacts/adapters/** and **artifacts/merged_adapters/**: Contain `.gitkeep` placeholders. These directories will be filled with generated LoRA checkpoint files when you run the training notebooks. (No pre-trained checkpoints are included in the repository.)
- **analysis/** and **results/**: Contain CSVs, PNGs, and aggregated results produced by the notebooks.

## Notebook descriptions

- **`00_setup_common.ipynb`**: environment checks and directory setup.  
- **`01_prepare_data.ipynb`**: data preparation. Not needed when using `artifacts/prepared_data/`.
- **`02_train_cpt_lora.ipynb`**: CPT adapter training. 
- **`03_train_sft_lora_base.ipynb`**: task adapter and multitask training from the base model.
- **`04_train_sft_lora_on_cpt.ipynb`**: fine-tune new task adapters (Logs and QA) on top of the merged CPT adapter.  
- **`05_eval_baselines.ipynb`**: baseline evaluation. 
- **`06_task_vector_analysis.ipynb`**: adapter geometry and audit analysis.  
- **`07_merge_adapters.ipynb`**: create merged adapters using various methods.  
- **`08_eval_merged_and_grid.ipynb`**: evaluate merged adapters.  
- **`09_aggregate_results.ipynb`**: aggregate raw metrics and produce summary tables.
- **`10_coefficient_sweep.ipynb`**: isolated coefficient sweep for the log update.

## Environment and installation

The experiments were developed and reproduced using the following environment:

| Component              | Version/Value                      |
|------------------------|------------------------------------|
| Python                 | 3.10.11                            |
| CUDA (runtime)         | 12.8                               |
| PyTorch                | 2.10.0+cu128                       |
| Transformers           | 5.5.0                              |
| Unsloth                | 2026.7.1                           |
| xFormers               | 0.0.35                             |
| Training precision     | bfloat16                           |

The current training configuration requires a CUDA-capable GPU with bfloat16 support. The repository does not define a minimum VRAM requirement.  

### Setup

1. **Create and activate a virtual environment:**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

2. **Install dependencies using the provided `requirements.txt`:**

```bash
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

The repository pins the project-level dependencies used by the experiments in `requirements.txt`.  

### Verify installation

Run the following to verify your environment:


```python 
import torch
import unsloth
import transformers
import trl
import peft
import accelerate
import datasets
import bitsandbytes
import xformers
import safetensors
import sentence_transformers
import sklearn
import pandas as pd
import numpy as np
import matplotlib
import tqdm
import huggingface_hub


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
print(f"bitsandbytes:      {bitsandbytes.__version__}")
print(f"xFormers:          {xformers.__version__}")
print(f"sentence-transformers: {sentence_transformers.__version__}")
print(f"scikit-learn:      {sklearn.__version__}")
print(f"pandas:            {pd.__version__}")
print(f"numpy:             {np.__version__}")
print(f"safetensors:       {safetensors.__version__}")
print(f"matplotlib:        {matplotlib.__version__}")
print(f"tqdm:              {tqdm.__version__}")
print(f"huggingface-hub:   {huggingface_hub.__version__}")
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
bitsandbytes:      0.49.2
xFormers:          0.0.35
sentence-transformers: 5.6.0
scikit-learn:      1.7.2
pandas:            2.3.3
numpy:             1.24.3
safetensors:       0.8.0
matplotlib:        3.10.9
tqdm:              4.68.3
huggingface-hub:   1.22.0
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
Default batch:     2
Gradient accum:    8  (effective batch size = 16 per GPU)
Checkpoint criterion: lowest validation loss
```

## LoRA merge convention

The repository supports merging adapters in two spaces:

1. **Factor space**
   
   Merge operators are applied directly to LoRA adapter tensors (`lora_A` and `lora_B`).  

   Implemented operators:
   - Model Soup
   - Task Arithmetic
   - TIES
   - DARE combinations (implemented as DARE + operator)

2. **Delta-W space**

   LoRA adapters can also be converted into induced weight updates:

   \[
   \Delta W = scaling \cdot (B @ A)
   \]

   The Delta-W implementation first reconstructs the induced weight updates, merges these updates, and then converts the result back into a low-rank LoRA representation using SVD.

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

The main evaluation uses **first 2000** examples from `logs_test_eval.jsonl`. Semantic similarity is computed on 500 QA examples and LLM-as-Judge is computed on the first 100 QA examples from `qa_test_eval.jsonl`.

### Checksum verification

To ensure data integrity, the repository provides SHA-256 checksums for each prepared file. Before running experiments, you can verify them:

```bash
sha256sum artifacts/prepared_data/*
```

Expected SHA-256 sums:

```text
503b69563b6ea24ba42bfa3b96d3cb27f8f3d0c89124919b04fd22eb5ab8237f *artifacts/prepared_data/cpt_train.jsonl
75097f0fe1bd7687992792c3c6cd2af29602e38c7bab77cf71fb18dd74d151b5 *artifacts/prepared_data/cpt_val.jsonl
e404da263e35461dbacd6974646d9b85205412d50d5f48f8c7c3cb60a127716c *artifacts/prepared_data/data_manifest.json
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

2. **Run notebooks in order (skip 01):**  
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
           ↓
   10_coefficient_sweep.ipynb
   ```

   - **Skip `01_prepare_data.ipynb`** if you are using the provided `artifacts/prepared_data/` files (recommended).  
   - After training notebooks (02–04), checkpoint files will be saved under `artifacts/adapters/` (e.g., `artifacts/adapters/cpt/seed_42/`, etc.).  

### Reproducibility artifacts

The repository stores intermediate and final artifacts required to inspect and reproduce the reported results.

Important outputs include:

* `artifacts/results/*_raw.csv` - raw metrics for individual runs and seeds;
* `artifacts/results/*_mean_std.csv` - aggregated results with mean and standard deviation;
* `artifacts/analysis/` - task-vector statistics, merge diagnostics, and coefficient sweep outputs;
* `artifacts/prepared_data/` - prepared datasets used by downstream experiments.

All main experiments are run with seeds: `42`, `123`, `777`.
For configurations with high variance across seeds, per-seed results should be inspected instead of relying only on aggregate averages.

## Evaluation settings

- **Evaluation set sizes:**  
  - Logs (anomaly detection): `EVAL_LOGS_N = 2000`  
  - QA: `EVAL_QA_N = 500`  

- **Batch sizes:**  
  - Logs evaluation: `BATCH_LOGS = 16`  
  - QA evaluation: `BATCH_QA = 4`  

- **Log evaluation:** Log anomaly detection is evaluated as a constrained generation task. The model is prompted to output exactly one label (`OK` or `Anomaly`) and is evaluated with greedy decoding (`do_sample=False`, `max_new_tokens=3`). 

  Primary metrics use strict exact-label parsing:
  - `OK` → `OK`
  - `Anomaly` → `Anomaly`
  - any other output → invalid/off-format prediction

  Keyword-based parsing is reported only as a secondary sensitivity analysis. It maps outputs containing anomaly-related keywords (`anomaly`, `abnormal`, `error`, `fault`, `fail`) to `Anomaly`, and outputs containing normality-related keywords (`ok`, `normal`, `benign`) to `OK`.

  The evaluation reports strict coverage and off-format rate in addition to classification metrics. Raw generations and parsed predictions are stored in evaluation JSON outputs.  

- **QA evaluation:** Semantic similarity is computed using the `sentence-transformers/all-MiniLM-L6-v2` model. QA responses are additionally evaluated using an LLM-as-Judge approach with the `unsloth/Llama-3.1-8B-Instruct` model configured in the current evaluation code.

## Hyperparameter sweep

Notebook `10_coefficient_sweep.ipynb` runs a parameter sweep with:

```python
LAMBDAS = [0.25, 0.33, 0.5, 0.75, 1.0]
```

Sweep is done for seeds `42`, `123`, `777`. Output CSVs are in `artifacts/analysis/coefficient_sweep`.

## Files intentionally omitted

- The repo **does not include** trained LoRA checkpoint files in `artifacts/adapters/` or `artifacts/merged_adapters/`. These will be generated when you run the training notebooks.
- Directories `data/`, `artifacts/adapters/`, and `artifacts/merged_adapters/` contain `.gitkeep` placeholders to ensure they appear in the repo even if empty.

## Troubleshooting & notes
  
- **ALPHA parameter:** The `07_merge_adapters.ipynb` default cell sets `ALPHA = 1.0`.  

- **Reproducing archived results:**  
  The files in `artifacts/results/` contain the saved CSVs used in the manuscript. For example, `artifacts/results/all_metrics_raw.csv` has the full set of raw metrics for each model and seed. You can load these CSVs directly (no training needed) to recompute tables.

## Reproducibility checklist

1. **Verify data**: Check SHA-256 sums and row counts of `artifacts/prepared_data/*`.  
2. **Set up environment**: Create Python 3.10 venv and install dependencies. Verify versions.  
3. **Run notebooks in order** (skipping 01): 00 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10.  
4. **Monitor outputs**: Training notebooks will write LoRA adapter files to `artifacts/adapters/`; merged adapters go to `artifacts/merged_adapters/`.  
5. **Compute metrics**: Evaluation notebooks will write CSVs under `artifacts/results/`.  

Following the above steps using the provided code and prepared data will reproduce the experiment pipeline without requiring additional dataset files.