from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import torch

try:
    import unsloth 
except Exception:
    unsloth = None

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
PREPARED_DIR = ARTIFACTS_DIR / "prepared_data"
ADAPTERS_DIR = ARTIFACTS_DIR / "adapters"
MERGED_DIR = ARTIFACTS_DIR / "merged_adapters"
RESULTS_DIR = ARTIFACTS_DIR / "results"
ANALYSIS_DIR = ARTIFACTS_DIR / "analysis"
CACHE_DIR = ARTIFACTS_DIR / "cache"
for d in [ARTIFACTS_DIR, PREPARED_DIR, ADAPTERS_DIR, MERGED_DIR, RESULTS_DIR, ANALYSIS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507"
JUDGE_MODEL = "unsloth/Llama-3.1-8B-Instruct"
SEEDS = [42, 123, 777]
DEFAULT_MAX_SEQ_LENGTH = 2048
LOG_LABELS = ["OK", "Anomaly"]
TRAIN_DTYPE = torch.bfloat16

DEFAULT_SFT_LR = 2e-4
DEFAULT_CPT_LR = 1e-4
DEFAULT_SFT_EPOCHS = 1.0
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 8
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_EVAL_LOGS_N = 2000
DEFAULT_EVAL_QA_N = 500
DEFAULT_JUDGE_N = 100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path, max_rows: Optional[int] = None) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sample_balanced_logs(path: Path, per_class: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    buckets = {label: [] for label in LOG_LABELS}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            y = str(r.get("output", "")).strip()
            if y in buckets:
                buckets[y].append(r)
    n = min(per_class, *(len(v) for v in buckets.values()))
    out = []
    for label in LOG_LABELS:
        out.extend(rng.sample(buckets[label], n))
    rng.shuffle(out)
    return out


def sample_rows(path: Path, n: int, seed: int) -> List[dict]:
    rows = read_jsonl(path)
    rng = random.Random(seed)
    if len(rows) <= n:
        rng.shuffle(rows)
        return rows
    return rng.sample(rows, n)


def log_prompt(x: str) -> str:
    return (
        "You are an expert in HDFS log analysis. "
        "Classify the given log message as either normal or anomalous. "
        "Respond with exactly one label: OK or Anomaly. "
        "Do not provide an explanation.\n\n"
        f"Log message:\n{x}\n\nLabel:"
    )


def qa_prompt(x: str) -> str:
    return f"Answer the following question accurately:\n\n{x}\n\nAnswer:"


def format_sft_example(example: dict, task: str) -> dict:
    if task == "logs":
        prompt = log_prompt(example["input"])
    elif task == "qa":
        prompt = qa_prompt(example["input"])
    else:
        raise ValueError(task)
    return {"text": prompt + " " + str(example["output"]).strip()}


def format_cpt_example(example: dict) -> dict:
    return {"text": str(example["text"]).strip()}


def load_unsloth_model(
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
    model_name: Optional[str] = None,
):
    from unsloth import FastLanguageModel
    model_name = model_name or BASE_MODEL
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=TRAIN_DTYPE,
        load_in_4bit=load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def add_lora(
    model,
    r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    random_state: int = 42,
):
    from unsloth import FastLanguageModel
    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=random_state,
        use_rslora=False,
        loftq_config=None,
    )


def train_lora_adapter(
    dataset_rows: List[dict],
    eval_rows: List[dict],
    output_dir: Path,
    seed: int,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    epochs: float = DEFAULT_SFT_EPOCHS,
    max_steps: int = -1,
    per_device_train_batch_size: int = DEFAULT_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRAD_ACCUM,
    learning_rate: float = DEFAULT_SFT_LR,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    save_steps: int = 500,
    base_adapter_to_merge: Optional[Path] = None,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    overwrite: bool = False,
):
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from peft import PeftModel

    set_seed(seed)
    output_dir = Path(output_dir)
    adapter_out = output_dir / "adapter"
    if adapter_out.exists() and (adapter_out / "adapter_model.safetensors").exists() and not overwrite:
        raise FileExistsError(
            f"Adapter already exists: {adapter_out}. Set overwrite=True or remove the old run explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    use_4bit = base_adapter_to_merge is None
    model, tokenizer = load_unsloth_model(
        max_seq_length=max_seq_length,
        load_in_4bit=use_4bit,
    )

    if base_adapter_to_merge is not None:
        base_adapter_to_merge = Path(base_adapter_to_merge)
        model = PeftModel.from_pretrained(model, str(base_adapter_to_merge), is_trainable=False)
        model = model.merge_and_unload()

    model = add_lora(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        random_state=seed,
    )

    ds = Dataset.from_list(dataset_rows)
    use_eval = bool(eval_rows)
    ds_eval = Dataset.from_list(eval_rows) if use_eval else None

    args = SFTConfig(
        output_dir=str(output_dir / "trainer_state"),
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        save_only_model=True,
        eval_steps=50 if use_eval else None,
        eval_strategy="steps" if use_eval else "no",
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False,
        seed=seed,
        data_seed=seed,
        report_to="none",
        remove_unused_columns=True,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        eval_dataset=ds_eval,
        args=args,
    )
    trainer.train()
    trainer.save_model(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))

    metadata = {
        "base_model": BASE_MODEL,
        "seed": seed,
        "n_train": len(dataset_rows),
        "n_eval": len(eval_rows),
        "max_seq_length": max_seq_length,
        "epochs": epochs,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "bf16": True,
        "fp16": False,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "base_adapter_to_merge": str(base_adapter_to_merge) if base_adapter_to_merge else None,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    del trainer, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return adapter_out


def load_adapter_state(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    adapter_dir = Path(adapter_dir)
    st_path = adapter_dir / "adapter_model.safetensors"
    if st_path.exists():
        from safetensors.torch import load_file
        return load_file(str(st_path), device="cpu")
    pt_path = adapter_dir / "adapter_model.bin"
    if pt_path.exists():
        return torch.load(pt_path, map_location="cpu")
    raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin in {adapter_dir}")


def load_adapter_config(adapter_dir: Path) -> dict:
    p = Path(adapter_dir) / "adapter_config.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing adapter_config.json: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def lora_scaling(adapter_dir: Path) -> float:
    cfg = load_adapter_config(adapter_dir)
    r = float(cfg.get("r", DEFAULT_LORA_R))
    alpha = float(cfg.get("lora_alpha", DEFAULT_LORA_ALPHA))
    use_rslora = bool(cfg.get("use_rslora", False))
    return alpha / math.sqrt(r) if use_rslora else alpha / r


def save_adapter_state_like(
    reference_adapter_dir: Path,
    state: Dict[str, torch.Tensor],
    out_dir: Path,
    metadata: dict,
    config_override: Optional[dict] = None,
):
    from safetensors.torch import save_file
    reference_adapter_dir = Path(reference_adapter_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in [
        "adapter_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]:
        src = reference_adapter_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    if config_override is not None:
        (out_dir / "adapter_config.json").write_text(
            json.dumps(config_override, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "merge_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _merge_keys(states: List[Dict[str, torch.Tensor]]) -> List[str]:
    if not states:
        return []
    keys = sorted(set.intersection(*[set(s.keys()) for s in states]))
    return [k for k in keys if torch.is_floating_point(states[0][k])]


def model_soup(states: List[Dict[str, torch.Tensor]], weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("states must not be empty")
    if weights is None:
        weights = [1.0] * len(states)
    if len(weights) != len(states):
        raise ValueError("weights and states must have equal length")
    wsum = float(sum(weights))
    if wsum == 0:
        raise ValueError("weights must not sum to zero")
    weights = [float(w) / wsum for w in weights]
    out = {k: states[0][k].clone() for k in states[0].keys()}
    for k in _merge_keys(states):
        out[k] = sum(w * s[k].float() for w, s in zip(weights, states)).to(states[0][k].dtype)
    return out


def task_arithmetic(
    states: List[Dict[str, torch.Tensor]],
    alpha: float = 1.0,
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Unnormalised Task Arithmetic in adapter-factor space.

    With weights=None and two states, alpha=1 gives state_1 + state_2.
    This is deliberately different from Model Soup, which normalises weights.
    """
    if not states:
        raise ValueError("states must not be empty")
    if weights is None:
        weights = [1.0] * len(states)
    if len(weights) != len(states):
        raise ValueError("weights and states must have equal length")
    out = {k: states[0][k].clone() for k in states[0].keys()}
    for k in _merge_keys(states):
        merged = sum(float(w) * s[k].float() for w, s in zip(weights, states))
        out[k] = (float(alpha) * merged).to(states[0][k].dtype)
    return out


def _ties_tensor(stack: torch.Tensor, top_k: float, alpha: float = 1.0) -> torch.Tensor:
    if not (0 < top_k <= 1):
        raise ValueError("top_k must be in (0, 1]")
    trimmed = []
    for t in stack:
        if top_k < 1:
            kth = max(1, int(t.numel() * top_k))
            thresh = torch.topk(t.flatten().abs(), kth, sorted=False).values.min()
            t = torch.where(t.abs() >= thresh, t, torch.zeros_like(t))
        trimmed.append(t)
    stack = torch.stack(trimmed, dim=0)
    sign = torch.sign(stack.sum(dim=0))
    same = torch.where(torch.sign(stack) == sign.unsqueeze(0), stack, torch.zeros_like(stack))
    denom = (same != 0).sum(dim=0).clamp(min=1)
    merged = same.sum(dim=0) / denom
    return alpha * merged


def ties_merge(states: List[Dict[str, torch.Tensor]], top_k: float = 0.2, alpha: float = 1.0) -> Dict[str, torch.Tensor]:
    out = {k: states[0][k].clone() for k in states[0].keys()}
    for k in _merge_keys(states):
        stack = torch.stack([s[k].float() for s in states], dim=0)
        out[k] = _ties_tensor(stack, top_k=top_k, alpha=alpha).to(states[0][k].dtype)
    return out


def dare_state(state: Dict[str, torch.Tensor], drop_rate: float = 0.5, seed: int = 42) -> Dict[str, torch.Tensor]:
    if not (0 <= drop_rate < 1):
        raise ValueError("drop_rate must be in [0,1)")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    keep_prob = 1.0 - drop_rate
    out = {k: v.clone() for k, v in state.items()}
    for k, v in state.items():
        if torch.is_floating_point(v):
            mask = torch.rand(v.shape, generator=g) < keep_prob
            out[k] = torch.where(mask, v.float() / keep_prob, torch.zeros_like(v.float())).to(v.dtype)
    return out


def _lora_pairs(state: Dict[str, torch.Tensor]) -> Dict[str, Tuple[str, str]]:
    """Return module prefix -> (A_key, B_key) for LoRA tensors."""
    pairs = {}
    for k in state:
        if ".lora_A" in k and k.endswith(".weight"):
            b_key = k.replace(".lora_A", ".lora_B")
            if b_key in state:
                prefix = k.split(".lora_A", 1)[0]
                pairs[prefix] = (k, b_key)
    return pairs


def state_to_delta_w(state: Dict[str, torch.Tensor], scaling: float) -> Dict[str, torch.Tensor]:
    pairs = _lora_pairs(state)
    if not pairs:
        raise ValueError("No LoRA A/B tensor pairs found in adapter state")
    delta = {}
    for module, (a_key, b_key) in pairs.items():
        A = state[a_key].float()
        B = state[b_key].float()
        if B.ndim != 2 or A.ndim != 2:
            raise ValueError(f"Expected 2-D LoRA matrices for {module}, got {B.shape}, {A.shape}")
        delta[module] = scaling * (B @ A)
    return delta


def delta_w_stats(delta: Dict[str, torch.Tensor]) -> dict:
    vals = torch.cat([v.float().flatten() for v in delta.values()])
    return tensor_stats(vals)


def _weighted_delta_merge(
    deltas: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    keys = sorted(set.intersection(*[set(d.keys()) for d in deltas]))
    return {k: sum(float(w) * d[k] for w, d in zip(weights, deltas)) for k in keys}


def _dare_delta(delta: Dict[str, torch.Tensor], drop_rate: float, seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    keep_prob = 1.0 - drop_rate
    out = {}
    for k, v in delta.items():
        mask = torch.rand(v.shape, generator=g) < keep_prob
        out[k] = torch.where(mask, v / keep_prob, torch.zeros_like(v))
    return out


def _ties_delta(deltas: List[Dict[str, torch.Tensor]], top_k: float, alpha: float = 1.0) -> Dict[str, torch.Tensor]:
    keys = sorted(set.intersection(*[set(d.keys()) for d in deltas]))
    return {k: _ties_tensor(torch.stack([d[k] for d in deltas], dim=0), top_k, alpha) for k in keys}


def delta_w_merge(
    adapter_dirs: List[Path],
    method: str,
    weights: Optional[List[float]] = None,
    alpha: float = 1.0,
    top_k: float = 0.2,
    drop_rate: float = 0.5,
    seed: int = 42,
    target_rank: int = DEFAULT_LORA_R,
) -> Tuple[Dict[str, torch.Tensor], dict]:

    adapter_dirs = [Path(p) for p in adapter_dirs]
    states = [load_adapter_state(p) for p in adapter_dirs]
    scalings = [lora_scaling(p) for p in adapter_dirs]
    deltas = [state_to_delta_w(s, sc) for s, sc in zip(states, scalings)]

    method_l = method.lower().replace("_", "+")
    effective_weights = weights
    if effective_weights is None:
        effective_weights = [1.0] * len(deltas)

    if method_l in {"soup", "model+soup", "model soups"}:
        wsum = sum(effective_weights)
        effective_weights = [w / wsum for w in effective_weights]
        merged = _weighted_delta_merge(deltas, effective_weights)
    elif method_l in {"ta", "task+arithmetic", "task+arithmetic+"}:
        merged = _weighted_delta_merge(deltas, effective_weights)
        merged = {k: alpha * v for k, v in merged.items()}
    elif method_l == "ties":
        merged = _ties_delta(deltas, top_k=top_k, alpha=alpha)
    elif method_l in {"dare+ta", "dare+task+arithmetic"}:
        dd = [_dare_delta(d, drop_rate, seed + i) for i, d in enumerate(deltas)]
        merged = _weighted_delta_merge(dd, effective_weights)
        merged = {k: alpha * v for k, v in merged.items()}
    elif method_l == "dare+ties":
        dd = [_dare_delta(d, drop_rate, seed + i) for i, d in enumerate(deltas)]
        merged = _ties_delta(dd, top_k=top_k, alpha=alpha)
    elif method_l in {"dare+soup", "dare+model+soup"}:
        dd = [_dare_delta(d, drop_rate, seed + i) for i, d in enumerate(deltas)]
        wsum = sum(effective_weights)
        if wsum == 0:
            raise ValueError("weights must not sum to zero")
        effective_weights = [w / wsum for w in effective_weights]
        merged = _weighted_delta_merge(dd, effective_weights)
    else:
        raise ValueError(f"Unknown Delta-W merge method: {method}")

    ref_state = states[0]
    ref_cfg = load_adapter_config(adapter_dirs[0])
    out = {k: ref_state[k].clone() for k in ref_state.keys()}
    pairs = _lora_pairs(ref_state)
    reconstruction = []

    for module, (a_key, b_key) in pairs.items():
        M = merged[module].float()
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        r = min(int(target_rank), S.numel())
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        approx = (U_r * S_r.unsqueeze(0)) @ Vh_r
        denom = torch.linalg.vector_norm(M).item()
        rel_err = float(torch.linalg.vector_norm(M - approx).item() / denom) if denom > 0 else 0.0
        reconstruction.append({"module": module, "target_rank": r, "relative_fro_error": rel_err})

        sc = lora_scaling(adapter_dirs[0])
        B_new = U_r * S_r.unsqueeze(0) / sc
        A_new = Vh_r
        out[a_key] = A_new.to(ref_state[a_key].dtype)
        out[b_key] = B_new.to(ref_state[b_key].dtype)

    meta = {
        "space": "delta_W",
        "method": method,
        "adapter_dirs": [str(p) for p in adapter_dirs],
        "seed": seed,
        "alpha": alpha,
        "top_k": top_k,
        "drop_rate": drop_rate,
        "target_rank": target_rank,
        "source_scalings": scalings,
        "reconstruction": reconstruction,
        "mean_relative_fro_error": float(np.mean([x["relative_fro_error"] for x in reconstruction])) if reconstruction else None,
        "max_relative_fro_error": float(np.max([x["relative_fro_error"] for x in reconstruction])) if reconstruction else None,
    }
    return out, meta


def merge_adapters(
    adapter_dirs: List[Path],
    out_dir: Path,
    method: str,
    seed: int = 42,
    alpha: float = 1.0,
    top_k: float = 0.2,
    drop_rate: float = 0.5,
    space: str = "factor",
    target_rank: int = DEFAULT_LORA_R,
):
    adapter_dirs = [Path(p) for p in adapter_dirs]
    if space == "delta_w":
        merged, metadata = delta_w_merge(
            adapter_dirs,
            method=method,
            alpha=alpha,
            top_k=top_k,
            drop_rate=drop_rate,
            seed=seed,
            target_rank=target_rank,
        )
        metadata["space"] = "delta_W"
    else:
        states = [load_adapter_state(p) for p in adapter_dirs]
        method_l = method.lower()
        dare_prefix = method_l.startswith("dare+")
        if dare_prefix:
            inner = method_l.split("+", 1)[1]
            states = [dare_state(s, drop_rate=drop_rate, seed=seed + i) for i, s in enumerate(states)]
            method_l = inner
        if method_l in {"model_soup", "soup", "model soups", "model_soups"}:
            merged = model_soup(states)
        elif method_l in {"task_arithmetic", "task arithmetic", "ta"}:
            merged = task_arithmetic(states, alpha=alpha)
        elif method_l == "ties":
            merged = ties_merge(states, top_k=top_k, alpha=alpha)
        else:
            raise ValueError(f"Unknown factor-space merge method: {method}")
        metadata = {
            "space": "factor",
            "method": method,
            "adapter_dirs": [str(p) for p in adapter_dirs],
            "seed": seed,
            "alpha": alpha,
            "top_k": top_k,
            "drop_rate": drop_rate,
        }
    save_adapter_state_like(adapter_dirs[0], merged, out_dir, metadata)
    return Path(out_dir)


def tensor_stats(t: torch.Tensor) -> dict:
    x = t.detach().float().flatten()
    if x.numel() == 0:
        return {}
    return {
        "numel": int(x.numel()),
        "nnz": int((x != 0).sum().item()),
        "zero_frac": float((x == 0).float().mean().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "abs_mean": float(x.abs().mean().item()),
        "l2": float(torch.linalg.vector_norm(x).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _global_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float(torch.dot(a, b).item() / denom.item()) if denom.item() > 0 else float("nan")


def adapter_vector_analysis(adapter_a: Path, adapter_b: Path, out_prefix: Path) -> pd.DataFrame:
    a, b = load_adapter_state(adapter_a), load_adapter_state(adapter_b)
    keys = _merge_keys([a, b])
    rows = []
    flat_a, flat_b = [], []
    flat_A, flat_B = [], []
    for k in keys:
        ta, tb = a[k].float().flatten(), b[k].float().flatten()
        n = min(ta.numel(), tb.numel())
        ta, tb = ta[:n], tb[:n]
        denom = torch.linalg.vector_norm(ta) * torch.linalg.vector_norm(tb)
        cos = float(torch.dot(ta, tb).item() / denom.item()) if denom.item() > 0 else float("nan")
        rows.append({
            "key": k,
            "shape": str(tuple(a[k].shape)),
            "cosine": cos,
            "conflict_frac": float(((ta * tb) < 0).float().mean().item()) if n else 0.0,
            "a_l2": tensor_stats(a[k]).get("l2"),
            "b_l2": tensor_stats(b[k]).get("l2"),
            "a_zero_frac": tensor_stats(a[k]).get("zero_frac"),
            "b_zero_frac": tensor_stats(b[k]).get("zero_frac"),
            "a_abs_mean": tensor_stats(a[k]).get("abs_mean"),
            "b_abs_mean": tensor_stats(b[k]).get("abs_mean"),
        })
        if ".lora_A" in k:
            flat_A.append(ta); flat_B.append(tb)
        flat_a.append(ta); flat_b.append(tb)

    df = pd.DataFrame(rows)
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_prefix) + "_per_tensor.csv", index=False)

    summary = {
        "global_cosine_factor_all": _global_cosine(torch.cat(flat_a), torch.cat(flat_b)) if flat_a else None,
        "global_conflict_frac_factor_all": float(((torch.cat(flat_a) * torch.cat(flat_b)) < 0).float().mean().item()) if flat_a else None,
        "a_factor": tensor_stats(torch.cat(flat_a)) if flat_a else {},
        "b_factor": tensor_stats(torch.cat(flat_b)) if flat_b else {},
    }

    if flat_A:
        A1, A2 = torch.cat(flat_A), torch.cat(flat_B)
        summary["global_cosine_A"] = _global_cosine(A1, A2)
        summary["global_conflict_frac_A"] = float(((A1 * A2) < 0).float().mean().item())

    B1, B2 = [], []
    for k in keys:
        if ".lora_B" in k:
            B1.append(a[k].float().flatten()); B2.append(b[k].float().flatten())
    if B1:
        b1, b2 = torch.cat(B1), torch.cat(B2)
        summary["global_cosine_B"] = _global_cosine(b1, b2)
        summary["global_conflict_frac_B"] = float(((b1 * b2) < 0).float().mean().item())

    da = state_to_delta_w(a, lora_scaling(adapter_a))
    db = state_to_delta_w(b, lora_scaling(adapter_b))
    dflat_a = torch.cat([v.flatten() for v in da.values()])
    dflat_b = torch.cat([v.flatten() for v in db.values()])
    summary["global_cosine_delta_W"] = _global_cosine(dflat_a, dflat_b)
    summary["global_conflict_frac_delta_W"] = float(((dflat_a * dflat_b) < 0).float().mean().item())
    summary["a_delta_W"] = delta_w_stats(da)
    summary["b_delta_W"] = delta_w_stats(db)

    (Path(str(out_prefix) + "_summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return df


def generate_texts(model, tokenizer, prompts: List[str], max_new_tokens: int = 128, batch_size: int = 8) -> List[str]:

    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = max_new_tokens

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    outs = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=DEFAULT_MAX_SEQ_LENGTH,
            ).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
            outs.extend([d.strip() for d in decoded])
    finally:
        tokenizer.padding_side = old_padding_side
    return outs


def load_model_for_eval(
    adapter_dir: Optional[Path] = None,
    base_adapter_dir: Optional[Path] = None,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    merge_adapter: bool = False,
):
    from peft import PeftModel

    use_4bit = base_adapter_dir is None
    model, tokenizer = load_unsloth_model(
        max_seq_length=max_seq_length,
        load_in_4bit=use_4bit,
    )

    if base_adapter_dir is not None:
        base_adapter_dir = Path(base_adapter_dir)
        model = PeftModel.from_pretrained(model, str(base_adapter_dir), is_trainable=False)
        model = model.merge_and_unload()

    if adapter_dir is not None:
        adapter_dir = Path(adapter_dir)
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        if merge_adapter:
            model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def load_judge_model(max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH):
    return load_unsloth_model(max_seq_length=max_seq_length, load_in_4bit=True, model_name=JUDGE_MODEL)


def parse_log_label_strict(text: str) -> Optional[str]:
    t = text.strip().lower()
    if t == "ok":
        return "OK"
    if t == "anomaly":
        return "Anomaly"
    return None


def normalize_log_label_keyword(text: str) -> Optional[str]:
    t = text.strip().lower()
    if re.search(r"\banomaly\b|abnormal|error|fault|fail", t):
        return "Anomaly"
    if re.search(r"\bok\b|normal|benign", t):
        return "OK"
    return None


def evaluate_logs(
    adapter_dir: Optional[Path],
    rows: List[dict],
    out_path: Path,
    batch_size: int = 16,
    max_new_tokens: int = 3,
    base_adapter_dir: Optional[Path] = None,
) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    model, tokenizer = load_model_for_eval(adapter_dir, base_adapter_dir=base_adapter_dir)
    prompts = [log_prompt(r["input"]) for r in rows]
    generations = generate_texts(model, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
    y_true = [r["output"] for r in rows]
    y_pred_strict = [parse_log_label_strict(g) for g in generations]
    y_pred_keyword = [normalize_log_label_keyword(g) for g in generations]

    valid = [p is not None for p in y_pred_strict]
    y_true_valid = [y for y, v in zip(y_true, valid) if v]
    y_pred_valid = [p for p in y_pred_strict if p is not None]
    if y_pred_valid:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_valid, y_pred_valid, labels=LOG_LABELS, average="binary", pos_label="Anomaly", zero_division=0
        )
        acc = accuracy_score(y_true_valid, y_pred_valid)
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=LOG_LABELS).tolist()
    else:
        precision = recall = f1 = acc = float("nan")
        cm = [[0, 0], [0, 0]]

    keyword_valid = [p is not None for p in y_pred_keyword]
    ytv_k = [y for y, v in zip(y_true, keyword_valid) if v]
    ypk = [p for p in y_pred_keyword if p is not None]
    keyword_acc = accuracy_score(ytv_k, ypk) if ypk else float("nan")

    pred_anomaly_rate = float(np.mean([p == "Anomaly" for p in y_pred_strict if p is not None])) if y_pred_valid else float("nan")
    gold_anomaly_rate = float(np.mean([y == "Anomaly" for y in y_true_valid])) if y_true_valid else float("nan")
    off_format_rate = float(np.mean([p is None for p in y_pred_strict]))

    result = {
        "accuracy": float(acc),
        "precision_anomaly": float(precision),
        "recall_anomaly": float(recall),
        "f1_anomaly": float(f1),
        "coverage_strict": float(1.0 - off_format_rate),
        "off_format_rate": off_format_rate,
        "pred_anomaly_rate": pred_anomaly_rate,
        "gold_anomaly_rate": gold_anomaly_rate,
        "keyword_accuracy_sensitivity": float(keyword_acc),
        "confusion_matrix_labels": LOG_LABELS,
        "confusion_matrix": cm,
        "parser": "strict_exact_label_primary; keyword_sensitivity_secondary",
        "base_adapter_dir": str(base_adapter_dir) if base_adapter_dir else None,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "metrics": result,
                "predictions": [
                    {
                        "input": r["input"],
                        "gold": r["output"],
                        "pred_strict": ps,
                        "pred_keyword": pk,
                        "raw": g,
                    }
                    for r, ps, pk, g in zip(rows, y_pred_strict, y_pred_keyword, generations)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _parse_judge_score(text: str) -> Optional[float]:

    t = text.strip()
    if not t:
        return None

    m = re.match(r"(?is)^\s*SCORE\s*[:=]\s*([0-5](?:\.0|\.5)?)\b", t)
    if m:
        return float(m.group(1))

    m = re.match(r"^\s*([0-5](?:\.0|\.5)?)(?=\s|[.\-]|$)", t)
    if m:
        return float(m.group(1))

    first_line = next((line.strip() for line in t.splitlines() if line.strip()), "")
    if re.fullmatch(r"[0-5]", first_line):
        return float(first_line)
    return None



def llm_judge_qa(
    inputs: List[str],
    refs: List[str],
    preds: List[str],
    judge_sample: int = DEFAULT_JUDGE_N,
    batch_size: int = 4,
) -> Tuple[float, float, List[Optional[float]], float, List[str]]:
    n = min(judge_sample, len(inputs))
    prompts = []
    for x, ref, pred in zip(inputs[:n], refs[:n], preds[:n]):
        prompts.append(
            "You are an impartial evaluator of technical question-answering. "
            "Score the candidate answer from 0 to 5 using only the question and reference answer. "
            "Do not reward verbosity and do not penalize concise correct answers. "
            "Return exactly one score from 0 to 5; half-point scores are allowed (0, 0.5, 1, ..., 5). "
            "Output the score first and provide no explanation before it.\n\n"
            "Rubric:\n"
            "0 = completely incorrect or irrelevant\n"
            "1 = mostly incorrect; major errors\n"
            "2 = partially correct; major omissions/errors\n"
            "3 = substantially correct; some omissions or minor errors\n"
            "4 = correct and complete with minor issues\n"
            "5 = fully correct, complete, and precise\n\n"
            f"Question:\n{x}\n\nReference answer:\n{ref}\n\nCandidate answer:\n{pred}\n\nScore:"
        )

    judge_model, judge_tokenizer = load_judge_model()
    raw_scores = generate_texts(judge_model, judge_tokenizer, prompts, max_new_tokens=8, batch_size=batch_size)
    del judge_model, judge_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scores = [_parse_judge_score(t) for t in raw_scores]
    valid = [s for s in scores if s is not None]
    invalid_rate = float(1.0 - len(valid) / len(scores)) if scores else 1.0
    if not valid:
        return float("nan"), float("nan"), scores, invalid_rate, raw_scores
    return float(np.mean(valid)), float(np.std(valid)), scores, invalid_rate, raw_scores


def evaluate_qa(
    adapter_dir: Optional[Path],
    rows: List[dict],
    out_path: Path,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    judge_sample: int = DEFAULT_JUDGE_N,
    use_llm_judge: bool = True,
    base_adapter_dir: Optional[Path] = None,
) -> dict:
    from sentence_transformers import SentenceTransformer

    model, tokenizer = load_model_for_eval(adapter_dir, base_adapter_dir=base_adapter_dir)
    prompts = [qa_prompt(r["input"]) for r in rows]
    generations = generate_texts(model, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    pred_emb = emb_model.encode(generations, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    ref_emb = emb_model.encode([r["output"] for r in rows], batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    sims = np.sum(pred_emb * ref_emb, axis=1)
    del emb_model

    if use_llm_judge:
        judge_mean, judge_std, judge_scores, judge_invalid_rate, judge_raw_outputs = llm_judge_qa(
            [r["input"] for r in rows], [r["output"] for r in rows], generations,
            judge_sample=judge_sample, batch_size=batch_size,
        )
        judge_note = f"External fixed judge: {JUDGE_MODEL}; fixed first {min(judge_sample, len(rows))} examples."
    else:
        judge_scores = []
        judge_raw_outputs = []
        judge_mean = judge_std = float("nan")
        judge_invalid_rate = float("nan")
        judge_note = "LLM judge disabled."

    result = {
        "semantic_similarity_mean": float(np.mean(sims)),
        "semantic_similarity_std": float(np.std(sims)),
        "llm_judge_mean_0_5": float(judge_mean),
        "llm_judge_std_0_5": float(judge_std),
        "judge_invalid_rate": float(judge_invalid_rate),
        "judge_sample_n": int(min(judge_sample, len(rows))) if use_llm_judge else 0,
        "judge_model": JUDGE_MODEL if use_llm_judge else None,
        "judge_note": judge_note,
        "base_adapter_dir": str(base_adapter_dir) if base_adapter_dir else None,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "metrics": result,
                "judge_scores": judge_scores,
                "judge_raw_outputs": judge_raw_outputs,
                "predictions": [
                    {
                        "input": r["input"],
                        "gold": r["output"],
                        "pred": p,
                        "semantic_similarity": float(s),
                    }
                    for r, p, s in zip(rows, generations, sims)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def compute_audit(
    base_adapter: Path,
    task_adapter: Path,
    merged_adapter: Optional[Path] = None,
) -> dict:

    base = load_adapter_state(base_adapter)
    task = load_adapter_state(task_adapter)
    db = state_to_delta_w(base, lora_scaling(base_adapter))
    dt = state_to_delta_w(task, lora_scaling(task_adapter))
    keys = sorted(set(db) & set(dt))
    b = torch.cat([db[k].flatten() for k in keys])
    t = torch.cat([dt[k].flatten() for k in keys])
    coverage = float((t.abs() > 0).float().mean().item())
    base_norm = torch.linalg.vector_norm(b).item()
    task_norm = torch.linalg.vector_norm(t).item()
    effective_amplitude = task_norm / base_norm if base_norm > 0 else float("nan")

    result = {
        "coverage": coverage,
        "task_l2": task_norm,
        "base_l2": base_norm,
        "effective_amplitude": effective_amplitude,
        "task_base_cosine": _global_cosine(t, b),
    }

    if merged_adapter is not None:
        merged = load_adapter_state(merged_adapter)
        dm = state_to_delta_w(merged, lora_scaling(merged_adapter))
        m = torch.cat([dm[k].flatten() for k in keys])
        m_norm = torch.linalg.vector_norm(m).item()
        result["merged_l2"] = m_norm
        result["self_contribution_dot_over_merged_sq"] = float(torch.dot(t, m).item() / (m_norm ** 2)) if m_norm > 0 else float("nan")
        result["task_share_of_merged_norm"] = float(task_norm / m_norm) if m_norm > 0 else float("nan")
    return result



def merge_operator_audit(
    adapter_dirs: List[Path],
    merged_adapter: Path,
    method: str,
    space: str = "delta_w",
    top_k: float = 0.2,
    drop_rate: float = 0.5,
    seed: int = 42,
) -> dict:

    dirs = [Path(p) for p in adapter_dirs]
    merged = state_to_delta_w(load_adapter_state(merged_adapter), lora_scaling(merged_adapter))
    if space == "delta_w":
        sources = [state_to_delta_w(load_adapter_state(p), lora_scaling(p)) for p in dirs]
    else:
        sources = [state_to_delta_w(load_adapter_state(p), lora_scaling(p)) for p in dirs]

    keys = sorted(set.intersection(*[set(x.keys()) for x in sources]) & set(merged.keys()))
    source_stack = torch.cat([torch.stack([s[k].flatten() for s in sources], dim=0) for k in keys], dim=1)
    source_sum = source_stack.sum(dim=0)

    method_l = method.lower().replace("_", "+")
    work = source_stack.clone()
    if method_l.startswith("dare+"):
        keep = 1.0 - drop_rate
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        mask = torch.rand(work.shape, generator=g) < keep
        work = torch.where(mask, work / keep, torch.zeros_like(work))

    if "ties" in method_l:
        trim_mask = torch.zeros_like(work, dtype=torch.bool)
        for i in range(work.shape[0]):
            flat = work[i]
            if top_k >= 1:
                trim_mask[i] = flat != 0
            else:
                kth = max(1, int(flat.numel() * top_k))
                thresh = torch.topk(flat.abs(), kth, sorted=False).values.min()
                trim_mask[i] = flat.abs() >= thresh
        trimmed = torch.where(trim_mask, work, torch.zeros_like(work))
        c_trim = float((trimmed != 0).float().mean().item())
        sign = torch.sign(trimmed.sum(dim=0))
        c_elect = float((sign != 0).float().mean().item())
    else:
        c_trim = float((work != 0).float().mean().item())
        c_elect = float((work.sum(dim=0) != 0).float().mean().item())

    merged_flat = torch.cat([merged[k].flatten() for k in keys])

    source_sum_flat = torch.cat([
        sum((s[k].flatten() for s in sources), torch.zeros_like(sources[0][k].flatten()))
        for k in keys
    ])
    denom = torch.linalg.vector_norm(source_sum_flat).item()
    gamma = float(torch.linalg.vector_norm(merged_flat).item() / denom) if denom > 0 else float("nan")

    return {
        "method": method,
        "space": space,
        "c_trim_empirical": c_trim,
        "c_elect_empirical": c_elect,
        "gamma_norm_ratio": gamma,
        "merged_l2": float(torch.linalg.vector_norm(merged_flat).item()),
        "source_sum_l2": float(denom),
        "top_k": top_k,
        "drop_rate": drop_rate,
        "seed": seed,
    }

def summarize_json_metrics(pattern: str, out_csv: Path):
    rows = []
    for p in sorted(Path(".").glob(pattern)):
        obj = json.loads(p.read_text(encoding="utf-8"))
        m = obj.get("metrics", obj)
        row = {"path": str(p)}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def clean_result_artifacts() -> None:
    for p in [RESULTS_DIR, MERGED_DIR, ANALYSIS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
        for child in p.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
