from __future__ import annotations
import unsloth
import os, re, json, math, random, shutil, gc, time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch

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

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path, max_rows: Optional[int] = None) -> List[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def sample_balanced_logs(path: Path, per_class: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    buckets = {label: [] for label in LOG_LABELS}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            y = str(r.get('output', '')).strip()
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
        "Your task is to classify the given log message as either normal or anomalous. "
        "Respond with exactly one of the following labels: OK or Anomaly.\n\n"
        f"Log message:\n{x}\n\nLabel:"
    )


def qa_prompt(x: str) -> str:
    return f"Answer the following question accurately:\n\n{x}\n\nAnswer:"



def format_sft_example(example: dict, task: str) -> dict:
    if task == 'logs':
        prompt = log_prompt(example['input'])
    elif task == 'qa':
        prompt = qa_prompt(example['input'])
    else:
        raise ValueError(task)
    return {"text": prompt + " " + str(example['output']).strip()}

def format_cpt_example(example: dict) -> dict:
    return {"text": str(example['text']).strip()}


def load_unsloth_model(max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH, load_in_4bit: bool = True, model_name: str = None):
    from unsloth import FastLanguageModel
    if model_name is None:
        model_name = BASE_MODEL
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=TRAIN_DTYPE,
        load_in_4bit=load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return model, tokenizer


def add_lora(model, r=16, lora_alpha=32, lora_dropout=0.05):
    from unsloth import FastLanguageModel
    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )


def train_lora_adapter(
    dataset_rows: List[dict],
    eval_rows: List[dict],
    output_dir: Path,
    seed: int,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    epochs: float = 1.0,
    max_steps: int = -1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    save_steps: int = 500,
    base_adapter_to_merge: Optional[Path] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from peft import PeftModel
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_unsloth_model(max_seq_length=max_seq_length, load_in_4bit=True)

    # For CPT -> FT experiments: load CPT adapter, merge it into base in memory, then attach a fresh LoRA adapter.
    # The merged full model is NOT saved; only the newly trained adapter is saved.
    if base_adapter_to_merge is not None:
        model = PeftModel.from_pretrained(model, str(base_adapter_to_merge), is_trainable=False)
        model = model.merge_and_unload()

    model = add_lora(model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    ds = Dataset.from_list(dataset_rows)
    use_eval = eval_rows and len(eval_rows) > 0
    ds_eval = Dataset.from_list(eval_rows) if use_eval else None

    args = SFTConfig(
        output_dir=str(output_dir / 'trainer_state'),
        dataset_text_field='text',
        max_seq_length=max_seq_length,
        packing=False,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type='cosine',
        optim='adamw_8bit',
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
        report_to='none',
        remove_unused_columns=True,
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, eval_dataset=ds_eval, args=args)
    trainer.train()
    model.save_pretrained(str(output_dir / 'adapter'))
    tokenizer.save_pretrained(str(output_dir / 'adapter'))
    with (output_dir / 'metadata.json').open('w', encoding='utf-8') as f:
        json.dump({
            'base_model': BASE_MODEL,
            'seed': seed,
            'n_train': len(dataset_rows),
            'max_seq_length': max_seq_length,
            'epochs': epochs,
            'max_steps': max_steps,
            'learning_rate': learning_rate,
            'bf16': True,
            'fp16': False,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'base_adapter_to_merge': str(base_adapter_to_merge) if base_adapter_to_merge else None,
        }, f, ensure_ascii=False, indent=2)
    del trainer, model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    return output_dir / 'adapter'


def load_adapter_state(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    adapter_dir = Path(adapter_dir)
    st_path = adapter_dir / 'adapter_model.safetensors'
    if st_path.exists():
        from safetensors.torch import load_file
        return load_file(str(st_path), device='cpu')
    pt_path = adapter_dir / 'adapter_model.bin'
    if pt_path.exists():
        return torch.load(pt_path, map_location='cpu')
    raise FileNotFoundError(f'No adapter_model.safetensors or adapter_model.bin in {adapter_dir}')


def save_adapter_state_like(reference_adapter_dir: Path, state: Dict[str, torch.Tensor], out_dir: Path, metadata: dict):
    from safetensors.torch import save_file
    reference_adapter_dir = Path(reference_adapter_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ['adapter_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json']:
        src = reference_adapter_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(out_dir / 'adapter_model.safetensors'))
    with (out_dir / 'merge_metadata.json').open('w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _merge_keys(states: List[Dict[str, torch.Tensor]]) -> List[str]:
    keys = sorted(set.intersection(*[set(s.keys()) for s in states]))
    return [k for k in keys if torch.is_floating_point(states[0][k])]


def model_soup(states: List[Dict[str, torch.Tensor]], weights: Optional[List[float]]=None) -> Dict[str, torch.Tensor]:
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    wsum = sum(weights)
    weights = [w / wsum for w in weights]
    out = {k: states[0][k].clone() for k in states[0].keys()}
    for k in _merge_keys(states):
        out[k] = sum(w * s[k].float() for w, s in zip(weights, states)).to(states[0][k].dtype)
    return out


def task_arithmetic(states: List[Dict[str, torch.Tensor]], alpha: float=1.0, weights: Optional[List[float]]=None) -> Dict[str, torch.Tensor]:
    # In LoRA-adapter space the adapter itself is the saved task-specific delta proxy.
    soup = model_soup(states, weights=weights)
    out = {k: v.clone() for k, v in soup.items()}
    for k in _merge_keys(states):
        out[k] = (alpha * soup[k].float()).to(soup[k].dtype)
    return out


def ties_merge(states: List[Dict[str, torch.Tensor]], top_k: float=0.2, alpha: float=1.0) -> Dict[str, torch.Tensor]:
    assert 0 < top_k <= 1
    out = {k: states[0][k].clone() for k in states[0].keys()}
    for k in _merge_keys(states):
        stack = torch.stack([s[k].float() for s in states], dim=0)
        # Trim: keep largest magnitudes per tensor.
        trimmed = []
        for t in stack:
            flat = t.flatten()
            if top_k < 1:
                kth = max(1, int(flat.numel() * top_k))
                thresh = torch.topk(flat.abs(), kth, sorted=False).values.min()
                t = torch.where(t.abs() >= thresh, t, torch.zeros_like(t))
            trimmed.append(t)
        stack = torch.stack(trimmed, dim=0)
        # Elect sign by aggregate sign, disjoint merge same-sign params.
        sign = torch.sign(stack.sum(dim=0))
        same = torch.where(torch.sign(stack) == sign.unsqueeze(0), stack, torch.zeros_like(stack))
        denom = (same != 0).sum(dim=0).clamp(min=1)
        merged = same.sum(dim=0) / denom
        out[k] = (alpha * merged).to(states[0][k].dtype)
    return out


def dare_state(state: Dict[str, torch.Tensor], drop_rate: float=0.5, seed: int=42) -> Dict[str, torch.Tensor]:
    assert 0 <= drop_rate < 1
    g = torch.Generator(device='cpu').manual_seed(seed)
    keep_prob = 1.0 - drop_rate
    out = {k: v.clone() for k, v in state.items()}
    for k, v in state.items():
        if torch.is_floating_point(v):
            mask = torch.rand(v.shape, generator=g) < keep_prob
            out[k] = torch.where(mask, v.float() / keep_prob, torch.zeros_like(v.float())).to(v.dtype)
    return out


def merge_adapters(adapter_dirs: List[Path], out_dir: Path, method: str, seed: int=42, alpha: float=1.0, top_k: float=0.2, drop_rate: float=0.5):
    states = [load_adapter_state(Path(p)) for p in adapter_dirs]
    method = method.lower()
    if method.startswith('dare+'):
        inner = method.split('+', 1)[1]
        states = [dare_state(s, drop_rate=drop_rate, seed=seed+i) for i, s in enumerate(states)]
        method = inner
    if method in ['model_soup', 'soup', 'model soups', 'model_soups']:
        merged = model_soup(states)
    elif method in ['task_arithmetic', 'task arithmetic', 'ta']:
        merged = task_arithmetic(states, alpha=alpha)
    elif method == 'ties':
        merged = ties_merge(states, top_k=top_k, alpha=alpha)
    else:
        raise ValueError(f'Unknown merge method: {method}')
    metadata = {'adapter_dirs': [str(p) for p in adapter_dirs], 'method': method, 'seed': seed, 'alpha': alpha, 'top_k': top_k, 'drop_rate': drop_rate}
    save_adapter_state_like(adapter_dirs[0], merged, out_dir, metadata)
    return out_dir


def tensor_stats(t: torch.Tensor) -> dict:
    x = t.detach().float().flatten()
    if x.numel() == 0:
        return {}
    return {
        'numel': int(x.numel()),
        'nnz': int((x != 0).sum().item()),
        'zero_frac': float((x == 0).float().mean().item()),
        'mean': float(x.mean().item()),
        'std': float(x.std(unbiased=False).item()),
        'abs_mean': float(x.abs().mean().item()),
        'l2': float(torch.linalg.vector_norm(x).item()),
        'min': float(x.min().item()),
        'max': float(x.max().item()),
    }



def adapter_vector_analysis(adapter_a: Path, adapter_b: Path, out_prefix: Path) -> pd.DataFrame:
    a, b = load_adapter_state(adapter_a), load_adapter_state(adapter_b)
    keys = _merge_keys([a, b])
    rows = []
    flat_a, flat_b = [], []
    for k in keys:
        ta, tb = a[k].float().flatten(), b[k].float().flatten()
        n = min(ta.numel(), tb.numel())
        ta, tb = ta[:n], tb[:n]
        conflict = ((ta * tb) < 0).float().mean().item() if n else 0.0
        denom = (torch.linalg.vector_norm(ta) * torch.linalg.vector_norm(tb)).item()
        cos = float(torch.dot(ta, tb).item() / denom) if denom > 0 else float('nan')
        st_a = tensor_stats(a[k]); st_b = tensor_stats(b[k])
        rows.append({
            'key': k,
            'shape': tuple(a[k].shape),
            'cosine': cos,
            'conflict_frac': conflict,
            'logs_l2': st_a.get('l2'),
            'qa_l2': st_b.get('l2'),
            'logs_zero_frac': st_a.get('zero_frac'),
            'qa_zero_frac': st_b.get('zero_frac'),
            'logs_abs_mean': st_a.get('abs_mean'),
            'qa_abs_mean': st_b.get('abs_mean'),
        })
        flat_a.append(ta); flat_b.append(tb)
    df = pd.DataFrame(rows)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_prefix) + '_per_tensor.csv', index=False)
    if flat_a:
        va, vb = torch.cat(flat_a), torch.cat(flat_b)
        denom = (torch.linalg.vector_norm(va) * torch.linalg.vector_norm(vb)).item()
        summary = {
            'global_cosine': float(torch.dot(va, vb).item() / denom) if denom > 0 else float('nan'),
            'global_conflict_frac': float(((va * vb) < 0).float().mean().item()),
            'logs': tensor_stats(va),
            'qa': tensor_stats(vb),
        }
        with open(str(out_prefix) + '_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return df


def generate_texts(model, tokenizer, prompts: List[str], max_new_tokens: int=128, batch_size: int=8) -> List[str]:
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=DEFAULT_MAX_SEQ_LENGTH).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        outs.extend([d.strip() for d in decoded])
    return outs


def load_model_for_eval(adapter_dir: Optional[Path]=None, max_seq_length: int=DEFAULT_MAX_SEQ_LENGTH, merge_adapter: bool=False):
    from peft import PeftModel
    model, tokenizer = load_unsloth_model(max_seq_length=max_seq_length, load_in_4bit=True)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        if merge_adapter:
            model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def load_judge_model(max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH):
    """Load the model for LLM-as-Judge"""
    return load_unsloth_model(
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        model_name=JUDGE_MODEL
    )


def normalize_log_label(text: str) -> str:
    t = text.strip().lower()
    if re.search(r'\banomaly\b|abnormal|error|fault|fail', t):
        return 'Anomaly'
    if re.search(r'\bok\b|normal|benign', t):
        return 'OK'
    # Conservative fallback: many generative models answer with explanation; default to OK only if no anomaly cue.
    return 'OK'


def evaluate_logs(adapter_dir: Optional[Path], rows: List[dict], out_path: Path, batch_size: int=16, max_new_tokens: int=8) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    model, tokenizer = load_model_for_eval(adapter_dir)
    prompts = [log_prompt(r['input']) for r in rows]
    generations = generate_texts(model, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
    y_true = [r['output'] for r in rows]
    y_pred = [normalize_log_label(g) for g in generations]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LOG_LABELS, average='binary', pos_label='Anomaly', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=LOG_LABELS).tolist()
    result = {'accuracy': acc, 'precision_anomaly': precision, 'recall_anomaly': recall, 'f1_anomaly': f1, 'confusion_matrix_labels': LOG_LABELS, 'confusion_matrix': cm}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump({'metrics': result, 'predictions': [{'input': r['input'], 'gold': r['output'], 'pred': p, 'raw': g} for r, p, g in zip(rows, y_pred, generations)]}, f, ensure_ascii=False, indent=2)
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    return result


def _parse_judge_score(text: str) -> Optional[float]:
    m = re.search(r'\b([0-5](?:\.\d+)?)\b', text.strip())
    return float(m.group(1)) if m else None


def llm_judge_qa(inputs: List[str], refs: List[str], preds: List[str], judge_sample: int=80, batch_size: int=4) -> Tuple[float, float, List[Optional[float]]]:
    n = min(judge_sample, len(inputs))
    prompts = []
    for x, ref, pred in zip(inputs[:n], refs[:n], preds[:n]):
        prompts.append(
            "You are an impartial evaluator. Rate the candidate answer from 0 to 5. "
            "Use only the reference answer and the question. Output only one number.\n\n"
            f"Question:\n{x}\n\nReference answer:\n{ref}\n\nCandidate answer:\n{pred}\n\nScore:"
        )
    
    # Load the judge model (Llama)
    judge_model, judge_tokenizer = load_judge_model()
    
    # Generate evaluation scores
    raw_scores = generate_texts(judge_model, judge_tokenizer, prompts, max_new_tokens=4, batch_size=batch_size)
    
    # Free memory
    del judge_model, judge_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # Parse scores
    scores = [_parse_judge_score(t) for t in raw_scores]
    valid = [s for s in scores if s is not None]
    if not valid:
        return float('nan'), float('nan'), scores
    return float(np.mean(valid)), float(np.std(valid)), scores


def evaluate_qa(adapter_dir: Optional[Path], rows: List[dict], out_path: Path, batch_size: int=4, max_new_tokens: int=256, judge_sample: int=80, use_llm_judge: bool=True) -> dict:
    from sentence_transformers import SentenceTransformer
    model, tokenizer = load_model_for_eval(adapter_dir)
    prompts = [qa_prompt(r['input']) for r in rows]
    generations = generate_texts(model, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    pred_emb = emb_model.encode(generations, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    ref_emb = emb_model.encode([r['output'] for r in rows], batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    sims = np.sum(pred_emb * ref_emb, axis=1)

    judge_scores = []
    if use_llm_judge:
        judge_mean, judge_std, judge_scores = llm_judge_qa(
            [r['input'] for r in rows], [r['output'] for r in rows], generations,
            judge_sample=judge_sample, batch_size=batch_size,
        )
        judge_note = f'LLM-as-Judge with {JUDGE_MODEL} on first {min(judge_sample, len(rows))} examples.'
    else:
        proxy = np.clip((sims - 0.2) / 0.6 * 5, 0, 5)
        judge_scores = [float(x) for x in proxy[:judge_sample]]
        judge_mean, judge_std = float(np.mean(judge_scores)), float(np.std(judge_scores))
        judge_note = 'Proxy judge: semantic similarity mapped to 0..5 because use_llm_judge=False.'

    result = {
        'semantic_similarity_mean': float(np.mean(sims)),
        'semantic_similarity_std': float(np.std(sims)),
        'llm_judge_mean_0_5': judge_mean,
        'llm_judge_std_0_5': judge_std,
        'judge_note': judge_note,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump({'metrics': result, 'judge_scores': judge_scores, 'predictions': [{'input': r['input'], 'gold': r['output'], 'pred': p, 'semantic_similarity': float(s)} for r, p, s in zip(rows, generations, sims)]}, f, ensure_ascii=False, indent=2)
    return result

def summarize_json_metrics(pattern: str, out_csv: Path):
    rows = []
    for p in sorted(Path('.').glob(pattern)):
        with p.open('r', encoding='utf-8') as f:
            obj = json.load(f)
        m = obj.get('metrics', obj)
        row = {'path': str(p)}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
