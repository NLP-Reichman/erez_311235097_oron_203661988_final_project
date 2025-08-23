import os
import torch.multiprocessing as _mp

import torch
from torch import nn
from safetensors.torch import save_model, load_model

import numpy as np
import pandas as pd
import math
import random
import json
from typing import Any, Dict, Tuple, List, Sequence, Optional
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    T5TokenizerFast as T5Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    T5ForSequenceClassification,
)

import matplotlib
# Make sure a headless backend is used (important when running under DDP)
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# LoRA imports
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    
# Rank helper (so only rank-0 does heavy I/O/plotting)
IS_RANK0 = os.environ.get("LOCAL_RANK", "0") == "0"

# Bind each rank to its own GPU early (prevents end-of-run barrier stalls)
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)

class T5DecoderTokenClassifier(nn.Module):
    """
    T5 sequence classification that uses the decoder 1-token head (HuggingFace 'T5ForSequenceClassification').
    """
    def __init__(self, model_name: str, num_labels: int = 2):
        super().__init__()
       # Load T5 with a sequence-classification head
        self.model = T5ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="single_label_classification",
            low_cpu_mem_usage=True,
        )
        # Training (especially with grad ckecpoint) must disable KV cache
        self.model.config.use_cache = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None
    ):
        """
        Forward pass delegates to the built-in T5ForSequenceClassification.
        Returns:
          - If labels provided: (loss, logits)
          - Else: logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,               # Trainer apply CrossEntropyLoss + any label_smoothing
            return_dict=True
        )
        # T5ForSequenceClassification returns a SequenceClassifierOutput with loss and logits fields
        if labels is not None:
            return outputs.loss, outputs.logits
        return outputs.logits


def serialize_row(
    row: pd.Series,
    label_col: str,
    task_def: str
) -> Tuple[str, int]:
    """
    Serialize a DataFrame row into a prompt + label, dynamically:
    - task_def: short instruction, e.g. "Classify flight delay"
    - row: one record (including the label column)
    Returns:
      - prompt_str: "<task_def> | { <col1>:<val1>,... }"
      - label: int(row[label_col])
    """
    # build per-column values, keeping floats to 4 decimal places
    features = {}
    # skip metadata split-flag columns when serializing features
    flag_cols = {
        "test", "validation",
        "train_50","train_100","train_200","train_500",
        "train_1K","train_5K","train_10K",
        "train_50K","train_100K","train_500K",
        "train_1M","train_5M"
    }
    for col, val in row.items():
        if col == label_col or col in flag_cols:
            continue
        if pd.isna(val):
            features[col] = None
        elif isinstance(val, int):
            features[col] = val
        elif isinstance(val, float):
            features[col] = round(val, 4)
        else:
            features[col] = str(val)
    record_json = json.dumps(features, separators=(',',':'))
    
    # 2. Prepend the instruction
    prompt_str = f"{task_def} | {record_json}"
    
    # 3. Extract the integer label
    label = int(row[label_col])
    
    return prompt_str, label


def preprocess_batch(
    df: pd.DataFrame,
    tokenizer,
    label_col: str,
    task_def: str,
    max_length: int = 320
) -> List[Dict[str, torch.Tensor]]:
    """
    Tokenize prompts and set integer labels for classification.
    
    Parameters:
    - df: DataFrame including the label_col and feature columns.
    - tokenizer: T5Tokenizer or compatible tokenizer.
    - label_col: name of the column with binary labels (0/1).
    - task_def: short instruction prefix for every prompt.
    - max_length: max token length (truncates if exceeded).
    
    Returns:
    - A list of dicts, each with keys 'input_ids', 'attention_mask', and 'labels'.
    """
    inputs: List[Dict[str, torch.Tensor]] = []
    for _, row in df.iterrows():
        prompt, label = serialize_row(row, label_col, task_def)
        enc = tokenizer(
            prompt,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        inputs.append({
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        })
    return inputs


def compute_metrics(
    pred_out: Any,
    ratio: float,
    train_size: int,
    val_size: int,
    test_size: int
) -> Dict[str, float]:
    """
    Compute classification metrics from logits and print a concise summary.

    Parameters:
    - pred_out: output from Trainer.predict(), with .predictions and .label_ids
    - ratio: float, subsample ratio used in this experiment
    - train_size: int, number of training examples
    - val_size: int, number of validation examples
    - test_size: int, number of test examples

    Returns:
    - dict of metrics: accuracy, precision, recall, f1, auc
    """
    # Extract logits and true labels
    logits = pred_out.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = logits if not hasattr(logits, "detach") else logits.detach().cpu().numpy()
    y_pred = np.argmax(logits, axis=-1)
    y_true = pred_out.label_ids.reshape(-1)

    # Compute metrics
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    try:
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        auc = roc_auc_score(y_true, probs[:, 1])
    except Exception:
        auc = float('nan')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    title: str = ''
) -> None:
    """
    Plot a confusion matrix for binary or multi-class classification.
    
    Parameters:
    - y_true: True labels (list or array of ints).
    - y_pred: Predicted labels (list or array of ints).
    - title: Title for the plot (e.g., "ratio=0.01").
    """
    # Determine label set
    labels = sorted(set(y_true))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Create plot
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel='Predicted',
        ylabel='True',
        title=title
    )
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Annotate cells
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black'
            )
    
    plt.tight_layout()
    # plt.show()
    # instead of show, return the figure for external saving
    return fig


def run_2_t5_dec_head_hf_experiments(
    df: pd.DataFrame,
    label_col: str,
    task_def: str,
    model_name: str,
    output_dir: str,
    subsample_ratios: Optional[List[Optional[float]]] = None,
    subsample_sets: Optional[List[str]] = None,
    test_size: Optional[float] = None,
    val_ratio: Optional[float] = None,
    random_state: int = 42,
    num_train_epochs: int = 4,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    optim: str = "adamw_torch",
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.1,
    #fp16: bool = torch.cuda.is_available(),
    fp16: Optional[bool] = None,
    bf16: bool = False,
    dataloader_num_workers: int = 4,
    max_grad_norm: float = 1.0,
    use_lora: bool = True,
    lora_r: int = 4,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    #label_smoothing_factor: float = 0.0,
    eval_strategy: str = "steps",
    eval_steps: int = 200,
    logging_strategy: str = "steps",
    logging_steps: int = 50,
    save_strategy: str = "steps",
    save_steps: int = None,
    save_total_limit: int = 2,
    overwrite_output_dir: bool = True,
    save_safetensors: bool = False,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "f1",
    greater_is_better: bool = True,
    early_stopping_patience: int = 3,
    max_length: int = 320,
    run_tag: Optional[str] = None,
) -> Dict[str, Any]:
    # Set seeds
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # Speed over strict reproducibility
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # only now, after spawn has been set, decide fp16
    if fp16 is None:
        fp16 = torch.cuda.is_available()

    # Tokenizer & split
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    # prepare splits
    if subsample_sets:
        missing = [s for s in subsample_sets if f"train_{s}" not in df.columns]
        if missing:
            raise ValueError(f"Missing train_* columns: {missing}")
        df_test = df[df['test']]
        df_val  = df[df['validation']]
    else:
        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df[label_col]
        )

    results = {"confusion_matrices": []}
    cms: List[Tuple[float, List[int], List[int]]] = []

    # build experiments
    exp_iter = [(r, None) for r in subsample_ratios] if subsample_sets is None else [(None, s) for s in subsample_sets]
    base_classifier = T5DecoderTokenClassifier(model_name, num_labels=2)
    # base_classifier = torch.compile(base_classifier)
    for ratio, suffix in exp_iter:
        print(f"\n=== Processing ratio {ratio} ===")

        # Initialize model
        classifier = copy.deepcopy(base_classifier)
        
        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                # T5 attention dense names are q,k,v,o; q+v is a common efficient choice
                # target_modules=["q", "k", "v", "o"],
                target_modules = ["q","k","v","o","wi","wo","wi_0","wi_1"],
                # keep the classification layer trainable & saved
                modules_to_save=["classifier"],
            )
            classifier.model = get_peft_model(classifier.model, peft_config)
            # Memory/graph savers for training
            if hasattr(classifier.model, "gradient_checkpointing_enable"):
                classifier.model.gradient_checkpointing_enable()
            if hasattr(classifier.model, "enable_input_require_grads"):
                classifier.model.enable_input_require_grads()
            classifier.model.config.use_cache = False
            # just in case PEFT froze it, ensure head stays trainable
            for n, p in classifier.model.named_parameters():
                if ("classifier" in n) or ("score" in n):
                    p.requires_grad = True
            torch.cuda.empty_cache()
            # DEBUG: LoRA trainable parameters
            total_params     = sum(p.numel() for p in classifier.model.parameters())
            trainable_params = sum(p.numel() for p in classifier.model.parameters() if p.requires_grad)
            print("LoRA applied. Trainable parameters:")
            print(f"  Total: {total_params:,}")
            print(f"  Trainable: {trainable_params:,} ({100 * trainable_params/total_params:.1f}%)")

        # select training subset
        if suffix is not None:
            sub_df = df[df[f"train_{suffix}"]]
            df_tr  = sub_df
            df_val = df_val
        else:
            if ratio < 1.0:
                sub_df, _ = train_test_split(
                    df_train, train_size=ratio, random_state=random_state, stratify=df_train[label_col]
                )
            else:
                sub_df = df_train.copy()
            df_tr, df_val = train_test_split(
                sub_df, test_size=val_ratio, random_state=random_state, stratify=sub_df[label_col]
            )

        # Preprocess
        # Batch‐tokenize training and validation sets
        # 1. Serialize prompts
        prompts_tr = [serialize_row(row, label_col, task_def)[0] 
                      for _, row in df_tr.iterrows()]
        prompts_val = [serialize_row(row, label_col, task_def)[0] 
                       for _, row in df_val.iterrows()]

        # 2. Tokenize (no global padding; let the collator pad per batch)
        enc_tr = tokenizer(
            prompts_tr,
            max_length=max_length,
            padding=False,
            truncation=True,
        )
        enc_val = tokenizer(
            prompts_val,
            max_length=max_length,
            padding=False,
            truncation=True,
        )

        # 3. Attach labels
        labels_tr = torch.tensor(df_tr[label_col].values, dtype=torch.long)
        labels_val = torch.tensor(df_val[label_col].values, dtype=torch.long)

        # 4. Build lists of feature dicts for Trainer
        train_data = [
            {
                "input_ids": enc_tr["input_ids"][i],
                "attention_mask": enc_tr["attention_mask"][i],
                "labels": labels_tr[i],
            }
            for i in range(len(labels_tr))
        ]
        val_data = [
            {
                "input_ids": enc_val["input_ids"][i],
                "attention_mask": enc_val["attention_mask"][i],
                "labels": labels_val[i],
            }
            for i in range(len(labels_val))
        ]

         # DEBUG: token-lengths & sample
        lengths = [len(f["input_ids"]) for f in train_data]
        print(f"DEBUG: token-lengths in this batch → min={min(lengths)}, max={max(lengths)}")
        sample_prompt = tokenizer.decode(train_data[0]["input_ids"], skip_special_tokens=True)
        sample_label  = train_data[0]["labels"].item()
        print(f"DEBUG: Sample training prompt: {sample_prompt}")
        print(f"DEBUG: Sample training label: {sample_label}")
        print(f"DEBUG: Training data size: {len(train_data)}")
        print(f"DEBUG: Validation data size: {len(val_data)}")

        # Data collator
        collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

        # PRE-TRAINING DEBUG
        print("\n=== PRE-TRAINING DEBUG ===")
        test_batch = collator(train_data[:2])
        with torch.no_grad():
            classifier.model.eval()
            out = classifier.model(**test_batch)
        classifier.model.train()
        test_loss   = out.loss
        test_logits = out.logits
        print(f"DEBUG: Pre-training loss: {test_loss.item():.4f}")
        print(f"DEBUG: Pre-training logits: {test_logits}")
        print(f"DEBUG: Pre-training predictions: {torch.argmax(test_logits, dim=-1)}")

        # Choose a readable per-experiment folder (no 'ratio_None')
        if suffix is not None:
            subdir = f"set_{suffix}"
        elif ratio is None:
            subdir = "full"
        else:
            subdir = f"ratio_{ratio}"
        # Nest under the per-run folder so checkpoints live with the PDF/log
        train_out_dir = os.path.join(output_dir, run_tag, subdir) if run_tag else os.path.join(output_dir, subdir)

        # Training args
        args = TrainingArguments(
            output_dir=train_out_dir,
            overwrite_output_dir=overwrite_output_dir,
            seed=random_state,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_safetensors=save_safetensors,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            logging_strategy=logging_strategy,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            save_steps=save_steps or eval_steps,
            save_total_limit=save_total_limit,
            save_on_each_node=True,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optim=optim,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            max_grad_norm=max_grad_norm,
            fp16=fp16,
            bf16=bf16,
            #label_smoothing_factor=label_smoothing_factor,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_persistent_workers=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            disable_tqdm=(os.environ.get("LOCAL_RANK", "0") != "0"),
            report_to=[],
        )

        # Label smoothing
        trainer_kwargs = {"model": classifier, "args": args,
                          "train_dataset": train_data, "eval_dataset": val_data,
                          "tokenizer": tokenizer, "data_collator": collator,
                          "compute_metrics": lambda pred: compute_metrics(
                              pred, ratio, len(train_data), len(val_data), len(df_test)
                          ),
                          "callbacks": [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
                         }
        #if label_smoothing_factor > 0:
        #    trainer_kwargs["label_smoother"] = LabelSmoother(label_smoothing_factor=label_smoothing_factor)

        trainer = Trainer(**trainer_kwargs)

        # Train
        trainer.train()

        # Evaluate on test across ALL ranks (gathered)
        test_data = preprocess_batch(df_test, tokenizer, label_col, task_def, max_length)
        pred_out = trainer.predict(test_data)

        if trainer.is_world_process_zero():
            key = suffix if (subsample_sets is not None and suffix is not None) else ratio
            metrics = {k.replace("test_", ""): v
                       for k, v in pred_out.metrics.items()
                       if k.startswith("test_")}
            print(f"Test metrics (ratio={key}, train={len(train_data)}, val={len(val_data)}, test={len(df_test)}):")
            print(
                f"  accuracy={metrics.get('accuracy', float('nan')):.4f}, "
                f"precision={metrics.get('precision', float('nan')):.4f}, "
                f"recall={metrics.get('recall', float('nan')):.4f}, "
                f"f1={metrics.get('f1', float('nan')):.4f}, "
                f"auc={metrics.get('auc', float('nan')):.4f}"
            )
            results["confusion_matrices"].append((key, metrics))

            y_pred = np.argmax(pred_out.predictions, axis=-1)
            y_true = pred_out.label_ids.reshape(-1)
            cms.append((key, y_true.tolist(), y_pred.tolist()))

        # FREE GPU MEMORY BEFORE NEXT EXPERIMENT
        try:
            # move model off GPU, drop Trainer and big tensors
            trainer.model.to("cpu")
        except Exception:
            pass
        del trainer, classifier
        del enc_tr, enc_val, labels_tr, labels_val
        del train_data, val_data, test_data, pred_out
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # optional: reset peak stats for cleaner monitoring
            torch.cuda.reset_peak_memory_stats()
        # keep ranks in lockstep so one rank doesn’t start early
        # if torch.distributed.is_initialized():
        #     torch.distributed.barrier()

    # Metrics vs ratio - return as a figure
    def plot_metrics_vs_ratio(results):
        fig, ax = plt.subplots(figsize=(8,6))
        def _to_num(k):
            if isinstance(k, str) and k[-1] in ('K','M'):
                return float(k[:-1])*(1e3 if k[-1]=='K' else 1e6)
            return float(k)
        metrics_map = {k: m for k,m in results["confusion_matrices"]}
        keys = sorted(metrics_map.keys(), key=_to_num)
        for metric in metrics_map[keys[0]].keys():
            ax.plot(keys, [metrics_map[k][metric] for k in keys],
                    marker="o", label=metric)
        ax.set(xlabel="Subsample Ratio", ylabel="Metric",
               title="Metrics vs Subsample Ratio")
        ax.legend()
        fig.tight_layout()
        return fig

    # Create figures only on rank-0 to avoid mem/teardown issues across ranks
    report_figs = []
    if IS_RANK0:
        for ratio, y_true, y_pred in cms:
            report_figs.append(
                plot_confusion_matrix(y_true, y_pred, title=f"ratio={ratio}")
            )
        if results["confusion_matrices"]:
            report_figs.append(plot_metrics_vs_ratio(results))

    # include raw confusion data for combined plotting in the runner
    results["raw_cms"] = cms
    return results, report_figs