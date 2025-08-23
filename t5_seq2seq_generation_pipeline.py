import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, List
import matplotlib.pyplot as plt
import wandb
import torch.cuda
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from peft import LoraConfig, get_peft_config, get_peft_model, TaskType
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)


def serialize_row(row: pd.Series, label_col: str) -> Tuple[str, str]:
    """
    Build a prompt with a clear classification instruction and a single-token target.
    """
    features = [f"{col} = {val}" for col, val in row.items() if col != label_col]
    body = ", ".join(features)

    # Flights dataset
    if label_col == "label":
        prompt = f"Given the following flight information, determine whether the flight is delayed or on-time:\n{body}\n \
                   Based on this information, is the flight delayed or on-time (yes or no)?"
    # Higgs dataset
    elif label_col == "is_higgs_event":
        prompt = f"Classify whether this collision is a Higgs event (yes or no).\nFeatures: {body}"
    else:
        raise ImportError("Unsupported dataset, please provide the required prompt")

    # single-token target: "0" or "1"
    target = str(int(row[label_col]))
    return prompt, target


def compute_metrics(pred_out: Any) -> Dict[str, float]:
    """
    Decode generated token IDs into ints and compute classification metrics.
    """
    true_labels, pred_labels = calc_true_and_pred_labels(pred_out)

    # metrics
    res = {
        "accuracy":  accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "recall":    recall_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "f1":        f1_score(true_labels, pred_labels, average="weighted", zero_division=0),
    }
    # AUC
    try:
        raw_preds = pred_out.predictions
        # reconstruct class probabilities if available
        if not (isinstance(raw_preds, np.ndarray) and raw_preds.ndim == 2):
            # we still have logits
            if isinstance(raw_preds, tuple):
                logits = raw_preds[0]
            else:
                logits = raw_preds
            if hasattr(logits, "detach"):
                logits = logits.detach().cpu().numpy()
            if logits.ndim == 3:
                logits = logits[:, 0, :]
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            proba = exp / exp.sum(axis=1, keepdims=True)
            uniq = np.unique(true_labels)
            if len(uniq) == 2:
                res["auc"] = roc_auc_score(true_labels, proba[:, 1])
            else:
                res["auc"] = roc_auc_score(true_labels, proba, multi_class="ovr")
    except Exception:
        pass

    print(res)
    return res


def preprocess_batch(
    df: pd.DataFrame,
    tokenizer: T5Tokenizer,
    label_col: str,
    max_length: int = 350,
    target_length: int = 1
) -> List[Dict[str, np.ndarray]]:
    """
    Tokenize prompts and targets into input_ids, attention_mask, and labels.
    """

    inputs = []
    for _, row in df.iterrows():
        prompt, target = serialize_row(row, label_col)
        bool_target = "yes" if target == "1" else "no"

        enc = tokenizer(
            prompt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt = tokenizer(
            bool_target,
            max_length=target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,   # don’t insert EOS/BOS
        )

        label_token_id = tgt.input_ids[0].tolist()

        inputs.append({
            "input_ids": enc.input_ids[0],
            "attention_mask": enc.attention_mask[0],
            "labels": label_token_id,
        })

    return inputs

def plot_confusion_matrix(
    y_true: List[Any],
    y_pred: List[Any],
    title: str = ''
) -> None:
    """
    Plot a single confusion matrix.
    """
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel='Predicted', ylabel='True', title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j],
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else 'black'
            )
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()
    plt.close()
    print(cm)


def calc_true_and_pred_labels(pred_out: Any):
    """
    Decode generated token IDs into ints and compute true and predicted labels.
    """

    # Extract predictions and labels
    pred_label_token_ids = pred_out.predictions[:, 1]
    true_label_token_ids = pred_out.label_ids[:, 0]

    # Map token IDs back to token strings if desired
    pred_label_tokens = np.array([tokenizer.decode(seq) for seq in pred_label_token_ids])
    true_label_tokens = np.array([tokenizer.decode(seq) for seq in true_label_token_ids])

    pred_labels = []
    for token in pred_label_tokens:
        if token == "yes":
            pred_labels.append(1)
        elif token == "no":
            pred_labels.append(0)
        else:
            print(f"Warning! predicted label token ({token}) is other than \"yes\" or \"no\"")
            pred_labels.append(0)

    assert np.all((true_label_tokens == "yes") | (true_label_tokens == "no")), \
        "True label tokens contain IDs other than \"yes\" or \"no\""

    true_labels = [1 if token == "yes" else 0 for token in true_label_tokens]

    return true_labels, pred_labels

def run_t5_seq2seq_generation_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_size: str,
    label_col: str,
    model_name: str,
    output_dir: str,
    random_state: int,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    learning_rate: float,
    use_lora: bool = False,
    lora_r: int = 4,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    warmup_ratio: float = 0.1,
) -> Dict[str, Any]:
    """
    Runs a T5 sequence-to-sequence binary classification experiment with optional LoRA fine-tuning.

    This function fine-tunes a T5 model for a binary classification task (yes/no) using a
    sequence-to-sequence approach, where the model generates either "yes" or "no" as output.
    Supports parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation).

    Args:
        train_df: Training dataset as a DataFrame containing text and labels
        val_df: Validation dataset as a DataFrame containing text and labels
        test_df: Test dataset as a DataFrame containing text and labels
        train_size: Identifier for the training set size (for logging and naming purposes)
        label_col: Name of the column containing binary labels (typically 'yes'/'no' or 1/0)
        model_name: Name or path of the pre-trained T5 model to use (e.g., 't5-small', 't5-base')
        output_dir: Directory to save model checkpoints, logs, and outputs
        random_state: Seed for reproducibility of random processes
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device during training
        per_device_eval_batch_size: Batch size per device during evaluation
        learning_rate: Learning rate for the optimizer
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning (default: False)
        lora_r: LoRA rank parameter (determines the size of low-rank matrices)
        lora_alpha: LoRA alpha parameter (scaling factor for LoRA weights)
        lora_dropout: Dropout probability for LoRA layers
        warmup_ratio: Ratio of total training steps for linear learning rate warmup

    Returns:
        Dictionary containing:
        - metrics: Dictionary with accuracy, precision, recall, and F1-score
        - true_labels: List of true labels from the test set
        - pred_labels: List of predicted labels from the model
    """

    device = torch.device("cuda:0")

    if torch.cuda.is_available():
        print(f"CUDA is available, using {device}")
    else:
        print(f"CUDA is not available, using CPU")

    wandb.init(
        project='NLP Project',
        name=f"Experiment_{model_name}_{train_size}",
        group='model_comparison',
        config={
            'model_name': model_name,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'batch_size': per_device_train_batch_size,
            'epochs': num_train_epochs,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'warmup_ratio': warmup_ratio,
        }
    )

    # make tokenizer global for compute_metrics
    global tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model     = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("no", add_special_tokens=False)[0]
    print(f"[yes_id, no_id] = {[yes_id, no_id]}")

    def only_yes_no(batch_id, input_ids):
        return [yes_id, no_id]

    # apply LoRA if requested
    if use_lora:
        print("--------------- using lora -------------")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        model = get_peft_model(model, peft_config)

    results = {"metrics": None, "true_labels": None, "pred_labels": None}
    cms: List[Tuple[float, List[int], List[int]]] = []

    # tokenize into lists of example‐dicts
    train_data = preprocess_batch(train_df, tokenizer, label_col)
    val_data   = preprocess_batch(val_df, tokenizer, label_col)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=2,  # room for BOS + one new token
        seed=random_state,
        report_to="wandb",
        overwrite_output_dir=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        gradient_accumulation_steps=1,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        optim="adamw_torch",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # fine-tune
    trainer.train()

    # prepare test set and do a single, constrained generation pass
    test_data = preprocess_batch(test_df, tokenizer, label_col)

    test_predictions = trainer.predict(
        test_data,
        prefix_allowed_tokens_fn=only_yes_no
    )

    # compute metrics
    results["metrics"] = compute_metrics(test_predictions)
    print("Results metrics:", results["metrics"])
    wandb.log({"accuracy": results["metrics"]["accuracy"],
               "precision": results["metrics"]["precision"],
               "recall": results["metrics"]["recall"],
               "f1": results["metrics"]["f1"]})

    true_labels, pred_labels = calc_true_and_pred_labels(test_predictions)
    results["true_labels"] = true_labels
    results["pred_labels"] = pred_labels

    plot_confusion_matrix(results["true_labels"], results["pred_labels"], title=f"Confusion Matrix for {model_name} with {train_size}")

    wandb.finish()  # end current run

    return results
