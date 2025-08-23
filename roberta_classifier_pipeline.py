import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import wandb
from typing import Any, Dict, Tuple, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from peft import LoraConfig, get_peft_config, get_peft_model, TaskType
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

def serialize_row(row: pd.Series, label_col: str) -> Tuple[str, int]:
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
    target = int(row[label_col])
    return prompt, target


def compute_metrics(pred_out: Any) -> Dict[str, float]:
    """
    Compute classification metrics.
    """
    print(f"compute_metrics: pred_out.predictions are {pred_out.predictions}")
    predictions = np.argmax(pred_out.predictions, axis=1)
    print(f"compute_metrics: predictions are {predictions}")
    labels = pred_out.label_ids
    print(f"compute_metrics: labels are {labels}")

    res = {
        "accuracy":  accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall":    recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1":        f1_score(labels, predictions, average="weighted", zero_division=0),
    }
    # AUC
    try:
        # For binary classification, roc_auc_score expects scores for the positive class
        # Assuming the positive class is at index 1
        if pred_out.predictions.shape[1] == 2:
             res["auc"] = roc_auc_score(labels, pred_out.predictions[:, 1])
        else:
             # For multi-class, use 'ovr' strategy
             res["auc"] = roc_auc_score(labels, pred_out.predictions, multi_class="ovr")
    except Exception as e:
        print(f"Could not compute AUC: {e}")
        pass # Handle cases where AUC cannot be computed (e.g., single class in predictions)

    print(res)
    return res


def plot_confusion_matrix(
    y_true: List[Any],
    y_pred: List[Any],
    title: str = ''
) -> None:
    """
    Plot a single confusion matrix.
    """
    labels = sorted(list(set(y_true) | set(y_pred))) # Ensure all labels are included
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


def preprocess_batch(
    df: pd.DataFrame,
    tokenizer: RobertaTokenizer,
    label_col: str,
    max_length: int = 320,
) -> List[Dict[str, np.ndarray]]:
    """
    Tokenize prompts and targets into input_ids, attention_mask, and labels.
    Labels are padded to target_length and pad token IDs set to -100.
    """
    inputs = []
    for _, row in df.iterrows():
        prompt, target = serialize_row(row, label_col)

        enc = tokenizer(
            prompt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        assert ((target == 0) | (target == 1)), \
            f"True label {target} is neither 1 nor 0"

        inputs.append({
            "input_ids": enc.input_ids[0],
            "attention_mask": enc.attention_mask[0],
            "labels": target,
        })

    return inputs


def run_roberta_experiment(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_size: str,
    label_col: str,
    model_name: str,
    output_dir: str,
    random_state: int,
    # TrainingArguments kwargs
    num_train_epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    learning_rate: float,
    # LoRA kwargs
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    warmup_ratio: float = 0.1,
) -> Dict[str, float]:
    """
    Runs a RoBERTa classification experiment with optional LoRA fine-tuning.

    Args:
        train_df: Training dataset as a DataFrame
        val_df: Validation dataset as a DataFrame
        test_df: Test dataset as a DataFrame
        train_size: Identifier for the training set size (for logging purposes)
        label_col: Name of the target column containing class labels
        model_name: Name of the pre-trained RoBERTa model to use
        output_dir: Directory to save model checkpoints and logs
        random_state: Random state for reproducibility
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device during training
        per_device_eval_batch_size: Batch size per device during evaluation
        learning_rate: Learning rate for the optimizer
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning
        lora_r: LoRA attention dimension (rank parameter)
        lora_alpha: LoRA alpha parameter (scaling factor)
        lora_dropout: Dropout probability for LoRA layers
        warmup_ratio: Ratio of training steps for linear learning rate warmup

    Returns:
        Dictionary containing evaluation metrics on the test set.
        Metrics include accuracy, precision, recall, F1-score and AUC-ROC.
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

    print(f"\nRunning experiment with training set: {train_size}")

    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "no", 1: "yes"}
    )

    # Configure LoRA if enabled
    if use_lora:
        print("--------------- using lora -------------")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, # Specify the task type for classification
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, peft_config)

    # tokenize into lists of example‐dicts
    tokenized_train_dataset = preprocess_batch(train_df, tokenizer, label_col)
    tokenized_val_dataset   = preprocess_batch(val_df, tokenizer, label_col)
    tokenized_test_dataset  = preprocess_batch(test_df, tokenizer, label_col)

    print("\n=== Data Validation ===")
    sample_data = tokenized_train_dataset[:2]
    for i, data in enumerate(sample_data):
        text = tokenizer.decode(data["input_ids"], skip_special_tokens=False)
        print(f"Sample {i + 1}:")
        print(f"  Text: {text}")
        print(f"  Label: {data['labels']}")
        print("-" * 50)

    # Training arguments
    training_args = TrainingArguments(
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
        seed=random_state,
        report_to="wandb",
        overwrite_output_dir=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,
        fp16=torch.cuda.is_available(),
        warmup_ratio=warmup_ratio,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        optim="adamw_torch",
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest', max_length=512, pad_to_multiple_of=8)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    test_predictions = trainer.predict(tokenized_test_dataset)

    # Extract results
    results = {
        "metrics": compute_metrics(test_predictions),
        "true_labels": test_predictions.label_ids,
        "pred_labels": np.argmax(test_predictions.predictions, axis=1)
    }

    wandb.log({"accuracy": results["metrics"]["accuracy"],
               "precision": results["metrics"]["precision"],
               "recall": results["metrics"]["recall"],
               "f1": results["metrics"]["f1"],
               "auc": results["metrics"]["auc"]})

    plot_confusion_matrix(results["true_labels"], results["pred_labels"], title=f"Confusion Matrix for {model_name} with {train_size}")

    wandb.finish()  # end current run

    return results