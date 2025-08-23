#!/usr/bin/env python3
"""
run_experiments.py

Launches one of your T5 experiments across 8 GPUs via accelerate.
Includes:
 - CPU thread pinning
 - Tokenizer parallelism disabled
 - Logging/tee to file
 - Parametric dataset selection (HIGGS or Flights with stage)
"""
import os
import argparse
import logging
import sys
import math
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import logging as hf_logging
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix
from v1_t5_enc_head_pool import run_1_t5_enc_head_pool_experiments
from v2_t5_dec_head_hf import run_2_t5_dec_head_hf_experiments
from v3_t5_fused_head_custom import run_3_t5_fused_head_custom_experiments
import matplotlib.pyplot as plt

# Only let local_rank 0 print to the console
local_rank = int(os.environ.get("LOCAL_RANK", 0))
# if local_rank != 0:
#     # suppress all logging below ERROR
#     logging.getLogger().setLevel(logging.ERROR)
#     # drop their stdout/stderr
#     sys.stdout = open(os.devnull, "w")
#     sys.stderr = open(os.devnull, "w")

def setup_environment():
    # Pin CPU threads so DataLoader workers do the parallelism
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    # Less OOM from fragmentation on long multi-experiment runs
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    # Enable TF32 in script-level environment too
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32   = True


def _load_higgs(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    selected_cols = [
        "label",
        "dijet_invariant_mass",
        "trijet_invariant_mass",
        "lepton_missing_energy_mass",
        "dijet_lepton_missing_energy_mass",
        "bjet_pair_invariant_mass",
        "wboson_bjet_pair_mass",
        "wboson_bjet_pair_plus_jet_mass",
        "leading_jet_b_tag",
        "subleading_jet_b_tag",
        "third_jet_b_tag",
        "fourth_jet_b_tag",
        "missing_transverse_energy",
        "leading_jet_transverse_momentum",
        "lepton_transverse_momentum",
        "fourth_jet_transverse_momentum",
        "subleading_jet_transverse_momentum",
        # split markers
        "test","validation",
        "train_50","train_100","train_200","train_500",
        "train_1K","train_5K","train_10K","train_50K",
        "train_100K","train_500K","train_1M","train_5M",
    ]
    return df[selected_cols]

def _load_flights(path: str, stage: str) -> pd.DataFrame:
    """
    stage ∈ {"post_arrival","post_departure","pre_departure"}
    """
    df = pd.read_parquet(path)
    # 1) remove cancellation-only features
    flights_post_arrival = df.drop(columns=[
        "DIVERTED","CANCELLED","CANCELLATION_REASON"
    ], errors="ignore")
    # 2) drop post-arrival features - usable at (or just after) departure time
    flights_post_departure = flights_post_arrival.drop(columns=[
        "ELAPSED_TIME","AIR_TIME","WHEELS_ON","TAXI_IN","ARRIVAL_TIME",
        "AIR_SYSTEM_DELAY","SECURITY_DELAY","AIRLINE_DELAY",
        "LATE_AIRCRAFT_DELAY","WEATHER_DELAY",
    ], errors="ignore")
    # 3) drop immediate post-departure features - pre-departure only
    flights_pre_departure = flights_post_departure.drop(columns=[
        "DEPARTURE_TIME","DEPARTURE_DELAY","TAXI_OUT","WHEELS_OFF",
    ], errors="ignore")
    if stage == "post_arrival":
        return flights_post_arrival
    if stage == "pre_departure":
        return flights_pre_departure
    # default "post_departure"
    return flights_post_departure


def setup_logging(log_path: str, local_rank: int):
    # clean root handlers (avoid duplicates on reruns)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG if local_rank == 0 else logging.ERROR)

    if local_rank == 0:
        # Console: INFO+ without timestamps
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(console)

        # File: DEBUG with timestamps (fresh file per run)
        fileh = logging.FileHandler(log_path, mode="w")
        fileh.setLevel(logging.DEBUG)
        fileh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)5s | %(message)s", datefmt="%H:%M:%S"
        ))
        root.addHandler(fileh)

        # Tee print() and stderr into the same log file (and keep console)
        log_fp = open(log_path, "a", buffering=1)
        class _Tee:
            def __init__(self, *streams): self.streams = streams
            def write(self, data):
                for s in self.streams: s.write(data); s.flush()
            def flush(self):
                for s in self.streams: s.flush()
        sys.stdout = _Tee(sys.stdout, log_fp)
        sys.stderr = _Tee(sys.stderr, log_fp)
    else:
        # silence libraries on worker ranks
        hf_logging.set_verbosity_error()
        # and drop all print()/stderr from workers
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    # Transformers logging for rank 0
    if local_rank == 0:
        hf_logging.set_verbosity_info()
        hf_logging.enable_default_handler()
        hf_logging.enable_explicit_format()


def main():
    setup_environment()
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        choices=["enc_pool", "dec_head_hf", "fused_head"],
        default="enc_pool",
        help="Which architecture to run this time."
    )
    parser.add_argument(
        "--dataset",
        choices=["higgs","flights"],
        default="higgs",
        help="Dataset to load and preprocess."
    )
    parser.add_argument(
        "--flights_stage",
        choices=["post_arrival","post_departure","pre_departure"],
        default="post_departure",
        help="Which feature availability cut for Flights (only used with --dataset flights)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Optional override for dataset parquet path."
    )
    
    parser.add_argument(
        "--pooling",
        choices=["masked_mean", "attention"],
        default="masked_mean",
        help="Pooling for enc_pool architecture (ignored otherwise)."
    )
    cli = parser.parse_args()

    # Decide tags/paths FIRST so logs live next to artifacts, then init logging
    dataset_tag = "higgs_top_ft" if cli.dataset == "higgs" else f"flights_{cli.flights_stage}"
    arch_name = (
        f"enc_pool_{cli.pooling}" if cli.arch == "enc_pool"
        else ("dec_head_hf" if cli.arch == "dec_head_hf" else "fused_head_custom")
    )
    final_out_dir = os.path.join("./t5_runs", dataset_tag, arch_name)
    os.makedirs(final_out_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{dataset_tag}__{arch_name}_{run_id}"
    pdf_path = os.path.join(final_out_dir, f"{base_name}.pdf")
    log_path = os.path.join(final_out_dir, f"{base_name}.log")
    setup_logging(log_path, local_rank)
    logger = logging.getLogger(__name__)
    logger.info("=== Starting run_experiments script ===")

    # 1) Load dataset (parametric) — AFTER logging is initialized
    if cli.dataset == "higgs":
        data_path = cli.data_path or "/home/ubuntu/oron/nlp/data/higgs/HIGGS_with_splits_nested.parquet"
        logger.info(f"[dataset=higgs] Loading {data_path}")
        df_use = _load_higgs(data_path)
        task_def = "Higgs Boson Event Classification"
    else:
        data_path = cli.data_path or "/home/ubuntu/oron/nlp/data/flights/flights_with_splits_nested.parquet"
        logger.info(f"[dataset=flights/{cli.flights_stage}] Loading {data_path}")
        df_use = _load_flights(data_path, cli.flights_stage)
        task_def = "Flights Delay Classification"

    # 2) Build the experiment arguments (modify as needed)
    args = {
        "df":                         df_use,
        "label_col":                  "label",
        "task_def":                   task_def,
        "subsample_sets":             ['1M'],#['50','100','200','500','1K','5K','10K','50K'],
        "model_name":                 "t5-base",
        "output_dir":                 final_out_dir,
        "random_state":               42,
        "num_train_epochs":           5, #8, #12,
        "per_device_train_batch_size":64,#16,
        "per_device_eval_batch_size": 32,
        "gradient_accumulation_steps":1,#2,
        "learning_rate":              3e-4, #3e-4, #5e-4,
        "weight_decay":               0.01,
        "optim":                      "adamw_torch",
        "lr_scheduler_type":          "cosine",#"linear",
        "warmup_ratio":               0.05, #0.1,
        "fp16":                       False, #True,
        "bf16":                       True, #False,
        "dataloader_num_workers":     16, #8,
        "use_lora":                   True,
        "lora_r":                     32, #16,
        "lora_alpha":                 64, #32,
        "lora_dropout":               0.0,
        
        "eval_strategy":              "steps",
        "eval_steps":                 400,
        "logging_strategy":           "steps",
        "logging_steps":              400,
        "save_steps":                 None,
        
        #"eval_strategy":              "epoch",  # only at epoch end
        #"logging_strategy":           "epoch",  # only at epoch end
        #"save_strategy":              "epoch",  # only at epoch end
        "save_total_limit":           2,
        "overwrite_output_dir":       True,
        "save_safetensors":           False,
        "load_best_model_at_end":     True,
        "metric_for_best_model":      "f1",
        "greater_is_better":          True,
        "early_stopping_patience":    3,
        "max_length":                 300,
    }
    # Pick one architecture
    if cli.arch == "enc_pool":
        fn = run_1_t5_enc_head_pool_experiments
        args["pooling"] = cli.pooling   # used by v1
    elif cli.arch == "dec_head_hf":
        fn = run_2_t5_dec_head_hf_experiments
    else:  # fused_head
        fn = run_3_t5_fused_head_custom_experiments

    # Pass the per-run tag to the training function so it can nest checkpoints under it
    args["run_tag"] = base_name
    logger.info(f"Arguments prepared, launching: {arch_name}")
    results, report_figs = fn(**args)
    logger.info(f"=== {arch_name} complete ===")
    logger.info(f"Final metrics: {results['confusion_matrices']}")

    # 4) Save PDF – page 1: table + metrics chart; page 2: all CMs in a grid
    # pdf_path = os.path.join(args["output_dir"], f"{dataset_tag}__{arch_name}_{run_id}.pdf")
    # build metrics DataFrame
    summary_df = pd.DataFrame(
        results['confusion_matrices'], columns=["ratio", "metrics"]
    )
    metrics_df = pd.DataFrame(
        summary_df['metrics'].to_list(), index=summary_df['ratio']
    ).reset_index()
    # Keep a numeric copy for plotting
    metrics_df_plot = metrics_df.copy()
    # Make a formatted copy for the table (4 decimals as strings)
    metric_cols = [c for c in metrics_df.columns if c != "index" and c != "ratio"]
    metrics_df_table = metrics_df.copy()
    metrics_df_table[metric_cols] = metrics_df_table[metric_cols].applymap(
        lambda v: f"{float(v):.4f}"
    )
    # raw CM tuples: (ratio, y_true, y_pred)
    raw_cms = results["raw_cms"]

    if local_rank == 0:
        with PdfPages(pdf_path) as pp:
            # Page 1: side‐by‐side table + summary chart
            import matplotlib.gridspec as gridspec
            fig = plt.figure(figsize=(12, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
    
            # Table on left
            ax_tab = fig.add_subplot(gs[0])
            ax_tab.axis('off')
            tbl = ax_tab.table(
                cellText=metrics_df_table.values,
                colLabels=metrics_df_table.columns,
                cellLoc='center',
                loc='center'
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            # Let Matplotlib widen columns to fit the shortened text
            try:
                tbl.auto_set_column_width(col=list(range(metrics_df_table.shape[1])))
            except Exception:
                pass
    
            # Summary chart on right
            ax_chart = fig.add_subplot(gs[1])
            for m in ["accuracy", "precision", "recall", "f1", "auc"]:
                ax_chart.plot(
                    metrics_df_plot['ratio'],
                    metrics_df_plot[m],
                    marker='o',
                    label=m
                )
            ax_chart.set(
                xlabel="Subsample Ratio",
                ylabel="Metric Value",
                title="Metrics vs Subsample Ratio"
            )
            ax_chart.legend()
            fig.tight_layout()
            pp.savefig(fig)
            plt.close(fig)
    
            # Page 2: all confusion matrices in a grid
            n = len(raw_cms)
            cols = min(4, n)
            rows = math.ceil(n / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
            axes = np.array(axes).flatten()
            # blank out any extra axes
            for ax in axes[n:]:
                ax.axis('off')
    
            for i, (ratio, y_true, y_pred) in enumerate(raw_cms):
                cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
                ax = axes[i]
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.set(
                    title=f"ratio={ratio}",
                    xlabel="Pred",
                    ylabel="True"
                )
                # annotate
                thresh = cm.max()/2 if cm.max() else 0
                for r in range(cm.shape[0]):
                    for c in range(cm.shape[1]):
                        ax.text(
                            c, r, str(cm[r, c]),
                            ha='center', va='center',
                            color='white' if cm[r, c] > thresh else 'black'
                        )
            fig.tight_layout()
            pp.savefig(fig)
            plt.close(fig)
    
            # Page 3: hyperparameters list
            # turn the args dict into a 2-column DataFrame
            hparams_df = pd.DataFrame(
                list(args.items()), 
                columns=["Parameter", "Value"]
            )
    
            # size it tall enough to fit all rows
            fig = plt.figure(figsize=(8, max(2, 0.3 * len(hparams_df))))
            ax = fig.add_subplot(111)
            ax.axis("off")
    
            # render as a table
            tbl = ax.table(
                cellText=hparams_df.values,
                colLabels=hparams_df.columns,
                cellLoc="left",
                loc="center"
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            fig.tight_layout()
            pp.savefig(fig)
            plt.close(fig)

    if local_rank == 0:
        logger.info(f"[{arch_name}] Saved PDF report to {pdf_path}")
        print("=== DONE ===")

if __name__ == "__main__":
    main()
