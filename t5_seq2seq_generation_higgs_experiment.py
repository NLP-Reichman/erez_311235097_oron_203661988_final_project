import pandas as pd
import os

from t5_seq2seq_generation_pipeline import run_t5_seq2seq_generation_experiment

if __name__ == '__main__':
    print("Running T5 seq2seq generation on Higgs dataset")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    df_splits = pd.read_parquet("/home/erez/NLP_project/data/higgs/HIGGS_with_splits_nested.parquet")

    print(df_splits.head(5))

    print("Rows:", len(df_splits), "Cols:", len(df_splits.columns))
    print(df_splits.dtypes.value_counts())
    print(df_splits["label"].value_counts(normalize=True))

    df_splits.rename(columns={'label': 'is_higgs_event'}, inplace=True)

    # List of the 17 high-level features plus the label
    selected_cols = [
        "is_higgs_event",
        "dijet_invariant_mass",
        "trijet_invariant_mass",
        "lepton_missing_energy_mass",
        "dijet_lepton_missing_energy_mass",
        "bjet_pair_invariant_mass",
        "wboson_bjet_pair_mass",
        "wboson_bjet_pair_plus_jet_mass",

        # Categorical features (b-tagging information)
        "leading_jet_b_tag",  # jet1_btag - 3 unique values
        "subleading_jet_b_tag",  # jet2_btag - 3 unique values
        "third_jet_b_tag",  # jet3_btag - 3 unique values
        "fourth_jet_b_tag",  # jet4_btag - 3 unique values

        # Transverse momentum features
        "missing_transverse_energy",
        "leading_jet_transverse_momentum",
        "lepton_transverse_momentum",
        "fourth_jet_transverse_momentum",
        "subleading_jet_transverse_momentum",
    ]

    # Show the resulting DataFrame shape and columns
    print(f"df_splits shape: {df_splits.shape}")
    print("Columns in df_splits:", df_splits.columns.tolist())

    val_df = df_splits[df_splits['validation']][selected_cols]
    test_df = df_splits[df_splits['test']][selected_cols]

    print("Validation size:", val_df.shape)
    print("Test size:", test_df.shape)

    # Define your training sizes
    train_sizes = [
        'train_50',
        'train_100',
        'train_200',
        'train_500',
        'train_1K',
        'train_5K',
        'train_10K',
        'train_50K',
        'train_100K',
        'train_500K',
        'train_1M',
    ]

    results = {}

    # Loop over each size
    for train_col in train_sizes:
        print(f"\nRunning experiment with training set: {train_col}")

        # Filter dataset to only include training samples for current size
        train_df = df_splits[df_splits[train_col]][selected_cols]

        result = run_t5_seq2seq_generation_experiment(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_size=train_col,
            label_col="is_higgs_event",
            model_name='t5-base',
            output_dir='./t5_results',
            random_state=42,
            # below are T5 training kwargs,
            num_train_epochs=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=5e-4,
            use_lora=True,
            lora_r=8,
            lora_alpha=32,
            lora_dropout=0,
        )

        results[train_col] = result
