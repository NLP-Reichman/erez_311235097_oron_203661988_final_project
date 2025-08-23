import pandas as pd
import os
from roberta_classifier_pipeline import run_roberta_experiment

if __name__ == '__main__':
    print("Running RoBERTa classifier on Flights dataset")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    data_path = "/home/erez/NLP_project/data/FlightsDelay/flights_with_splits_nested.parquet"
    flights_post_arrival = pd.read_parquet(data_path).drop(columns=[
        # Irrelevnt cancellation feature
        'DIVERTED',
        'CANCELLED',
        'CANCELLATION_REASON'
    ], inplace=False)

    flights_post_departure = flights_post_arrival.drop(columns=[
        'ELAPSED_TIME',
        'AIR_TIME',
        'WHEELS_ON',
        'TAXI_IN',
        'ARRIVAL_TIME',
        'AIR_SYSTEM_DELAY',
        'SECURITY_DELAY',
        'AIRLINE_DELAY',
        'LATE_AIRCRAFT_DELAY',
        'WEATHER_DELAY',
    ], inplace=False)

    flights_pre_departure = flights_post_departure.drop(columns=[
        # Post-departure features
        'DEPARTURE_TIME',
        'DEPARTURE_DELAY',
        'TAXI_OUT',
        'WHEELS_OFF',
    ], inplace=False)

    print("Rows:", len(flights_post_departure), "Cols:", len(flights_post_departure.columns))
    print(flights_post_departure.dtypes.value_counts())
    print(flights_post_departure["label"].value_counts(normalize=True))

    # Show the resulting DataFrame shape and columns
    print(f"df_splits shape: {flights_post_departure.shape}")
    print("Columns in df_splits:", flights_post_departure.columns.tolist())

    val_df = flights_post_departure[flights_post_departure['validation']]
    test_df = flights_post_departure[flights_post_departure['test']]

    print("Validation size:", val_df.shape)
    print("Test size:", test_df.shape)

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
        train_df = flights_post_departure[flights_post_departure[train_col]]

        result = run_roberta_experiment(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_size=train_col,
            label_col="label",
            model_name='roberta-base',
            output_dir='./roberta_results',
            random_state=42,  # for reproducibility
            # below are T5 training kwargs,
            num_train_epochs=4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            use_lora=True,  # turn on LoRA
            lora_r=8,  # optionally tune these
            lora_alpha=32,
            lora_dropout=0.01,
            warmup_ratio=0.1,
        )

        results[train_col] = result
