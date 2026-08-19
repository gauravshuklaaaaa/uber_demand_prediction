import logging
from pathlib import Path
import pandas as pd

# Create logger
logger = logging.getLogger("feature_processing")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

if __name__ == "__main__":
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    data_path = root_path / "data/processed/resampled_data.csv"
    
    # 1. Read data
    df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")
    
    # 2. Sort data strictly by region and datetime BEFORE lag creation
    df.sort_values(by=["region", "tpep_pickup_datetime"], inplace=True)
    
    # 3. Extract Datetime features
    df["day_of_week"] = df["tpep_pickup_datetime"].dt.day_of_week
    df["month"] = df["tpep_pickup_datetime"].dt.month
    logger.info("Datetime features extracted successfully")
    
    # 4. Generate Lag features safely (without indexing issues)
    for p in range(1, 5):
        df[f"lag_{p}"] = df.groupby("region")["total_pickups"].shift(p)
    logger.info("Lag features generated successfully")
    
    # 5. Drop NaN rows created by shift()
    df.dropna(inplace=True)
    
    # 6. Set datetime index AFTER lag creation
    df.set_index("tpep_pickup_datetime", inplace=True)
    logger.info("Datetime column set as index successfully")
    
    # 7. Explicit Feature Selection (Avoids relying on column order)
    feature_cols = [f"lag_{i}" for i in range(1, 5)] + ["month", "day_of_week"]
    
    # Split train & test
    trainset = df.loc[df["month"].isin([1, 2]), feature_cols]
    testset = df.loc[df["month"].isin([3]), feature_cols]
    
    # 8. Save output
    train_save_path = root_path / "data/processed/train.csv"
    test_save_path = root_path / "data/processed/test.csv"

    trainset.to_csv(train_save_path, index=True)
    logger.info("Train data saved successfully")
    
    testset.to_csv(test_save_path, index=True)
    logger.info("Test data saved successfully")