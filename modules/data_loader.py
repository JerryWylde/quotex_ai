import pandas as pd

def load_raw_data(csv_path: str):
    """
    Load raw FX data (e.g., EURUSD) from CSV safely.
    Expects CSV with 'Timestamp' column parseable to datetime.
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=['Timestamp'])
        df.sort_values('Timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # ✅ Optional safety check
        required_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")
