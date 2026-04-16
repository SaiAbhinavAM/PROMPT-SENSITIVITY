import json
import pandas as pd
from pathlib import Path


def load_dataset(path: str) -> pd.DataFrame:
    """Load the dataset JSON file into a pandas DataFrame.

    Expected JSON format:
    [
        {
            "input_text": "...",
            "reference_output": "...",
            "topic_label": "..."
        },
        ...
    ]
    """
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)
