import json
import os
from datetime import datetime

def generate_run_id():
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")

def save_json(data, path):
    """Save dictionary to JSON safely."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
