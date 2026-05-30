import json
import os
import random
from datetime import datetime

def generate_run_id():
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")

def save_json(data, path):
    """Save dictionary to JSON safely."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


class JsonlCheckpointWriter:
    """Append-only JSONL writer with fsync per record (Flaw §4.4).

    `baseline.json`, `pirc.json`, and `ifi_metrics.json` were written all-at-once
    at the end of each phase — a crash inside the loop discarded every completed
    article. This writer streams one record per article, flushed and fsync'd
    immediately, so partial runs always preserve their finished work.

    Usage:
        ckpt = JsonlCheckpointWriter(path)
        for article in articles:
            result = process(article)
            ckpt.write(result)
        ckpt.close()
        # Final aggregation reads the JSONL and emits the consolidated JSON.

    Read back with `JsonlCheckpointWriter.read_all(path)`.
    """

    def __init__(self, path: str):
        from pathlib import Path
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, record) -> None:
        self._f.write(json.dumps(record, default=str) + "\n")
        self._f.flush()
        try:
            os.fsync(self._f.fileno())
        except OSError:
            pass

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    @staticmethod
    def read_all(path: str):
        """Read a JSONL checkpoint file and return a list of records.

        Tolerates a trailing truncated line (last record cut off by crash):
        every well-formed line is returned, the malformed tail is skipped.
        """
        out = []
        if not os.path.exists(path):
            return out
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # Truncated last line from a crash — stop here, preserve
                    # earlier records.
                    break
        return out


def set_global_seed(seed: int = 42) -> None:
    """Seed every randomness source we touch (Flaw §4.2).

    Greedy decoding (`do_sample=False`) is already enforced in
    `ModelInterface`, but numpy / python-random / torch fallback paths
    (sampling for self-consistency baselines, dataset shuffling, NLI
    tie-breaks, etc.) need explicit seeding for full bit-reproducibility.

    Call once at the top of every `experiment_*.py` / `main.py` entry
    point. Safe to call before optional deps (torch) are imported —
    skipped silently when a backend is unavailable.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort determinism; some kernels remain non-deterministic.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
