import os
import json
import hashlib
from typing import Any
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir: str, enable_cache: bool = True):
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache
        self.cache_file = os.path.join(cache_dir, "cache.json")
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if not self.enable_cache:
            return {}
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        if not self.enable_cache:
            return
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _generate_key(self, prefix: str, **kwargs) -> str:
        # Sort keys to ensure deterministic hashing
        key_str = json.dumps(kwargs, sort_keys=True)
        hash_str = hashlib.md5(key_str.encode("utf-8")).hexdigest()
        return f"{prefix}_{hash_str}"

    def get(self, prefix: str, **kwargs) -> Any:
        if not self.enable_cache:
            return None
        key = self._generate_key(prefix, **kwargs)
        return self._cache.get(key)

    def set(self, value: Any, prefix: str, **kwargs):
        if not self.enable_cache:
            return
        key = self._generate_key(prefix, **kwargs)
        self._cache[key] = value
        self._save_cache()
