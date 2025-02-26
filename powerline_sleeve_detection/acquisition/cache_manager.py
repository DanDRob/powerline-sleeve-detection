import os
import json
import time
import hashlib
import shutil
from typing import Dict, Any, Optional, Union, Callable, TypeVar
import logging
from io import BytesIO
from PIL import Image
import pickle

from ..system.config import Config

T = TypeVar('T')  # Generic type for the cache value


class CacheManager:
    """
    Manages caching for various types of data with disk persistence.
    Supports API responses, images, and processed results.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("powerline_detector.cache_manager")

        self.enabled = config.cache.enabled
        self.cache_dir = config.cache.location
        self.max_size_mb = config.cache.max_size_mb
        self.expire_after = config.cache.expire_after

        # Initialize cache directories
        self._init_cache_dirs()

        # In-memory metadata about cached items
        self.cache_metadata = {}

        # Load existing cache metadata
        self._load_metadata()

        # Clean up expired and excess cache entries
        self._cleanup_cache()

    def _init_cache_dirs(self):
        """Initialize cache directory structure."""
        if not self.enabled:
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "api"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "metadata"), exist_ok=True)

    def _load_metadata(self):
        """Load metadata about cached items from disk."""
        if not self.enabled:
            return

        metadata_path = os.path.join(
            self.cache_dir, "metadata", "cache_index.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    self.cache_metadata = json.load(f)
                self.logger.info(
                    f"Loaded metadata for {len(self.cache_metadata)} cached items")
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
                self.cache_metadata = {}

    def _save_metadata(self):
        """Save metadata about cached items to disk."""
        if not self.enabled:
            return

        metadata_path = os.path.join(
            self.cache_dir, "metadata", "cache_index.json")
        try:
            with open(metadata_path, "w") as f:
                json.dump(self.cache_metadata, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache metadata: {e}")

    def _compute_cache_key(self, key_data: Any) -> str:
        """
        Compute a cache key from the provided data.

        Args:
            key_data: Data to use for generating the key (dict, string, etc.)

        Returns:
            String hash key
        """
        if isinstance(key_data, dict):
            # Sort keys for consistent hashing
            serialized = json.dumps(key_data, sort_keys=True)
        elif isinstance(key_data, str):
            serialized = key_data
        else:
            # Try to serialize with pickle for other types
            try:
                serialized = pickle.dumps(key_data)
            except:
                # Fall back to string representation
                serialized = str(key_data)

        # Compute MD5 hash
        if isinstance(serialized, str):
            key_hash = hashlib.md5(serialized.encode('utf-8')).hexdigest()
        else:
            key_hash = hashlib.md5(serialized).hexdigest()

        return key_hash

    def _cleanup_cache(self):
        """
        Clean up expired cache entries and ensure cache size is within limits.
        """
        if not self.enabled:
            return

        current_time = time.time()
        expired_keys = []

        # Find expired entries
        for key, metadata in self.cache_metadata.items():
            if current_time - metadata.get("timestamp", 0) > self.expire_after:
                expired_keys.append(key)

        # Remove expired entries
        for key in expired_keys:
            self._remove_cache_entry(key)

        self.logger.info(f"Removed {len(expired_keys)} expired cache entries")

        # Check cache size and remove oldest entries if necessary
        self._enforce_size_limit()

    def _enforce_size_limit(self):
        """Ensure cache size is within the configured limit."""
        if not self.enabled:
            return

        # Calculate current cache size
        total_size_bytes = 0
        entries_by_time = []

        for key, metadata in self.cache_metadata.items():
            size = metadata.get("size", 0)
            timestamp = metadata.get("timestamp", 0)
            total_size_bytes += size
            entries_by_time.append((key, timestamp))

        # Convert to MB
        total_size_mb = total_size_bytes / (1024 * 1024)

        # If over limit, remove oldest entries
        if total_size_mb > self.max_size_mb:
            # Sort by timestamp (oldest first)
            entries_by_time.sort(key=lambda x: x[1])

            # Remove oldest entries until under limit
            for key, _ in entries_by_time:
                if total_size_mb <= self.max_size_mb:
                    break

                entry_size_mb = self.cache_metadata[key].get(
                    "size", 0) / (1024 * 1024)
                self._remove_cache_entry(key)
                total_size_mb -= entry_size_mb

            self.logger.info(
                f"Removed oldest cache entries to maintain size limit of {self.max_size_mb}MB")

    def _remove_cache_entry(self, key: str):
        """Remove a cache entry."""
        if key not in self.cache_metadata:
            return

        try:
            cache_type = self.cache_metadata[key].get("type")
            filepath = self.cache_metadata[key].get("filepath")

            if filepath and os.path.exists(filepath):
                os.remove(filepath)

            del self.cache_metadata[key]
        except Exception as e:
            self.logger.warning(f"Error removing cache entry {key}: {e}")

    def get(self, key_data: Any, cache_type: str = "api") -> Optional[Any]:
        """
        Retrieve an item from the cache.

        Args:
            key_data: Data used to generate the cache key
            cache_type: Type of cached data ("api", "images", "processed")

        Returns:
            Cached item or None if not found/expired
        """
        if not self.enabled:
            return None

        key = self._compute_cache_key(key_data)

        if key not in self.cache_metadata:
            return None

        metadata = self.cache_metadata[key]

        # Check if expired
        if time.time() - metadata.get("timestamp", 0) > self.expire_after:
            self._remove_cache_entry(key)
            return None

        try:
            filepath = metadata.get("filepath")
            if not filepath or not os.path.exists(filepath):
                return None

            # Load based on cache type
            if cache_type == "images":
                with open(filepath, "rb") as f:
                    return Image.open(BytesIO(f.read()))
            elif cache_type == "processed":
                with open(filepath, "rb") as f:
                    return pickle.load(f)
            else:  # api and other types
                with open(filepath, "r") as f:
                    return json.load(f)

        except Exception as e:
            self.logger.warning(f"Error retrieving cache entry {key}: {e}")
            return None

    def set(self, key_data: Any, value: Any, cache_type: str = "api") -> bool:
        """
        Store an item in the cache.

        Args:
            key_data: Data used to generate the cache key
            value: Value to cache
            cache_type: Type of cached data ("api", "images", "processed")

        Returns:
            Success status
        """
        if not self.enabled:
            return False

        key = self._compute_cache_key(key_data)

        try:
            # Determine file path and extension based on type
            if cache_type == "images":
                extension = ".jpg"
                subdir = "images"
            elif cache_type == "processed":
                extension = ".pkl"
                subdir = "processed"
            else:  # api and other types
                extension = ".json"
                subdir = "api"

            filepath = os.path.join(
                self.cache_dir, subdir, f"{key}{extension}")

            # Save based on cache type
            if cache_type == "images":
                if isinstance(value, Image.Image):
                    value.save(filepath, format="JPEG")
                else:
                    return False
            elif cache_type == "processed":
                with open(filepath, "wb") as f:
                    pickle.dump(value, f)
            else:  # api and other types
                with open(filepath, "w") as f:
                    json.dump(value, f)

            # Get file size
            size = os.path.getsize(filepath)

            # Update metadata
            self.cache_metadata[key] = {
                "type": cache_type,
                "filepath": filepath,
                "timestamp": time.time(),
                "size": size
            }

            # Save metadata periodically (every 10 cache operations)
            if len(self.cache_metadata) % 10 == 0:
                self._save_metadata()

            return True

        except Exception as e:
            self.logger.warning(f"Error caching {cache_type} data: {e}")
            return False

    def clear(self, cache_type: Optional[str] = None):
        """
        Clear the cache.

        Args:
            cache_type: Type of cache to clear (None for all)
        """
        if not self.enabled:
            return

        if cache_type is None:
            # Clear all cache types
            subdirs = ["api", "images", "processed"]
            for subdir in subdirs:
                try:
                    shutil.rmtree(os.path.join(self.cache_dir, subdir))
                    os.makedirs(os.path.join(
                        self.cache_dir, subdir), exist_ok=True)
                except Exception as e:
                    self.logger.warning(f"Error clearing {subdir} cache: {e}")

            # Reset metadata
            self.cache_metadata = {}
            self._save_metadata()

        else:
            # Clear specific cache type
            keys_to_remove = []
            for key, metadata in self.cache_metadata.items():
                if metadata.get("type") == cache_type:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                self._remove_cache_entry(key)

            self._save_metadata()

        self.logger.info(
            f"Cache cleared: {cache_type if cache_type else 'all'}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled:
            return {"enabled": False}

        stats = {
            "enabled": True,
            "total_entries": len(self.cache_metadata),
            "size_bytes": 0,
            "by_type": {}
        }

        # Calculate stats
        for key, metadata in self.cache_metadata.items():
            cache_type = metadata.get("type", "unknown")
            size = metadata.get("size", 0)

            stats["size_bytes"] += size

            if cache_type not in stats["by_type"]:
                stats["by_type"][cache_type] = {
                    "count": 0,
                    "size_bytes": 0
                }

            stats["by_type"][cache_type]["count"] += 1
            stats["by_type"][cache_type]["size_bytes"] += size

        # Convert bytes to MB for easier reading
        stats["size_mb"] = stats["size_bytes"] / (1024 * 1024)
        for cache_type in stats["by_type"]:
            stats["by_type"][cache_type]["size_mb"] = stats["by_type"][cache_type]["size_bytes"] / \
                (1024 * 1024)

        return stats

    def cached(self, cache_type: str = "api"):
        """
        Decorator for caching function results.

        Args:
            cache_type: Type of cache to use

        Returns:
            Decorator function
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                # Create a key from function name, args, and kwargs
                key_data = {
                    "func": func.__name__,
                    "args": args,
                    "kwargs": kwargs
                }

                # Try to get from cache
                cached_result = self.get(key_data, cache_type)
                if cached_result is not None:
                    return cached_result

                # Call the function and cache the result
                result = func(*args, **kwargs)
                self.set(key_data, result, cache_type)

                return result
            return wrapper
        return decorator
