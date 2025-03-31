"""
Utility functions for the RAG system.

This module contains various helper functions used across different components
of the RAG pipeline.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Any, List, Optional, Union
from enum import Enum

from src.common.config import DEBUG_DIR
from src.common.models import ProcessedChunk

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_environment() -> None:
    """Set up environment variables and configurations."""
    os.environ["USER_AGENT"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"
    )
    logger.info("Environment set up completed")


def create_hash(data: Any) -> str:
    """
    Create a unique hash for the provided data.

    This is used for caching purposes to detect changes in source data.

    Args:
        data: Any data structure that can be serialized to JSON

    Returns:
        str: A hash string representing the data
    """

    def serialize(obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, (list, dict)):
            return (
                {k: serialize(v) for k, v in obj.items()}
                if isinstance(obj, dict)
                else [serialize(item) for item in obj]
            )
        return str(obj)

    if isinstance(data, (list, dict)):
        serialized = json.dumps(serialize(data), sort_keys=True)
    else:
        serialized = str(data)

    return hashlib.md5(serialized.encode()).hexdigest()


def save_to_cache(data: Any, cache_path: Union[str, Path], filename: str) -> Path:
    """
    Save data to the cache directory.

    Args:
        data: The data to cache
        cache_path: Path to the cache directory
        filename: Name of the cache file

    Returns:
        Path: The path where data was saved
    """
    if isinstance(cache_path, str):
        cache_path = Path(cache_path)

    os.makedirs(cache_path, exist_ok=True)

    file_path = cache_path / filename

    if isinstance(data, (dict, list)):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        with open(file_path, "w") as f:
            f.write(str(data))

    logger.info(f"Data saved to cache: {file_path}")
    return file_path


def load_from_cache(cache_path: Union[str, Path], filename: str) -> Optional[Any]:
    """
    Load data from the cache directory.

    Args:
        cache_path: Path to the cache directory
        filename: Name of the cache file

    Returns:
        Optional[Any]: The cached data if available, None otherwise
    """
    if isinstance(cache_path, str):
        cache_path = Path(cache_path)

    file_path = cache_path / filename

    if not file_path.exists():
        logger.info(f"Cache file not found: {file_path}")
        return None

    try:
        with open(file_path, "r") as f:
            if filename.endswith(".json"):
                return json.load(f)
            else:
                return f.read()
    except Exception as e:
        logger.error(f"Error loading from cache: {e}")
        return None


def save_chunk_debug_info(chunks: List[ProcessedChunk]) -> None:
    """
    Save debug information about processed chunks to a file.
    Args:
        chunks: List of processed chunks

    Returns:
        None
    """

    debug_file = DEBUG_DIR / "processed_chunks.txt"

    with open(debug_file, "w", encoding="utf-8") as f:
        f.write(f"Total chunks processed: {len(chunks)}\n\n")
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i + 1}/{len(chunks)}\n")
            f.write(f"ID: {chunk.id}\n")
            f.write(f"Document ID: {chunk.document_id}\n")
            f.write(f"Metadata: {chunk.metadata}\n")
            f.write("Content:\n")
            f.write(f"{chunk.content}\n")
            f.write("-" * 80 + "\n\n")

    logger.debug(f"Saved chunk debug info to {debug_file}")
