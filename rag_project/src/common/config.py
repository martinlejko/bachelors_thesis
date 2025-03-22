"""
Configuration settings for the RAG system.

This module contains paths, model settings, and other configuration variables
used throughout the project.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "src" / "data"
CACHE_DIR = DATA_DIR / "cache"
PRIVATE_DATA_DIR = DATA_DIR / "private"
PUBLIC_DATA_DIR = DATA_DIR / "public"
ITERATIONS_DIR = BASE_DIR / "src" / "iterations"

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PRIVATE_DATA_DIR, exist_ok=True)
os.makedirs(PUBLIC_DATA_DIR, exist_ok=True)

# Model settings
OLLAMA_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"

# Data processing settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Confluence API settings (to be configured by the user)
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL", "")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME", "")
CONFLUENCE_API_KEY = os.getenv("CONFLUENCE_API_KEY", "")

# Vector database settings
VECTOR_DB_PATH = CACHE_DIR / "vectorstore"

# Test results and reports
TEST_RESULTS_DIR = BASE_DIR / "test_results"
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# Cache settings
CACHE_ENABLED = True
CACHE_VERSION = "v1"  # Increment this when the caching mechanism changes

# URLs for testing
TEST_URLS = [
    "https://d3s.mff.cuni.cz/teaching/nswi200/teams/",
    "https://d3s.mff.cuni.cz/teaching/nprg035/",
]
