#!/usr/bin/env python3
"""
Main entry point for running the RAG pipeline.

This script ensures that the Python path is set up correctly for imports.
"""

import sys
import os
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.iterations.iter_0.pipeline import create_pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create the pipeline with default parameters
    pipeline = create_pipeline(use_web_urls=False, use_pdf=True, use_confluence=False, force_refresh=False)

    # Run a sample query
    query = "What is RAG?"
    result = pipeline.query(query)
    print("\nPipeline execution completed.")
    print(f"Question: {query}")
    print(f"Answer: {result.answer}")

    # Print retrieved documents for debugging
    print("\nRetrieved documents:")
    for i, doc in enumerate(result.retrieved_documents):
        print(f"Document {i + 1}:")
        print(f"Content: {doc.content[:150]}...")  # Print first 150 chars
        print(f"Score: {doc.score}")
        print("---")
