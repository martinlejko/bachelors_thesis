#!/usr/bin/env python3
"""
Main entry point for running the RAG pipeline.
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.iterations.iter_8.pipeline import create_pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create the pipeline with default parameters
    pipeline = create_pipeline(use_web_urls=False, use_pdf=True, use_confluence=False, force_refresh=True)

    # Run a sample query
    query = "What is RAG?"
    result = pipeline.query(query)
    print("\nPipeline execution completed.")
    print(f"Question: {query}")
    print(f"Answer: {result.answer}")

    print("\nRetrieved documents:")
    for i, doc in enumerate(result.retrieved_documents):
        print(f"Document {i + 1}:")
        print(f"Content: {doc.content[:150]}...")
        print(f"Score: {doc.score}")
        print("---")
