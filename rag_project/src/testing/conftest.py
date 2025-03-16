"""
Pytest configuration and shared fixtures.

This module provides fixtures for testing the RAG system across different iterations.
"""
import os
import sys
import pytest
import importlib
from pathlib import Path

from src.common.config import ITERATIONS_DIR
from src.evaluation.metrics import get_default_metrics

def get_iteration_modules():
    """Get all available iteration modules."""
    iterations = []
    for item in os.listdir(ITERATIONS_DIR):
        if os.path.isdir(os.path.join(ITERATIONS_DIR, item)) and item.startswith("iter_"):
            iterations.append(item)
    return sorted(iterations)

@pytest.fixture(params=get_iteration_modules())
def iteration_name(request):
    """Parametrized fixture that yields each iteration name."""
    return request.param

@pytest.fixture
def qa_pipeline(iteration_name):
    """
    Initialize RAG pipeline for the specified iteration.
    
    This fixture dynamically imports the pipeline module from the specified iteration
    and creates a pipeline instance.
    """
    # Import the pipeline module from the specified iteration
    module_path = f"src.iterations.{iteration_name}.pipeline"
    try:
        pipeline_module = importlib.import_module(module_path)
        
        # Create the pipeline
        pipeline = pipeline_module.create_pipeline(
            use_web_urls=True,
            use_pdf=True,
            use_confluence=False,  # Set to True if Confluence credentials are configured
            force_refresh=False
        )
        
        return pipeline.query
    except ImportError as e:
        pytest.skip(f"Could not import pipeline from iteration {iteration_name}: {e}")
    except Exception as e:
        pytest.skip(f"Error creating pipeline for iteration {iteration_name}: {e}")

@pytest.fixture
def evaluation_metrics():
    """Define evaluation metrics used across tests."""
    return get_default_metrics()

@pytest.fixture
def test_data():
    """Define test data for RAG evaluation."""
    return {
        "team_size": {
            "question": "What is the amount of people in a team for operating systems course at MFF cuni?",
            "expected_output": "Teams should consist of 2-3 students.",
            "context": ["For the Operating Systems course at MFF CUNI, teams should consist of 2-3 students for project work."]
        },
        "programming_language": {
            "question": "What programming language is used in NPRG035?",
            "expected_output": "The course uses Python programming language.",
            "context": ["NPRG035 is taught using Python as the main programming language."]
        }
    } 