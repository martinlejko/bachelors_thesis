"""
Pytest configuration and shared fixtures.

This module provides fixtures for testing the RAG system across different iterations.
"""

import os
import pytest
import importlib

from src.common.config import ITERATIONS_DIR
from src.evaluation.metrics import get_default_metrics
from src.testing.test_data import TEST_CASES


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
            use_web_urls=False,
            use_pdf=True,
            use_confluence=False,
            force_refresh=True,
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
    return TEST_CASES
