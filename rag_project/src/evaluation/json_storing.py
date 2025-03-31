"""
Json file generation module.

This module handles the generation of JSON report from evaluation result.
"""

import os
import json
import logging
from datetime import datetime
from src.common.config import TEST_RESULTS_DIR

logger = logging.getLogger(__name__)


def save_test_results(evaluation_result, iteration_name):
    """
    Save test results to a JSON file.

    Args:
        evaluation_result: Evaluation result from deepeval
        iteration_name: Name of the iteration being tested

    Returns:
        str: Path to the saved JSON file
    """
    all_results = []

    for test_result in evaluation_result.test_results:
        # Extract metrics data
        metrics_data = {}
        for metric_data in test_result.metrics_data:
            metrics_data[metric_data.name] = {
                "score": metric_data.score,
                "threshold": metric_data.threshold,
                "passed": metric_data.success,
            }

        # Create result dict
        result = {
            "test_name": test_result.name,
            "timestamp": datetime.now().isoformat(),
            "question": test_result.input,
            "actual_context": test_result.retrieval_context,
            "context": test_result.context,
            "actual_output": test_result.actual_output,
            "expected_output": test_result.expected_output,
            "metrics": metrics_data,
            "success": test_result.success,
            "iteration": iteration_name,
        }

        all_results.append(result)

    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{TEST_RESULTS_DIR}/results_{iteration_name}_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Test results saved to {filename}")
    return filename
