"""
RAG system evaluation tests.

This module contains tests for evaluating the RAG system across different iterations.
"""

import os
import json
import logging
from datetime import datetime

from deepeval import evaluate

from src.common.config import TEST_RESULTS_DIR
from src.evaluation.evaluation_factory import EvaluationDatasetFactory
from src.evaluation.report_generation import save_report

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

    # Create output directory if it doesn't exist
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    # Create filename with timestamp and iteration name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{TEST_RESULTS_DIR}/results_{iteration_name}_{timestamp}.json"

    # Save results to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Test results saved to {filename}")
    return filename


def test_rag_pipeline(qa_pipeline, evaluation_metrics, test_data, iteration_name):
    """
    Test the RAG pipeline using the evaluation dataset.

    This test evaluates the RAG pipeline on all test cases and generates
    both JSON and HTML reports.

    Args:
        qa_pipeline: Function that takes a question and returns an answer and retrieval context
        evaluation_metrics: List of evaluation metrics to use
        test_data: Dictionary of test cases
        iteration_name: Name of the iteration being tested
    """
    # Create evaluation dataset
    dataset = EvaluationDatasetFactory.create_from_dict_with_invocation(test_data, qa_pipeline)

    # Evaluate the dataset
    result = evaluate(
        test_cases=dataset.test_cases,
        metrics=evaluation_metrics,
    )

    # Save results to JSON file
    json_file = save_test_results(result, iteration_name)

    # Generate HTML report
    html_file = save_report(json_file)

    # Log results
    logger.info(f"Evaluation completed for iteration {iteration_name}")
    logger.info(f"JSON results saved to {json_file}")
    logger.info(f"HTML report saved to {html_file}")

    # Check if all tests passed
    assert result.success_rate > 0, f"All tests failed for iteration {iteration_name}"
