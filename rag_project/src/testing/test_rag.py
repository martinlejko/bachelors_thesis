"""
RAG system evaluation tests.

This module contains tests for evaluating the RAG system across different iterations.
"""

import logging

from deepeval import evaluate

from src.evaluation.evaluation_factory import EvaluationDatasetFactory
from src.evaluation.report_generation import save_report
from src.evaluation.json_storing import save_test_results

logger = logging.getLogger(__name__)


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
    logger.info(f"Creating evaluation dataset for iteration {iteration_name}")
    dataset = EvaluationDatasetFactory.create_from_dict_with_invocation(test_data, qa_pipeline)

    logger.info(f"Evaluating dataset for iteration {iteration_name}")
    
    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            result = evaluate(
                test_cases=dataset.test_cases,
                metrics=evaluation_metrics,
            )
            break
        except ValueError as e:
            if 'invalid JSON' in str(e):
                logger.warning(f"Attempt {attempt}/{max_retries} failed due to invalid JSON. Retrying...")
                if attempt == max_retries:
                    raise
            else:
                raise

    # Save results to JSON file
    json_file = save_test_results(result, iteration_name)

    # Generate HTML report
    html_file = save_report(json_file)

    # Log results
    logger.info(f"Evaluation completed for iteration {iteration_name}")
    logger.info(f"JSON results saved to {json_file}")
    logger.info(f"HTML report saved to {html_file}")
