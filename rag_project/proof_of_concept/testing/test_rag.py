"""
RAG system evaluation tests.

This module contains pytest tests that evaluate the RAG pipeline's performance
using test cases and metrics defined in the project. The tests use DeepEval
for running evaluations and generating test reports.
"""

from deepeval import evaluate
from src.testing.test_data import POC_TEST_CASES
from src.evaluation.evaluation_factory import EvaluationDatasetFactory
from src.evaluation.report_generation import report_from_latest_json
from src.evaluation.json_storing import save_test_results


def test_all_dataset(qa_pipeline, evaluation_metrics):
    """
    Test the RAG pipeline using EvaluationDatasetFactory on all test cases.

    This test:
    1. Creates a test dataset from the defined test cases
    2. Evaluates the RAG pipeline using the configured metrics
    3. Saves test results and generates a report

    Args:
        qa_pipeline: Fixture providing the configured RAG pipeline
        evaluation_metrics: Fixture providing the evaluation metrics
    """
    dataset = EvaluationDatasetFactory.create_from_dict_with_invocation(POC_TEST_CASES, qa_pipeline)

    result = evaluate(
        test_cases=dataset.test_cases,
        metrics=evaluation_metrics,
    )

    save_test_results(result, "proof_of_concept")
    report_from_latest_json()
