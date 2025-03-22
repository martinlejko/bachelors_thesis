"""RAG system evaluation tests"""

from deepeval import evaluate
from proof_of_concept.testing.test_data import TEST_CASES
from src.evaluation.evaluation_factory import EvaluationDatasetFactory
from src.evaluation.report_generation import report_from_latest_json
from src.evaluation.json_storing import save_test_results


def test_all_dataset(qa_pipeline, evaluation_metrics):
    """Test the RAG pipeline using EvaluationDatasetFactory on all test cases."""
    dataset = EvaluationDatasetFactory.create_from_dict_with_invocation(TEST_CASES, qa_pipeline)

    result = evaluate(
        test_cases=dataset.test_cases,
        metrics=evaluation_metrics,
    )

    save_test_results(result, "proof_of_concept")
    report_from_latest_json()
