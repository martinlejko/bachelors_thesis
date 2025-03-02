"""
EvaluationDatasetFactory module.

This module implements a factory design pattern for creating an EvaluationDataset 
from a set of test case configurations.

Each test case configuration should be a dict with keys:
    - question (str): The input for the LLMTestCase.
    - expected_output (str): The expected output.
    - context (Optional[str]): Additional context for the test case.
    - actual_output (Optional[str]): The actual output if available, else an empty string.

Usage:

    from evaluation.evaluation_factory import EvaluationDatasetFactory
    
    test_cases_config = {
        "team_size": {
            "question": "How many people are in a team?",
            "expected_output": "A typical team consists of 3 to 5 members",
            "context": "team organization",
            "actual_output": "3"
        },
        "programming_language": {
            "question": "What is the best programming language?",
            "expected_output": "Depends on the requirements",
            "context": "software engineering",
            "actual_output": "Python"
        }
    }
    
    dataset = EvaluationDatasetFactory.create_from_dict(test_cases_config)
"""

from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

class EvaluationDatasetFactory:
    @staticmethod
    def create_from_dict_with_invocation(test_cases_config: dict, qa_pipeline) -> EvaluationDataset:
        """
        Create an EvaluationDataset from a dictionary of test case configurations by invoking the qa_pipeline
        to obtain the actual_output. 
        """
        test_cases = []
        for name, config in test_cases_config.items():
            question = config.get("question", "")
            output_and_context = qa_pipeline(question)
            test_case = LLMTestCase(
                input=question,
                context=config.get("context", ""),
                retrieval_context=output_and_context["retrieval_context"],
                actual_output=output_and_context["answer"],
                expected_output=config.get("expected_output", "")
            )
            test_cases.append(test_case)
        return EvaluationDataset(test_cases=test_cases)
