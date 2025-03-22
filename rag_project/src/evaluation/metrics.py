"""
Evaluation metrics module.

This module defines the evaluation metrics used for RAG system evaluation.
"""

import logging
from typing import List

from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCaseParams

logger = logging.getLogger(__name__)


def get_default_metrics() -> List:
    """
    Get the default evaluation metrics for RAG evaluation.

    Returns:
        List: List of deepeval metrics
    """
    return [
        GEval(
            name="Answer Correctness",
            criteria="Determine if the actual output correctly answers the question based on the expected output.",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            threshold=0.7,
        ),
        GEval(
            name="Answer Relevancy",
            criteria="Evaluate if the response is directly relevant to the question asked.",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.7,
        ),
        GEval(
            name="Conciseness",
            criteria="Check if the response is within three sentences and provides a clear answer.",
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.8,
        ),
        GEval(
            name="Context Relevancy",
            criteria="Evaluate if the retrieved context contains information relevant to answering the query. The context should contain facts or information directly related to what the question is asking about.",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            threshold=0.7,
        ),
        FaithfulnessMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.7),
    ]
