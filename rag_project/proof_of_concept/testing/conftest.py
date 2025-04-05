"""Pytest configuration and shared fixtures"""

import pytest
from deepeval.metrics import GEval
from src.common.config import TEST_URLS
from deepeval.test_case import LLMTestCaseParams
from proof_of_concept.proof_of_concept import (
    load_data,
    process_data,
    create_vectorstore,
    setup_model,
    create_prompt_template,
    create_qa_chain,
)


@pytest.fixture(scope="session")
def qa_pipeline():
    """Initialize RAG pipeline once for all tests."""
    raw_data = load_data(TEST_URLS)
    processed_data = process_data(raw_data)
    vectorstore = create_vectorstore(processed_data)
    model = setup_model()
    prompt = create_prompt_template()
    return create_qa_chain(vectorstore, model, prompt)


@pytest.fixture(scope="session")
def evaluation_metrics():
    """Define evaluation metrics used across tests."""

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
    ]
