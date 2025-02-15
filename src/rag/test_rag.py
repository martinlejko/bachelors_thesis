from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_ollama import ChatOllama
from proof_of_concept import (
    setup_environment,
    load_data,
    process_data,
    create_vectorstore,
    setup_model,
    create_prompt_template,
    create_qa_chain,
)

# Initialize RAG pipeline in a controlled way
def initialize_rag():
    """Initialize RAG pipeline once."""
    setup_environment()
    urls = [
        "https://d3s.mff.cuni.cz/teaching/nswi200/teams/",
        "https://d3s.mff.cuni.cz/teaching/nprg035/",
    ]
    
    raw_data = load_data(urls)
    processed_data = process_data(raw_data)
    vectorstore = create_vectorstore(processed_data)
    model = setup_model()
    prompt = create_prompt_template()
    return create_qa_chain(vectorstore, model, prompt)

# Initialize once at module level
qa_pipeline = initialize_rag()

# Define metrics
metrics = [
    GEval(
        name="Answer Correctness",
        criteria="Determine if the actual output correctly answers the question based on the expected output and context.",
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
        criteria="Evaluate if the response is directly relevant to the question asked, regardless of correctness.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.7,
    ),
    GEval(
        name="Conciseness",
        criteria="Check if the response is within three sentences and provides a clear, concise answer without unnecessary information.",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.8,
    )
]

def test_team_size():
    """Test the RAG pipeline for team size question."""
    question = "What is the amount of people in a team for operating systems course at MFF cuni?"
    actual_output = qa_pipeline.invoke(question)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output="Teams should consist of 2-3 students.",
        context=["For the Operating Systems course at MFF CUNI, teams should consist of 2-3 students for project work."]
    )
    
    evaluate(
        test_cases=[test_case],
        metrics=metrics
    )

def test_programming_language():
    """Test the RAG pipeline for programming language question."""
    question = "What programming language is used in NPRG035?"
    actual_output = qa_pipeline.invoke(question)
    
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output="The course uses Python programming language.",
        context=["NPRG035 is taught using Python as the main programming language."]
    )
    
    evaluate(
        test_cases=[test_case],
        metrics=metrics
    ) 