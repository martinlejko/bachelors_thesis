"""
Test case definitions and expected results for RAG system evaluation.

This module defines test cases used for evaluating the RAG system's performance.
Each test case contains:
- question: The query to be answered by the RAG system
- expected_output: The expected answer from the system
- context: Sample context data that contains the information needed to answer correctly
"""

TEST_CASES = {
    "team_size": {
        "question": "What is the amount of people in a team for operating systems course at MFF cuni?",
        "expected_output": "Teams should consist of 2-3 students.",
        "context": [
            "For the Operating Systems course at MFF CUNI, teams should consist of 2-3 students for project work."
        ],
    },
    "programming_language": {
        "question": "What programming language is used in NPRG035?",
        "expected_output": "The course uses C# programming language.",
        "context": ["NPRG035 is taught using C# as the main programming language."],
    },
}
