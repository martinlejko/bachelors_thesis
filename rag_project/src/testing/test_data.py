"""
Test case definitions and expected results for RAG system evaluation.

This module defines test cases used for evaluating the RAG system's performance.
Each test case contains:
- question: The query to be answered by the RAG system
- expected_output: The expected answer from the system
- context: Sample context data that contains the information needed to answer correctly
"""

POC_TEST_CASES = {
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

TEST_CASES = {
    "market_trends_2023": {
        "question": "What were the key market trends in 2023?",
        "expected_output": "The stock market saw a strong rebound driven by technology stocks and easing inflation.",
        "context": ["Investors regained confidence as inflation cooled and central banks slowed interest rate hikes."],
    },
    "inflation_decline_causes": {
        "question": "What were the main reasons for the decline in inflation?",
        "expected_output": "Supply chain improvements and higher interest rates helped reduce inflation.",
        "context": [
            "The Federal Reserve's monetary policy and stabilization of global supply chains played key roles."
        ],
    },
    "recession_fears_2023": {
        "question": "Why were recession fears high in 2023?",
        "expected_output": "Rising interest rates and slowing economic growth led to concerns of a recession.",
        "context": ["Analysts debated the impact of tighter monetary policy on economic expansion."],
    },
}
