"""
Test case definitions and expected results for RAG system evaluation.

This module defines test cases used for evaluating the RAG system's performance.
Each test case contains:
- question: The query to be answered by the RAG system
- expected_output: The expected answer from the system
- context: Sample context data that contains the information needed to answer correctly
"""

TEST_CASES = {
    "market_trends_2023": {
        "question": "What were the key market trends in 2023?",
        "expected_output": "The stock market saw a strong rebound driven by technology stocks and easing inflation.",
        "context": [
            "Investors regained confidence as inflation cooled and central banks slowed interest rate hikes."
        ],
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
        "context": [
            "Analysts debated the impact of tighter monetary policy on economic expansion."
        ],
    },
    "job_market_stability": {
        "question": "How did the job market remain stable despite economic challenges?",
        "expected_output": "Strong demand for workers in key industries helped sustain low unemployment rates.",
        "context": [
            "Tech layoffs were offset by growth in healthcare and hospitality sectors."
        ],
    },
    "crypto_market_volatility": {
        "question": "What caused high volatility in the cryptocurrency market?",
        "expected_output": "Regulatory uncertainty and fluctuating investor sentiment contributed to volatility.",
        "context": [
            "New regulations and shifts in institutional investment influenced crypto price movements."
        ],
    }
}

