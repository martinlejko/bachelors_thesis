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
    "investor_support": {
    "question": "What factors helped investors during the 12-month period ending April 30, 2024?",
    "expected_output": "Continued economic growth and cooling inflation provided a supportive backdrop for investors.",
    "context": [
        "The combination of continued economic growth and cooling inflation provided a supportive backdrop for investors during the 12-month reporting period ended April 30, 2024.",
        "Higher interest rates helped to rein in inflation, and the Consumer Price Index decelerated substantially while remaining above pre-pandemic levels.",
        "Wage and job growth powered robust consumer spending, backstopping the economy."
    ]},
    "asia_pacific_performance": {
    "question": "What factors contributed to the performance of Asia-Pacific region stocks?",
    "expected_output": "The strong performance of Japanese equities and India's economic growth contributed to the performance of Asia-Pacific region stocks.",
    "context": [
        "Asia-Pacific region stocks advanced, helped by strong Japanese equities and India's economic growth.",
        "Japanese stocks were bolstered by solid exports, rising profits, and corporate reforms.",
        "India saw significant gains due to strong growth and robust corporate earnings."
    ]},
    "emerging_markets_etf": {
    "question": "What does the iShares Emerging Markets Dividend ETF aim to track?",
    "expected_output": "It aims to track the investment results of an index composed of relatively high dividend paying equities in emerging markets.",
    "context": [
        "The iShares Emerging Markets Dividend ETF seeks to track the investment results of an index composed of relatively high dividend paying equities in emerging markets.",
        "The Index is the Dow Jones Emerging Markets Select Dividend Index.",
        "The Fund uses representative sampling and may not hold all securities in the Index."
    ],
    },
    "emerging_markets_allocation":
    {
        "question": "What is the percentage allocation of the iShares Emerging Markets Dividend ETF into energy?",
        "expected_output": "The ETF has a 21.1% allocation to energy.",
        "context": [
            "The iShares Emerging Markets Dividend ETF has a 21.1% allocation to energy.",
            "The ETF invests in stocks from various sectors, including financials, consumer discretionary, and energy."
        ],
    },
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
    "ai_investments_2023": {
        "question": "Why did AI-related investments increase in 2023?",
        "expected_output": "Businesses and investors saw AI as a transformative technology.",
        "context": ["Companies focused on AI adoption, boosting tech sector valuations."],
    },
    "market_outlook": {
    "question": "What is the investment stance on U.S., Japanese, and European stocks?",
    "expected_output": "There is an overweight stance on U.S. and Japanese stocks, and an underweight stance on European stocks.",
    "context": [
        "There is an overweight stance on U.S. stocks overall, particularly due to emerging AI technologies.",
        "Japanese stocks are also overweight due to shareholder-friendly policies and increased investor interest.",
        "European stocks are underweight in the current investment stance."
    ]
    }
}

ALL_TEST_CASES = {
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
    "job_market_stability": {
        "question": "How did the job market remain stable despite economic challenges?",
        "expected_output": "Strong demand for workers in key industries helped sustain low unemployment rates.",
        "context": ["Tech layoffs were offset by growth in healthcare and hospitality sectors."],
    },
    "crypto_market_volatility": {
        "question": "What caused high volatility in the cryptocurrency market?",
        "expected_output": "Regulatory uncertainty and fluctuating investor sentiment contributed to volatility.",
        "context": ["New regulations and shifts in institutional investment influenced crypto price movements."],
    },
    "tech_stock_performance_2023": {
        "question": "How did technology stocks perform in 2023?",
        "expected_output": "Technology stocks, especially AI-driven companies, led the market rally in 2023.",
        "context": [
            "Investors were optimistic about the long-term potential of artificial intelligence and cloud computing."
        ],
    },
    "federal_reserve_policy": {
        "question": "How did the Federal Reserve’s policy affect markets in 2023?",
        "expected_output": "The Federal Reserve’s decision to pause interest rate hikes helped stabilize market confidence.",
        "context": ["Investors anticipated potential rate cuts in 2024, leading to a rally in bonds and equities."],
    },
    "geopolitical_tensions_markets": {
        "question": "How did geopolitical tensions impact financial markets in late 2023?",
        "expected_output": "Rising geopolitical tensions, including conflicts in the Middle East, increased market volatility.",
        "context": ["Investors became cautious due to uncertainty in global trade and energy markets."],
    },
    "treasury_yields_2023": {
        "question": "What happened to U.S. Treasury yields in 2023?",
        "expected_output": "U.S. Treasury yields rose significantly before moderating toward the year’s end.",
        "context": [
            "Investors adjusted their expectations based on inflation trends and Federal Reserve policy shifts."
        ],
    },
    "sectors_benefitting_high_rates": {
        "question": "Which market sectors benefited from higher interest rates?",
        "expected_output": "Financials and energy sectors performed well due to higher interest rates and energy demand.",
        "context": [
            "Bank profitability improved as lending rates increased, and global energy prices remained elevated."
        ],
    },
    "consumer_spending_2023": {
        "question": "What was the trend in consumer spending in 2023?",
        "expected_output": "Consumer spending remained strong despite rising interest rates.",
        "context": ["A resilient job market and wage growth supported household consumption."],
    },
    "housing_market_interest_rates": {
        "question": "How did higher interest rates affect the housing market?",
        "expected_output": "Housing affordability declined as mortgage rates increased.",
        "context": ["Home sales slowed down due to higher borrowing costs and limited supply."],
    },
    "bond_market_recovery": {
        "question": "What led to the bond market recovery in late 2023?",
        "expected_output": "Expectations of future Federal Reserve rate cuts drove bond market gains.",
        "context": ["Investors anticipated a slowdown in rate hikes, leading to increased bond demand."],
    },
    "stock_market_sectors_2023": {
        "question": "Which sectors outperformed in the stock market in 2023?",
        "expected_output": "Technology and consumer discretionary sectors led market gains.",
        "context": ["AI advancements and strong consumer demand fueled sector performance."],
    },
    "labor_market_resilience": {
        "question": "How did the labor market perform in 2023?",
        "expected_output": "The labor market remained resilient, with low unemployment rates.",
        "context": ["Job openings stayed high, and wage growth continued despite economic uncertainties."],
    },
    "energy_prices_impact": {
        "question": "How did energy prices impact inflation in 2023?",
        "expected_output": "Rising energy prices contributed to inflationary pressures.",
        "context": ["Supply constraints and geopolitical issues led to higher oil and gas prices."],
    },
    "ai_investments_2023": {
        "question": "Why did AI-related investments increase in 2023?",
        "expected_output": "Businesses and investors saw AI as a transformative technology.",
        "context": ["Companies focused on AI adoption, boosting tech sector valuations."],
    },
    "gold_performance_2023": {
        "question": "How did gold perform as an asset in 2023?",
        "expected_output": "Gold prices increased as investors sought safety amid uncertainty.",
        "context": ["Inflation fears and geopolitical tensions led to higher demand for safe-haven assets."],
    },
    "cryptocurrency_rebound": {
        "question": "What caused the cryptocurrency market rebound in 2023?",
        "expected_output": "Institutional adoption and regulatory clarity helped crypto prices recover.",
        "context": ["Bitcoin and Ethereum saw gains as investor confidence improved."],
    },
    "corporate_earnings_2023": {
        "question": "How did corporate earnings influence the stock market in 2023?",
        "expected_output": "Strong corporate earnings supported stock market growth.",
        "context": ["Companies in key sectors reported better-than-expected profits, driving investor confidence."],
    },
}
