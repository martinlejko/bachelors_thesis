"""
Report generation module.

This module handles the generation of HTML reports from JSON evaluation results.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Counter
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from src.common.config import TEST_RESULTS_DIR

logger = logging.getLogger(__name__)


def generate_html_report(json_data: List[Dict[str, Any]]) -> str:
    """
    Generate an HTML report from JSON evaluation results.

    Args:
        json_data: List of test result dictionaries

    Returns:
        str: HTML report content
    """
    # Basic test statistics
    total_tests = len(json_data)
    passed_tests = sum(1 for test in json_data if test.get("success", False))
    failed_tests = total_tests - passed_tests
    overall_success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    # Collect metrics across all tests
    all_metrics = set()
    metrics_totals = defaultdict(lambda: {"total": 0, "passed": 0, "scores": []})
    
    for test in json_data:
        if "metrics" in test:
            for metric_name, metric_data in test["metrics"].items():
                all_metrics.add(metric_name)
                metrics_totals[metric_name]["total"] += 1
                if metric_data.get("passed", False):
                    metrics_totals[metric_name]["passed"] += 1
                metrics_totals[metric_name]["scores"].append(metric_data.get("score", 0))

    # Calculate average scores for each metric
    metrics_avg = {}
    for metric_name, data in metrics_totals.items():
        scores = data["scores"]
        metrics_avg[metric_name] = sum(scores) / len(scores) if scores else 0

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG System Evaluation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .summary {{
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .summary-stats {{
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
                margin-top: 20px;
            }}
            .stat-box {{
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                min-width: 150px;
                margin: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                flex: 1;
            }}
            .metric-summary {{
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                margin: 15px 0;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .test-case {{
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }}
            .metrics-table th, .metrics-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            .metrics-table th {{
                background-color: #f2f2f2;
            }}
            .success {{
                color: #28a745;
                font-weight: bold;
            }}
            .failure {{
                color: #dc3545;
                font-weight: bold;
            }}
            .progress-bar {{
                height: 20px;
                background-color: #e9ecef;
                border-radius: 5px;
                margin-top: 10px;
                overflow: hidden;
            }}
            .progress {{
                height: 100%;
                background-color: #28a745;
                border-radius: 5px;
                width: {overall_success_rate}%;
            }}
            .test-summary {{
                margin-top: 10px;
                padding: 10px;
                background-color: #f2f2f2;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>RAG System Evaluation Report</h1>
        <p>Generated on: {timestamp}</p>
        
        <div class="summary">
            <h2>Overall Summary</h2>
            <div class="progress-bar">
                <div class="progress"></div>
            </div>
            <div class="summary-stats">
                <div class="stat-box">
                    <h3>Total Tests</h3>
                    <p>{total_tests}</p>
                </div>
                <div class="stat-box">
                    <h3>Passed</h3>
                    <p class="success">{passed_tests}</p>
                </div>
                <div class="stat-box">
                    <h3>Failed</h3>
                    <p class="failure">{failed_tests}</p>
                </div>
                <div class="stat-box">
                    <h3>Success Rate</h3>
                    <p>{overall_success_rate:.2f}%</p>
                </div>
            </div>
            
            <h2>Metrics Summary</h2>
    """
    
    # Add metrics summary section
    for metric_name in sorted(all_metrics):
        metric_data = metrics_totals[metric_name]
        pass_rate = (metric_data["passed"] / metric_data["total"]) * 100 if metric_data["total"] > 0 else 0
        avg_score = metrics_avg[metric_name]
        
        status_class = "success" if pass_rate >= 70 else "failure"
        
        html += f"""
            <div class="metric-summary">
                <h3>{metric_name}</h3>
                <p><strong>Pass Rate:</strong> <span class="{status_class}">{metric_data["passed"]}/{metric_data["total"]} ({pass_rate:.2f}%)</span></p>
                <p><strong>Average Score:</strong> {avg_score:.2f}</p>
                <div class="progress-bar">
                    <div class="progress" style="width: {pass_rate}%"></div>
                </div>
            </div>
        """
    
    html += """
        </div>
        
        <h2>Individual Test Results</h2>
    """

    for test in json_data:
        status_class = "success" if test.get("success", False) else "failure"
        status_text = "PASSED" if test.get("success", False) else "FAILED"

        # Calculate average score for this test if metrics exist
        avg_test_score = 0
        if "metrics" in test:
            scores = [data.get("score", 0) for data in test["metrics"].values()]
            avg_test_score = sum(scores) / len(scores) if scores else 0

        html += f"""
        <div class="test-case">
            <h3>{test.get("test_name", "Unnamed Test")} - <span class="{status_class}">{status_text}</span></h3>
            <div class="test-summary">
                <p><strong>Average Score: </strong>{avg_test_score:.2f}</p>
            </div>
            <p><strong>Question:</strong> {test.get("question", "")}</p>
            <div><strong>Actual context:</strong> <pre>{test.get("actual_context", test.get("actual_conext", ""))}</pre></div>
            <p><strong>Expected Output:</strong> {test.get("expected_output", "")}</p>
            <p><strong>Actual Output:</strong> {test.get("actual_output", "")}</p>
            
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Score</th>
                    <th>Threshold</th>
                    <th>Status</th>
                </tr>
        """

        if "metrics" in test:
            for metric, data in test["metrics"].items():
                status_class = "success" if data.get("passed", False) else "failure"
                status_text = "PASSED" if data.get("passed", False) else "FAILED"

                html += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{data.get("score", 0):.2f}</td>
                    <td>{data.get("threshold", 0):.2f}</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
                """

        html += """
            </table>
        </div>
        """

    html += """
        </div>
    </body>
    </html>
    """

    return html


def save_report(json_data_file: str, output_file: Optional[str] = None) -> str:
    """
    Save an HTML report from a JSON results file.

    Args:
        json_data_file: Path to the JSON results file
        output_file: Path to save the HTML report, if None a default path is used

    Returns:
        str: Path to the saved HTML report
    """
    try:
        with open(json_data_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        html_content = generate_html_report(json_data)

        if output_file is None:
            input_path = Path(json_data_file)
            output_file = str(TEST_RESULTS_DIR / f"{input_path.stem}.html")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML report saved to {output_file}")
        return output_file

    except Exception as e:
        logger.error(f"Error generating HTML report: {e}")
        raise


def report_from_latest_json() -> Optional[str]:
    """
    Generate an HTML report from the latest JSON results file.

    Returns:
        Optional[str]: Path to the saved HTML report, or None if no JSON files found
    """
    try:
        json_files = list(TEST_RESULTS_DIR.glob("*.json"))
        if not json_files:
            logger.warning("No JSON result files found")
            return None

        latest_file = max(json_files, key=os.path.getmtime)

        return save_report(str(latest_file))

    except Exception as e:
        logger.error(f"Error generating report from latest JSON: {e}")
        return None
