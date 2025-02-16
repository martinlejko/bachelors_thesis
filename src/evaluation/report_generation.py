import json
from datetime import datetime

def generate_html_report(json_data):
    # Calculate statistics
    total_tests = len(json_data)
    passed_tests = sum(1 for test in json_data if test['success'])
    failed_tests = total_tests - passed_tests
    
    metrics_summary = {}
    for test in json_data:
        for metric, data in test['metrics'].items():
            if metric not in metrics_summary:
                metrics_summary[metric] = {'passed': 0, 'failed': 0}
            if data['passed']:
                metrics_summary[metric]['passed'] += 1
            else:
                metrics_summary[metric]['failed'] += 1

    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .summary-box {{
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .test-case {{
                background-color: white;
                padding: 20px;
                margin-bottom: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            .metrics-table th, .metrics-table td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .success {{
                color: #28a745;
            }}
            .failure {{
                color: #dc3545;
            }}
            .metric-bar {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-top: 5px;
            }}
            .metric-progress {{
                height: 20px;
                background-color: #f8d7da;
                border-radius: 4px;
                overflow: hidden;
            }}
            .metric-progress-bar {{
                height: 100%;
                background-color: #28a745;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Test Results Summary</h1>
            
            <div class="summary-box">
                <h2>Overall Statistics</h2>
                <p>Total Tests: {total_tests}</p>
                <p>Passed Tests: <span class="success">{passed_tests}</span></p>
                <p>Failed Tests: <span class="failure">{failed_tests}</span></p>
                
                <h3>Metrics Summary</h3>
                <div class="metric-bars">
    """
    
    # Add metric bars
    for metric, data in metrics_summary.items():
        total = data['passed'] + data['failed']
        pass_percentage = (data['passed'] / total) * 100 if total > 0 else 0
        html += f"""
                    <div class="metric-bar">
                        <div style="width: 200px;">{metric}:</div>
                        <div class="metric-progress" style="width: 300px;">
                            <div class="metric-progress-bar" style="width: {pass_percentage}%;"></div>
                        </div>
                        <div>{data['passed']}/{total} passed ({pass_percentage:.1f}%)</div>
                    </div>
        """

    html += """
            </div>
        </div>
            
        <h2>Individual Test Cases</h2>
    """
    
    # Add individual test cases
    for test in json_data:
        status_class = "success" if test['success'] else "failure"
        status_text = "PASSED" if test['success'] else "FAILED"
        
        html += f"""
        <div class="test-case">
            <h3>{test['test_name']} - <span class="{status_class}">{status_text}</span></h3>
            <p><strong>Question:</strong> {test['question']}</p>
            <p><strong>Expected Output:</strong> {test['expected_output']}</p>
            <p><strong>Actual Output:</strong> {test['actual_output']}</p>
            
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Score</th>
                    <th>Threshold</th>
                    <th>Status</th>
                </tr>
        """
        
        for metric, data in test['metrics'].items():
            status_class = "success" if data['passed'] else "failure"
            status_text = "PASSED" if data['passed'] else "FAILED"
            
            html += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{data['score']:.2f}</td>
                    <td>{data['threshold']:.2f}</td>
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

def save_report(json_data_file, output_file="test_report.html"):
    with open(json_data_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    html_content = generate_html_report(json_data)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return output_file

if __name__ == "__main__":
    save_report("/src/test_results/all_test_results_20250216_230657.json")