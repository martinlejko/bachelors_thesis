"""RAG system evaluation tests"""
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from .test_cases import TEST_CASES
from datetime import datetime
import json
import os

def save_test_results(test_name, test_case, evaluation_result):
    """Save test results to a JSON file."""
    # Extract the metrics data from evaluation result
    metrics_data = {}
    for test_result in evaluation_result.test_results:
        for metric_data in test_result.metrics_data:
            metrics_data[metric_data.name] = {
                'score': metric_data.score,
                'threshold': metric_data.threshold,
                'passed': metric_data.success
            }

    result = {
        'test_name': test_name,
        'timestamp': datetime.now().isoformat(),
        'question': test_case.input,
        'actual_output': test_case.actual_output,
        'expected_output': test_case.expected_output,
        'context': test_case.context,
        'metrics': metrics_data,
        'success': evaluation_result.test_results[0].success 
    }
    
    os.makedirs('test_results', exist_ok=True)
    
    filename = f"test_results/{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    return filename

def test_team_size(qa_pipeline, evaluation_metrics):
    """Test the RAG pipeline for team size question."""
    case = TEST_CASES["team_size"]
    actual_output = qa_pipeline.invoke(case["question"])
    
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=actual_output,
        expected_output=case["expected_output"],
        context=case["context"]
    )
    
    result = evaluate(
        test_cases=[test_case],
        metrics=evaluation_metrics,
    )
    
    save_test_results("team_size", test_case, result)
    

def test_programming_language(qa_pipeline, evaluation_metrics):
    """Test the RAG pipeline for programming language question."""
    case = TEST_CASES["programming_language"]
    actual_output = qa_pipeline.invoke(case["question"])
    
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=actual_output,
        expected_output=case["expected_output"],
        context=case["context"]
    )
    
    result = evaluate(
        test_cases=[test_case],
        metrics=evaluation_metrics,
    )

    save_test_results("programming_language", test_case, result)
    
