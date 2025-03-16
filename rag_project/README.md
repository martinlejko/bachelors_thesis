# RAG System Project

A modular Retrieval-Augmented Generation (RAG) system that ingests data from multiple sources, processes it, and provides a question-answering interface using a local LLM.

## Project Structure

```
src/
├── common/              # Common utilities and models
├── data/                # Data storage
│   ├── cache/           # Cache for processed data
│   ├── private/         # Private data (Confluence DOCX files)
│   └── public/          # Public data (PDF files)
├── evaluation/          # Evaluation framework
├── ingestion/           # Data ingestion modules
├── iterations/          # Iterative development folders
│   └── iter_0/          # Initial implementation
├── processing/          # Document processing
├── rag_pipeline/        # RAG pipeline components
└── testing/             # Testing framework
```

## Features

- **Modular Architecture**: Easily swap components like embedding methods or storage solutions
- **Multiple Data Sources**: Ingest data from Confluence API (DOCX) and local PDF files
- **Caching Mechanism**: Avoid reprocessing unchanged data
- **Iterative Development**: Organize improvements in separate iterations
- **Evaluation Framework**: Test and compare different iterations
- **HTML Reports**: Generate visual reports from evaluation results

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure environment variables for Confluence API (if using):
   ```
   export CONFLUENCE_URL="your-confluence-url"
   export CONFLUENCE_USERNAME="your-username"
   export CONFLUENCE_API_KEY="your-api-key"
   ```

3. Place PDF files in `src/data/public/` directory

## Usage

### Running the RAG Pipeline

```python
from src.iterations.iter_0.pipeline import create_pipeline

# Create pipeline with default settings
pipeline = create_pipeline(
    use_web_urls=True,
    use_pdf=True,
    use_confluence=False,  # Set to True if Confluence credentials are configured
    force_refresh=False
)

# Query the pipeline
result = pipeline.query("What is the amount of people in a team?")
print(result.answer)
```

### Running Tests

Run tests for all iterations:
```
pytest src/testing/test_rag.py -v
```

Run tests for a specific iteration:
```
pytest src/testing/test_rag.py -v -k "iter_0"
```

## Extending the Project

### Adding a New Data Source

1. Create a new class that inherits from `DataIngestionSource` in `src/ingestion/`
2. Implement the required methods: `get_source_type()`, `get_source_info()`, `load_data()`, and `has_changed()`
3. Add the new source to the pipeline in your iteration

### Creating a New Iteration

1. Create a new directory in `src/iterations/` (e.g., `iter_1/`)
2. Copy and modify the pipeline from the previous iteration
3. Implement your improvements
4. Tests will automatically run against all iterations

## Evaluation

The evaluation framework uses the Deepeval library to assess:

- Answer correctness
- Answer relevancy
- Conciseness
- Context relevancy
- Faithfulness
- Contextual relevancy

Results are saved as JSON files and HTML reports in the `test_results/` directory.

## License

This project is licensed under the MIT License - see the LICENSE file for details.