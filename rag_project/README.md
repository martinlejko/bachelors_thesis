# RAG System Project for Bachelor Thesis

This repository contains the source code for a Bachelor's thesis project focused on building and evaluating a modular Retrieval-Augmented Generation (RAG) system. The system ingests data from various sources (mostly financial data), processes it, and utilizes a RAG pipeline with local LLMs (via Ollama) to answer questions based on the ingested context. The project also includes a comprehensive testing and evaluation pipeline using the DeepEval library to compare different RAG configurations (iterations).

## Project Goal

The primary goal is to experiment with different configurations of a RAG pipeline (e.g., varying data cleaning methods, chunking strategies, embedding models, retrieval techniques) and systematically evaluate their performance on a specific question-answering task using DeepEval metrics. This allows for identifying the most effective RAG setup for the given domain and data.

## Features

*   **Modular RAG Pipeline**: Core components (ingestion, processing, vector store, LLM interaction) are designed for easy modification and extension.
*   **Multiple Data Sources**: Supports ingestion from Web URLs, local PDF files, and Confluence. Easily extensible for other sources.
*   **Iterative Development**: Each significant change or experiment is encapsulated within its own iteration (`src/iterations/iter_X`), allowing for clear comparison.
*   **Local LLM Integration**: Leverages Ollama to run language models locally, ensuring privacy and control.
*   **Comprehensive Evaluation**: Uses DeepEval (`deepeval`) to measure various aspects of the RAG output (e.g., Faithfulness, Answer Relevancy, Context Relevancy).
*   **Task Automation**: Uses `poe the poet` for streamlined execution of common tasks like testing, running pipelines, and setup.
*   **Dependency Management**: Uses Poetry for robust dependency management.
*   **Documentation**: Uses MkDocs for generating project documentation.

## Project Structure

```
.

├── debug/               # Log files for debugging (e.g., pytest logs, chunked files)
├── docs/                # MkDocs documentation source files
├── proof_of_concept/    # Initial proof-of-concept script with some testing
├── pyproject.toml       # Poetry configuration, dependencies, and poe tasks
├── src/                 # Main source code
│   ├── data/            # Data storage 
│   │   ├── cache/       # Cache for processed data and vector stores
│   │   ├── private/     # Private data (e.g., Confluence exports)
│   │   └── public/      # Public data (e.g., PDFs)
│   ├── common/          # Shared utilities, configurations, models
│   ├── evaluation/      # Evaluation datasets and logic (using deepeval)
│   ├── ingestion/       # Data ingestion modules (PDF, Web, Confluence)
│   ├── iterations/      # Different RAG pipeline implementations/experiments
│   │   ├── iter_0/      # Baseline implementation
│   │   ├── iter_1/      # Example: Iteration with advanced cleaning
│   │   └── ...          # Further iterations
│   ├── processing/      # Document cleaning, chunking, processing logic
│   ├── rag_pipeline/    # Core RAG pipeline components (vector store, retrieval, generation)
│   └── testing/         # Test suite (using pytest and deepeval)
└── test_results/        # Evaluation results (JSON, HTML reportsd)
```

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone git@github.com:martinlejko/bachelors_thesis.git
    cd bachelors_thesis/rag_project 
    ```

2.  **Install Ollama:**
    Follow the official Ollama installation instructions for your operating system: [https://ollama.com/](https://ollama.com/)

3.  **Pull Required Ollama Models:**
    This project requires specific LLM and embedding models to be available via Ollama. Check the configuration (e.g., `src/common/config.py`) for the model names specified in `OLLAMA_MODEL` and `EMBEDDING_MODEL`. Pull them using:
    ```bash
    ollama pull <model_name> # e.g., ollama pull lllama3.1:8b
    ollama pull <embedding_model_name> # e.g., ollama pull nomic-embed-text
    ```
    *Note: Ensure the models defined in your config are pulled.*

4.  **Run Ollama Service:**
    Ollama needs to be running in the background to serve the models. Open a separate terminal and run:
    ```bash
    ollama serve
    ```
    Keep this terminal running while you use the RAG pipeline. Ensure that the models are running in the same environment as the project.

5.  **Install Project Dependencies using Poetry:**
    If you don't have Poetry installed, follow the instructions [here](https://python-poetry.org/docs/#installation).
    *Note: I would suggest downloading also the Poetry plugin for `poetry shell` to make it easier to work with the virtual environment. [here](https://github.com/python-poetry/poetry-plugin-shell)*

    ```bash
    poetry install
    ```
    This command creates a virtual environment and installs all necessary dependencies listed in `pyproject.toml`.

6.  **Configure Environment Variables (Optional):**
    If using Confluence as a data source, create a `.env` file in the project root or set the following environment variables:
    ```bash
    CONFLUENCE_URL="your-confluence-url"
    CONFLUENCE_USERNAME="your-username"
    CONFLUENCE_API_KEY="your-api-key"
    ```

7.  **Download NLTK Data:**
    Some processing steps might require NLTK data (e.g., for tokenization, stop words). Run the setup script using Poe:
    ```bash
    poetry run nltk-setup
    ```

8.  **Place Data Files:**
    *   Place any public PDF files you want to ingest into the `data/public/` directory.
    *   If using Confluence exports or other private data, place them accordingly (e.g., `data/private/`) and ensure they are listed in your `.gitignore` if necessary.

## Usage

All commands should be run from the project's root directory using `poe <task_name>`.

*   **Runing Proof of Concept:**
    If you want to try out the functionality you can simply run the proof of concept script:
    ```bash
    poe run-proof
    ```

*   **Run Evaluation Tests for PoC:**
    If you are curios about the evaluation and would like to see a quick example. You can run the testing pipeline for the proof of concept:
    ```bash
    poe test-proof
    ```
    Test results will be saved in the `test_results/` directory. There will be a Html report generated for all the iterations from the original JSON file. Logs are saved to `debug/`.

*   **Run a Specific RAG Pipeline Iteration:**
    This typically involves initializing the pipeline and querying it. Check the `src/testing/run_pipeline.py` script. Where you can test out iterations by changing the import path.
    And run the pipeline using:
    ```bash
    poe run-pipeline
    ```

*   **Run Evaluation Tests:**
    Execute the test suite using pytest and DeepEval. This will run tests against all implemented iterations.
    ```bash
    poe test-iterations
    ```
    Test results will be saved in the `test_results/` directory. There will be a Html report generated for all the iterations from the original JSON file. Logs are saved to `debug/pytest.log`.


*   **Serve Documentation Locally:**
    View the project documentation generated by MkDocs.
    ```bash
    mkdocs serve
    ```
    Then open your browser to `http://127.0.0.1:8000`.

*   **Build Documentation:**
    Generate the static HTML documentation site.
    ```bash
    mkdocs build
    ```
    The output will be in the `site/` directory.

## Evaluation Framework

The evaluation process (`src/testing/test_rag.py` and `src/evaluation/`) uses the `deepeval` library. It defines test cases (questions and expected context/answers) and runs them against each RAG pipeline iteration (`src/iterations/iter_X`).

Key metrics measured might include:

*   **Faithfulness**: How factually consistent the answer is with the retrieved context.
*   **Answer Relevancy**: How relevant the answer is to the given question.
*   **Context Relevancy**: How relevant the retrieved context chunks are to the question.
*   **Conciseness**: How concise the answer is.

The framework is designed to produce comparable results across iterations, helping to identify the impact of specific changes on the RAG system's performance. Results are typically logged and saved, including detailed HTML report.

## Extending the Project

*   **Adding a New Data Source**: Create a new class inheriting from `DataIngestionSource` in `src/ingestion/`, implement its methods, and integrate it into a pipeline iteration.
*   **Creating a New Iteration**:
    1.  Create a new directory under `src/iterations/` (e.g., `iter_M/`) for the new iteration.
    2.  Modify the pipeline logic (`pipeline.py`) or associated components (e.g., processing functions) within the new iteration directory (`src/iterations/iter_M/`).
    3.  Ensure the new iteration's `create_pipeline` function is correctly imported and used in the testing/evaluation framework (`src/testing/test_rag.py`). Tests should automatically pick up the new iteration if structured correctly.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if one exists, otherwise specify license).