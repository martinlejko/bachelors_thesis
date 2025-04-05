# RAG Project Documentation

This is the documentation for the Retrieval-Augmented Generation (RAG) project, mostly focused on the financial documents. It should serve as a testing ground where we iterate on the solution and use our RAG testing pipeline to evaluate the changes. It was developed as part of a Bachelor's thesis.

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that combines the power of large language models with a retrieval mechanism to provide accurate and contextually relevant responses to user queries. The system retrieves information from a knowledge base before generating responses, improving the accuracy and relevance of the answers. It also contains a testing pipeline to evaluate the performance of the RAG system. Using GEval, to evaluate as close to the human like evaluation as possible.

# Repository Structure

The project is organized into two main folders:

- **proof_of_concept/**  
  This folder contains the initial proof-of-concept for the RAG pipeline. It includes:
  - **Core RAG Functionality:** Basic implementation of the retrieval-augmented generation workflow.
  - **Document Processing:** Proof-of-concept techniques for splitting and processing documents.
  - **Vector Database Integration:** Early integration with a vector store for handling document chunks.
  - **Question Answering Chain:** A simple QA chain to demonstrate how the system can answer questions based on retrieved documents.
  - **Evaluation:** Initial evaluation metrics and methods to assess the performance of the RAG system.

- **src/**  
  This folder contains the full-fledged implementation of the RAG pipeline and its testing framework. It includes:
  - **Data Ingestion:** Modules that load data (e.g., PDF ingestion) into the system.
  - **Document Processing:** More advanced methods for cleaning, chunking, and processing documents.
  - **Vector Store Management:** Handling vector store creation and updates using local embedding models.
  - **QA & Pipeline:** The RAG pipeline classes that tie together data ingestion, processing, vector search, and query answering.
  - **Evaluation & Testing:** Test fixtures, evaluation metrics, and utilities for rigorously testing and evaluating the RAG system.

This structure allows you to iterate over the more robust and scalable implementation in `src/` for production tests and evaluation. While the `proof_of_concept/` folder serves as a try out of the basic functionality. 

## Getting Started

To run the RAG pipeline:

```python
from proof_of_concept.proof_of_concept import load_data, process_data, create_vectorstore, setup_model, create_prompt_template, create_qa_chain

# Example URLs
urls = ["https://example.com/document1", "https://example.com/document2"]

# Data pipeline
raw_data = load_data(urls)
processed_data = process_data(raw_data)
vectorstore = create_vectorstore(processed_data)

# Model and prompt setup
model = setup_model()
prompt = create_prompt_template()

# Create QA chain
qa_chain = create_qa_chain(vectorstore, model, prompt)

# Ask a question
question = "What is RAG?"
response = qa_chain(question)
print(f"Answer: {response['answer']}")
``` 