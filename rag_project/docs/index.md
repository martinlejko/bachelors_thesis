# RAG Project Documentation

This is the documentation for the Retrieval-Augmented Generation (RAG) project developed as part of a Bachelor's thesis.

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that combines the power of large language models with a retrieval mechanism to provide accurate and contextually relevant responses to user queries. The system retrieves information from a knowledge base before generating responses, improving the accuracy and relevance of the answers.

## Project Structure

- **proof_of_concept/**: Contains the RAG pipeline implementation
  - Core RAG functionality
  - Document processing
  - Vector database integration
  - Question answering chain
  
- **testing/**: Contains testing utilities and test cases
  - Test data definitions
  - Evaluation metrics
  - Test fixtures and utilities

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