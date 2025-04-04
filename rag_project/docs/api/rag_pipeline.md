# RAG Pipeline

::: src.rag_pipeline.pipeline

::: src.rag_pipeline.vector_store

This section documents the core RAG (Retrieval-Augmented Generation) pipeline components.

## Core Module

::: proof_of_concept.proof_of_concept

## Main Functions

The RAG pipeline consists of the following key components:

- `load_data()`: Loads data from URLs
- `process_data()`: Splits documents into chunks
- `create_vectorstore()`: Creates vector embeddings
- `setup_model()`: Initializes the language model
- `create_prompt_template()`: Creates the RAG prompt
- `create_qa_chain()`: Builds the question-answering chain 