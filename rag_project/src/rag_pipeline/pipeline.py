"""
RAG pipeline module.

This module implements the main Retrieval-Augmented Generation pipeline,
combining document ingestion, processing, vector store management, and LLM generation.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.common.models import Document, ProcessedChunk, QueryResult, RetrievedDocument
from src.common.config import (
    OLLAMA_MODEL,
    CACHE_ENABLED
)
from src.common.utils import setup_environment
from src.ingestion.base import DataIngestionSource
from src.processing.processor import DocumentProcessor
from src.rag_pipeline.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class RagPipeline:
    """Main RAG pipeline for the system."""
    
    def __init__(
        self,
        data_sources: Optional[List[DataIngestionSource]] = None,
        processor: Optional[DocumentProcessor] = None,
        vector_store_manager: Optional[VectorStoreManager] = None,
        model_name: str = OLLAMA_MODEL,
        force_refresh: bool = False
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            data_sources: List of data sources, if None pipeline expects manual document loading
            processor: Document processor, if None a default one will be created
            vector_store_manager: Vector store manager, if None a default one will be created
            model_name: Name of the LLM model to use
            force_refresh: Force refresh of data sources and vector store
        """
        # Set up environment
        setup_environment()
        
        # Initialize components
        self.data_sources = data_sources or []
        self.processor = processor or DocumentProcessor()
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        self.model_name = model_name
        self.force_refresh = force_refresh
        
        # Initialize the LLM
        self.llm = ChatOllama(model=model_name)
        
        # Initialize the prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Vector store and chain will be initialized when needed
        self.vector_store = None
        self.qa_chain = None
    
    def initialize(self) -> None:
        """
        Initialize the pipeline.
        
        This loads data from all sources, processes it, creates the vector store,
        and initializes the QA chain.
        """
        # Check if we need to reload data
        need_reload = self.force_refresh
        
        if not need_reload:
            for source in self.data_sources:
                if source.has_changed():
                    logger.info(f"Source {source.get_source_type()} has changed, reloading data")
                    need_reload = True
                    break
        
        if need_reload or not self.vector_store:
            # Load and process data
            documents = self._load_data_from_sources()
            chunks = self.processor.process_documents(documents, force_reprocess=self.force_refresh)
            
            # Create vector store
            self.vector_store = self.vector_store_manager.create_or_update_vector_store(chunks)
        else:
            # Load existing vector store
            self.vector_store = self.vector_store_manager.load_vector_store()
            
            if not self.vector_store:
                # No existing vector store, create a new one
                logger.info("No existing vector store, creating a new one")
                documents = self._load_data_from_sources()
                chunks = self.processor.process_documents(documents)
                self.vector_store = self.vector_store_manager.create_or_update_vector_store(chunks)
        
        # Initialize QA chain
        self._initialize_qa_chain()
    
    def load_documents(self, documents: List[Document]) -> None:
        """
        Load documents manually into the pipeline.
        
        Args:
            documents: List of documents to load
        """
        # Process documents
        chunks = self.processor.process_documents(documents)
        
        # Update vector store
        self.vector_store = self.vector_store_manager.create_or_update_vector_store(chunks)
        
        # Re-initialize QA chain
        self._initialize_qa_chain()
    
    def query(self, query: str) -> QueryResult:
        """
        Execute a query against the RAG pipeline.
        
        Args:
            query: User query string
            
        Returns:
            QueryResult: The query result including answer and retrieved documents
        """
        # Ensure the pipeline is initialized
        if not self.qa_chain:
            self.initialize()
        
        # Execute the query
        result = self.qa_chain(query)
        
        return QueryResult(
            query=query,
            answer=result["answer"],
            retrieved_documents=result["retrieved_documents"]
        )
    
    def _load_data_from_sources(self) -> List[Document]:
        """
        Load data from all configured sources.
        
        Returns:
            List[Document]: Combined list of documents from all sources
        """
        all_documents = []
        
        for source in self.data_sources:
            try:
                logger.info(f"Loading data from source: {source.get_source_type()}")
                documents = source.load_data()
                all_documents.extend(documents)
                logger.info(f"Loaded {len(documents)} documents from {source.get_source_type()}")
            except Exception as e:
                logger.error(f"Error loading data from source {source.get_source_type()}: {e}")
        
        logger.info(f"Loaded {len(all_documents)} documents in total")
        return all_documents
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the RAG prompt template.
        
        Returns:
            ChatPromptTemplate: The prompt template
        """
        template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved 
        context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        
        <context>
        {context}
        </context>
        
        Question: {question}
        """
        
        return ChatPromptTemplate.from_template(template)
    
    def _format_docs(self, docs: List[Any]) -> str:
        """
        Format retrieved documents into a single string.
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            str: Formatted context string
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _initialize_qa_chain(self) -> None:
        """Initialize the QA chain using the vector store and LLM."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call initialize() first.")
        
        # Create retriever
        retriever = self.vector_store.as_retriever()
        
        # Create the QA chain
        qa_chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        # Wrap the chain to include retrieved documents in the output
        def invoke_with_retrieval_context(question):
            # Get retrieved documents
            retrieved_docs = retriever.invoke(question)
            
            # Get answer
            answer = qa_chain.invoke(question)
            
            # Convert to RetrievedDocument objects
            retrieved_documents = []
            for i, doc in enumerate(retrieved_docs):
                retrieved_documents.append(
                    RetrievedDocument(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        score=getattr(doc, 'score', 0.0)
                    )
                )
            
            return {
                "answer": answer,
                "retrieved_documents": retrieved_documents
            }
        
        self.qa_chain = invoke_with_retrieval_context
        logger.info("QA chain initialized") 