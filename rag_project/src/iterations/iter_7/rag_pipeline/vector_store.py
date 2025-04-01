"""
Vector store module.

This module handles the creation and management of vector stores for the RAG system.
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangchainDocument

from src.common.models import ProcessedChunk
from src.common.config import EMBEDDING_MODEL_EXPERIMENTAL, VECTOR_DB_PATH

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manager for vector stores in the RAG system."""

    def __init__(self, embedding_model: str = EMBEDDING_MODEL_EXPERIMENTAL, persist_directory: Optional[Path] = None):
        """
        Initialize the vector store manager.

        Args:
            embedding_model: Name of the embedding model to use
            persist_directory: Directory where the vector store is persisted
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory or VECTOR_DB_PATH
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)

    def create_or_update_vector_store(self, chunks: List[ProcessedChunk]) -> Chroma:
        """
        Create or update the vector store with the given chunks.

        Args:
            chunks: List of processed chunks

        Returns:
            Chroma: The vector store
        """
        try:
            # Convert our chunks to LangChain Documents
            documents = []
            for chunk in chunks:
                doc = LangchainDocument(
                    page_content=chunk.content,
                    metadata={**chunk.metadata, "chunk_id": chunk.id, "document_id": chunk.document_id},
                )
                documents.append(doc)

            # Create or update the vector store
            logger.info(f"Creating vector store with {len(documents)} chunks")

            # Check if the vector store already exists
            if os.path.exists(self.persist_directory) and len(os.listdir(self.persist_directory)) > 0:
                # Load existing vector store and update it
                vector_store = Chroma(persist_directory=str(self.persist_directory), embedding_function=self.embeddings)

                # Add new documents
                vector_store.add_documents(documents)
                logger.info(f"Updated existing vector store with {len(documents)} chunks")
            else:
                # Create new vector store
                vector_store = Chroma.from_documents(
                    documents=documents, embedding=self.embeddings, persist_directory=str(self.persist_directory)
                )
                logger.info(f"Created new vector store with {len(documents)} chunks")

            return vector_store

        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def load_vector_store(self) -> Optional[Chroma]:
        """
        Load an existing vector store.

        Returns:
            Optional[Chroma]: The loaded vector store or None if it doesn't exist
        """
        try:
            if os.path.exists(self.persist_directory) and len(os.listdir(self.persist_directory)) > 0:
                vector_store = Chroma(persist_directory=str(self.persist_directory), embedding_function=self.embeddings)
                logger.info(f"Loaded existing vector store from {self.persist_directory}")
                return vector_store
            else:
                logger.warning(f"No existing vector store found at {self.persist_directory}")
                return None
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return None
