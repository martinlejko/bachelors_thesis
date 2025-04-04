"""
Initial RAG pipeline implementation.

This module implements the fourth iteration of the RAG pipeline,
based on the proof of concept code. Aiming to improve the quality of the data by implementing more advanced cleaning strategy.
"""

import logging
from typing import Optional

from src.common.config import TEST_URLS, OLLAMA_MODEL, EMBEDDING_MODEL
from src.common.utils import setup_environment
from src.ingestion.base import DataIngestionSource
from src.processing.processor import DocumentProcessor
from src.iterations.iter_9.rag_pipeline.pipeline import RagPipeline
from src.rag_pipeline.vector_store import VectorStoreManager
from src.ingestion.pdf import PdfFolderSource
from src.common.models import Document

try:
    from src.ingestion.confluence import ConfluenceSource

    HAS_CONFLUENCE = True
except Exception:
    HAS_CONFLUENCE = False

logger = logging.getLogger(__name__)


def create_pipeline(
    use_web_urls: bool = True,
    use_pdf: bool = True,
    use_confluence: bool = False,
    confluence_space: Optional[str] = None,
    model_name: str = OLLAMA_MODEL,
    embedding_model: str = EMBEDDING_MODEL,
    force_refresh: bool = False,
) -> RagPipeline:
    """
    Create the RAG pipeline for the first iteration (improved cleaning).

    Args:
        use_web_urls: Whether to use web URLs as a data source
        use_pdf: Whether to use PDF files as a data source
        use_confluence: Whether to use Confluence as a data source
        confluence_space: Confluence space key if using Confluence
        model_name: Name of the LLM model to use
        embedding_model: Name of the embedding model to use
        force_refresh: Force refresh of data sources and vector store
    Returns:
        RagPipeline: The initialized RAG pipeline
    """
    setup_environment()

    data_sources = []

    if use_web_urls:
        try:
            from langchain_community.document_loaders import WebBaseLoader

            class WebSource(DataIngestionSource):
                def __init__(self, urls):
                    self.urls = urls

                def get_source_type(self):
                    from src.common.models import DocumentSource

                    return DocumentSource.WEB

                def get_source_info(self):
                    return {"urls": self.urls}

                def has_changed(self):
                    return False

                def load_data(self):
                    loader = WebBaseLoader(self.urls)
                    docs = loader.load()

                    # Convert to our Document model
                    return [
                        Document(
                            id=f"web_{i}",
                            content=doc.page_content,
                            metadata=doc.metadata,
                            source=self.get_source_type(),
                        )
                        for i, doc in enumerate(docs)
                    ]

            data_sources.append(WebSource(TEST_URLS))
            logger.info(f"Added web source with URLs: {TEST_URLS}")
        except Exception as e:
            logger.error(f"Error adding web source: {e}")

    if use_pdf:
        try:
            data_sources.append(PdfFolderSource())
            logger.info("Added PDF source")
        except Exception as e:
            logger.error(f"Error adding PDF source: {e}")

    if use_confluence and HAS_CONFLUENCE and confluence_space:
        try:
            data_sources.append(ConfluenceSource(space_key=confluence_space))
            logger.info(f"Added Confluence source for space: {confluence_space}")
        except Exception as e:
            logger.error(f"Error adding Confluence source: {e}")

    # Initialize processor
    processor = DocumentProcessor()

    # Initialize vector store manager
    vector_store_manager = VectorStoreManager(embedding_model=embedding_model)

    # Create pipeline
    pipeline = RagPipeline(
        data_sources=data_sources,
        processor=processor,
        vector_store_manager=vector_store_manager,
        model_name=model_name,
        force_refresh=force_refresh,
    )

    pipeline.initialize()

    return pipeline
