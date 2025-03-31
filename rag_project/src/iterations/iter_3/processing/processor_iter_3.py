"""
Document processor module.

This module handles the processing of documents, including text cleaning,
splitting into chunks, and caching of processed data. This module is designed to clean the data in moderate detail.
"""

import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import unicodedata


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from src.common.models import Document, ProcessedChunk, DocumentSource
from src.common.config import CHUNK_SIZE, CHUNK_OVERLAP, CACHE_DIR, CACHE_ENABLED, CACHE_VERSION
from src.common.utils import create_hash, save_chunk_debug_info, save_to_cache, load_from_cache

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processor for documents in the RAG system."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = CACHE_DIR / "processed"
        os.makedirs(self.cache_dir, exist_ok=True)

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def process_documents(
        self, documents: List[Document], force_reprocess: bool = False, source_info: Optional[Dict[str, Any]] = None
    ) -> List[ProcessedChunk]:
        """
        Process a list of documents.

        Args:
            documents: List of documents to process
            force_reprocess: Force reprocessing even if cached data exists
            source_info: Information about the source for cache identification

        Returns:
            List[ProcessedChunk]: List of processed chunks
        """
        if CACHE_ENABLED and not force_reprocess:
            logger.info("Checking cache for processed chunks...")
            data_hash = create_hash([doc.model_dump() for doc in documents])
            cache_key = f"processed_{data_hash}_{self.chunk_size}_{self.chunk_overlap}_{CACHE_VERSION}.json"

            cached_data = load_from_cache(self.cache_dir, cache_key)
            if cached_data:
                try:
                    chunks = [ProcessedChunk(**chunk) for chunk in cached_data["chunks"]]
                    logger.info(f"Loaded {len(chunks)} processed chunks from cache")
                    return chunks
                except Exception as e:
                    logger.error(f"Error loading chunks from cache: {e}")

        chunks = []

        for doc in documents:
            try:
                cleaned_content = self.clean_text_moderate(doc.content)

                lc_doc = LangchainDocument(page_content=cleaned_content, metadata=doc.metadata)

                split_docs = self.text_splitter.split_documents([lc_doc])

                for i, split_doc in enumerate(split_docs):
                    chunk_id = f"{doc.id}_chunk_{i}"
                    chunk_content = split_doc.page_content.strip()
                    if not chunk_content:
                        logger.debug(f"Skipping empty chunk {i} from doc {doc.id}")
                        continue

                    metadata = split_doc.metadata.copy()
                    metadata.update(
                        {
                            "chunk_index": i,
                            "chunk_count": len(split_docs),
                            "source": doc.source.value if isinstance(doc.source, DocumentSource) else str(doc.source),
                        }
                    )

                    chunk = ProcessedChunk(
                        id=chunk_id, content=split_doc.page_content, document_id=doc.id, metadata=metadata
                    )

                    chunks.append(chunk)

            except Exception as e:
                logger.error(f"Error processing document {doc.id}: {e}")

        logger.info(f"Processed {len(documents)} documents into {len(chunks)} chunks")

        if CACHE_ENABLED:
            try:
                cache_data = {
                    "chunks": [chunk.model_dump() for chunk in chunks],
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "document_count": len(documents),
                        "chunk_count": len(chunks),
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "cache_version": CACHE_VERSION,
                        "source_info": source_info or {},
                    },
                }

                data_hash = create_hash([doc.model_dump() for doc in documents])
                cache_key = f"processed_{data_hash}_{self.chunk_size}_{self.chunk_overlap}_{CACHE_VERSION}.json"

                save_to_cache(cache_data, self.cache_dir, cache_key)
            except Exception as e:
                logger.error(f"Error caching processed chunks: {e}")

        save_chunk_debug_info(chunks)

        return chunks

    def clean_text_moderate(self, text: str) -> str:
        """Moderate cleaning for financial documents: normalize Unicode, preserve key symbols,
        remove unnecessary special characters, and consolidate whitespace.

        Args:
            text (str): Raw financial text.

        Returns:
            str: Cleaned financial text.
        """

        # Normalize Unicode characters
        text = unicodedata.normalize("NFKC", text)

        # Standardize common typographical characters
        text = text.replace("’", "'").replace("“", '"').replace("”", '"')
        text = text.replace("\u2013", "-").replace("\u00a0", " ")

        cleaned = text.lower()

        cleaned = re.sub(r"[^a-z0-9\s.,?!:%€£$±=()/-]", "", text)

        # Normalize spaces and blank lines
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", cleaned)

        return cleaned
