"""
Document processor module.

This module handles the processing of documents, including text cleaning,
splitting into chunks, and caching of processed data.
"""

import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from src.common.models import Document, ProcessedChunk, DocumentSource
from src.common.config import CHUNK_SIZE, CHUNK_OVERLAP, CACHE_DIR, CACHE_ENABLED, CACHE_VERSION
from src.common.utils import create_hash, save_to_cache, load_from_cache

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
        # Check cache if enabled and not forcing reprocess
        if CACHE_ENABLED and not force_reprocess:
            logger.info("Checking cache for processed chunks...")
            # Create a unique cache key based on documents content
            data_hash = create_hash([doc.model_dump() for doc in documents])
            cache_key = f"processed_{data_hash}_{self.chunk_size}_{self.chunk_overlap}_{CACHE_VERSION}.json"

            # Try to load from cache
            cached_data = load_from_cache(self.cache_dir, cache_key)
            if cached_data:
                try:
                    # Convert cached data to ProcessedChunk objects
                    chunks = [ProcessedChunk(**chunk) for chunk in cached_data["chunks"]]
                    logger.info(f"Loaded {len(chunks)} processed chunks from cache")
                    return chunks
                except Exception as e:
                    logger.error(f"Error loading chunks from cache: {e}")

        # Process documents
        chunks = []

        for doc in documents:
            try:
                # Clean the document content first
                cleaned_content = self.clean_text_minimal(doc.content)

                # Convert to LangChain Document for splitting
                lc_doc = LangchainDocument(page_content=cleaned_content, metadata=doc.metadata)

                # Split into chunks
                split_docs = self.text_splitter.split_documents([lc_doc])

                # Convert back to our model
                for i, split_doc in enumerate(split_docs):
                    chunk_id = f"{doc.id}_chunk_{i}"
                    chunk_content = split_doc.page_content.strip()
                    if not chunk_content:
                        logger.debug(f"Skipping empty chunk {i} from doc {doc.id}")
                        continue

                    # Update metadata with chunk info
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

        # Cache the processed chunks if enabled
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

                # Create a unique cache key based on documents content
                data_hash = create_hash([doc.model_dump() for doc in documents])
                cache_key = f"processed_{data_hash}_{self.chunk_size}_{self.chunk_overlap}_{CACHE_VERSION}.json"

                save_to_cache(cache_data, self.cache_dir, cache_key)
            except Exception as e:
                logger.error(f"Error caching processed chunks: {e}")

        debug_file = "debug/processed_chunks.txt"

        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(f"Total chunks processed: {len(chunks)}\n\n")
            for i, chunk in enumerate(chunks):
                f.write(f"Chunk {i + 1}/{len(chunks)}\n")
                f.write(f"ID: {chunk.id}\n")
                f.write(f"Document ID: {chunk.document_id}\n")
                f.write(f"Metadata: {chunk.metadata}\n")
                f.write("Content:\n")
                f.write(f"{chunk.content}\n")
                f.write("-" * 80 + "\n\n")

        logger.debug(f"Saved chunk debug info to {debug_file}")
        return chunks

    def clean_text_minimal(self, text: str) -> str:
        """Minimal cleaning: lowercase, consolidate whitespace, basic unicode normalization.
        Args:
            text: Raw text

        Returns:
            str: Cleaned text
        """
        cleaned = text.lower()
        cleaned = re.sub(r"[^a-z0-9\s.,?!-]", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned
