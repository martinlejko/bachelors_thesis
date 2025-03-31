"""
Document processor module.

This module handles the processing of documents, including text cleaning,
splitting into chunks, and caching of processed data. It implementes regex-based cleaning, where we are trying to match some patterns that occur in the documents and remove them.
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

# --- Configuration for Clening ---
# Set to True to remove lines matching footer/identifier patterns
REMOVE_IDENTIFIERS_FOOTERS = True
# Set to True to remove lines looking like TOC entries (Text ..... PageNumber)
REMOVE_TOC_LINES = True
# Set to True to reformat lines like 'Sector ..... Value%' into 'Sector Value%'
REFORMAT_ALLOCATION_LINES = True
# How to handle complex lines (e.g., 'Metric .... Val1 Val2 Val3'):
# 'reformat': Change to 'Metric Val1 Val2 Val3' (keeps data, may be messy)
# 'remove': Delete these lines entirely (safer if data is duplicated elsewhere)
# 'keep_original': Leave these lines untouched
HANDLE_COMPLEX_DOT_LEADER_LINES = "reformat"

# --- Regex Patterns (Compile for efficiency) ---

# Identifiers (Customize based on your documents)
RE_IDENTIFIER = re.compile(r"^\s*(?:BNM\S+|ISIN\s+[A-Z0-9]+|CUSIP\s+[A-Z0-9]+)\s*$", re.IGNORECASE)

# Footers (Customize heavily based on your documents)
RE_FOOTERS = [
    re.compile(r"^\s*\d+\s+[A-Z\s]+(?:REPORT|SHAREHOLDERS|INC|LLC|LTD)", re.IGNORECASE),
    re.compile(r"^\s*page\s+\d+\s*(?:of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:www\.|https?://).+\.(?:com|org|net)\b", re.IGNORECASE),
    re.compile(r"^\s*\d{1,2}\s+\d{2,4}\s+I\s?S\s?H\s?A\s?R\s?E\s?S\s+A\s?N\s?N\s?U\s?A\s?L", re.IGNORECASE),
]

RE_DOT_LEADER = re.compile(r"^(.*?)\s*((?:[\s.]\.){5,}|[\s.]{5,})\s*(.+)$")

# Pattern to identify simple TOC lines (ending in digits only)
RE_TOC_SIMPLE = re.compile(r"^\s*\d+\s*$")

# Pattern to identify single allocation values (number, maybe %, $, parens)
RE_ALLOCATION_VALUE = re.compile(r"^\s*\(?[$€£]?\s?[\d,.' -]+?\s?\)?%?\s*$")


def remove_specific_lines(text: str) -> str:
    """Removes lines matching identifier or footer patterns if REMOVE_IDENTIFIERS_FOOTERS is True."""
    if not REMOVE_IDENTIFIERS_FOOTERS:
        return text

    cleaned_lines = []
    lines_removed_count = 0
    for line in text.splitlines():
        removed = False
        if RE_IDENTIFIER.match(line):
            removed = True
        else:
            for pattern in RE_FOOTERS:
                if pattern.search(line):
                    removed = True
                    break
        if removed:
            lines_removed_count += 1
            logger.debug(f"Removed identifier/footer line: '{line}'")
            continue
        cleaned_lines.append(line)

    if lines_removed_count > 0:
        logger.debug(f"Removed {lines_removed_count} identifier/footer lines.")
    return "\n".join(cleaned_lines)


def clean_dot_leader_lines(text: str) -> str:
    """
    Intelligently handles lines with dot leaders based on global config flags.
    """
    cleaned_lines = []
    lines_processed_count = 0

    for line in text.splitlines():
        match = RE_DOT_LEADER.match(line)
        if match:
            label = match.group(1).strip()
            value_part = match.group(3).strip()
            lines_processed_count += 1

            if not label:
                cleaned_lines.append(line)
                continue

            if REMOVE_TOC_LINES and RE_TOC_SIMPLE.fullmatch(value_part):
                logger.debug(f"Removed TOC line: '{line}'")
                continue

            if REFORMAT_ALLOCATION_LINES and RE_ALLOCATION_VALUE.fullmatch(value_part) and len(value_part) < 25:
                cleaned_value = re.sub(r"\s+", " ", value_part).strip()
                reformatted_line = f"{label} {cleaned_value}"
                cleaned_lines.append(reformatted_line)
                logger.debug(f"Reformatted allocation line: '{line}' -> '{reformatted_line}'")
                continue

            if HANDLE_COMPLEX_DOT_LEADER_LINES == "reformat":
                cleaned_value = re.sub(r"\s+", " ", value_part).strip()
                reformatted_line = f"{label} {cleaned_value}"
                cleaned_lines.append(reformatted_line)
                logger.debug(f"Reformatted complex dot leader line: '{line}' -> '{reformatted_line}'")
                continue
            elif HANDLE_COMPLEX_DOT_LEADER_LINES == "remove":
                logger.debug(f"Removed complex dot leader line: '{line}'")
                continue
            else:
                cleaned_lines.append(line)

        else:
            cleaned_lines.append(line)

    if lines_processed_count > 0:
        logger.debug(f"Processed {lines_processed_count} dot-leader lines based on configuration.")
    return "\n".join(cleaned_lines)


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
                content_step1 = remove_specific_lines(doc.content)
                if not content_step1.strip():
                    logger.warning(f"Content for doc {doc.id} became empty after removing specific lines.")
                    continue

                content_step2 = clean_dot_leader_lines(content_step1)
                if not content_step2.strip():
                    logger.warning(f"Content for doc {doc.id} became empty after cleaning dot leaders.")
                    continue

                cleaned_content = self.clean_text_moderate(content_step2)
                if not cleaned_content.strip():
                    logger.warning(f"Content for doc {doc.id} became empty after minimal cleaning.")
                    continue

                lc_doc = LangchainDocument(page_content=cleaned_content, metadata=doc.metadata)

                split_docs = self.text_splitter.split_documents([lc_doc])

                for i, split_doc in enumerate(split_docs):
                    chunk_id = f"{doc.id}_chunk_{i}"

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

        # Standardize common typographic variations
        text = text.replace("’", "'").replace("“", '"').replace("”", '"')
        text = text.replace("\u2013", "-").replace("\u00a0", " ")

        cleaned = text.lower()
        # Remove unwanted special characters but keep financial symbols
        cleaned = re.sub(r"[^a-z0-9\s.,?!:%€£$±=()/-]", "", text)

        # Normalize spaces and blank lines
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", cleaned)

        return cleaned
