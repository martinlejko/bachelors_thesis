"""
PDF data ingestion module.

This module handles loading PDF files from a designated folder and converting
them to the Document model format.
"""

import os
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from pypdf import PdfReader

from src.common.models import Document, DocumentSource
from src.common.config import PUBLIC_DATA_DIR, CACHE_DIR
from src.common.utils import create_hash, save_to_cache, load_from_cache
from src.ingestion.base import DataIngestionSource

logger = logging.getLogger(__name__)


class PdfFolderSource(DataIngestionSource):
    """Data ingestion source for PDF files from a folder."""

    def __init__(self, folder_path: Optional[Path] = None, file_pattern: str = "*.pdf"):
        """
        Initialize the PDF folder source.

        Args:
            folder_path: Path to the folder containing PDF files, defaults to PUBLIC_DATA_DIR
            file_pattern: Glob pattern for PDF files
        """
        self.folder_path = folder_path or PUBLIC_DATA_DIR
        self.file_pattern = file_pattern
        self.cache_dir = CACHE_DIR / "pdf"
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_source_type(self) -> DocumentSource:
        """Get the source type."""
        return DocumentSource.PDF

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the source."""
        return {
            "folder_path": str(self.folder_path),
            "file_pattern": self.file_pattern,
            "timestamp": datetime.now().isoformat(),
        }

    def has_changed(self) -> bool:
        """Check if the PDF files have changed."""
        try:
            current_files = self._get_file_metadata()

            cache_file = "pdf_files_metadata.json"
            cached_files = load_from_cache(self.cache_dir, cache_file)

            if cached_files is None:
                logger.info("No cached PDF file metadata found, considering as changed")
                return True

            current_hash = create_hash(current_files)
            cached_hash = create_hash(cached_files)

            changed = current_hash != cached_hash
            if changed:
                logger.info("PDF files have changed")
            else:
                logger.info("PDF files have not changed")

            return changed

        except Exception as e:
            logger.error(f"Error checking if PDF files have changed: {e}")
            return True

    def load_data(self) -> List[Document]:
        """Load data from PDF files."""
        try:
            files_metadata = self._get_file_metadata()

            cache_file = "pdf_files_metadata.json"
            save_to_cache(files_metadata, self.cache_dir, cache_file)

            documents = []

            for file_info in files_metadata:
                try:
                    file_path = file_info["path"]

                    text = self._extract_text_from_pdf(file_path)

                    if text:
                        doc_id = f"pdf_{file_info['id']}"
                        metadata = {
                            "filename": file_info["filename"],
                            "path": file_path,
                            "size": file_info["size"],
                            "modified": file_info["modified"],
                        }

                        document = Document(id=doc_id, content=text, metadata=metadata, source=DocumentSource.PDF)

                        documents.append(document)
                except Exception as e:
                    logger.error(f"Error processing PDF file {file_info.get('path')}: {e}")

            logger.info(f"Loaded {len(documents)} documents from PDF files")
            return documents

        except Exception as e:
            logger.error(f"Error loading data from PDF files: {e}")
            return []

    def _get_file_metadata(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all PDF files in the folder.

        Returns:
            List[Dict[str, Any]]: List of file metadata
        """
        files_metadata = []

        for file_path in self.folder_path.glob(self.file_pattern):
            if file_path.is_file():
                stat = file_path.stat()
                files_metadata.append(
                    {
                        "id": str(uuid.uuid4()),
                        "filename": file_path.name,
                        "path": str(file_path),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }
                )

        return files_metadata

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            str: Extracted text
        """
        try:
            reader = PdfReader(file_path)
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            return ""
