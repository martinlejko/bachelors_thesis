"""
Base classes for data ingestion.

This module defines the abstract interfaces that all data ingestion
sources must implement.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from src.common.models import Document, DocumentSource

class DataIngestionSource(ABC):
    """Abstract base class for data ingestion sources."""
    
    @abstractmethod
    def get_source_type(self) -> DocumentSource:
        """
        Get the type of the source.
        
        Returns:
            DocumentSource: The source type
        """
        pass
    
    @abstractmethod
    def get_source_info(self) -> Dict[str, Any]:
        """
        Get information about the source for caching purposes.
        
        Returns:
            Dict[str, Any]: Source information
        """
        pass
    
    @abstractmethod
    def load_data(self) -> List[Document]:
        """
        Load data from the source.
        
        Returns:
            List[Document]: List of documents
        """
        pass
    
    @abstractmethod
    def has_changed(self) -> bool:
        """
        Check if the source data has changed since the last ingestion.
        
        Returns:
            bool: True if the data has changed, False otherwise
        """
        pass 