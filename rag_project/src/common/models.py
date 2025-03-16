"""
Data models for the RAG system.

This module contains data models for representing different data structures
in the application.
"""
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class DocumentSource(Enum):
    """Enumeration of possible document sources."""
    CONFLUENCE = "confluence"
    PDF = "pdf"
    WEB = "web"
    UNKNOWN = "unknown"

class Document(BaseModel):
    """Model representing a document in the system."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: DocumentSource = DocumentSource.UNKNOWN

class ProcessedChunk(BaseModel):
    """Model representing a processed chunk of a document."""
    id: str
    content: str
    document_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievedDocument(BaseModel):
    """Model representing a document retrieved from the vector store."""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0

class QueryResult(BaseModel):
    """Model representing the result of a query to the RAG system."""
    query: str
    answer: str
    retrieved_documents: List[RetrievedDocument] = Field(default_factory=list)

class CacheInfo(BaseModel):
    """Model representing cache information."""
    hash: str
    timestamp: str
    source_info: Dict[str, Any] = Field(default_factory=dict) 