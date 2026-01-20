"""Document chunking using LlamaIndex.

Provides intelligent document chunking that preserves semantic boundaries
and includes metadata for provenance tracking.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A chunk of document text with metadata."""

    id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="The chunk text content")
    document_id: str = Field(..., description="Source document identifier")
    document_path: str = Field(default="", description="Path to source document")
    chunk_index: int = Field(..., description="Index of this chunk in the document")
    total_chunks: int = Field(default=0, description="Total chunks in document")
    start_char: int = Field(default=0, description="Start character position")
    end_char: int = Field(default=0, description="End character position")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def __hash__(self) -> int:
        """Make chunk hashable by ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Compare chunks by ID."""
        if isinstance(other, DocumentChunk):
            return self.id == other.id
        return False


class DocumentChunker:
    """Intelligent document chunking with semantic awareness.

    Uses LlamaIndex for sophisticated chunking that respects document
    structure and semantic boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
    ) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            separator: Primary separator for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self._llama_index_available = self._check_llama_index()

    def _check_llama_index(self) -> bool:
        """Check if LlamaIndex is available."""
        try:
            from llama_index.core.node_parser import SentenceSplitter
            return True
        except ImportError:
            return False

    def _generate_chunk_id(self, document_id: str, chunk_index: int, text: str) -> str:
        """Generate a unique ID for a chunk."""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{document_id}_chunk_{chunk_index}_{content_hash}"

    def _generate_document_id(self, path: str, content: str) -> str:
        """Generate a unique ID for a document."""
        if path:
            return hashlib.md5(path.encode()).hexdigest()[:12]
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def chunk_text(
        self,
        text: str,
        document_id: str | None = None,
        document_path: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Chunk a text string.

        Args:
            text: The text to chunk
            document_id: Optional document identifier
            document_path: Optional path to source document
            metadata: Optional metadata to attach to chunks

        Returns:
            List of DocumentChunk objects
        """
        if not text.strip():
            return []

        doc_id = document_id or self._generate_document_id(document_path, text)
        metadata = metadata or {}

        if self._llama_index_available:
            return self._chunk_with_llama_index(text, doc_id, document_path, metadata)
        else:
            return self._chunk_simple(text, doc_id, document_path, metadata)

    def _chunk_with_llama_index(
        self,
        text: str,
        document_id: str,
        document_path: str,
        metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        """Chunk using LlamaIndex's SentenceSplitter."""
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import Document

        # Create LlamaIndex document
        doc = Document(text=text, metadata={"source": document_path, **metadata})

        # Use sentence splitter for semantic chunking
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        nodes = splitter.get_nodes_from_documents([doc])

        chunks = []
        for i, node in enumerate(nodes):
            chunk = DocumentChunk(
                id=self._generate_chunk_id(document_id, i, node.text),
                text=node.text,
                document_id=document_id,
                document_path=document_path,
                chunk_index=i,
                total_chunks=len(nodes),
                start_char=node.start_char_idx or 0,
                end_char=node.end_char_idx or len(node.text),
                metadata={**metadata, **node.metadata},
            )
            chunks.append(chunk)

        return chunks

    def _chunk_simple(
        self,
        text: str,
        document_id: str,
        document_path: str,
        metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        """Simple chunking fallback without LlamaIndex."""
        chunks = []

        # Split by separator first
        segments = text.split(self.separator)
        current_chunk = ""
        current_start = 0
        chunk_index = 0

        for segment in segments:
            segment_with_sep = segment + self.separator

            if len(current_chunk) + len(segment_with_sep) <= self.chunk_size:
                current_chunk += segment_with_sep
            else:
                if current_chunk.strip():
                    chunks.append(
                        DocumentChunk(
                            id=self._generate_chunk_id(document_id, chunk_index, current_chunk),
                            text=current_chunk.strip(),
                            document_id=document_id,
                            document_path=document_path,
                            chunk_index=chunk_index,
                            start_char=current_start,
                            end_char=current_start + len(current_chunk),
                            metadata=metadata,
                        )
                    )
                    chunk_index += 1

                # Handle overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap :]
                    current_chunk = overlap_text + segment_with_sep
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                else:
                    current_start += len(current_chunk)
                    current_chunk = segment_with_sep

        # Add final chunk
        if current_chunk.strip():
            chunks.append(
                DocumentChunk(
                    id=self._generate_chunk_id(document_id, chunk_index, current_chunk),
                    text=current_chunk.strip(),
                    document_id=document_id,
                    document_path=document_path,
                    chunk_index=chunk_index,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata=metadata,
                )
            )

        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def chunk_file(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Chunk a file.

        Args:
            file_path: Path to the file
            metadata: Optional metadata to attach

        Returns:
            List of DocumentChunk objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        metadata = metadata or {}
        metadata["filename"] = file_path.name
        metadata["file_extension"] = file_path.suffix

        # Handle different file types
        if file_path.suffix.lower() == ".pdf":
            text = self._read_pdf(file_path)
        elif file_path.suffix.lower() in (".txt", ".md", ".rst"):
            text = file_path.read_text(encoding="utf-8")
        else:
            # Try to read as text
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raise ValueError(f"Cannot read file as text: {file_path}")

        return self.chunk_text(
            text,
            document_path=str(file_path),
            metadata=metadata,
        )

    def _read_pdf(self, file_path: Path) -> str:
        """Read text from a PDF file."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            text_parts = []

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

            return "\n\n".join(text_parts)

        except ImportError:
            raise ImportError("pypdf is required for PDF support. Install with: pip install pypdf")

    def chunk_directory(
        self,
        directory: str | Path,
        extensions: list[str] | None = None,
        recursive: bool = True,
    ) -> list[DocumentChunk]:
        """Chunk all files in a directory.

        Args:
            directory: Path to the directory
            extensions: File extensions to process (e.g., ['.txt', '.pdf'])
            recursive: Whether to process subdirectories

        Returns:
            List of DocumentChunk objects from all files
        """
        directory = Path(directory)

        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        extensions = extensions or [".txt", ".md", ".pdf", ".rst"]
        extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions]

        all_chunks = []

        pattern = "**/*" if recursive else "*"

        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    chunks = self.chunk_file(file_path)
                    all_chunks.extend(chunks)
                except (ValueError, UnicodeDecodeError) as e:
                    print(f"Warning: Could not process {file_path}: {e}")

        return all_chunks

    def get_chunk_statistics(self, chunks: list[DocumentChunk]) -> dict[str, Any]:
        """Get statistics about a set of chunks.

        Args:
            chunks: List of chunks to analyze

        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "avg_chunk_size": 0,
                "unique_documents": 0,
            }

        sizes = [len(c.text) for c in chunks]
        documents = set(c.document_id for c in chunks)

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(sizes),
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "unique_documents": len(documents),
        }
