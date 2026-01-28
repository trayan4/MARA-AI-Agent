"""
Document chunking utilities for splitting text into manageable chunks.
Supports multiple splitting strategies and overlap for context preservation.
"""

from typing import List, Optional
from dataclasses import dataclass

from loguru import logger

from config import settings


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    
    text: str
    chunk_id: int
    start_char: int
    end_char: int
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __len__(self):
        return len(self.text)
    
    def __repr__(self):
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Chunk(id={self.chunk_id}, len={len(self.text)}, text='{preview}')"


class TextChunker:
    """
    Split text into overlapping chunks using configurable separators.
    Preserves context through overlap and smart splitting at natural boundaries.
    """
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Number of overlapping characters between chunks
            separators: List of separators to try (in order of preference)
        """
        self.chunk_size = chunk_size or settings.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap
        self.separators = separators or settings.chunking.separators
        
        # Validation
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        logger.debug(
            f"TextChunker initialized: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, separators={self.separators}"
        )
    
    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            metadata: Optional metadata to attach to all chunks
        
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to chunker")
            return []
        
        # Start with the full text
        chunks = self._split_with_separators(text, self.separators)
        
        # Create Chunk objects with metadata
        result = []
        current_pos = 0
        
        for idx, chunk_text in enumerate(chunks):
            chunk = Chunk(
                text=chunk_text,
                chunk_id=idx,
                start_char=current_pos,
                end_char=current_pos + len(chunk_text),
                metadata=metadata.copy() if metadata else {}
            )
            result.append(chunk)
            
            # Update position (accounting for overlap)
            current_pos += len(chunk_text) - self.chunk_overlap
        
        logger.info(
            f"Split text into {len(result)} chunks "
            f"(original length: {len(text)} chars)"
        )
        
        return result
    
    def _split_with_separators(
        self, 
        text: str, 
        separators: List[str]
    ) -> List[str]:
        """
        Recursively split text using separators in priority order.
        
        Args:
            text: Text to split
            separators: List of separators to try
        
        Returns:
            List of text chunks
        """
        if not separators:
            # No more separators, split by character
            return self._split_by_characters(text)
        
        # Try current separator
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        splits = text.split(separator) if separator else [text]
        
        # Process each split
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # Add separator back (except for empty separator)
            split_with_sep = split + separator if separator else split
            
            # If adding this split would exceed chunk_size
            if len(current_chunk) + len(split_with_sep) > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunks.append(current_chunk.rstrip(separator))
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap(current_chunk, separator)
                    current_chunk = overlap_text
                
                # If split itself is too large, recursively split it
                if len(split_with_sep) > self.chunk_size:
                    # Recursively split with remaining separators
                    sub_chunks = self._split_with_separators(
                        split_with_sep.rstrip(separator), 
                        remaining_separators
                    )
                    
                    for sub_chunk in sub_chunks:
                        if current_chunk and len(current_chunk) + len(sub_chunk) > self.chunk_size:
                            chunks.append(current_chunk)
                            current_chunk = self._get_overlap(current_chunk, separator)
                        
                        current_chunk += sub_chunk + (separator if separator else "")
                else:
                    current_chunk += split_with_sep
            else:
                current_chunk += split_with_sep
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.rstrip(separator))
        
        return chunks
    
    def _split_by_characters(self, text: str) -> List[str]:
        """
        Split text by characters when no separators work.
        Used as fallback for very long continuous text.
        
        Args:
            text: Text to split
        
        Returns:
            List of character-split chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def _get_overlap(self, text: str, separator: str) -> str:
        """
        Get overlap text from the end of a chunk.
        
        Args:
            text: Text to get overlap from
            separator: Separator being used
        
        Returns:
            Overlap text
        """
        if len(text) <= self.chunk_overlap:
            return text
        
        # Get last chunk_overlap characters
        overlap = text[-self.chunk_overlap:]
        
        # Try to start at a separator boundary for cleaner overlap
        if separator:
            separator_idx = overlap.find(separator)
            if separator_idx != -1:
                overlap = overlap[separator_idx + len(separator):]
        
        return overlap
    
    def merge_small_chunks(
        self, 
        chunks: List[Chunk], 
        min_size: int = 100
    ) -> List[Chunk]:
        """
        Merge chunks that are too small.
        
        Args:
            chunks: List of chunks to process
            min_size: Minimum chunk size
        
        Returns:
            List of chunks with small ones merged
        """
        if not chunks:
            return []
        
        merged = []
        current = chunks[0]
        
        for next_chunk in chunks[1:]:
            if len(current.text) < min_size:
                # Merge with next chunk
                current.text += " " + next_chunk.text
                current.end_char = next_chunk.end_char
            else:
                merged.append(current)
                current = next_chunk
        
        # Add final chunk
        merged.append(current)
        
        # Re-index chunks
        for idx, chunk in enumerate(merged):
            chunk.chunk_id = idx
        
        logger.debug(f"Merged {len(chunks)} chunks into {len(merged)} chunks")
        
        return merged


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    metadata: Optional[dict] = None
) -> List[Chunk]:
    """
    Convenience function to chunk text with default settings.
    
    Args:
        text: Text to chunk
        chunk_size: Optional chunk size override
        chunk_overlap: Optional overlap override
        metadata: Optional metadata for chunks
    
    Returns:
        List of Chunk objects
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.split_text(text, metadata=metadata)


# Example usage
if __name__ == "__main__":
    # Test the chunker
    sample_text = """
    This is a sample document that needs to be chunked.
    
    It has multiple paragraphs and different sections.
    The chunker should split this intelligently at natural boundaries.
    
    This helps preserve context and makes retrieval more effective.
    Each chunk will have some overlap with adjacent chunks to maintain continuity.
    """
    
    chunks = chunk_text(sample_text, chunk_size=100, chunk_overlap=20)
    
    print(f"\nCreated {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"{chunk}\n")