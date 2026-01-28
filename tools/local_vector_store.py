"""
Local vector store using SQLite for storage and FAISS for vector search.
Supports hybrid search combining semantic (vector) and lexical (BM25) retrieval.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import pickle

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from loguru import logger

from config import settings
from tools.chunking import Chunk


class VectorStore:
    """
    Local vector store with hybrid search capabilities.
    Combines dense vector search (FAISS) with sparse lexical search (BM25).
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        use_hybrid: Optional[bool] = None,
        hybrid_alpha: Optional[float] = None,
    ):
        """
        Initialize vector store.
        
        Args:
            db_path: Path to SQLite database
            embedding_model: HuggingFace model name for embeddings
            use_hybrid: Enable hybrid search (vector + BM25)
            hybrid_alpha: Weight for vector search (0.0 = all BM25, 1.0 = all vector)
        """
        self.db_path = db_path or settings.vector_store.path
        self.embedding_model_name = embedding_model or settings.embeddings.model
        self.use_hybrid = use_hybrid if use_hybrid is not None else settings.vector_store.use_hybrid
        self.hybrid_alpha = hybrid_alpha if hybrid_alpha is not None else settings.vector_store.hybrid_alpha
        self.dimension = settings.embeddings.dimension
        
        # Ensure data directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize SQLite database
        self._init_database()
        
        # Initialize FAISS index
        self.faiss_index = None
        self._init_faiss_index()
        
        # Initialize BM25 (loaded on-demand)
        self.bm25 = None
        self.bm25_corpus = []
        
        logger.info(f"Vector store initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    start_char INTEGER,
                    end_char INTEGER,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON chunks(chunk_id)")
            
            conn.commit()
        
        logger.debug("Database schema initialized")
    
    def _init_faiss_index(self):
        """Initialize or load FAISS index."""
        faiss_path = Path(self.db_path).parent / "faiss.index"
        
        if faiss_path.exists():
            # Load existing index
            self.faiss_index = faiss.read_index(str(faiss_path))
            logger.info(f"Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
        else:
            # Create new index (using L2 distance)
            self.faiss_index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"Created new FAISS index (dimension={self.dimension})")
    
    def _save_faiss_index(self):
        """Save FAISS index to disk."""
        faiss_path = Path(self.db_path).parent / "faiss.index"
        faiss.write_index(self.faiss_index, str(faiss_path))
        logger.debug(f"FAISS index saved to {faiss_path}")
    
    def _init_bm25(self):
        """Initialize BM25 index from database."""
        if self.bm25 is not None:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM chunks ORDER BY id")
            rows = cursor.fetchall()
        
        if not rows:
            logger.warning("No documents in database for BM25 initialization")
            self.bm25_corpus = []
            self.bm25 = None
            return
        
        # Tokenize corpus for BM25
        self.bm25_corpus = [row[0].lower().split() for row in rows]
        self.bm25 = BM25Okapi(self.bm25_corpus)
        
        logger.info(f"BM25 index initialized with {len(self.bm25_corpus)} documents")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.astype('float32')
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            Array of embedding vectors
        """
        batch_size = settings.embeddings.batch_size
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.astype('float32')
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        chunks: List[Chunk],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a document and its chunks to the vector store.
        
        Args:
            doc_id: Unique document identifier
            content: Full document content
            chunks: List of document chunks
            metadata: Optional metadata
        
        Returns:
            True if successful
        """
        try:
            # Generate embeddings for all chunks
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = self.embed_batch(chunk_texts)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert document
                cursor.execute(
                    "INSERT OR REPLACE INTO documents (doc_id, content, metadata) VALUES (?, ?, ?)",
                    (doc_id, content, json.dumps(metadata or {}))
                )
                
                # Insert chunks
                for chunk, embedding in zip(chunks, embeddings):
                    cursor.execute(
                        """
                        INSERT INTO chunks 
                        (doc_id, chunk_id, content, embedding, start_char, end_char, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc_id,
                            chunk.chunk_id,
                            chunk.text,
                            pickle.dumps(embedding),
                            chunk.start_char,
                            chunk.end_char,
                            json.dumps(chunk.metadata)
                        )
                    )
                
                conn.commit()
            
            # Add embeddings to FAISS index
            self.faiss_index.add(embeddings)
            self._save_faiss_index()
            
            # Reset BM25 (will be reinitialized on next search)
            self.bm25 = None
            
            logger.info(f"Added document '{doc_id}' with {len(chunks)} chunks")
            return True
        
        except Exception as e:
            logger.error(f"Error adding document '{doc_id}': {e}")
            return False
    
    def search_vector(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Similarity threshold (0-1, lower is more similar for L2)
        
        Returns:
            List of search results with scores
        """
        top_k = top_k or settings.vector_store.top_k
        threshold = threshold or settings.vector_store.similarity_threshold
        
        # Generate query embedding
        query_embedding = self.embed_text(query).reshape(1, -1)
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, top_k * 2)
        
        # Get chunks from database
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                # Convert L2 distance to similarity score (0-1, higher is better)
                similarity = 1 / (1 + distance)
                
                # Skip if below threshold
                if similarity < threshold:
                    continue
                
                # Get chunk from database (idx corresponds to row number)
                cursor.execute(
                    "SELECT doc_id, chunk_id, content, metadata FROM chunks ORDER BY id LIMIT 1 OFFSET ?",
                    (int(idx),)
                )
                row = cursor.fetchone()
                
                if row:
                    results.append({
                        "doc_id": row[0],
                        "chunk_id": row[1],
                        "content": row[2],
                        "metadata": json.loads(row[3]),
                        "score": float(similarity),
                        "search_type": "vector"
                    })
                
                if len(results) >= top_k:
                    break
        
        logger.debug(f"Vector search returned {len(results)} results")
        return results
    
    def search_bm25(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Lexical search using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of search results with scores
        """
        top_k = top_k or settings.vector_store.top_k
        
        # Initialize BM25 if needed
        if self.bm25 is None:
            self._init_bm25()
        
        if self.bm25 is None or not self.bm25_corpus:
            logger.warning("BM25 index is empty")
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Get chunks from database
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for idx in top_indices:
                score = scores[idx]
                
                if score <= 0:
                    continue
                
                cursor.execute(
                    "SELECT doc_id, chunk_id, content, metadata FROM chunks ORDER BY id LIMIT 1 OFFSET ?",
                    (int(idx),)
                )
                row = cursor.fetchone()
                
                if row:
                    results.append({
                        "doc_id": row[0],
                        "chunk_id": row[1],
                        "content": row[2],
                        "metadata": json.loads(row[3]),
                        "score": float(score),
                        "search_type": "bm25"
                    })
        
        logger.debug(f"BM25 search returned {len(results)} results")
        return results
    
    def search_hybrid(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and BM25 results.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for vector search (0.0 = all BM25, 1.0 = all vector)
        
        Returns:
            List of search results with combined scores
        """
        top_k = top_k or settings.vector_store.top_k
        alpha = alpha if alpha is not None else self.hybrid_alpha
        
        # Get results from both methods
        vector_results = self.search_vector(query, top_k=top_k * 2)
        bm25_results = self.search_bm25(query, top_k=top_k * 2)
        
        # Normalize scores to 0-1 range
        if vector_results:
            max_vector_score = max(r["score"] for r in vector_results)
            for r in vector_results:
                r["normalized_score"] = r["score"] / max_vector_score if max_vector_score > 0 else 0
        
        if bm25_results:
            max_bm25_score = max(r["score"] for r in bm25_results)
            for r in bm25_results:
                r["normalized_score"] = r["score"] / max_bm25_score if max_bm25_score > 0 else 0
        
        # Combine results
        combined = {}
        
        for result in vector_results:
            key = (result["doc_id"], result["chunk_id"])
            combined[key] = {
                **result,
                "vector_score": result["normalized_score"],
                "bm25_score": 0.0,
                "search_type": "hybrid"
            }
        
        for result in bm25_results:
            key = (result["doc_id"], result["chunk_id"])
            if key in combined:
                combined[key]["bm25_score"] = result["normalized_score"]
            else:
                combined[key] = {
                    **result,
                    "vector_score": 0.0,
                    "bm25_score": result["normalized_score"],
                    "search_type": "hybrid"
                }
        
        # Calculate hybrid scores
        for result in combined.values():
            result["score"] = (
                alpha * result["vector_score"] + 
                (1 - alpha) * result["bm25_score"]
            )
        
        # Sort by hybrid score and return top-k
        results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        
        logger.debug(f"Hybrid search returned {len(results)} results")
        return results
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_hybrid: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the vector store.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_hybrid: Use hybrid search (overrides default)
        
        Returns:
            List of search results
        """
        use_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid
        
        if use_hybrid:
            return self.search_hybrid(query, top_k=top_k)
        else:
            return self.search_vector(query, top_k=top_k)
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and its chunks.
        Note: This requires rebuilding the FAISS index.
        
        Args:
            doc_id: Document ID to delete
        
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                conn.commit()
            
            # Rebuild FAISS index (expensive operation)
            logger.warning("Rebuilding FAISS index after deletion...")
            self._rebuild_faiss_index()
            
            # Reset BM25
            self.bm25 = None
            
            logger.info(f"Deleted document '{doc_id}'")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting document '{doc_id}': {e}")
            return False
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from all embeddings in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM chunks ORDER BY id")
            rows = cursor.fetchall()
        
        if not rows:
            self.faiss_index = faiss.IndexFlatL2(self.dimension)
            return
        
        # Load all embeddings
        embeddings = np.array([pickle.loads(row[0]) for row in rows])
        
        # Create new index
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.faiss_index.add(embeddings)
        self._save_faiss_index()
        
        logger.info(f"Rebuilt FAISS index with {len(embeddings)} vectors")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            num_docs = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM chunks")
            num_chunks = cursor.fetchone()[0]
        
        return {
            "num_documents": num_docs,
            "num_chunks": num_chunks,
            "faiss_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
            "embedding_dimension": self.dimension,
            "hybrid_search_enabled": self.use_hybrid,
            "hybrid_alpha": self.hybrid_alpha
        }


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create singleton vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store