"""
RAG Agent - Retrieval Augmented Generation for document-based question answering.
Retrieves relevant chunks from vector store and generates grounded responses.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
from loguru import logger
import sqlite3
from config import settings
from tools.openai_client import get_openai_client
from tools.local_vector_store import get_vector_store
from tools.chunking import chunk_text, Chunk


@dataclass
class RetrievalResult:
    """Represents a retrieval result with context."""
    
    query: str
    chunks: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'chunks': self.chunks,
            'answer': self.answer,
            'sources': self.sources,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class RAGAgent:
    """
    Retrieval Augmented Generation agent.
    Retrieves relevant document chunks and generates grounded answers.
    """
    
    def __init__(self):
        """Initialize RAG Agent."""
        self.client = get_openai_client()
        self.vector_store = get_vector_store()
        
        self.max_chunks = settings.agents.rag.max_chunks
        self.use_rerank = settings.agents.rag.rerank
        self.max_retries = settings.agents.rag.max_retries
        self.timeout = settings.agents.rag.timeout
        
        logger.info(f"RAG Agent initialized (max_chunks={self.max_chunks}, rerank={self.use_rerank})")
    
    def search_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_hybrid: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_hybrid: Use hybrid search (vector + BM25)
            filters: Optional metadata filters
        
        Returns:
            List of relevant chunks with metadata
        """
        top_k = top_k or self.max_chunks
        
        logger.info(f"Searching documents for: '{query[:100]}...'")
        
        # Search vector store
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            use_hybrid=use_hybrid
        )
        
        # Apply metadata filters if provided
        if filters:
            results = [
                r for r in results
                if all(r['metadata'].get(k) == v for k, v in filters.items())
            ]
        
        # Rerank if enabled
        if self.use_rerank and len(results) > 1:
            results = self._rerank_results(query, results)
        
        logger.info(f"Retrieved {len(results)} relevant chunks")
        
        return results[:top_k]
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder for better relevance.
        
        Args:
            query: Original query
            results: Initial retrieval results
        
        Returns:
            Reranked results
        """
        try:
            from sentence_transformers import CrossEncoder
            
            # Load reranker model (cached after first load)
            reranker_model = settings.agents.rag.reranker_model
            reranker = CrossEncoder(reranker_model)
            
            # Create query-document pairs
            pairs = [(query, r['content']) for r in results]
            
            # Get reranking scores
            scores = reranker.predict(pairs)
            
            # Add rerank scores and sort
            for result, score in zip(results, scores):
                result['rerank_score'] = float(score)
            
            results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            logger.debug("Results reranked successfully")
            return results
        
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, using original order")
            return results
    
    def answer_question(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_hybrid: bool = True,
        include_sources: bool = True
    ) -> RetrievalResult:
        """
        Answer a question using retrieved documents.
        
        Args:
            query: Question to answer
            top_k: Number of chunks to retrieve
            use_hybrid: Use hybrid search
            include_sources: Include source citations in answer
        
        Returns:
            RetrievalResult with answer and sources
        """
        logger.info(f"Answering question: '{query[:100]}...'")
        
        # Retrieve relevant chunks
        chunks = self.search_documents(query, top_k=top_k, use_hybrid=use_hybrid)
        
        if not chunks:
            logger.warning("No relevant documents found")
            return RetrievalResult(
                query=query,
                chunks=[],
                answer="I couldn't find any relevant information in the knowledge base to answer this question.",
                sources=[],
                confidence=0.0,
                metadata={'retrieval_count': 0}
            )
        
        # Build context from chunks
        context = self._build_context(chunks)
        
        # Generate answer
        answer, confidence = self._generate_answer(query, context, chunks)
        
        # Extract sources
        sources = list(set([c['doc_id'] for c in chunks]))
        
        result = RetrievalResult(
            query=query,
            chunks=chunks,
            answer=answer,
            sources=sources,
            confidence=confidence,
            metadata={
                'retrieval_count': len(chunks),
                'unique_sources': len(sources),
                'search_type': chunks[0]['search_type'] if chunks else 'none'
            }
        )
        
        logger.info(f"Answer generated (confidence: {confidence:.2f})")
        return result
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks.
        
        Args:
            chunks: Retrieved chunks
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['doc_id']}]\n{chunk['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _generate_answer(
        self,
        query: str,
        context: str,
        chunks: List[Dict[str, Any]]
    ) -> tuple[str, float]:
        """
        Generate answer from context using LLM.
        
        Args:
            query: User question
            context: Retrieved context
            chunks: Original chunks (for citation)
        
        Returns:
            (answer, confidence_score)
        """
        # Build prompt
        prompt = f"""You are a helpful assistant that answers questions based on provided context.

**Context:**
{context}

**Question:**
{query}

**Instructions:**
1. Answer the question using ONLY information from the context above
2. If the context doesn't contain enough information, say so
3. Cite sources using [Source N] notation
4. Be concise but complete
5. If you're uncertain, indicate your confidence level

Provide your answer:"""

        messages = [
            {
                "role": "system",
                "content": "You are a precise question-answering assistant. Always ground your answers in the provided context and cite sources."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=0.3,  # Low temp for factual accuracy
                max_tokens=1000
            )
            
            answer = self.client.extract_content(response)
            
            # Estimate confidence based on answer characteristics
            confidence = self._estimate_confidence(answer, chunks)
            
            return answer, confidence
        
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}", 0.0
    
    def _estimate_confidence(
        self,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Estimate confidence in the answer.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
        
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence
        
        # Boost if answer cites sources
        if "[Source" in answer:
            confidence += 0.2
        
        # Boost based on retrieval scores
        if chunks:
            avg_score = sum(c['score'] for c in chunks) / len(chunks)
            confidence += avg_score * 0.2
        
        # Reduce if answer indicates uncertainty
        uncertainty_phrases = [
            "I'm not sure",
            "I don't know",
            "cannot be determined",
            "not enough information",
            "unclear"
        ]
        if any(phrase.lower() in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.3
        
        # Reduce if very short answer (likely insufficient context)
        if len(answer) < 50:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> bool:
        """
        Add a document to the knowledge base.
        
        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional metadata (author, date, etc.)
            chunk_size: Override default chunk size
            chunk_overlap: Override default overlap
        
        Returns:
            True if successful
        """
        logger.info(f"Adding document: {doc_id}")
        
        # Chunk the document
        chunks = chunk_text(
            content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=metadata
        )
        
        # Add to vector store
        success = self.vector_store.add_document(
            doc_id=doc_id,
            content=content,
            chunks=chunks,
            metadata=metadata
        )
        
        if success:
            logger.info(f"Successfully added document '{doc_id}' with {len(chunks)} chunks")
        else:
            logger.error(f"Failed to add document '{doc_id}'")
        
        return success
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the knowledge base.
        
        Args:
            doc_id: Document ID to delete
        
        Returns:
            True if successful
        """
        logger.info(f"Deleting document: {doc_id}")
        return self.vector_store.delete_document(doc_id)
    
    def list_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List documents in the knowledge base.
        
        Args:
            limit: Maximum number to return
        
        Returns:
            List of document metadata
        """
        
        with sqlite3.connect(self.vector_store.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT doc_id, metadata, created_at FROM documents ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
        
        import json
        documents = [
            {
                'doc_id': row[0],
                'metadata': json.loads(row[1]),
                'created_at': row[2]
            }
            for row in rows
        ]
        
        return documents
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full document by ID.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Document dict or None if not found
        """
        logger.info(f"Retrieving document: {doc_id}")
        
        # Query vector store for the document
        conn = sqlite3.connect(self.vector_store.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT content, metadata 
            FROM documents 
            WHERE doc_id = ?
        """, (doc_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            logger.warning(f"Document '{doc_id}' not found")
            return None
        
        content, metadata_json = row
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        return {
            'doc_id': doc_id,
            'content': content,
            'metadata': metadata
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG agent statistics."""
        vector_stats = self.vector_store.get_stats()
        
        return {
            **vector_stats,
            'max_chunks_per_query': self.max_chunks,
            'reranking_enabled': self.use_rerank
        }


# Singleton instance
_rag_agent: Optional[RAGAgent] = None


def get_rag_agent() -> RAGAgent:
    """Get or create singleton RAG Agent instance."""
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RAGAgent()
    return _rag_agent


# Example usage
if __name__ == "__main__":
    agent = RAGAgent()
    
    # Add a sample document
    sample_doc = """
    Artificial Intelligence (AI) has made remarkable progress in recent years.
    Large Language Models like GPT-4 can understand and generate human-like text.
    Computer vision systems can now identify objects with superhuman accuracy.
    AI is being applied in healthcare, finance, education, and many other domains.
    However, challenges remain including bias, interpretability, and safety.
    """
    
    agent.add_document(
        doc_id="ai_overview",
        content=sample_doc,
        metadata={"topic": "AI", "date": "2024-01-01"}
    )
    
    # Test question answering
    result = agent.answer_question("What are the challenges in AI?")
    
    print(f"\nQuestion: {result.query}")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Sources: {result.sources}")