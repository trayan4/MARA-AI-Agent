"""
Web Search Agent - Retrieves real-time information from the internet.
Useful for current events, recent data, and information beyond the knowledge cutoff.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import requests
from datetime import datetime

from loguru import logger

from config import settings
from tools.openai_client import get_openai_client


@dataclass
class SearchResult:
    """Represents a web search result."""
    
    query: str
    results: List[Dict[str, Any]]
    summary: str
    sources: List[str]
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'results': self.results,
            'summary': self.summary,
            'sources': self.sources,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class WebSearchAgent:
    """
    Web search agent for retrieving real-time information.
    Uses search APIs to find current information beyond training data.
    """
    
    def __init__(self, search_api: str = "duckduckgo"):
        """
        Initialize Web Search Agent.
        
        Args:
            search_api: Search provider ('duckduckgo', 'serper', 'brave')
        """
        self.client = get_openai_client()
        self.search_api = search_api
        
        # API keys (optional, depending on provider)
        self.serper_api_key = None  # Set if using Serper
        self.brave_api_key = None   # Set if using Brave
        
        logger.info(f"Web Search Agent initialized (provider={search_api})")
    
    def search(
        self,
        query: str,
        num_results: int = 5,
        time_range: Optional[str] = None
    ) -> SearchResult:
        """
        Search the web for information.
        
        Args:
            query: Search query
            num_results: Number of results to return
            time_range: Time filter ('day', 'week', 'month', 'year', None)
        
        Returns:
            SearchResult with findings
        """
        logger.info(f"Searching web for: '{query}'")
        
        # Perform search based on provider
        if self.search_api == "duckduckgo":
            results = self._search_duckduckgo(query, num_results)
        elif self.search_api == "serper":
            results = self._search_serper(query, num_results)
        elif self.search_api == "brave":
            results = self._search_brave(query, num_results)
        else:
            raise ValueError(f"Unknown search provider: {self.search_api}")
        
        if not results:
            logger.warning("No search results found")
            return SearchResult(
                query=query,
                results=[],
                summary="No relevant results found.",
                sources=[],
                confidence=0.0,
                metadata={'provider': self.search_api, 'result_count': 0}
            )
        
        # Generate summary from results
        summary = self._generate_summary(query, results)
        
        # Extract sources
        sources = [r.get('url', r.get('link', '')) for r in results]
        
        # Estimate confidence
        confidence = self._estimate_confidence(results)
        
        search_result = SearchResult(
            query=query,
            results=results,
            summary=summary,
            sources=sources,
            confidence=confidence,
            metadata={
                'provider': self.search_api,
                'result_count': len(results),
                'timestamp': datetime.now().isoformat(),
                'time_range': time_range
            }
        )
        
        logger.info(f"Found {len(results)} results (confidence: {confidence:.2f})")
        return search_result
    
    def _search_duckduckgo(
        self,
        query: str,
        num_results: int
    ) -> List[Dict[str, Any]]:
        """
        Search using DuckDuckGo (free, no API key needed).
        
        Args:
            query: Search query
            num_results: Number of results
        
        Returns:
            List of search results
        """
        try:
            # from duckduckgo_search import DDGS
            from ddgs import DDGS
            
            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=num_results)
                
                for result in search_results:
                    results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('body', ''),
                        'url': result.get('href', ''),
                        'source': 'duckduckgo'
                    })
            
            return results
        
        except ImportError:
            logger.error("duckduckgo_search not installed. Run: pip install duckduckgo-search")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    def _search_serper(
        self,
        query: str,
        num_results: int
    ) -> List[Dict[str, Any]]:
        """
        Search using Serper API (requires API key).
        
        Args:
            query: Search query
            num_results: Number of results
        
        Returns:
            List of search results
        """
        if not self.serper_api_key:
            logger.error("Serper API key not set")
            return []
        
        try:
            url = "https://google.serper.dev/search"
            
            payload = {
                "q": query,
                "num": num_results
            }
            
            headers = {
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            for item in data.get('organic', [])[:num_results]:
                results.append({
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'url': item.get('link', ''),
                    'source': 'serper'
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Serper search failed: {e}")
            return []
    
    def _search_brave(
        self,
        query: str,
        num_results: int
    ) -> List[Dict[str, Any]]:
        """
        Search using Brave Search API (requires API key).
        
        Args:
            query: Search query
            num_results: Number of results
        
        Returns:
            List of search results
        """
        if not self.brave_api_key:
            logger.error("Brave API key not set")
            return []
        
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            
            params = {
                "q": query,
                "count": num_results
            }
            
            headers = {
                "X-Subscription-Token": self.brave_api_key,
                "Accept": "application/json"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            for item in data.get('web', {}).get('results', [])[:num_results]:
                results.append({
                    'title': item.get('title', ''),
                    'snippet': item.get('description', ''),
                    'url': item.get('url', ''),
                    'source': 'brave'
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Brave search failed: {e}")
            return []
    
    def _generate_summary(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate summary from search results using LLM.
        
        Args:
            query: Original query
            results: Search results
        
        Returns:
            Summary text
        """
        # Compile snippets
        snippets = []
        for i, result in enumerate(results[:5], 1):
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            snippets.append(f"{i}. {title}\n   {snippet}")
        
        snippets_text = "\n\n".join(snippets)
        
        prompt = f"""Based on these search results, provide a comprehensive answer to the query.

**Query:** {query}

**Search Results:**
{snippets_text}

**Instructions:**
1. Synthesize information from multiple sources
2. Be factual and cite sources when relevant
3. If results conflict, mention the discrepancy
4. Keep answer concise (3-5 sentences)
5. Focus on answering the query directly

Provide your answer:"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a web research assistant. Synthesize search results into clear, accurate answers."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            summary = self.client.extract_content(response)
            return summary
        
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Fallback: return first snippet
            if results:
                return results[0].get('snippet', 'No summary available.')
            return "No summary available."
    
    def _estimate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """
        Estimate confidence in search results.
        
        Args:
            results: Search results
        
        Returns:
            Confidence score (0-1)
        """
        if not results:
            return 0.0
        
        confidence = 0.6  # Base confidence for web search
        
        # Boost if multiple results
        if len(results) >= 3:
            confidence += 0.1
        
        # Boost if results have substantive snippets
        avg_snippet_length = sum(len(r.get('snippet', '')) for r in results) / len(results)
        if avg_snippet_length > 100:
            confidence += 0.1
        
        # Boost if titles are relevant (simple heuristic)
        # Check if query terms appear in titles
        # This is simplified - could use more sophisticated relevance scoring
        
        return min(1.0, confidence)
    
    def answer_with_sources(
        self,
        query: str,
        num_results: int = 5
    ) -> Dict[str, Any]:
        """
        Answer query with cited sources.
        
        Args:
            query: Question to answer
            num_results: Number of sources to use
        
        Returns:
            Answer with source citations
        """
        # Search
        search_result = self.search(query, num_results=num_results)
        
        if not search_result.results:
            return {
                'answer': "I couldn't find relevant information on the web.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Build cited answer
        answer_parts = [search_result.summary, "\n\n**Sources:**"]
        
        for i, result in enumerate(search_result.results, 1):
            title = result.get('title', 'Unknown')
            url = result.get('url', '')
            answer_parts.append(f"{i}. [{title}]({url})")
        
        answer = "\n".join(answer_parts)
        
        return {
            'answer': answer,
            'sources': search_result.sources,
            'confidence': search_result.confidence,
            'raw_results': search_result.results
        }


# Singleton instance
_web_search_agent: Optional[WebSearchAgent] = None


def get_web_search_agent(provider: str = "duckduckgo") -> WebSearchAgent:
    """Get or create singleton Web Search Agent instance."""
    global _web_search_agent
    if _web_search_agent is None:
        _web_search_agent = WebSearchAgent(search_api=provider)
    return _web_search_agent


# Example usage
if __name__ == "__main__":
    agent = WebSearchAgent(search_api="duckduckgo")
    
    # Test search
    result = agent.search("latest AI developments 2024", num_results=3)
    
    print(f"Query: {result.query}")
    print(f"\nSummary: {result.summary}")
    print(f"\nSources:")
    for i, source in enumerate(result.sources, 1):
        print(f"{i}. {source}")
    print(f"\nConfidence: {result.confidence:.2f}")