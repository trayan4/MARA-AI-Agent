"""
Critic Agent - Verifies outputs, detects hallucinations, checks consistency.
Acts as quality control for the MARA system.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

from loguru import logger

from config import settings
from tools.openai_client import get_openai_client


@dataclass
class CriticResult:
    """Represents a critic verification result."""
    
    passed: bool
    confidence: float
    issues: List[str]
    suggestions: List[str]
    verification_details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'confidence': self.confidence,
            'issues': self.issues,
            'suggestions': self.suggestions,
            'verification_details': self.verification_details
        }


class CriticAgent:
    """
    Quality control agent that verifies outputs from other agents.
    Performs hallucination detection, consistency checks, and citation validation.
    """
    
    def __init__(self):
        """Initialize Critic Agent."""
        self.client = get_openai_client()
        
        self.confidence_threshold = settings.agents.critic.confidence_threshold
        self.hallucination_check = settings.agents.critic.hallucination_check
        self.consistency_check = settings.agents.critic.consistency_check
        self.citation_validation = settings.agents.critic.citation_validation
        self.trigger_replan_threshold = settings.agents.critic.trigger_replan_threshold
        
        logger.info(
            f"Critic Agent initialized (confidence_threshold={self.confidence_threshold}, "
            f"replan_threshold={self.trigger_replan_threshold})"
        )
    
    def verify_output(
        self,
        agent_type: str,
        output: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> CriticResult:
        """
        Verify output from an agent.
        
        Args:
            agent_type: Type of agent (rag, vision, data, report)
            output: Agent output to verify
            context: Optional context (query, retrieved data, etc.)
        
        Returns:
            CriticResult with verification details
        """
        logger.info(f"Verifying {agent_type} agent output")
        
        issues = []
        suggestions = []
        verification_details = {}
        
        # Perform different checks based on agent type
        if agent_type == "rag":
            rag_result = self._verify_rag_output(output, context)
            issues.extend(rag_result['issues'])
            suggestions.extend(rag_result['suggestions'])
            verification_details.update(rag_result['details'])
        
        elif agent_type == "vision":
            vision_result = self._verify_vision_output(output, context)
            issues.extend(vision_result['issues'])
            suggestions.extend(vision_result['suggestions'])
            verification_details.update(vision_result['details'])
        
        elif agent_type == "data":
            data_result = self._verify_data_output(output, context)
            issues.extend(data_result['issues'])
            suggestions.extend(data_result['suggestions'])
            verification_details.update(data_result['details'])
        
        # General checks applicable to all
        general_result = self._verify_general(output, context)
        issues.extend(general_result['issues'])
        suggestions.extend(general_result['suggestions'])
        verification_details.update(general_result['details'])
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(output, issues)
        
        # Determine if verification passed
        passed = (
            confidence >= self.confidence_threshold and
            len(issues) == 0
        )
        
        result = CriticResult(
            passed=passed,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions,
            verification_details=verification_details
        )
        
        logger.info(
            f"Verification {'passed' if passed else 'failed'} "
            f"(confidence: {confidence:.2f}, issues: {len(issues)})"
        )
        
        return result
    
    def _verify_rag_output(
        self,
        output: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify RAG agent output."""
        issues = []
        suggestions = []
        details = {}
        
        # Check if answer is grounded in retrieved chunks
        if self.hallucination_check and 'answer' in output and 'chunks' in output:
            answer = output['answer']
            chunks = output.get('chunks', [])
            
            if chunks:
                grounding_check = self._check_grounding(answer, chunks)
                details['grounding_score'] = grounding_check['score']
                
                if grounding_check['score'] < 0.7:
                    issues.append("Answer may not be fully grounded in retrieved context")
                    suggestions.append("Verify claims against source documents")
            else:
                issues.append("No chunks retrieved to support answer")
                suggestions.append("Increase retrieval top_k or adjust query")
        
        # Check citation validity
        if self.citation_validation and 'answer' in output:
            citation_check = self._validate_citations(
                output['answer'],
                output.get('chunks', [])
            )
            details['citation_validity'] = citation_check['valid']
            
            if not citation_check['valid']:
                issues.append(citation_check['reason'])
                suggestions.append("Add proper source citations")
        
        # Check for conflicting information in chunks
        if self.consistency_check and len(output.get('chunks', [])) > 1:
            consistency = self._check_chunk_consistency(output['chunks'])
            details['chunk_consistency'] = consistency['score']
            
            if consistency['score'] < 0.8:
                issues.append("Retrieved chunks may contain conflicting information")
                suggestions.append("Review sources for consistency")
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'details': details
        }
    
    def _verify_vision_output(
        self,
        output: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify Vision agent output."""
        issues = []
        suggestions = []
        details = {}
        
        # Check for uncertainty markers
        if 'analysis' in output:
            analysis = output['analysis']
            uncertainty_markers = [
                'might be', 'possibly', 'unclear', 'difficult to see',
                'cannot determine', 'appears to be', 'seems like'
            ]
            
            uncertainty_count = sum(
                1 for marker in uncertainty_markers
                if marker.lower() in analysis.lower()
            )
            
            details['uncertainty_markers'] = uncertainty_count
            
            if uncertainty_count > 3:
                issues.append("High uncertainty in vision analysis")
                suggestions.append("Request higher detail level or better quality image")
        
        # Check if analysis is too short (likely insufficient detail)
        if 'analysis' in output and len(output['analysis']) < 50:
            issues.append("Vision analysis is very brief")
            suggestions.append("Request more detailed analysis")
        
        # Verify extracted data structure (if chart extraction)
        if output.get('extracted_data'):
            if not isinstance(output['extracted_data'], dict):
                issues.append("Extracted data format is invalid")
                suggestions.append("Verify data extraction logic")
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'details': details
        }
    
    def _verify_data_output(
        self,
        output: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify Data agent output."""
        issues = []
        suggestions = []
        details = {}
        
        # Check for NaN or infinite values in numerical results
        if 'result' in output:
            result = output['result']
            if isinstance(result, dict):
                # Check for invalid numerical values
                invalid_values = self._check_invalid_numbers(result)
                details['invalid_values'] = invalid_values
                
                if invalid_values > 0:
                    issues.append(f"Found {invalid_values} invalid numerical values (NaN/Inf)")
                    suggestions.append("Handle missing data or extreme values")
        
        # Verify insights are non-empty
        if 'insights' in output:
            if not output['insights'] or len(output['insights']) == 0:
                issues.append("No insights generated from data analysis")
                suggestions.append("Add interpretations of statistical results")
        
        # Check if operation succeeded
        if 'metadata' in output and 'error' in output['metadata']:
            issues.append(f"Data operation error: {output['metadata']['error']}")
            suggestions.append("Review data format and operation parameters")
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'details': details
        }
    
    def _verify_general(
        self,
        output: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """General verification checks applicable to all agents."""
        issues = []
        suggestions = []
        details = {}
        
        # Check confidence score if present
        if 'confidence' in output:
            confidence = output['confidence']
            details['agent_confidence'] = confidence
            
            if confidence < self.confidence_threshold:
                issues.append(f"Low confidence score: {confidence:.2f}")
                suggestions.append("Consider alternative approaches or gather more data")
        
        # Check for empty or None results
        if 'result' in output and output['result'] is None:
            issues.append("Result is None/empty")
            suggestions.append("Verify agent execution completed successfully")
        
        # Check metadata completeness
        if 'metadata' not in output:
            suggestions.append("Add metadata for better traceability")
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'details': details
        }
    
    def _check_grounding(
        self,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check if answer is grounded in retrieved chunks using LLM.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
        
        Returns:
            Grounding check result with score
        """
        # Build context from chunks
        context = "\n\n".join([chunk['content'] for chunk in chunks[:5]])
        
        prompt = f"""You are a fact-checking assistant. Verify if the answer is grounded in the provided context.

**Context:**
{context}

**Answer to verify:**
{answer}

**Task:**
Check if each claim in the answer can be found in or reasonably inferred from the context.

Respond in JSON format:
{{
    "grounded": true/false,
    "score": 0.0-1.0,
    "unsupported_claims": ["claim1", "claim2"],
    "reasoning": "brief explanation"
}}"""
        
        messages = [
            {"role": "system", "content": "You are a precise fact-checker. Respond only in JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=500
            )
            
            content = self.client.extract_content(response)
            
            # Parse JSON response
            import json
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            
            result = json.loads(content.strip())
            
            return {
                'score': result.get('score', 0.5),
                'grounded': result.get('grounded', False),
                'unsupported_claims': result.get('unsupported_claims', []),
                'reasoning': result.get('reasoning', '')
            }
        
        except Exception as e:
            logger.error(f"Grounding check failed: {e}")
            return {'score': 0.5, 'grounded': True, 'unsupported_claims': [], 'reasoning': ''}
    
    def _validate_citations(
        self,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that citations in answer match available sources.
        
        Args:
            answer: Answer text with citations
            chunks: Retrieved chunks
        
        Returns:
            Validation result
        """
        # Find citation patterns like [Source 1] or [1]
        citation_pattern = r'\[(?:Source\s+)?(\d+)\]'
        citations = re.findall(citation_pattern, answer)
        
        if not citations:
            # No citations found - this might be an issue for factual queries
            if len(chunks) > 0:
                return {
                    'valid': False,
                    'reason': 'No source citations found in answer'
                }
            else:
                return {'valid': True, 'reason': 'No chunks available to cite'}
        
        # Check if cited sources exist
        max_citation = max([int(c) for c in citations])
        if max_citation > len(chunks):
            return {
                'valid': False,
                'reason': f'Citation [Source {max_citation}] exceeds available sources ({len(chunks)})'
            }
        
        return {'valid': True, 'reason': 'All citations valid'}
    
    def _check_chunk_consistency(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check if retrieved chunks are consistent with each other.
        
        Args:
            chunks: List of retrieved chunks
        
        Returns:
            Consistency check result
        """
        # For now, use a simple heuristic
        # In production, could use LLM to detect contradictions
        
        # Check if chunks are from same document (likely more consistent)
        doc_ids = [chunk['doc_id'] for chunk in chunks]
        unique_docs = len(set(doc_ids))
        
        # More unique documents = potentially more inconsistency
        consistency_score = max(0.5, 1.0 - (unique_docs - 1) * 0.1)
        
        return {
            'score': consistency_score,
            'unique_documents': unique_docs,
            'total_chunks': len(chunks)
        }
    
    def _check_invalid_numbers(self, data: Any) -> int:
        """
        Recursively check for NaN/Inf in data structure.
        
        Args:
            data: Data to check
        
        Returns:
            Count of invalid numbers
        """
        import math
        
        invalid_count = 0
        
        if isinstance(data, dict):
            for value in data.values():
                invalid_count += self._check_invalid_numbers(value)
        elif isinstance(data, list):
            for item in data:
                invalid_count += self._check_invalid_numbers(item)
        elif isinstance(data, (int, float)):
            if math.isnan(data) or math.isinf(data):
                invalid_count += 1
        
        return invalid_count
    
    def _calculate_confidence(
        self,
        output: Dict[str, Any],
        issues: List[str]
    ) -> float:
        """
        Calculate overall confidence score.
        
        Args:
            output: Agent output
            issues: List of detected issues
        
        Returns:
            Confidence score (0-1)
        """
        # Start with agent's own confidence if available
        base_confidence = output.get('confidence', 0.7)
        
        # Reduce confidence for each issue
        confidence = base_confidence - (len(issues) * 0.1)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def should_trigger_replan(self, critic_result: CriticResult) -> bool:
        """
        Determine if verification failure should trigger replanning.
        
        Args:
            critic_result: Critic verification result
        
        Returns:
            True if should replan
        """
        should_replan = (
            not critic_result.passed and
            critic_result.confidence < self.trigger_replan_threshold
        )
        
        if should_replan:
            logger.warning(
                f"Triggering replan due to low confidence: {critic_result.confidence:.2f}"
            )
        
        return should_replan
    
    def generate_verification_report(
        self,
        results: Dict[str, CriticResult]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive verification report for all agents.
        
        Args:
            results: Dict of {agent_type: CriticResult}
        
        Returns:
            Verification report
        """
        total_issues = sum(len(r.issues) for r in results.values())
        avg_confidence = sum(r.confidence for r in results.values()) / len(results)
        all_passed = all(r.passed for r in results.values())
        
        report = {
            'overall_passed': all_passed,
            'average_confidence': avg_confidence,
            'total_issues': total_issues,
            'agent_results': {
                agent: result.to_dict()
                for agent, result in results.items()
            },
            'recommendation': 'APPROVE' if all_passed else 'REVIEW_REQUIRED'
        }
        
        return report


# Singleton instance
_critic_agent: Optional[CriticAgent] = None


def get_critic_agent() -> CriticAgent:
    """Get or create singleton Critic Agent instance."""
    global _critic_agent
    if _critic_agent is None:
        _critic_agent = CriticAgent()
    return _critic_agent


# Example usage
if __name__ == "__main__":
    agent = CriticAgent()
    
    # Example: Verify RAG output
    rag_output = {
        'query': 'What is AI?',
        'answer': 'AI is artificial intelligence. [Source 1]',
        'chunks': [
            {'doc_id': 'doc1', 'content': 'Artificial intelligence (AI) is...'},
        ],
        'confidence': 0.9
    }
    
    result = agent.verify_output('rag', rag_output)
    print(f"Verification passed: {result.passed}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Issues: {result.issues}")
    print(f"Suggestions: {result.suggestions}")