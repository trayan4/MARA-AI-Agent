"""
Report Agent - Synthesizes outputs from all agents into structured enterprise reports.
Generates JSON, Markdown, or HTML reports with evidence and confidence scores.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json

from loguru import logger

from config import settings
from tools.openai_client import get_openai_client


@dataclass
class Report:
    """Represents a complete MARA report."""
    
    query: str
    executive_summary: str
    visual_insights: Dict[str, Any]
    data_insights: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    recommendations: List[str]
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'executive_summary': self.executive_summary,
            'visual_insights': self.visual_insights,
            'data_insights': self.data_insights,
            'evidence': self.evidence,
            'recommendations': self.recommendations,
            'confidence': self.confidence,
            'metadata': self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export report as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_markdown(self) -> str:
        """Export report as Markdown."""
        md = f"""# MARA Analysis Report

**Query:** {self.query}

**Generated:** {self.metadata.get('timestamp', 'N/A')}

**Overall Confidence:** {self.confidence:.1%}

---

## Executive Summary

{self.executive_summary}

---

## Visual Insights

"""
        if self.visual_insights:
            for key, value in self.visual_insights.items():
                md += f"### {key.replace('_', ' ').title()}\n\n{value}\n\n"
        else:
            md += "*No visual insights available*\n\n"
        
        md += """---

## Data Insights

"""
        if self.data_insights:
            for key, value in self.data_insights.items():
                md += f"### {key.replace('_', ' ').title()}\n\n{value}\n\n"
        else:
            md += "*No data insights available*\n\n"
        
        md += """---

## Evidence

"""
        if self.evidence:
            for i, ev in enumerate(self.evidence, 1):
                source = ev.get('source', 'Unknown')
                content = ev.get('content', '')
                md += f"{i}. **Source:** {source}\n   - {content}\n\n"
        else:
            md += "*No evidence cited*\n\n"
        
        md += """---

## Recommendations

"""
        if self.recommendations:
            for i, rec in enumerate(self.recommendations, 1):
                md += f"{i}. {rec}\n"
        else:
            md += "*No recommendations*\n"
        
        md += f"""

---

## Metadata

- **Agents Used:** {', '.join(self.metadata.get('agents_used', []))}
- **Execution Time:** {self.metadata.get('execution_time', 'N/A')}s
- **Timestamp:** {self.metadata.get('timestamp', 'N/A')}
"""
        
        return md


class ReportAgent:
    """
    Report generation agent that synthesizes outputs from all agents.
    Creates structured enterprise reports with evidence and recommendations.
    """
    
    def __init__(self):
        """Initialize Report Agent."""
        self.client = get_openai_client()
        
        self.format = settings.agents.report.format
        self.include_evidence = settings.agents.report.include_evidence
        self.include_metadata = settings.agents.report.include_metadata
        
        logger.info(f"Report Agent initialized (format={self.format})")
    
    def generate_report(
        self,
        query: str,
        agent_outputs: Dict[str, Any],
        execution_metadata: Optional[Dict[str, Any]] = None
    ) -> Report:
        """
        Generate comprehensive report from agent outputs.
        
        Args:
            query: Original user query
            agent_outputs: Dict of {agent_type: agent_result}
            execution_metadata: Optional execution metadata (time, agents used, etc.)
        
        Returns:
            Complete Report object
        """
        logger.info(f"Generating report for query: '{query[:100]}...'")
        
        # Extract outputs from each agent
        rag_output = agent_outputs.get('rag')
        vision_output = agent_outputs.get('vision')
        data_output = agent_outputs.get('data')
        critic_output = agent_outputs.get('critic')
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            query, agent_outputs
        )
        
        # Extract insights
        visual_insights = self._extract_visual_insights(vision_output)
        data_insights = self._extract_data_insights(data_output)
        
        # Collect evidence
        evidence = []
        if self.include_evidence:
            evidence = self._collect_evidence(agent_outputs)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            query, agent_outputs
        )
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(agent_outputs, critic_output)
        
        # Build metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'agents_used': list(agent_outputs.keys()),
            'execution_time': execution_metadata.get('execution_time', 0) if execution_metadata else 0,
            'query_length': len(query),
            'evidence_count': len(evidence)
        }
        
        if self.include_metadata and execution_metadata:
            metadata.update(execution_metadata)
        
        report = Report(
            query=query,
            executive_summary=executive_summary,
            visual_insights=visual_insights,
            data_insights=data_insights,
            evidence=evidence,
            recommendations=recommendations,
            confidence=confidence,
            metadata=metadata
        )
        
        logger.info(f"Report generated (confidence: {confidence:.2f})")
        return report
    
    def _generate_executive_summary(
        self,
        query: str,
        agent_outputs: Dict[str, Any]
    ) -> str:
        """
        Generate executive summary using LLM.
        
        Args:
            query: Original query
            agent_outputs: All agent outputs
        
        Returns:
            Executive summary text
        """
        # Compile findings from all agents
        findings = []
        # DEBUG: Log what we're receiving
        logger.info(f"DEBUG: agent_outputs keys = {agent_outputs.keys()}")
        if 'rag' in agent_outputs:
            logger.info(f"DEBUG: rag = {agent_outputs['rag']}")
        
        if 'rag' in agent_outputs and agent_outputs['rag']:
            rag = agent_outputs['rag']
            if isinstance(rag, dict) and 'answer' in rag:
                # Use the RAG answer directly as the main finding
                findings.append(rag['answer'])
        
        if 'vision' in agent_outputs and agent_outputs['vision']:
            vision = agent_outputs['vision']
            if isinstance(vision, dict) and 'analysis' in vision:
                findings.append(f"**Visual Analysis:** {vision['analysis'][:200]}...")
        
        if 'data' in agent_outputs and agent_outputs['data']:
            data = agent_outputs['data']
            if isinstance(data, dict):
                summary = data.get('summary', '')
                insights = data.get('insights', [])
                if summary:
                    findings.append(f"**Data Analysis:** {summary}")
                if insights:
                    findings.append(f"**Key Insights:** {'; '.join(insights[:3])}")

        if 'web_search' in agent_outputs and agent_outputs['web_search']:
            web = agent_outputs['web_search']
            if isinstance(web, dict) and 'summary' in web:
                findings.append(web['summary'])
        
        if not findings:
            return "No significant findings from agent analysis."
        
        # If we have a RAG answer, just use that
        if 'rag' in agent_outputs and agent_outputs['rag'] and 'answer' in agent_outputs['rag']:
            print('test rag answer:', agent_outputs['rag']['answer'])
            return agent_outputs['rag']['answer']
        
        # Otherwise synthesize
        findings_text = "\n\n".join(findings)
        
        # Use LLM to synthesize executive summary
        prompt = f"""Based on the following findings, create a concise executive summary (2-3 sentences) that answers the user's query.

**User Query:** {query}

**Findings:**
{findings_text}

**Instructions:**
- Be concise and factual
- Focus on answering the query directly
- Highlight the most important insights
- Use business-appropriate language

Provide only the executive summary (no preamble):"""
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert at synthesizing technical findings into executive summaries."
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
                max_tokens=300
            )
            
            summary = self.client.extract_content(response).strip()
            return summary
        
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            # Fallback: use first finding
            return findings[0] if findings else "Analysis completed."
    
    def _extract_visual_insights(
        self,
        vision_output: Optional[Any]
    ) -> Dict[str, Any]:
        """Extract visual insights from vision agent output."""
        if not vision_output:
            return {}
        
        insights = {}
        
        if isinstance(vision_output, dict):
            if 'analysis' in vision_output:
                insights['analysis'] = vision_output['analysis']
            
            if 'extracted_data' in vision_output and vision_output['extracted_data']:
                insights['extracted_data'] = vision_output['extracted_data']
            
            if 'metadata' in vision_output:
                insights['image_details'] = vision_output['metadata']
        
        elif isinstance(vision_output, list):
            # Multiple vision results
            insights['analyses'] = [
                v.get('analysis', '') for v in vision_output if isinstance(v, dict)
            ]
        
        return insights
    
    def _extract_data_insights(
        self,
        data_output: Optional[Any]
    ) -> Dict[str, Any]:
        """Extract data insights from data agent output."""
        if not data_output:
            return {}
        
        insights = {}
        
        if isinstance(data_output, dict):
            # Add summary
            if 'summary' in data_output:
                insights['summary'] = data_output['summary']
            
            # Add insights
            if 'insights' in data_output:
                insights['key_findings'] = data_output['insights']
            
            # Add statistics if available
            if 'result' in data_output:
                result = data_output['result']
                if isinstance(result, dict):
                    # Sample key statistics
                    insights['statistics'] = {
                        k: v for k, v in list(result.items())[:5]
                    }
            
            # Add operation type
            if 'operation' in data_output:
                insights['analysis_type'] = data_output['operation']
        
        return insights
    
    def _collect_evidence(
        self,
        agent_outputs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Collect evidence from all agent outputs."""
        evidence = []
        
        # RAG evidence (retrieved chunks)
        if 'rag' in agent_outputs and agent_outputs['rag']:
            rag = agent_outputs['rag']
            if isinstance(rag, dict):
                # Check for chunks in the result
                chunks = rag.get('chunks', [])
                if not chunks:
                    # Maybe it's stored differently - check sources
                    sources = rag.get('sources', [])
                    if sources:
                        for source in sources[:5]:
                            evidence.append({
                                'source': source,
                                'content': '',
                                'score': 1.0,
                                'type': 'document'
                            })
                else:
                    for chunk in chunks[:5]:  # Top 5 chunks
                        evidence.append({
                            'source': chunk.get('doc_id', chunk.get('source', 'Unknown')),
                            'content': chunk.get('content', chunk.get('text', ''))[:200] + '...',
                            'score': chunk.get('score', chunk.get('confidence', 0)),
                            'type': 'document'
                        })
        
        # Vision evidence (image paths)
        if 'vision' in agent_outputs and agent_outputs['vision']:
            vision = agent_outputs['vision']
            if isinstance(vision, dict) and 'image_path' in vision:
                evidence.append({
                    'source': vision['image_path'],
                    'content': vision.get('analysis', '')[:200] + '...',
                    'score': vision.get('confidence', 0),
                    'type': 'image'
                })
        
        # Data evidence (data sources)
        if 'data' in agent_outputs and agent_outputs['data']:
            data = agent_outputs['data']
            if isinstance(data, dict) and 'data_path' in data:
                evidence.append({
                    'source': data['data_path'],
                    'content': data.get('summary', ''),
                    'score': data.get('confidence', 0),
                    'type': 'data'
                })
        
        return evidence
    
    def _generate_recommendations(
        self,
        query: str,
        agent_outputs: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Extract existing suggestions from agents
        for agent_type, output in agent_outputs.items():
            if isinstance(output, dict):
                # Check for suggestions/recommendations in output
                if 'suggestions' in output:
                    recommendations.extend(output['suggestions'])
                if 'recommendations' in output:
                    recommendations.extend(output['recommendations'])
        
        # If no recommendations found, generate using LLM
        if not recommendations:
            recommendations = self._generate_llm_recommendations(query, agent_outputs)
        
        # Limit to top 5 recommendations
        return recommendations[:5]
    
    def _generate_llm_recommendations(
        self,
        query: str,
        agent_outputs: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations using LLM."""
        # Build context from outputs
        context = []
        
        for agent_type, output in agent_outputs.items():
            if isinstance(output, dict):
                summary = output.get('summary', '')
                if summary:
                    context.append(f"{agent_type}: {summary}")
        
        if not context:
            return []
        
        context_text = "\n".join(context)
        
        prompt = f"""Based on the analysis results, provide 3-5 actionable recommendations.

**Original Query:** {query}

**Analysis Results:**
{context_text}

**Instructions:**
- Provide specific, actionable recommendations
- Focus on next steps or improvements
- Be concise (1 sentence per recommendation)
- Number each recommendation

Provide recommendations:"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a strategic advisor providing actionable recommendations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=0.4,
                max_tokens=400
            )
            
            content = self.client.extract_content(response)
            
            # Parse numbered recommendations
            import re
            recommendations = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\n*$)', content, re.DOTALL)
            recommendations = [r.strip() for r in recommendations]
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    def _calculate_overall_confidence(
        self,
        agent_outputs: Dict[str, Any],
        critic_output: Optional[Any]
    ) -> float:
        """Calculate overall confidence score."""
        confidences = []
        
        # Collect confidence scores from agents
        for output in agent_outputs.values():
            if isinstance(output, dict) and 'confidence' in output:
                confidences.append(output['confidence'])
        
        # Include critic confidence if available
        if critic_output and isinstance(critic_output, dict):
            if 'confidence' in critic_output:
                confidences.append(critic_output['confidence'])
        
        if not confidences:
            return 0.7  # Default moderate confidence
        
        # Use weighted average (critic gets more weight)
        if critic_output and isinstance(critic_output, dict) and 'confidence' in critic_output:
            # 60% critic, 40% agents
            agent_avg = sum(confidences[:-1]) / len(confidences[:-1]) if len(confidences) > 1 else 0.7
            critic_conf = confidences[-1]
            overall = 0.4 * agent_avg + 0.6 * critic_conf
        else:
            # Simple average
            overall = sum(confidences) / len(confidences)
        
        return round(overall, 2)
    
    def format_report(
        self,
        report: Report,
        output_format: Optional[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Format report in requested format.
        
        Args:
            report: Report object
            output_format: Output format (json, markdown, html)
        
        Returns:
            Formatted report
        """
        format_type = output_format or self.format
        
        if format_type == "json":
            return report.to_json()
        elif format_type == "markdown":
            return report.to_markdown()
        elif format_type == "html":
            return self._to_html(report)
        else:
            # Default to dict
            return report.to_dict()
    
    def _to_html(self, report: Report) -> str:
        """Convert report to HTML format."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>MARA Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .metadata {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .confidence {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
        .evidence {{ background: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; }}
        .recommendation {{ background: #e8f5e9; padding: 10px; margin: 5px 0; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>MARA Analysis Report</h1>
    
    <div class="metadata">
        <p><strong>Query:</strong> {report.query}</p>
        <p><strong>Generated:</strong> {report.metadata.get('timestamp', 'N/A')}</p>
        <p><strong>Confidence:</strong> <span class="confidence">{report.confidence:.1%}</span></p>
    </div>
    
    <h2>Executive Summary</h2>
    <p>{report.executive_summary}</p>
    
    <h2>Visual Insights</h2>
    {''.join([f'<h3>{k.replace("_", " ").title()}</h3><p>{v}</p>' for k, v in report.visual_insights.items()]) or '<p>No visual insights</p>'}
    
    <h2>Data Insights</h2>
    {''.join([f'<h3>{k.replace("_", " ").title()}</h3><p>{v}</p>' for k, v in report.data_insights.items()]) or '<p>No data insights</p>'}
    
    <h2>Evidence</h2>
    {''.join([f'<div class="evidence"><strong>{ev["source"]}</strong><br>{ev["content"]}</div>' for ev in report.evidence]) or '<p>No evidence</p>'}
    
    <h2>Recommendations</h2>
    {''.join([f'<div class="recommendation">{i}. {rec}</div>' for i, rec in enumerate(report.recommendations, 1)]) or '<p>No recommendations</p>'}
    
</body>
</html>"""
        
        return html


# Singleton instance
_report_agent: Optional[ReportAgent] = None


def get_report_agent() -> ReportAgent:
    """Get or create singleton Report Agent instance."""
    global _report_agent
    if _report_agent is None:
        _report_agent = ReportAgent()
    return _report_agent


# Example usage
if __name__ == "__main__":
    agent = ReportAgent()
    
    # Example agent outputs
    agent_outputs = {
        'rag': {
            'answer': 'AI has made significant progress in recent years.',
            'chunks': [{'doc_id': 'doc1', 'content': 'AI progress...', 'score': 0.9}],
            'confidence': 0.9
        },
        'data': {
            'summary': 'Analyzed 1000 data points',
            'insights': ['Growth rate: 15%', 'Peak in Q3'],
            'confidence': 0.85
        }
    }
    
    report = agent.generate_report(
        query="What are the AI trends?",
        agent_outputs=agent_outputs
    )
    
    print(report.to_markdown())