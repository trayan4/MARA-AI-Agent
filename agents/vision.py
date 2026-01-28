"""
Vision Agent - Analyzes images, charts, screenshots, and visual content.
Uses GPT-4 Vision for multimodal understanding.
"""

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import mimetypes

from PIL import Image
from loguru import logger

from config import settings
from tools.openai_client import get_openai_client


@dataclass
class VisionResult:
    """Represents a vision analysis result."""
    
    image_path: str
    query: Optional[str]
    analysis: str
    extracted_data: Optional[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'image_path': self.image_path,
            'query': self.query,
            'analysis': self.analysis,
            'extracted_data': self.extracted_data,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class VisionAgent:
    """
    Vision analysis agent using GPT-4 Vision.
    Handles image analysis, chart extraction, OCR, and visual comparisons.
    """
    
    def __init__(self):
        """Initialize Vision Agent."""
        self.client = get_openai_client()
        
        self.max_image_size = settings.agents.vision.max_image_size
        self.supported_formats = settings.agents.vision.supported_formats
        self.detail_level = settings.agents.vision.detail_level
        self.max_retries = settings.agents.vision.max_retries
        self.timeout = settings.agents.vision.timeout
        
        logger.info(
            f"Vision Agent initialized (max_size={self.max_image_size}px, "
            f"detail={self.detail_level})"
        )
    
    def _validate_image(self, image_path: Union[str, Path]) -> tuple[bool, Optional[str]]:
        """
        Validate image file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            (is_valid, error_message)
        """
        image_path = Path(image_path)
        
        # Check if file exists
        if not image_path.exists():
            return False, f"Image file not found: {image_path}"
        
        # Check file format
        extension = image_path.suffix.lower().lstrip('.')
        if extension not in self.supported_formats:
            return False, f"Unsupported format: {extension}. Supported: {self.supported_formats}"
        
        # Check file size and dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                max_dimension = max(width, height)
                
                if max_dimension > self.max_image_size:
                    logger.warning(
                        f"Image {width}x{height} exceeds max size {self.max_image_size}. "
                        "Will resize."
                    )
        except Exception as e:
            return False, f"Failed to open image: {e}"
        
        return True, None
    
    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Encode image to base64.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Base64 encoded image string
        """
        image_path = Path(image_path)
        
        # Open and potentially resize
        with Image.open(image_path) as img:
            width, height = img.size
            max_dimension = max(width, height)
            
            if max_dimension > self.max_image_size:
                # Calculate new dimensions
                scale = self.max_image_size / max_dimension
                new_size = (int(width * scale), int(height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized image from {width}x{height} to {new_size}")
            
            # Encode to base64 (inside the with block!)
            import io
            buffer = io.BytesIO()
            img.save(buffer, format=img.format or 'PNG')
            buffer.seek(0)
            
            encoded = base64.b64encode(buffer.read()).decode('utf-8')
            return encoded
    
    def _get_mime_type(self, image_path: Union[str, Path]) -> str:
        """Get MIME type for image."""
        mime_type, _ = mimetypes.guess_type(str(image_path))
        return mime_type or 'image/png'
    
    def analyze_image(
        self,
        image_path: Union[str, Path],
        query: Optional[str] = None,
        detail_level: Optional[str] = None
    ) -> VisionResult:
        """
        Analyze an image with optional specific query.
        
        Args:
            image_path: Path to image file
            query: Optional specific question about the image
            detail_level: 'low', 'high', or 'auto'
        
        Returns:
            VisionResult with analysis
        """
        logger.info(f"Analyzing image: {image_path}")
        
        # Validate image
        is_valid, error = self._validate_image(image_path)
        if not is_valid:
            logger.error(f"Image validation failed: {error}")
            return VisionResult(
                image_path=str(image_path),
                query=query,
                analysis=f"Error: {error}",
                extracted_data=None,
                confidence=0.0,
                metadata={'error': error}
            )
        
        # Encode image
        try:
            image_base64 = self._encode_image(image_path)
            mime_type = self._get_mime_type(image_path)
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return VisionResult(
                image_path=str(image_path),
                query=query,
                analysis=f"Error encoding image: {e}",
                extracted_data=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
        
        # Build prompt
        detail = detail_level or self.detail_level
        
        if query:
            prompt = f"Please analyze this image and answer the following question:\n\n{query}"
        else:
            prompt = """Please provide a detailed analysis of this image including:
1. What you see (objects, people, text, etc.)
2. The context or setting
3. Any notable details or patterns
4. If it's a chart/graph: extract the data and insights
5. Overall purpose or message

Be specific and thorough."""
        
        # Create vision message
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}",
                            "detail": detail
                        }
                    }
                ]
            }
        ]
        
        try:
            # Use vision model
            response = self.client.chat_completion(
                messages=messages,
                model=self.client.vision_model,
                max_tokens=1500
            )
            
            analysis = self.client.extract_content(response)
            
            # Estimate confidence
            confidence = self._estimate_confidence(analysis)
            
            result = VisionResult(
                image_path=str(image_path),
                query=query,
                analysis=analysis,
                extracted_data=None,  # Will be populated by specialized methods
                confidence=confidence,
                metadata={
                    'detail_level': detail,
                    'model': self.client.vision_model
                }
            )
            
            logger.info(f"Image analysis complete (confidence: {confidence:.2f})")
            return result
        
        except Exception as e:
            logger.error(f"Vision API call failed: {e}")
            return VisionResult(
                image_path=str(image_path),
                query=query,
                analysis=f"Error during analysis: {e}",
                extracted_data=None,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def extract_chart_data(
        self,
        image_path: Union[str, Path],
        chart_type: str = "auto"
    ) -> VisionResult:
        """
        Extract numerical data from charts/graphs.
        
        Args:
            image_path: Path to chart image
            chart_type: Type of chart (bar, line, pie, scatter, auto)
        
        Returns:
            VisionResult with extracted data
        """
        logger.info(f"Extracting chart data from: {image_path}")
        
        prompt = f"""This is a {chart_type} chart. Please extract the data in a structured format.

Provide:
1. Chart title and axis labels
2. All data points as a list or table
3. Key insights (trends, outliers, patterns)
4. Data in JSON format if possible

Be precise with numbers."""
        
        result = self.analyze_image(image_path, query=prompt, detail_level="high")
        
        # Try to parse structured data from response
        extracted_data = self._parse_chart_data(result.analysis)
        result.extracted_data = extracted_data
        
        return result
    
    def _parse_chart_data(self, analysis: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse structured data from chart analysis.
        
        Args:
            analysis: Raw analysis text
        
        Returns:
            Parsed data dictionary or None
        """
        import json
        import re
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', analysis, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: return None (data is in analysis text)
        return None
    
    def ocr_extract_text(
        self,
        image_path: Union[str, Path],
        language: str = "eng"
    ) -> VisionResult:
        """
        Extract text from images using OCR.
        
        Args:
            image_path: Path to image with text
            language: Language code (default: eng)
        
        Returns:
            VisionResult with extracted text
        """
        logger.info(f"Extracting text from image: {image_path}")
        
        prompt = """Please extract all text visible in this image.

Provide:
1. All text content, preserving formatting when possible
2. Location of text (top, bottom, center, etc.)
3. Any relevant context (headings, captions, labels)

Output the text exactly as it appears."""
        
        result = self.analyze_image(image_path, query=prompt, detail_level="high")
        
        # Extract just the text content
        result.extracted_data = {'text': result.analysis}
        
        return result
    
    def compare_images(
        self,
        image_paths: List[Union[str, Path]],
        comparison_aspect: Optional[str] = None
    ) -> VisionResult:
        """
        Compare multiple images and identify similarities/differences.
        
        Args:
            image_paths: List of 2-5 image paths
            comparison_aspect: Specific aspect to compare
        
        Returns:
            VisionResult with comparison analysis
        """
        if len(image_paths) < 2:
            raise ValueError("Need at least 2 images to compare")
        if len(image_paths) > 5:
            raise ValueError("Can compare maximum 5 images at once")
        
        logger.info(f"Comparing {len(image_paths)} images")
        
        # Encode all images
        image_contents = []
        for img_path in image_paths:
            is_valid, error = self._validate_image(img_path)
            if not is_valid:
                logger.error(f"Invalid image {img_path}: {error}")
                continue
            
            try:
                image_base64 = self._encode_image(img_path)
                mime_type = self._get_mime_type(img_path)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}",
                        "detail": "high"
                    }
                })
            except Exception as e:
                logger.error(f"Failed to encode {img_path}: {e}")
        
        if len(image_contents) < 2:
            raise ValueError("Failed to load enough valid images for comparison")
        
        # Build comparison prompt
        if comparison_aspect:
            prompt = f"""Compare these images focusing on: {comparison_aspect}

Provide:
1. Similarities between the images
2. Key differences
3. Notable patterns or trends
4. Overall assessment"""
        else:
            prompt = """Compare these images and identify:
1. What's similar across all images
2. What's different between them
3. Any progression or pattern
4. Overall relationship between the images

Be specific and detailed."""
        
        # Create message with all images
        content = [{"type": "text", "text": prompt}] + image_contents
        
        messages = [{"role": "user", "content": content}]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                model=self.client.vision_model,
                max_tokens=2000
            )
            
            analysis = self.client.extract_content(response)
            confidence = self._estimate_confidence(analysis)
            
            result = VisionResult(
                image_path=f"{len(image_paths)} images",
                query=comparison_aspect,
                analysis=analysis,
                extracted_data={'image_count': len(image_paths)},
                confidence=confidence,
                metadata={
                    'images': [str(p) for p in image_paths],
                    'comparison_aspect': comparison_aspect
                }
            )
            
            logger.info(f"Image comparison complete (confidence: {confidence:.2f})")
            return result
        
        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            raise
    
    def _estimate_confidence(self, analysis: str) -> float:
        """
        Estimate confidence in vision analysis.
        
        Args:
            analysis: Generated analysis
        
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.7  # Base confidence for vision tasks
        
        # Reduce if analysis is very short
        if len(analysis) < 100:
            confidence -= 0.2
        
        # Reduce if uncertainty phrases detected
        uncertainty_phrases = [
            "unclear",
            "difficult to see",
            "cannot determine",
            "possibly",
            "might be",
            "appears to be",
            "seems like"
        ]
        
        uncertainty_count = sum(
            1 for phrase in uncertainty_phrases 
            if phrase.lower() in analysis.lower()
        )
        
        confidence -= uncertainty_count * 0.1
        
        # Boost if specific details provided
        if any(word in analysis.lower() for word in ['specifically', 'precisely', 'exactly']):
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))


# Singleton instance
_vision_agent: Optional[VisionAgent] = None


def get_vision_agent() -> VisionAgent:
    """Get or create singleton Vision Agent instance."""
    global _vision_agent
    if _vision_agent is None:
        _vision_agent = VisionAgent()
    return _vision_agent


# Example usage
if __name__ == "__main__":
    agent = VisionAgent()
    
    # Example: Analyze an image (you'd need an actual image file)
    # result = agent.analyze_image("path/to/image.jpg", query="What is shown in this image?")
    # print(f"Analysis: {result.analysis}")
    # print(f"Confidence: {result.confidence}")
    
    print("Vision Agent initialized successfully")