import hashlib
import time
import random
from typing import Dict, Any
from app.models import DetailedAnalysis

class MockAIServices:
    """Mock AI services for MVP - provides deterministic responses for demo"""
    
    @staticmethod
    def analyze_text(content: str) -> Dict[str, Any]:
        """Mock text analysis using content hash for deterministic results"""
        # Create deterministic hash-based analysis
        content_hash = hashlib.md5(content.lower().encode()).hexdigest()
        hash_int = int(content_hash[:8], 16)
        
        # Deterministic verdict based on content characteristics
        if any(keyword in content.lower() for keyword in ['fake', 'false', 'hoax', 'scam']):
            verdict = "false"
            confidence = 0.8 + (hash_int % 20) / 100  # 0.8-0.99
        elif any(keyword in content.lower() for keyword in ['verified', 'official', 'confirmed', 'reuters']):
            verdict = "true"
            confidence = 0.75 + (hash_int % 25) / 100  # 0.75-0.99
        else:
            verdict = "inconclusive"
            confidence = 0.4 + (hash_int % 40) / 100  # 0.4-0.79
        
        return {
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "summary": f"Text analysis completed. Content appears to be {verdict} based on language patterns and fact-checking.",
            "evidence": [
                "Language pattern analysis completed",
                "Cross-referenced with known misinformation patterns",
                "Checked against fact-checking databases"
            ]
        }

    @staticmethod
    def analyze_image(content: str) -> Dict[str, Any]:
        """Mock image analysis"""
        # Simulate processing time
        processing_delay = random.uniform(0.5, 2.0)
        time.sleep(processing_delay / 10)  # Reduced for MVP
        
        # Mock image analysis results
        return {
            "verdict": "inconclusive",
            "confidence": 0.65,
            "summary": "Image analysis completed. No clear signs of manipulation detected, but verification inconclusive.",
            "vision_analysis": "Image metadata analyzed, reverse image search performed, manipulation detection algorithms applied."
        }

    @staticmethod
    def analyze_url(content: str) -> Dict[str, Any]:
        """Mock URL analysis"""
        # Basic URL credibility assessment
        trusted_domains = ['bbc.com', 'reuters.com', 'cnn.com', 'pib.gov.in', 'indianexpress.com']
        suspicious_domains = ['fakenews.com', 'clickbait.net', 'conspiracy.org']
        
        if any(domain in content.lower() for domain in trusted_domains):
            verdict = "true"
            confidence = 0.90
            summary = "URL points to a trusted news source with high credibility."
        elif any(domain in content.lower() for domain in suspicious_domains):
            verdict = "false" 
            confidence = 0.85
            summary = "URL points to a source known for publishing misinformation."
        else:
            verdict = "inconclusive"
            confidence = 0.55
            summary = "URL credibility could not be definitively determined."
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "summary": summary,
            "sources": ["Domain reputation database", "Historical fact-check records"]
        }

class AnalysisEngine:
    """Main analysis orchestrator"""
    
    def __init__(self):
        self.ai_services = MockAIServices()
    
    def process_content(self, content_type: str, content: str, language: str = "en") -> Dict[str, Any]:
        """Process content and return analysis results"""
        start_time = time.time()
        
        # Route to appropriate analyzer
        if content_type == "text":
            result = self.ai_services.analyze_text(content)
        elif content_type == "image":
            result = self.ai_services.analyze_image(content)
        elif content_type == "url":
            result = self.ai_services.analyze_url(content)
        else:
            result = {
                "verdict": "inconclusive",
                "confidence": 0.0,
                "summary": f"Unsupported content type: {content_type}"
            }
        
        processing_time = round(time.time() - start_time, 2)
        
        # Create detailed analysis object
        detailed_analysis = DetailedAnalysis(
            evidence=result.get("evidence", []),
            sources=result.get("sources", []),
            gemini_analysis=result.get("summary", ""),
            vision_analysis=result.get("vision_analysis")
        )
        
        return {
            "verdict": result["verdict"],
            "confidence_score": result["confidence"],
            "summary": result["summary"],
            "processing_time": processing_time,
            "detailed_analysis": detailed_analysis
        }

# Global analysis engine instance
analysis_engine = AnalysisEngine()