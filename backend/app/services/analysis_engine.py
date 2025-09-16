import asyncio
import threading
from time import time
from typing import Dict, Any
from app.models import DetailedAnalysis
from app.services.text_service import analyze_text, get_mock_analysis_result

class AnalysisEngine:
    """Analysis engine with robust async handling and real API integration"""

    def process_content(self, content_type: str, content: str, language: str = "en") -> Dict[str, Any]:
        """Process content with real APIs when configured, fallback to mock"""
        start = time()

        if content_type == "text":
            # Try real analysis first, fallback to mock on error
            try:
                results = self._run_async_analysis(content, language)
            except Exception as e:
                print(f"Real API analysis failed: {e}, using mock")
                results = get_mock_analysis_result(content)
        else:
            # URL/image not implemented yet
            results = get_mock_analysis_result(content)

        # Smart aggregation based on real API results
        verdict, confidence, summary = self._aggregate_results(results)

        detailed_analysis = DetailedAnalysis(
            evidence=self._extract_evidence(results),
            sources=self._extract_sources(results),
            gemini_analysis=results.get("gemini", {}).get("normalized_claim", content),
            factcheck_results=results.get("factcheck", {}).get("verdicts", [])
        )

        return {
            "verdict": verdict,
            "confidence_score": confidence,
            "summary": summary,
            "processing_time": round(time() - start, 2),
            "detailed_analysis": detailed_analysis
        }

    def _run_async_analysis(self, content: str, language: str) -> Dict[str, Any]:
        """Run async analysis in a separate thread to avoid event loop conflicts"""
        result_container = {"result": None, "error": None}
        
        def run_in_thread():
            try:
                # Create new event loop in thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result_container["result"] = loop.run_until_complete(
                        analyze_text(content, language)
                    )
                finally:
                    loop.close()
            except Exception as e:
                result_container["error"] = e
        
        # Run in separate thread
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join(timeout=30)  # 30 second timeout
        
        if result_container["error"]:
            raise result_container["error"]
        if result_container["result"] is None:
            raise Exception("Analysis timed out")
            
        return result_container["result"]

    def _aggregate_results(self, results: Dict[str, Any]) -> tuple[str, float, str]:
        """Smart aggregation of API results"""
        # Check if this is mock data
        if results.get("metadata", {}).get("mock_data"):
            return "inconclusive", 0.65, "Mock analysis - configure API keys for real results"
        
        # Real API result processing
        verdict = "inconclusive"
        confidence = 0.5
        summary = "Analysis completed"

        try:
            # Priority 1: Fact-check results
            factcheck = results.get("factcheck", {})
            verdicts = factcheck.get("verdicts", [])
            
            for fact_verdict in verdicts:
                if fact_verdict.get("verdict") == "FALSE":
                    return "false", 0.95, f"Fact-check by {fact_verdict.get('source', 'fact-checker')} confirms this is FALSE"
                elif fact_verdict.get("verdict") == "TRUE":
                    return "true", 0.90, f"Fact-check by {fact_verdict.get('source', 'fact-checker')} confirms this is TRUE"
            
            # Priority 2: Gemini analysis
            gemini = results.get("gemini", {})
            if not gemini.get("mock"):
                gemini_confidence = gemini.get("confidence", 0.5)
                if gemini_confidence >= 0.8:
                    confidence = 0.80
                    summary = "High-confidence AI analysis suggests verification needed"
            
            # Priority 3: Toxicity detection
            perspective = results.get("perspective", {})
            if not perspective.get("mock"):
                toxicity = perspective.get("toxicity_score", 0.0)
                if toxicity >= 0.7:
                    verdict = "false"
                    confidence = 0.85
                    summary = "High toxicity detected - likely misinformation"

        except Exception as e:
            print(f"Error in aggregation: {e}")
        
        return verdict, round(confidence, 2), summary

    def _extract_evidence(self, results: Dict[str, Any]) -> list[str]:
        """Extract evidence from results"""
        evidence = []
        
        if results.get("metadata", {}).get("mock_data"):
            evidence.append("Mock analysis performed")
        else:
            factcheck = results.get("factcheck", {})
            if factcheck.get("fact_checks_found", 0) > 0:
                evidence.append(f"Found {factcheck['fact_checks_found']} fact-check(s)")
            
            perspective = results.get("perspective", {})
            if perspective.get("toxicity_score", 0) > 0.3:
                evidence.append(f"Toxicity: {perspective['toxicity_score']:.2f}")
        
        return evidence or ["Analysis completed"]

    def _extract_sources(self, results: Dict[str, Any]) -> list[str]:
        """Extract sources from results"""
        sources = []
        
        if results.get("metadata", {}).get("mock_data"):
            sources.append("Mock Data Service")
        else:
            if results.get("gemini", {}).get("status") == "success":
                sources.append("Google Gemini AI")
            if results.get("factcheck", {}).get("fact_checks_found", 0) > 0:
                sources.append("Google Fact Check API")
            if results.get("perspective", {}).get("status") == "success":
                sources.append("Google Perspective API")
        
        return sources or ["TruthLens Engine"]

# Global instance
analysis_engine = AnalysisEngine()
